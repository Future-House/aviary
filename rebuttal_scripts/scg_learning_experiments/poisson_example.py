import asyncio
import math
from collections.abc import Iterable
from typing import Any, cast

import numpy as np
import tree
import wandb
from ldp.graph.op_utils import CallID, compute_graph
from ldp.graph.ops import GradInType, Op, OpCtx, OpResult
from tqdm import tqdm


class FloatParamOp(Op):
    def __init__(self, init_param: float):
        self.param = init_param

    async def forward(self) -> float:
        return self.param

    @classmethod
    def backward(
        cls, ctx: OpCtx, input_args, input_kwargs, grad_output: Any, call_id: CallID
    ) -> GradInType:
        return [], {}


class PoissonSamplerOp(Op):
    @staticmethod
    def _probability(lam: float, k: int) -> float:
        if k < 0:
            # negative k can happen when taking a gradient
            return 0.0
        return np.exp(-lam) * lam**k / math.factorial(k)

    async def forward(self, lam: float) -> int:
        return np.random.poisson(max(0.01, lam))  # noqa: NPY002

    @classmethod
    def backward(
        cls,
        ctx: OpCtx,
        input_args,
        input_kwargs,  # forwards method kwargs
        grad_output: tree.Structure,
        call_id: CallID,
    ) -> GradInType:
        # This op has no internal parameters, so we just compute delta_{j,i}
        lam = max(0.01, input_kwargs["lam"])
        k = ctx.get(call_id, "output").value  # get output of the forward pass

        p_k = cls._probability(lam, k)
        p_km1 = cls._probability(lam, k - 1)

        # dp(k)/dlam
        grad_lam_p = p_km1 - p_k

        # d[lnp(k)]/dlam
        grad_lam_lnp = grad_lam_p / p_k

        # delta_{j,i}
        delta_lam = grad_lam_lnp

        # define dk/dlam in expectation: dE[k]/dlam = dlam/dlam = 1
        # grad_lam_k = 1.0

        # This term has no mathematical justification for now, refer to aviary notes for more details
        # delta_lam += (grad_lam_k * cast(float, grad_output))

        return [], {"lam": delta_lam}


class LossOp(Op):
    async def forward(self, prediction: float, target: float) -> float:
        return np.abs(prediction - target)

    @classmethod
    def backward(
        cls, ctx: OpCtx, input_args, input_kwargs, grad_output: Any, call_id: CallID
    ) -> GradInType:
        prediction = input_kwargs["prediction"]
        target = input_kwargs["target"]

        # d cost / d prediction
        if prediction > target:
            grad = 1.0
        elif prediction < target:
            grad = -1.0
        else:
            grad = 0.0

        return [], {"prediction": grad, "target": None}


class SGDOptimizer:
    def __init__(self, op: FloatParamOp, lr: float, lr_decay: float = 1.0):
        self.op = op
        self.lr = lr
        self.lr_decay = lr_decay
        self.accumulated_updates: list[float] = []

    def aggregate(self, samples: Iterable[OpResult]):
        for result in samples:
            call_ids = self.op.get_call_ids({result.call_id.run_id})
            grads = [
                cast(float, g)
                for g in (
                    self.op.ctx.get(call_id, "grad_output") for call_id in call_ids
                )
                if g is not None
            ]
            if not grads:
                continue

            loss = result.value
            self.accumulated_updates.append(loss * sum(grads))

    def update(self):
        self.op.param -= self.lr * cast(float, np.mean(self.accumulated_updates))
        self.accumulated_updates.clear()
        self.lr *= self.lr_decay


async def test_poisson_sgd(init_lam: float = 20):
    # Constants
    bsz = 8
    n_epochs = 3000
    target_lambda = 13

    # Graph Ops
    lam = FloatParamOp(init_lam)
    poisson = PoissonSamplerOp(target_lambda)
    loss = LossOp(target_lambda)
    opt = SGDOptimizer(lam, lr=0.01)

    # Create wandb logger
    logger = wandb.init(
        project="poisson_example",
        name="test",
        mode="online",
    )

    @compute_graph()
    async def fwd() -> OpResult[int]:
        return await loss(await poisson(await lam()), target_lambda)

    for _ in tqdm(range(n_epochs)):
        samples = await asyncio.gather(*[fwd() for _ in range(bsz)])
        training_batch = []
        training_loss = 0.0
        for k in samples:
            k.compute_grads()
            training_loss += k.value
            training_batch.append(k)
        training_loss /= bsz
        opt.aggregate(training_batch)
        grads = opt.update()
        logger.log({"loss": training_loss, "lambda": lam.param, "grads": grads})


if __name__ == "__main__":
    asyncio.run(test_poisson_sgd())
