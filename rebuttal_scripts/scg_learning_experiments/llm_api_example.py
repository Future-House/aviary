import asyncio
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import wandb
from aviary.message import Message
from ldp.graph.common_ops import EmbeddingOp, LLMCallOp
from ldp.graph.gradient_estimators import TorchParamBackwardEstimator
from ldp.graph.loss_ops import MSELossOp
from ldp.graph.op_utils import CallID, compute_graph
from ldp.graph.ops import (
    GradInType,
    Op,
    OpCtx,
    OpResult,
)
from ldp.graph.torch_ops import TorchOp
from tqdm import tqdm


@dataclass
class Config:
    embedding_size: int = 512
    batch_size: int = 8
    pretrain_learning_rate: float = 0.0005
    epoch_pretrain_iters: int = 10
    initial_train_iters: int = 2
    epoch_train_iters: int = 10
    num_epochs: int = 1000
    train_learning_rate: float = 10.0
    logging_mode: str = "offline"
    prompt_str: str = "Write 10 times HelloWorld"
    target_str: str = "HelloWorld " * 10
    start_max_tokens: int = 4
    range_max_tokens: int = 2


class ConfigOp(Op):
    def __init__(self, max_tokens: float = 2.0, range_max_tokens: int = 2):
        self.max_tokens = max_tokens
        self.range_max_tokens = range_max_tokens

    async def forward(self, train=True) -> dict:
        max_tokens = int(round(self.max_tokens))
        if train:
            max_tokens = random.randint(
                max_tokens - self.range_max_tokens, max_tokens + self.range_max_tokens
            )

        return {
            "name": "gpt-4o-mini-2024-07-18",
            "max_completion_tokens": max_tokens,
        }

    @classmethod
    def backward(
        cls, ctx: OpCtx, input_args, input_kwargs, grad_output: Any, call_id: CallID
    ) -> GradInType:
        return [], {}


class FixedPromptOp(Op):
    def __init__(self, prompt: str = "tell me a joke"):
        self.prompt = prompt

    async def forward(self) -> list[Message]:
        return [Message(content=self.prompt, role="user")]

    @classmethod
    def backward(
        cls, ctx: OpCtx, input_args, input_kwargs, grad_output: Any, call_id: CallID
    ) -> GradInType:
        return [], {}


class SurrogateNet(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_size + 1, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, output_size),
        )

    def forward(self, x: torch.Tensor, max_tokens: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x, max_tokens], dim=-1)  # Concatenate along the last dimension
        return self.layers(x)


class SurrogateAdamOptimizer:
    def __init__(self, mlp_op: TorchOp, lr: float = 0.01):
        self.lr = lr
        self.mlp_op = mlp_op
        self.internal_optim = torch.optim.Adam(mlp_op.module.parameters(), lr=lr)
        self.parameters = list(mlp_op.module.parameters())

    def aggregate(self, samples: list[OpResult]):
        for result in samples:
            call_ids = self.mlp_op.get_call_ids({result.call_id.run_id})
            grads_params = self.mlp_op.ctx.get(next(iter(call_ids)), "grad_params")
            for param, grad in zip(self.parameters, grads_params.values(), strict=True):
                if param.grad is None:
                    param.grad = grad.to(param.device)
                else:
                    param.grad += grad.to(param.device)

        for param in self.parameters:
            if param.grad is not None:
                param.grad = param.grad / float(len(samples))

    def update(self):
        self.internal_optim.step()
        self.internal_optim.zero_grad()


class MaxTokensSGDOptimizer:
    def __init__(self, param_op: ConfigOp, torch_op: TorchOp, lr: float = 0.1):
        self.param_op = param_op
        self.torch_op = torch_op
        self.lr = lr
        self.accumulated_updates: list[float] = []

    def aggregate(self, samples: list[OpResult]):
        for result in samples:
            call_ids = self.torch_op.get_call_ids({result.call_id.run_id})
            grad_input = self.torch_op.ctx.get(next(iter(call_ids)), "grad_input")
            grad_max_tokens = grad_input[1].get("max_tokens")
            self.accumulated_updates.append(grad_max_tokens)

    def update(self):
        gradient = np.mean(self.accumulated_updates)
        self.param_op.max_tokens -= self.lr * float(gradient)
        self.param_op.max_tokens = max(1.0, self.param_op.max_tokens)
        self.accumulated_updates.clear()
        return float(gradient)


async def test_llm_graph():  # noqa: PLR0915
    config = Config()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Create graph operations
    llm_config_op = ConfigOp(
        max_tokens=config.start_max_tokens, range_max_tokens=config.range_max_tokens
    )
    prompt_op = FixedPromptOp(config.prompt_str)
    llm_op = LLMCallOp()
    mlp_module = SurrogateNet(config.embedding_size, config.embedding_size).to(device)
    surrogate_op = TorchOp(mlp_module)
    surrogate_op.set_name("surrogate_op")
    embedding_op = EmbeddingOp(dense_embedding_dim=config.embedding_size)
    loss_op = MSELossOp()

    # Create gradient estimator
    estimator = TorchParamBackwardEstimator(mlp_module)

    # Create optimizers
    mlp_optim = SurrogateAdamOptimizer(surrogate_op, lr=config.pretrain_learning_rate)
    max_tokens_optim = MaxTokensSGDOptimizer(
        llm_config_op, surrogate_op, lr=config.train_learning_rate
    )

    # Create wandb logger
    logger = wandb.init(
        config=asdict(config),
        project="llm_graph_learning",
        name="llm_graph_surrogate",
        mode="offline",
    )

    @compute_graph()
    async def fwd_llm(numpy_prompt_embed) -> OpResult[int]:
        # Get prompt and config
        llm_config = await llm_config_op()
        prompt_embed = torch.tensor(
            numpy_prompt_embed, device=device, requires_grad=True
        )
        max_tokens = torch.tensor(
            llm_config.value["max_completion_tokens"],
            device=device,
            dtype=torch.float32,
            requires_grad=True,
        ).unsqueeze(-1)

        # Get llm completion
        target = await llm_op(llm_config, await prompt_op())
        target = await embedding_op(target.value.content)
        target_embed = torch.tensor(target.value, device=device, requires_grad=False)

        # Get MLP prediction
        pred = await surrogate_op(prompt_embed, max_tokens)
        return await loss_op(pred, target_embed)

    @compute_graph()
    async def fwd_surrogate(prompt_embed, target_embed) -> OpResult[int]:
        # Get prompt, target and config
        llm_config = await llm_config_op()
        prompt_embed = torch.tensor(prompt_embed, device=device, requires_grad=True)
        target_embed = torch.tensor(target_embed, device=device, requires_grad=False)
        max_tokens = torch.tensor(
            llm_config.value["max_completion_tokens"],
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        ).unsqueeze(-1)

        # Get MLP prediction
        pred = await surrogate_op(prompt_embed, max_tokens)
        return await loss_op(pred, target_embed)

    # Get the embeddings for the prompt and target strings
    prompt_embed = await embedding_op(config.prompt_str)
    prompt_embed = prompt_embed.value
    target_embed = await embedding_op(config.target_str)
    target_embed = target_embed.value

    for epoch in range(config.num_epochs):
        # Phase 1: Distill the LLM behavior with the surrogate network for the current max tokens value
        epoch_pretrain_iters = (
            config.epoch_pretrain_iters if epoch != 0 else config.initial_train_iters
        )
        pbar = tqdm(total=epoch_pretrain_iters, desc=f"Epoch {epoch + 1}, Phase 1")
        for _ in range(epoch_pretrain_iters):
            # Run forward pass
            start_forward = time.time()
            losses = await asyncio.gather(*[
                fwd_llm(prompt_embed) for _ in range(config.batch_size)
            ])
            forward_time = time.time() - start_forward

            # Run backward pass
            training_batch = []
            batch_loss = 0.0
            start_backward = time.time()
            for k in losses:
                k.compute_grads(backward_fns={"surrogate_op": estimator.backward})
                training_batch.append(k)
                batch_loss += k.value
            backward_time = time.time() - start_backward

            # Optimizer step
            batch_loss /= len(losses)
            mlp_optim.aggregate(training_batch)
            mlp_optim.update()

            # Log the results
            pbar.update()
            logger.log({
                "pretrain/loss": batch_loss,
                "pretrain/forward_time": forward_time,
                "pretrain/backward_time": backward_time,
            })

            # Clear operation contexts
            OpCtx.clear_contexts()

        # Phase 2: Update the max_tokens parameter with the surrogate network
        pbar = tqdm(total=config.epoch_train_iters, desc=f"Epoch {epoch + 1}, Phase 2")
        for _ in range(config.epoch_train_iters):
            # Run forward pass
            start_forward = time.time()
            losses = await asyncio.gather(*[
                fwd_surrogate(prompt_embed, target_embed)
                for _ in range(config.batch_size)
            ])
            forward_time = time.time() - start_forward

            # Run backward pass
            batch_loss = 0.0
            start_backward = time.time()
            training_batch = []
            for k in losses:
                k.compute_grads()
                training_batch.append(k)
                batch_loss += k.value
            backward_time = time.time() - start_backward

            # Optimizer step
            batch_loss /= len(losses)
            max_tokens_optim.aggregate(training_batch)
            gradient = max_tokens_optim.update()

            # Log the results
            pbar.update()
            logger.log({
                "train/loss": batch_loss,
                "train/forward_time": forward_time,
                "train/backward_time": backward_time,
                "train/max_tokens": llm_config_op.max_tokens,
                "train/gradient": gradient,
            })

            # Clear operation contexts
            OpCtx.clear_contexts()


if __name__ == "__main__":
    asyncio.run(test_llm_graph())
