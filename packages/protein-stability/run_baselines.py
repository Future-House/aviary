import asyncio
import json
from pathlib import Path

from aviary_internal.expts import baseline_eval
from aviary_internal.utils.storage import DataRepo
from ldp.alg.runners import EvaluatorConfig


async def ProteinBaselineEvalExpt(tmpdir) -> None:  # noqa: N802
    outdir = Path(tmpdir)
    eval_cfg = EvaluatorConfig(batch_size=40, max_rollout_steps=20)

    # Note that in most cases, users would call:
    # `run_expt aviary_internal.expts.baseline_eval.BaselineEvalExpt`
    # The average user would not instantiate the expt manually like we do here.
    expt = baseline_eval.BaselineEvalExpt(
        output_repo=DataRepo(
            name="proteincrow/aviary-storage/baseline_eval/pcrow_paper_1pd5",
            local_path=str(outdir),
        ),
        evaluator=eval_cfg,
        datasets={
            "proteincrow": {"enabled": True},
        },
        num_replicates=1,
    )

    await expt.run()

    jsons = list(outdir.rglob("*.json"))
    # 2 models x 2 agent architectures = 4

    for json_path in jsons:
        with json_path.open() as f:
            metrics = json.load(f)
        assert isinstance(metrics, dict), "Metrics should be stored as a dict"
        assert metrics, "Metrics should not be empty"


if __name__ == "__main__":
    tmpdir = Path("/Users/manu/1pd5_run_small/")
    asyncio.run(ProteinBaselineEvalExpt(tmpdir))
