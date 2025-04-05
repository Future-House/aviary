import logging
from pathlib import Path
from typing import ClassVar, Self

from aviary_internal import utils
from aviary_internal.expts.sn.rollout import SimpleAgentRolloutExpt
from ldp.agent.simple_agent import SimpleAgent
from ldp.graph.common_ops import LLMCallOp
from protein_stability.dataset import (
    DEFAULT_DATA_REPO_NAME,
    ProteinStabilityDataset,
    ProteinStabilityDatasetSplit,
)
from protein_stability.environment import ProteinStabilityEnvConfig
from pydantic import Field, model_validator


class StabilityDatasetConfig(ProteinStabilityEnvConfig):
    source_data: utils.DataRepo = Field(
        default_factory=lambda: utils.DataRepo(name=DEFAULT_DATA_REPO_NAME)
    )


class EvalStabilityDatasetConfig(StabilityDatasetConfig):
    split: ProteinStabilityDatasetSplit


class StabilitySimpleAgentRolloutExpt(SimpleAgentRolloutExpt):
    accuracy_threshold: ClassVar[float] = 0.00001  # small non-zero value

    env: EvalStabilityDatasetConfig

    @model_validator(mode="after")
    def set_log_level(self) -> Self:
        configure_expt_logs()
        return self

    async def make_dataset(self) -> ProteinStabilityDataset:
        return ProteinStabilityDataset(
            workspace_root=Path(self.output_repo.local_path) / "workspaces",
            split=self.env.split,
            source_data=self.env.source_data,
            env_config=self.env,
        )

    def make_agent(self, **kwargs) -> SimpleAgent:
        agent = super().make_agent(**kwargs)

        # hot-fix for messed up Anthropic tool requests. Upstream TODO: figure out
        # how SimpleAgent can configure this and/or figure out if the root cause is
        # fixable.
        if (
            "claude" in self.agent.llm_model.name
            and agent._llm_call_op.response_validator is None
        ):
            agent._llm_call_op.response_validator = (
                LLMCallOp.anthropic_response_validator
            )

        return agent


def configure_expt_logs():
    """Sets good default logging levels for experiments."""
    for mod in ("environment", "rewards", "tools"):
        utils.configure_logs(stdout_level=(f"protein_stability.{mod}", logging.WARNING))
