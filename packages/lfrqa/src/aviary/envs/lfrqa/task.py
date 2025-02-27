from collections.abc import Awaitable, Callable

from lmi import CommonLLMNames, LLMModel
from paperqa import Docs, Settings

from aviary.core import TASK_DATASET_REGISTRY, TaskDataset

from .env import LFRQAPairwiseEvalEnv, LFRQAQuestion


class LFRQATaskDataset(TaskDataset[LFRQAPairwiseEvalEnv]):
    """Task dataset for custom evaluation of non-multiple choice questions."""

    def __init__(
        self,
        data: list[LFRQAQuestion],
        settings: Settings | dict | None = None,
        base_docs: Docs | dict | None = None,
        pairwise_eval_llm: LLMModel | str = CommonLLMNames.GPT_4O.value,
        evaluation_callback: Callable[[dict], Awaitable] | None = None,
    ):
        self.data = data
        self.pairwise_eval_llm = pairwise_eval_llm

        if settings is None:
            settings = Settings()
        if isinstance(settings, dict):
            settings = Settings(**settings)
        self._settings = settings
        if base_docs is None:
            base_docs = Docs()
        if isinstance(base_docs, dict):
            base_docs = Docs(**base_docs)
        self._base_docs = base_docs
        self._rewards = {"win": 1, "tie": 0, "lose": -1}
        self._evaluation_callback = evaluation_callback

    def get_new_env_by_idx(self, idx: int) -> LFRQAPairwiseEvalEnv:
        """Create a new environment instance for the given index."""
        return LFRQAPairwiseEvalEnv(
            query=self.data[idx],
            pairwise_eval_llm=self.pairwise_eval_llm,
            settings=self._settings,
            docs=self._base_docs.model_copy(),
            rewards=self._rewards,
            evaluation_callback=self._evaluation_callback,
        )

    def __len__(self) -> int:
        return len(self.data)


TASK_DATASET_NAME = "lfrqa"
TASK_DATASET_REGISTRY[TASK_DATASET_NAME] = (
    LFRQATaskDataset.__module__,
    LFRQATaskDataset.__name__,
)
