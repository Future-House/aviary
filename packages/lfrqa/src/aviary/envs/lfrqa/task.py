from collections.abc import Awaitable, Callable

from llmclient import CommonLLMNames, LLMModel
from paperqa.settings import Settings

from aviary.core import TASK_DATASET_REGISTRY, TaskDataset

from .env import LFRQAPairwiseEvalEnv, LFRQAQuestion


class LFRQATaskDataset(TaskDataset[LFRQAPairwiseEvalEnv]):
    """Task dataset for custom evaluation of non-multiple choice questions."""

    def __init__(
        self,
        data: list[LFRQAQuestion],
        settings: Settings | dict | None = None,
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
        self._rewards = {"win": 1, "tie": 0, "lose": -1}
        self._evaluation_callback = evaluation_callback

    def get_new_env_by_idx(self, idx: int) -> LFRQAPairwiseEvalEnv:
        """Create a new environment instance for the given index."""
        question = self.data[idx]

        return LFRQAPairwiseEvalEnv(
            qid=question.qid,
            question=question.question,
            human_answer=question.answer,
            gt_doc_ids=question.gt_doc_ids,
            pairwise_eval_llm=self.pairwise_eval_llm,
            settings=self._settings,
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
