import os
from typing import TYPE_CHECKING

from aviary.core import TASK_DATASET_REGISTRY
from aviary.envs.labbench import PaperQATaskDataset

from .env import LFRQAPairwiseEvalEnv, LFRQAQuestion

if TYPE_CHECKING:
    import pandas as pd


def read_csv(path: str | os.PathLike) -> "pd.DataFrame":
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "Reading in a CSV requires the 'csv' extra for 'pandas'. Please:"
            " `pip install aviary.lfrqa[csv]`."
        ) from exc
    return pd.read_csv(path)


class LFRQATaskDataset(PaperQATaskDataset[LFRQAPairwiseEvalEnv]):
    """Task dataset for custom evaluation of non-multiple choice questions."""

    def __init__(self, data: str | os.PathLike | list[LFRQAQuestion], **kwargs):
        super().__init__(**kwargs)
        self.data: "list[LFRQAQuestion] | pd.DataFrame" = (  # noqa: UP037
            data if isinstance(data, list) else read_csv(data)
        )

    def _make_query(self, idx: int) -> LFRQAQuestion:
        if isinstance(self.data, list):
            return self.data[idx]
        return LFRQAQuestion(  # type: ignore[call-arg]
            qid=self.data.iloc[idx].qid,
            question=self.data.iloc[idx].question,
            answer=self.data.iloc[idx].answer,
            gold_doc_ids=self.data.iloc[idx].gold_doc_ids,
            **(self._question_kwargs or {}),
        )

    def get_new_env_by_idx(self, idx: int) -> LFRQAPairwiseEvalEnv:
        """Create a new environment instance for the given index."""
        return LFRQAPairwiseEvalEnv(
            query=self._make_query(idx),
            settings=self._settings,
            docs=self._base_docs.model_copy(),
            **self._env_kwargs,
        )

    def __len__(self) -> int:
        return len(self.data)


TASK_DATASET_NAME = "lfrqa"
TASK_DATASET_REGISTRY[TASK_DATASET_NAME] = (
    LFRQATaskDataset.__module__,
    LFRQATaskDataset.__name__,
)
