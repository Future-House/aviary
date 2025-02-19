import os

import pandas as pd

from aviary.envs.lfrqa.env import LFRQAPairwiseEvalEnv, LFRQAQuestion
from aviary.envs.lfrqa.task import LFRQATaskDataset


def test_env_construction() -> None:
    data: list[LFRQAQuestion] = [
        LFRQAQuestion(**row)  # type: ignore[misc]
        for row in pd.read_csv(
            os.path.join("packages", "lfrqa", "tests", "stub_data", "mini_lfrqa.csv")
        )[["qid", "question", "answer", "gold_doc_ids"]].to_dict(orient="records")
    ]

    dataset = LFRQATaskDataset(
        data=data,
    )

    env = dataset.get_new_env_by_idx(0)
    assert isinstance(env, LFRQAPairwiseEvalEnv)
