import pathlib

import pandas as pd
import pytest

from aviary.envs.lfrqa.env import LFRQAPairwiseEvalEnv, LFRQAQuestion
from aviary.envs.lfrqa.task import LFRQATaskDataset

TESTS_DIR = pathlib.Path(__file__).parent
STUB_DATA_DIR = TESTS_DIR / "stub_data"


@pytest.mark.asyncio
async def test_env_construction() -> None:
    data: list[LFRQAQuestion] = [
        LFRQAQuestion(**row)  # type: ignore[misc]
        for row in pd.read_csv(STUB_DATA_DIR / "mini_lfrqa.csv")[
            ["qid", "question", "answer", "gold_doc_ids"]
        ].to_dict(orient="records")
    ]

    dataset = LFRQATaskDataset(data=data)

    env = dataset.get_new_env_by_idx(0)  # noqa: FURB184
    assert isinstance(env, LFRQAPairwiseEvalEnv)
    assert await env.get_id() == "science-search-test-30"
