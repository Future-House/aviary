import pathlib
from unittest.mock import MagicMock

import pandas as pd
import pytest
from aviary.core import Environment
from ldp.alg import MeanMetricsCallback

from aviary.envs.lfrqa.env import LFRQAPairwiseEvalEnv, LFRQAQuestion
from aviary.envs.lfrqa.task import LFRQATaskDataset

TESTS_DIR = pathlib.Path(__file__).parent
STUB_DATA_DIR = TESTS_DIR / "stub_data"
MINI_LFRQA_CSV = STUB_DATA_DIR / "mini_lfrqa.csv"


def test_availability() -> None:
    assert "lfrqa" in Environment.available()


@pytest.fixture(name="mini_lfrqa", scope="module")
def fixture_mini_lfrqa() -> list[LFRQAQuestion]:
    return [
        LFRQAQuestion(**row)  # type: ignore[misc]
        for row in pd.read_csv(MINI_LFRQA_CSV)[
            ["qid", "question", "answer", "gold_doc_ids"]
        ].to_dict(orient="records")
    ]


@pytest.mark.parametrize("test_csv_read", [False, True])
@pytest.mark.asyncio
async def test_env_construction(
    test_csv_read: bool, mini_lfrqa: list[LFRQAQuestion]
) -> None:
    spy_callback = MagicMock()
    dataset = LFRQATaskDataset(
        data=MINI_LFRQA_CSV if test_csv_read else mini_lfrqa,
        evaluation_callback=spy_callback,
    )
    MeanMetricsCallback(eval_dataset=dataset)  # Confirm we could use this
    assert len(dataset) == 4

    env = dataset.get_new_env_by_idx(0)  # noqa: FURB184
    assert isinstance(env, LFRQAPairwiseEvalEnv)
    assert await env.get_id() == "science-search-test-30"
    assert env._evaluation_callback == spy_callback
