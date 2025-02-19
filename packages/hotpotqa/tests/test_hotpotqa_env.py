import os
import re
from uuid import UUID

import pytest

from aviary.core import Environment, TaskDataset
from aviary.envs.hotpotqa import HotPotQAEnv
from aviary.envs.hotpotqa.env import HotPotQADataset
from aviary.utils import EvalAnswerMode


def test_env_construction() -> None:
    hotpotqa_env: HotPotQAEnv = Environment.from_name(  # type: ignore[assignment]
        "hotpotqa",
        question_id=None,
        question=(
            "What is the formula for the volume of Abraham Lincoln's favorite hat?"
        ),
        correct_answer="pi*r^2*h",
    )
    assert isinstance(hotpotqa_env, HotPotQAEnv)


IN_GITHUB_ACTIONS: bool = os.getenv("GITHUB_ACTIONS") == "true"


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="Slow test")
def test_question_id_uniqueness() -> None:
    raw_dataset = HotPotQADataset.load_raw(split="dev")
    raw_ds_ids: set[str] = set()
    raw_ds_ids.update(row["id"] for row in raw_dataset)

    dataset = TaskDataset.from_name("hotpotqa", split="dev")

    question_ids: set[UUID] = set()
    for i in range(len(dataset)):
        env = dataset.get_new_env_by_idx(i)
        assert isinstance(env, HotPotQAEnv)
        assert isinstance(env.question_id, UUID)
        question_ids.add(env.question_id)

    assert len(raw_ds_ids) == len(dataset) == len(question_ids) == 7405, (
        'Expected 7405 examples in "dev" split'
    )
    converted_back_question_ids = {
        str(qid)[8:].replace("-", "") for qid in question_ids
    }
    assert converted_back_question_ids == raw_ds_ids, (
        "Should be able to restore original HotPotQA question ID"
    )


def test_dataset_from_name() -> None:
    dataset = TaskDataset.from_name("hotpotqa", split="dev")
    env_0 = dataset.get_new_env_by_idx(0)
    assert isinstance(dataset.get_new_env_by_idx(0), HotPotQAEnv)
    assert isinstance(env_0.question_id, UUID)

    # double-check we can load with various options
    dataset = TaskDataset.from_name(
        "hotpotqa",
        split="train",
        difficulty_level={"easy", "hard"},
        evaluation_mode=EvalAnswerMode.EXACT,
    )
    assert isinstance(dataset, HotPotQADataset)
    assert len(dataset) == 33633, 'Expected 33633 examples in "train[hard+easy]" split'
    assert dataset.get_new_env_by_idx(0).evaluation_mode == EvalAnswerMode.EXACT, (
        "evaluation_mode did not propagate to environment"
    )

    with pytest.raises(ValueError, match="answer"):
        TaskDataset.from_name("hotpotqa", split="test")


@pytest.mark.asyncio
async def test_tool_results() -> None:
    hotpotqa_env: HotPotQAEnv = Environment.from_name(  # type: ignore[assignment]
        "hotpotqa",
        question_id=None,
        question="Which country has a larger population: China or France?",
        correct_answer="China",
    )
    lookup_pattern = r"^\(Result \d+ / \d+\)\s*(.*)"

    _, _ = await hotpotqa_env.reset()
    obs1 = await hotpotqa_env.search("China")
    obs2 = hotpotqa_env.lookup("population")

    # Check lookup return format
    match = re.match(lookup_pattern, obs2)
    assert match  # Starts with the right pattern
    assert match.group(1) + "\n" in (
        hotpotqa_env.state.page or ""
    )  # Everything after the pattern should be a paragraph in current page

    obs3 = await hotpotqa_env.search("France")
    obs4 = hotpotqa_env.lookup("population")

    # Check lookup return format
    match = re.match(lookup_pattern, obs4)
    assert match, "Expected lookup"
    assert match.group(1) + "\n" in (hotpotqa_env.state.page or ""), (
        "Expected text after the match to be a paragraph"
    )

    obs5 = await hotpotqa_env.submit_answer("China")

    # Ensure that the observations are different
    assert obs1 != obs2 != obs3 != obs4 != obs5


@pytest.mark.parametrize(
    "evaluation_mode",
    [EvalAnswerMode.EXACT, EvalAnswerMode.CONTAINS, EvalAnswerMode.LLM],
)
@pytest.mark.asyncio
async def test_answer_evaluation_mode(evaluation_mode: EvalAnswerMode) -> None:
    correct_answer = "Golden Gate Bridge"
    incorrect_answer = "Bay Bridge"
    env = HotPotQAEnv(
        question_id=None,
        question="What is the reddest bridge in San Francisco?",
        correct_answer=correct_answer,
        evaluation_mode=evaluation_mode,
    )

    assert (await env.calculate_answer_reward(correct_answer)) == 1
    assert (await env.calculate_answer_reward(incorrect_answer)) == 0
