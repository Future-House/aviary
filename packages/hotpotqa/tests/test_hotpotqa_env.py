import re

import pytest

from aviary.core import Environment, TaskDataset
from aviary.envs.hotpotqa import HotPotQAEnv
from aviary.envs.hotpotqa.env import HotPotQADataset
from aviary.utils import EvalAnswerMode


def test_env_construction() -> None:
    hotpotqa_env: HotPotQAEnv = Environment.from_name(
        "hotpotqa",
        question=(
            "What is the formula for the volume of Abraham Lincoln's favorite hat?"
        ),
        correct_answer="pi*r^2*h",
    )
    assert isinstance(hotpotqa_env, HotPotQAEnv)


def test_dataset_from_name() -> None:
    dataset = TaskDataset.from_name("hotpotqa", split="dev")
    assert isinstance(dataset.get_new_env_by_idx(0), HotPotQAEnv)

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
    hotpotqa_env: HotPotQAEnv = Environment.from_name(
        "hotpotqa",
        question=("Which country has a larger population: China or France?"),
        correct_answer="China",
    )
    lookup_pattern = r"^\(Result \d+ / \d+\)\s*(.*)"

    _, _ = await hotpotqa_env.reset()
    obs1 = await hotpotqa_env.search("China")
    obs2 = hotpotqa_env.lookup("population")

    # Check lookup return format
    match = re.match(lookup_pattern, obs2)
    assert match  # Starts with the right pattern
    assert (
        match.group(1) + "\n" in hotpotqa_env.state.page
    )  # Everything after the pattern should be a paragraph in current page

    obs3 = await hotpotqa_env.search("France")
    obs4 = hotpotqa_env.lookup("population")

    # Check lookup return format
    match = re.match(lookup_pattern, obs4)
    assert match, "Expected lookup"
    assert match.group(1) + "\n" in hotpotqa_env.state.page, (
        "Expected text after the match to be a paragraph"
    )

    obs5 = hotpotqa_env.submit_answer("China")

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
        question="What is the reddest bridge in San Francisco?",
        correct_answer=correct_answer,
        evaluation_mode=evaluation_mode,
    )

    assert (await env.calculate_answer_reward(correct_answer)) == 1
    assert (await env.calculate_answer_reward(incorrect_answer)) == 0
