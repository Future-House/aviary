import pytest

from aviary.core import Environment, TaskDataset, ToolCall, ToolRequestMessage
from aviary.envs.gsm8k import CalculatorEnv, CalculatorEnvConfig


@pytest.mark.asyncio
async def test_calculator_env() -> None:
    problem = "What is the answer to the ultimate question of life, the universe, and everything?"
    answer = 42.0
    env = CalculatorEnv(
        problem_id="douglas-adams",
        problem=problem,
        answer=answer,
        config=CalculatorEnvConfig(
            correct_reward=1e4,
        ),
    )

    obs, tools = await env.reset()
    assert obs[0].content == problem
    assert len(tools) == 2

    # Run calculator
    response, reward, done, trunc = await env.step(
        ToolRequestMessage(tool_calls=[ToolCall.from_tool(tools[0], expr="4-3")])
    )
    assert not done
    assert reward == 0.0
    assert response[0].content == "1"

    # check answer
    response, reward, done, trunc = await env.step(
        ToolRequestMessage(tool_calls=[ToolCall.from_tool(tools[1], answer="42")])
    )
    assert reward == 1e4


def test_loading_from_name() -> None:
    env: CalculatorEnv = Environment.from_name(
        "calculator",
        problem_id="rhetorical",
        problem="I had a cake and I ate it. How many cakes do I have?",
        answer=0,
    )
    assert isinstance(env, CalculatorEnv)


@pytest.mark.parametrize(
    ("split", "first_answer"),
    [("train", 10.0), ("train_full", 72.0), ("val", 72.0), ("test", 18.0)],
)
def test_loading_gsm8k_from_name(split: str, first_answer: float) -> None:
    env = TaskDataset.from_name("gsm8k", split=split).get_new_env_by_idx(0)
    assert isinstance(env, CalculatorEnv)
    assert env.answer == first_answer
