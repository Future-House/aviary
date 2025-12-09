import asyncio
import time
from copy import deepcopy
from typing import cast
from unittest.mock import patch

import pytest
from aviary.core import MultipleChoiceEvaluation, ToolCall, ToolRequestMessage
from paperqa import Docs, Settings
from paperqa.agents import get_directory_index
from paperqa.agents.tools import Complete, GatherEvidence
from paperqa.prompts import CANNOT_ANSWER_PHRASE

from aviary.envs.labbench import GradablePaperQAEnvironment, make_discounted_returns


@pytest.fixture(name="stub_gradable_env")
def fixture_stub_gradable_env(
    agent_test_settings: Settings,
) -> GradablePaperQAEnvironment:
    return GradablePaperQAEnvironment(
        query="How can you use XAI for chemical property prediction?",
        settings=agent_test_settings,
        docs=Docs(),
    )


@pytest.mark.parametrize(
    ("evaluation", "expected_dreturns"),
    [
        (MultipleChoiceEvaluation.CORRECT, [0.25, 0.5, 1.0]),
        (MultipleChoiceEvaluation.INCORRECT, [-0.25, -0.5, -1.0]),
        (MultipleChoiceEvaluation.UNSURE, [0.025, 0.05, 0.1]),
    ],
)
def test_make_discounted_returns(
    evaluation: MultipleChoiceEvaluation, expected_dreturns: list[float]
) -> None:
    assert (
        make_discounted_returns(evaluation, num_steps=3, discount=0.5)
        == expected_dreturns
    )


class TestGradablePaperQAEnvironment:
    @pytest.mark.flaky(reruns=2, only_rerun=["AssertionError"])
    @pytest.mark.asyncio
    async def test_deepcopy_env(
        self,
        agent_test_settings: Settings,
        stub_gradable_env: GradablePaperQAEnvironment,
    ) -> None:
        await get_directory_index(settings=agent_test_settings)  # Trigger build

        # 1. Rollout until after gather evidence
        await stub_gradable_env.reset()
        for tool_call in (
            ToolCall.from_name(
                "paper_search",
                query="XAI for chemical property prediction",
                min_year=2018,
                max_year=2024,
            ),
            ToolCall.from_name(
                "gather_evidence", question=cast("str", stub_gradable_env._query)
            ),
        ):
            await stub_gradable_env.step(ToolRequestMessage(tool_calls=[tool_call]))

        # 2. Now we deepcopy the environment
        stub_gradable_env_copy = deepcopy(stub_gradable_env)
        assert stub_gradable_env.state == stub_gradable_env_copy.state

        # 3. Generate an answer and complete for both, and confirm they are identical
        gen_answer_action = ToolRequestMessage(
            tool_calls=[ToolCall.from_name("gen_answer")]
        )
        await stub_gradable_env.step(gen_answer_action)
        _, _, done, _ = await stub_gradable_env.step(
            ToolRequestMessage(
                tool_calls=[ToolCall.from_name("complete", has_successful_answer=True)]
            )
        )
        assert done
        assert len(stub_gradable_env.state.session.answer) > 10, "Expected an answer"
        assert stub_gradable_env.state.session.used_contexts
        await stub_gradable_env_copy.step(gen_answer_action)
        _, _, done, _ = await stub_gradable_env_copy.step(
            ToolRequestMessage(
                tool_calls=[ToolCall.from_name("complete", has_successful_answer=True)]
            )
        )
        assert done
        assert len(stub_gradable_env_copy.state.session.answer) > 10, (
            "Expected an answer"
        )
        assert stub_gradable_env_copy.state.session.used_contexts
        assert sorted(stub_gradable_env.state.session.used_contexts) == sorted(
            stub_gradable_env_copy.state.session.used_contexts
        )
        assert stub_gradable_env.state.session.tool_history == ([
            ["paper_search"],
            ["gather_evidence"],
            ["gen_answer"],
            ["complete"],
        ]), "Correct tool history was not saved in the session."
        assert stub_gradable_env_copy.state.query_tool_history("gen_answer"), (
            "Expected gen_answer tool to be in tool history"
        )

    @pytest.mark.asyncio
    async def test_empty_tool_calls(
        self, stub_gradable_env: GradablePaperQAEnvironment
    ) -> None:
        await stub_gradable_env.reset()
        obs, _, done, truncated = await stub_gradable_env.step(ToolRequestMessage())
        assert len(obs) == 1
        assert obs[0].content
        assert "no tool calls" in obs[0].content.lower()
        assert not done
        assert not truncated

    @pytest.mark.asyncio
    async def test_unsure_answer(
        self,
        agent_test_settings: Settings,
        stub_gradable_env: GradablePaperQAEnvironment,
    ) -> None:
        reset_obs, tools = await stub_gradable_env.reset()

        # 1. Immediately call gen_answer without paper search/evidence gathering
        answer_action = ToolRequestMessage(
            tool_calls=[ToolCall.from_name("gen_answer")]
        )
        answer_obs, _, done, truncated = await stub_gradable_env.step(answer_action)
        assert len(answer_obs) == 1
        assert answer_obs[0].content
        assert CANNOT_ANSWER_PHRASE in answer_obs[0].content
        assert not done
        assert not truncated

        # 2. Check this leads to us being unsure
        complete_action = await agent_test_settings.get_llm().select_tool(
            [*reset_obs, answer_action, *answer_obs],
            tools=tools,
            tool_choice=next(
                filter(lambda x: x.info.name == Complete.TOOL_FN_NAME, tools)
            ),
        )
        assert len(complete_action.tool_calls) == 1
        assert complete_action.tool_calls[0].function.arguments == {
            "has_successful_answer": False
        }, "Expected unsure"

    @pytest.mark.asyncio
    async def test_sequential_tool_calls(
        self, stub_gradable_env: GradablePaperQAEnvironment
    ) -> None:
        SLEEP_TIME = 2.0

        async def fake_gather_evidence(*args, **kwargs) -> str:  # noqa: ARG001
            await asyncio.sleep(SLEEP_TIME)
            return "fake evidence"

        _, tools = await stub_gradable_env.reset()

        gather_tool = next(
            tool for tool in tools if tool.info.name == GatherEvidence.TOOL_FN_NAME
        )

        with patch.object(gather_tool, "_tool_fn", fake_gather_evidence):
            tic = time.time()
            await stub_gradable_env.step(
                ToolRequestMessage(
                    tool_calls=[
                        ToolCall.from_name(
                            "gather_evidence",
                            question="XAI for chemical property prediction",
                        ),
                        ToolCall.from_name(
                            "gather_evidence",
                            question="XAI for chemical property prediction",
                        ),
                    ]
                )
            )

            assert time.time() - tic > 2 * SLEEP_TIME  # since they are sequential
