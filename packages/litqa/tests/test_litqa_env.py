import asyncio
import time
from collections.abc import Iterable
from copy import deepcopy
from typing import ClassVar, cast
from unittest.mock import patch

import pytest
from ldp.agent import SimpleAgent
from ldp.alg.callbacks import Callback, MeanMetricsCallback, StoreTrajectoriesCallback
from ldp.alg.runners import Evaluator, EvaluatorConfig
from paperqa import Docs, Settings
from paperqa.agents import get_directory_index
from paperqa.agents.env import PaperQAEnvironment
from paperqa.agents.tools import Complete, GatherEvidence, GenerateAnswer
from paperqa.types import PQASession
from pytest_subtests import SubTests

from aviary.core import (
    TASK_DATASET_REGISTRY,
    MultipleChoiceEvaluation,
    MultipleChoiceQuestion,
    TaskConfig,
    TaskDataset,
    ToolCall,
    ToolRequestMessage,
)
from aviary.envs.litqa import (
    DEFAULT_REWARD_MAPPING,
    GradablePaperQAEnvironment,
    LitQATaskDataset,
    LitQAv2TaskDataset,
    LitQAv2TaskSplit,
    make_discounted_returns,
    read_litqa_v2_from_hub,
)


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


def test_creating_litqa_questions() -> None:
    """Test making LitQA eval questions after downloading from Hugging Face Hub."""
    eval_split = read_litqa_v2_from_hub(seed=42)[1]
    assert len(eval_split) > 3
    assert [
        MultipleChoiceQuestion(
            question=cast(str, row.question),
            options=cast(list[str], row.distractors),
            ideal_answer=cast(str, row.ideal),
            shuffle_seed=42,
        ).question_prompt
        for row in eval_split[:3].itertuples()
    ] == [
        (
            "Q: Which of the following mutations in yeast Pbs2 increases its"
            " interaction with SH3?\n\nOptions:\nA) P97A\nB) N92S\nC) Insufficient"
            " information to answer this question\nD) K85W\nE) N92H\nF) I87W\nG) S83F"
        ),
        (
            "Q: What percentage of colorectal cancer-associated fibroblasts typically"
            " survive at 2 weeks if cultured with the platinum-based chemotherapy"
            " oxaliplatin?\n\nOptions:\nA) Insufficient information to answer this"
            " question\nB) 0%\nC) 50-80%\nD) 20-50%\nE) 1-20%\nF) 80-99%"
        ),
        (
            "Q: Which of the following genes shows the greatest difference in gene"
            " expression between homologous cell types in mouse and human"
            " brain?\n\nOptions:\nA) Htr1d\nB) Insufficient information to answer this"
            " question\nC) Htr6\nD) Htr5a\nE) Htr3a"
        ),
    ]


class StubLitQADataset(LitQATaskDataset):
    """Made up dataset of questions answerable from this repo's stub_data."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data: list[tuple[str, str | list[str], str, str]] = [
            (
                "Politician",
                ["Technologist", "Plumber"],
                "Who is Frederick Bates?",
                "bates.txt",
            ),
            (
                "Make molecular counterfactuals",
                [
                    "Generating images of cats",
                    "Simple explanations of internet searches",
                ],
                "How can you use XAI for chemical property prediction?",
                "paper.pdf",
            ),
            (
                "Maple Leaf",
                ["The Stars and Stripes", "The Blue and Yellow", "The Southern Cross"],
                "What is the national flag of Canada?",
                "flag_day.html",
            ),
        ]

    def get_new_env_by_idx(self, idx: int) -> GradablePaperQAEnvironment:
        return self._make_gradable_environment(
            ideal_answer=self.data[idx][0],
            distractors=self.data[idx][1],
            question=self.data[idx][2],
            sources=self.data[idx][3],
        )

    def __len__(self) -> int:
        return len(self.data)


STUB_TASK_DATASET_NAME = "stub-litqa"
TASK_DATASET_REGISTRY[STUB_TASK_DATASET_NAME] = (
    StubLitQADataset.__module__,
    StubLitQADataset.__name__,
)


class StoreEnvCallback(Callback):
    """Test utility to store instantiated environments."""

    def __init__(self):
        super().__init__()
        # NOTE: using question-to-env because too lazy to implement __hash__
        # for this being a set of envs
        self.query_to_envs: dict[str, PaperQAEnvironment] = {}

    async def before_rollout(self, traj_id: str, env) -> None:
        self.query_to_envs[
            env._query if isinstance(env._query, str) else env._query.question_prompt
        ] = env


class TestTaskDataset:
    EXPECTED_LENGTHS: ClassVar[tuple[int, ...]] = (159, 40, 49)

    @pytest.mark.parametrize(
        ("split", "expected_length"),
        [
            (LitQAv2TaskSplit.TRAIN, 159),
            (LitQAv2TaskSplit.EVAL, 40),
            (LitQAv2TaskSplit.TEST, 49),
        ],
    )
    @pytest.mark.asyncio
    async def test___len__(
        self,
        split: LitQAv2TaskSplit,
        expected_length: int,
        agent_task_settings: Settings,
    ) -> None:
        task_dataset = LitQAv2TaskDataset(
            settings=agent_task_settings,
            question_kwargs={"shuffle_seed": 42},
            read_data_kwargs={"seed": 42},
            split=split,
        )
        assert (
            len(task_dataset)
            == expected_length
            == self.EXPECTED_LENGTHS[split.get_index()]
        )

        # Now let's check we could use the sources in a validation
        for i in range(len(task_dataset)):
            env = task_dataset.get_new_env_by_idx(i)
            if i == 0 and split == LitQAv2TaskSplit.TRAIN:
                # Yes this assertion is somewhat brittle, but it reliably
                # checks the seeding's behavior so we keep it
                obs, _ = await env.reset()
                assert (
                    "Q: SLC14A1 been identified as a specific marker for endothelial"
                    " cells in which organ?\n\nOptions:\nA) liver\nB) Insufficient"
                    " information to answer this question\nC) prostate\nD) eye\nE)"
                    " heart" in (obs[0].content or "")
                )
            assert env.sources, "Sources need to be accessible"
            assert isinstance(env.sources, Iterable), (
                "Sources need to be at least iterable"
            )

    @pytest.mark.asyncio
    async def test_can_validate_stub_dataset_sources(
        self, agent_task_settings: Settings
    ) -> None:
        ds = StubLitQADataset(settings=agent_task_settings)
        await asyncio.gather(
            *(ds.get_new_env_by_idx(i).validate_sources() for i in range(len(ds)))
        )

    @pytest.mark.asyncio
    async def test_evaluation(
        self, subtests: SubTests, agent_task_settings: Settings
    ) -> None:
        await get_directory_index(settings=agent_task_settings)  # Build
        docs = Docs()
        raw_docs_deepcopy = deepcopy(docs)  # Preserve for later assertions
        # Why are we constructing a TaskConfig here using a serialized Settings and
        # Docs? It's to confirm everything works as if hydrating from a YAML config file
        task_config = TaskConfig(
            name=STUB_TASK_DATASET_NAME,
            eval_kwargs={
                "base_docs": docs.model_dump(
                    exclude={
                        "id",
                        "docnames",
                        "texts_index",
                        "index_path",
                        "deleted_dockeys",
                    }
                ),
                "question_kwargs": {
                    "shuffle_seed": MultipleChoiceQuestion.SEED_USING_QUESTION
                },
            },
        )
        # NOTE: set base_query after construction of the TaskConfig. because in
        # aviary 0.10 the TaskConfig Pydantic model has types `BaseModel | JsonValue`,
        # which lead to settings being cast into a BaseModel. This is probably a bug
        # in aviary, but for now let's just assign it after TaskConfig construction
        task_config.eval_kwargs["settings"] = agent_task_settings.model_dump()
        original_agent_task_settings = agent_task_settings.model_copy(deep=True)
        dataset = task_config.make_dataset(split="eval")  # noqa: FURB184
        assert isinstance(dataset, StubLitQADataset), "Test assertions depend on this"
        metrics_callback = MeanMetricsCallback(eval_dataset=dataset)
        store_env_callback = StoreEnvCallback()

        evaluator = Evaluator(
            config=EvaluatorConfig(batch_size=len(dataset.data), max_rollout_steps=10),
            agent=SimpleAgent(),
            dataset=dataset,
            callbacks=[metrics_callback, store_env_callback],
        )
        await evaluator.evaluate()

        assert agent_task_settings == original_agent_task_settings, (
            "Should not have mutated settings"
        )
        assert not docs.docs, "Should not have mutated docs in base docs"
        assert metrics_callback.eval_means["total_paper_count"] > 0, (
            "Expected some papers to help us answer questions"
        )
        correct_percentage = metrics_callback.eval_means["correct"]
        assert metrics_callback.eval_means["reward"] > 0, "Expected some wins"
        correct_reward, incorrect_reward = (
            DEFAULT_REWARD_MAPPING[evaluation.value]
            for evaluation in (
                MultipleChoiceEvaluation.CORRECT,
                MultipleChoiceEvaluation.INCORRECT,
            )
        )
        worst_case_reward_given_correct = (
            correct_reward * correct_percentage
            + incorrect_reward * (1 - correct_percentage)
        )
        assert (
            metrics_callback.eval_means["reward"] >= worst_case_reward_given_correct
        ), "Expected reward to be above worst case value"

        with subtests.test(msg="confirming-reset-works"):
            assert len(store_env_callback.query_to_envs) == len(dataset)
            for env in store_env_callback.query_to_envs.values():
                await env.reset()
                assert env.state.docs == raw_docs_deepcopy

        with subtests.test(msg="zero-shot"):
            # Confirm we can just directly call gen_answer
            agent_task_settings.agent.tool_names = {GenerateAnswer.gen_answer.__name__}
            agent_task_settings.answer.max_answer_attempts = 2
            agent_task_settings.answer.get_evidence_if_no_contexts = False
            dataset = LitQAv2TaskDataset(settings=agent_task_settings)
            dataset.data = dataset.data[:2]  # Save the world: just use two questions
            storage_callback = StoreTrajectoriesCallback()
            evaluator = Evaluator(
                config=EvaluatorConfig(batch_size=len(dataset), max_rollout_steps=4),
                agent=SimpleAgent(),
                dataset=dataset,
                callbacks=[storage_callback],
            )
            await evaluator.evaluate()
            for traj in storage_callback.eval_trajectories:
                assert not traj.failed
                assert traj.done
                for step in traj.steps:
                    assert all(
                        tc.function.name == GenerateAnswer.gen_answer.__name__
                        for tc in step.action.value.tool_calls
                    )

    @pytest.mark.vcr
    @pytest.mark.asyncio
    async def test_tool_failure(self, agent_task_settings: Settings) -> None:
        docs = Docs()
        dataset = TaskDataset.from_name(
            STUB_TASK_DATASET_NAME, settings=agent_task_settings, base_docs=docs
        )
        metrics_callback = MeanMetricsCallback(eval_dataset=dataset)

        evaluator = Evaluator(
            config=EvaluatorConfig(
                batch_size=1, num_eval_iterations=1, max_rollout_steps=2
            ),
            agent=SimpleAgent(),
            dataset=dataset,
            callbacks=[metrics_callback],
        )
        with patch(
            "paperqa.agents.search.SearchIndex",
            side_effect=Exception("Totally unexpected but retryable error."),
        ) as mock_SearchIndex:
            await evaluator.evaluate()  # Confirm this does not crash
        assert metrics_callback.eval_means["truncation_rate"] == 1.0, (
            "Expected 100% truncations due to max_rollout_steps"
        )
        mock_SearchIndex.assert_called(), "Expected failures to come from unit test"
        assert metrics_callback.eval_means["correct"] == 0.0
        assert metrics_callback.eval_means["correct_unsure"] == 0.0


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
                "gather_evidence", question=cast(str, stub_gradable_env._query)
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
        unsure_answer = "Based on the sources provided, it appears no one has done x."

        async def emulate_answered_but_unsure(  # noqa: RUF029
            *_, query: PQASession, **__
        ) -> PQASession:
            query.answer = unsure_answer
            return query

        reset_obs, tools = await stub_gradable_env.reset()

        # 1. Emulate being unsure after gen_answer
        answer_action = ToolRequestMessage(
            tool_calls=[ToolCall.from_name("gen_answer")]
        )
        with patch.object(
            type(stub_gradable_env.state.docs), "aquery", emulate_answered_but_unsure
        ):
            answer_obs, _, done, truncated = await stub_gradable_env.step(answer_action)
        assert len(answer_obs) == 1
        assert answer_obs[0].content
        assert unsure_answer in answer_obs[0].content
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
