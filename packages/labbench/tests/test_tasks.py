import asyncio
from collections.abc import Iterable
from copy import deepcopy
from typing import ClassVar, cast
from unittest.mock import patch
from uuid import UUID, uuid4

import pandas as pd
import pytest
from aviary.core import (
    TASK_DATASET_REGISTRY,
    Message,
    MultipleChoiceEvaluation,
    MultipleChoiceQuestion,
    TaskConfig,
    TaskDataset,
)
from ldp.agent import SimpleAgent
from ldp.alg.callbacks import Callback, MeanMetricsCallback, StoreTrajectoriesCallback
from ldp.alg.runners import Evaluator, EvaluatorConfig
from paperqa import Docs, Settings
from paperqa.agents import get_directory_index
from paperqa.agents.env import PaperQAEnvironment
from paperqa.agents.tools import GenerateAnswer
from pytest_subtests import SubTests

from aviary.envs.labbench import (
    DEFAULT_REWARD_MAPPING,
    LABBenchDatasets,
    PaperQATaskDataset,
    TextQATaskDataset,
    TextQATaskSplit,
)


class StubPaperQADataset(TextQATaskDataset):
    """Made up dataset of questions answerable from this repo's stub_data."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raw_data = [
            (
                "Politician",
                ["Technologist", "Plumber"],
                str(uuid4()),  # Emulate how datasets.load_dataset works, it gives a str
                "Who is Frederick Bates?",
                "bates.txt",
            ),
            (
                "Make molecular counterfactuals",
                [
                    "Generating images of cats",
                    "Simple explanations of internet searches",
                ],
                str(uuid4()),
                "How can you use XAI for chemical property prediction?",
                "paper.pdf",
            ),
            (
                "Maple Leaf",
                ["The Stars and Stripes", "The Blue and Yellow", "The Southern Cross"],
                str(uuid4()),
                "What is the national flag of Canada?",
                "flag_day.html",
            ),
        ]
        self.data = pd.DataFrame(
            raw_data, columns=["ideal", "distractors", "id", "question", "source"]
        )

    def _make_query(self, idx: int) -> MultipleChoiceQuestion:
        distractors = self.data.iloc[idx].distractors
        return MultipleChoiceQuestion(
            question_id=UUID(self.data.iloc[idx].id),
            question=self.data.iloc[idx].question,
            options=(
                distractors
                if isinstance(distractors, list)
                else MultipleChoiceQuestion.split_options(distractors)
            ),
            ideal_answer=self.data.iloc[idx].ideal,
            prompt_without_id=True,
            **(self._question_kwargs or {}),
        )

    def _make_sources(self, idx: int) -> str | list[str] | None:
        return self.data.iloc[idx].source

    def __len__(self) -> int:
        return len(self.data)


STUB_TASK_DATASET_NAME = "stub-labbench"
TASK_DATASET_REGISTRY[STUB_TASK_DATASET_NAME] = (
    StubPaperQADataset.__module__,
    StubPaperQADataset.__name__,
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


class TestPaperQATaskDataset:
    EXPECTED_LENGTHS: ClassVar[tuple[int, ...]] = (199, 49)

    @pytest.mark.parametrize(
        ("split", "expected_length"),
        [(TextQATaskSplit.TRAIN, 199), (TextQATaskSplit.TEST, 49)],
    )
    @pytest.mark.asyncio
    async def test___len__(
        self,
        split: TextQATaskSplit,
        expected_length: int,
        agent_task_settings: Settings,
    ) -> None:
        task_dataset = TextQATaskDataset(
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
            if i == 0 and split == TextQATaskSplit.TRAIN:
                expected_question_id = "dbfbae3d-62f6-4710-8d13-8ce4c8485567"
                # Getting ID can work before reset
                assert await env.get_id() == expected_question_id
                # Yes this assertion is somewhat brittle, but it reliably
                # checks the seeding's behavior so we keep it
                obs, _ = await env.reset()
                assert (
                    "Q: SLC14A1 been identified as a specific marker for endothelial"
                    " cells in which organ?\n\nOptions:\nA) liver\nB) Insufficient"
                    " information to answer this question\nC) prostate\nD) eye\nE)"
                    " heart" in (obs[0].content or "")
                )
                assert str(env.state.session.id) == expected_question_id, (
                    "Expected session ID to match the question ID, for readability"
                )
            assert env.sources, "Sources need to be accessible"
            assert isinstance(env.sources, Iterable), (
                "Sources need to be at least iterable"
            )

    @pytest.mark.asyncio
    async def test_can_validate_stub_dataset_sources(
        self, agent_task_settings: Settings
    ) -> None:
        ds = StubPaperQADataset(settings=agent_task_settings)
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
        assert isinstance(dataset, StubPaperQADataset), "Test assertions depend on this"
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
                assert await env.get_id() == str(env.state.session.id)

        with subtests.test(msg="zero-shot"):
            # Confirm we can just directly call gen_answer
            agent_task_settings.agent.tool_names = {GenerateAnswer.gen_answer.__name__}
            agent_task_settings.answer.max_answer_attempts = 2
            agent_task_settings.answer.get_evidence_if_no_contexts = False
            dataset = TextQATaskDataset(settings=agent_task_settings)
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


class TestTextQATaskDataset:
    @pytest.mark.parametrize(
        ("dataset", "split", "first_three"),
        [
            (
                LABBenchDatasets.FIG_QA,
                TextQATaskSplit.TRAIN,
                [
                    (
                        "Q: Expression of LAM genes most significantly decreased in"
                        " which group?\n\nOptions:\nA) GRN homozygous mutant\nB)"
                        " LPS\nC) GRN knockout\nD) Aging\nE) Insufficient information"
                        " to answer this question"
                    ),
                    (
                        "Q: How many cycles of current injection at 10 Hz does the"
                        " figure depict in panel D?\n\nOptions:\nA) 48\nB) 5\nC) 9\nD)"
                        " 7\nE) 40\nF) 0\nG) 8\nH) Insufficient information to answer"
                        " this question"
                    ),
                    (
                        "Q: Which of the following has the greatest in signal change in"
                        " maintenance?\n\nOptions:\nA) val/met\nB) met/val\nC)"
                        " met/met\nD) val/val\nE) Insufficient information to answer"
                        " this question"
                    ),
                ],
            ),
            (LABBenchDatasets.FIG_QA, TextQATaskSplit.TEST, None),
            (
                LABBenchDatasets.LIT_QA2,
                TextQATaskSplit.TRAIN,
                [
                    (
                        "Q: SLC14A1 been identified as a specific marker for"
                        " endothelial cells in which organ?\n\nOptions:\nA) liver\nB)"
                        " eye\nC) prostate\nD) heart\nE) Insufficient information to"
                        " answer this question"
                    ),
                    (
                        "Q: By what factor does MLH1dn expression increase editing"
                        " efficiency of the PE2 editing system on"
                        " average?\n\nOptions:\nA) 2.0x\nB) 4.2x\nC) 9.6x\nD) 7.7x\nE)"
                        " Insufficient information to answer this question"
                    ),
                    (
                        "Q: What is the cytoplasm biovolume of 4.27-mm Ca. T. magnifica"
                        " cell ?\n\nOptions:\nA) 2.91 x 10^-12 m3\nB) 3.56 x 10^-12"
                        " m3\nC) 1.39 x 10^-12 m3\nD) 5.91 x 10-12 m3\nE) Insufficient"
                        " information to answer this question"
                    ),
                ],
            ),
            (
                LABBenchDatasets.LIT_QA2,
                TextQATaskSplit.TEST,
                [
                    (
                        "Q: Immediately after birth in mice, removing the whiskers"
                        " results in neuronal redistribution of which of the"
                        " following?\n\nOptions:\nA) VGLUT1+ corticospinal axonal"
                        " boutons in L5\nB) GAD65+ interneuronal axonal boutons in"
                        " L2/3\nC) ChAT+ basal forebrain axonal boutons in L6\nD)"
                        " VGLUT2+ thalamocortical axonal boutons in L4\nE) Insufficient"
                        " information to answer this question"
                    ),
                    (
                        "Q: Which of the following tRNAs is enriched in extracellular"
                        " vesicles as opposed to within the cell in the"
                        " lymphoblastoidRN cell line?\n\nOptions:\nA) tRNA-TyrGTA\nB)"
                        " tRNA-ArgACG\nC) tRNA-ArgCCG\nD) tRNA-ProAGG\nE)"
                        " tRNA-ValAAC\nF) Insufficient information to answer this"
                        " question"
                    ),
                    (
                        "Q: Within the CTnano-cnt construct, which of the following"
                        " best describes the effect a N55D mutation in the"
                        " bacteriophage MS2 coat protein has on NLuc protein"
                        " production?\n\nOptions:\nA) Increased NLuc translation\nB) No"
                        " change in the level of NLuC translation\nC) Enhanced binding"
                        " to the TR hairpin\nD) Inability to block initiation of NLuC"
                        " translation\nE) Insufficient information to answer this"
                        " question"
                    ),
                ],
            ),
            (
                LABBenchDatasets.TABLE_QA,
                TextQATaskSplit.TRAIN,
                [
                    (
                        "Q: How many LNA antisense oligonucleotides are in Phase I"
                        " trials?\n\nOptions:\nA) 2\nB) 4\nC) 5\nD) 6\nE) 1\nF)"
                        " Insufficient information to answer this question"
                    ),
                    (
                        "Q: At which residue position is the cysteine liganded for the"
                        " one protein in the table where the its cysteine is not in the"
                        " active site?\n\nOptions:\nA) 315\nB) 131\nC) 277\nD) 481\nE)"
                        " 106\nF) 32\nG) 528\nH) Insufficient information to answer"
                        " this question"
                    ),
                    (
                        "Q: Which compound can target many different"
                        " targets?\n\nOptions:\nA) Reservatrol\nB) Varied\nC) HDAC\nD)"
                        " Flavones\nE) Insufficient information to answer this question"
                    ),
                ],
            ),
            (LABBenchDatasets.TABLE_QA, TextQATaskSplit.TEST, None),
            (
                LABBenchDatasets.FIG_QA2,
                TextQATaskSplit.TRAIN,
                [
                    (
                        "Q: A paper described how new granule cells (GCs) in the"
                        " dentate gyrus gradually recruit GABAergic feedback inhibition"
                        " that limits the activity of neighboring GCs, supporting"
                        " sparse coding. They found that this inhibitory control is"
                        " weak in young GCs, and computational modeling suggested that"
                        " the delayed coupling to feedback inhibition may be essential"
                        " for creating detailed representations of new inputs. In this"
                        " paper, in how many paired experiments between MTR and young"
                        " cells did the granule cell IPSC charge increase?"
                    ),
                    (
                        "Q: Inspired by the capacity for zebrafish Müller glia (MG) to"
                        " contribute stem cells that can regenerate damaged neurons in"
                        " the retina, a study used a mouse model to investigate the"
                        " regenerative capacity of MG across different retinal regions."
                        " In this study, using Gnat1rd17Gnat2cpfl3 mice, which quadrant"
                        " of the retina exhibited the highest density of MG-derived"
                        " rods per mm^2?"
                    ),
                    (
                        "Q: In a study using vesicular glutamate transporter-3"
                        " knock-out (Vglut3 KO) mice, researchers found that the"
                        " effects of noise exposure were different from those in"
                        " wild-type animals. Instead of showing potentiation, the KO"
                        " mice displayed reduced exocytosis and calcium influx, and the"
                        " calcium dependence of release stayed unchanged. These results"
                        " indicate that the noise-induced increase in exocytosis"
                        " depends on glutamate release through Vglut3. In this study,"
                        " what is the change in capacitance measurements (DCm) at -20"
                        " mV when comparing unexposed and 1 day after exposure"
                        " condition? Round to the closest integer."
                    ),
                ],
            ),
            (LABBenchDatasets.FIG_QA2, TextQATaskSplit.TEST, None),
            (
                LABBenchDatasets.TABLE_QA2,
                TextQATaskSplit.TRAIN,
                [
                    (
                        "Q: In a study introducing a toolset for efficient"
                        " semi-automated analysis of large-scale 3D electron microscopy"
                        " datasets for reconstruction of neural circuits, what was the"
                        " depth parameter used for whole-cell segmentations in the"
                        " cortex dataset?"
                    ),
                    (
                        "Q: In a recent study, researchers generated large-scale"
                        " datasets of lysine crotonylation and succinylation in"
                        " phytoplasma-infected jujube, revealing that Kcr modification"
                        " of ZjPHGPX2 enhances its activity. In this study, how many"
                        " whole proteins were upregulated when comparing diseased"
                        " versus healthy conditions?"
                    ),
                    (
                        "Q: In research developing covalent KRAS^G12C inhibitors"
                        " through structure-activity relationship studies, beginning"
                        " with compound 1 and employing iterative chemical optimization"
                        " to identify MRTX849 as a clinical candidate for KRAS^G12C"
                        " targeting, what was the Tmax (hours) observed for compound 7"
                        " after oral administration of 10 mg/kg in CD-1 mice?"
                    ),
                ],
            ),
            (LABBenchDatasets.TABLE_QA2, TextQATaskSplit.TEST, None),
        ],
    )
    def test_creating_questions(
        self,
        dataset: str | LABBenchDatasets,
        split: str | TextQATaskSplit,
        first_three: list[str] | None,
    ) -> None:
        """Test we can reliably make questions from Hugging Face Hub."""
        try:
            task_dataset = TextQATaskDataset(
                dataset=dataset, split=split, read_data_kwargs={"seed": 42}
            )
        except ValueError as exc:
            if first_three is None and "does not have" in str(exc):
                return
            raise
        assert [
            cast(
                MultipleChoiceQuestion, task_dataset.get_new_env_by_idx(i)._query
            ).question_prompt
            for i in range(3)
        ] == first_three

    @pytest.mark.parametrize(
        ("dataset", "expected_len"),
        [
            (LABBenchDatasets.FIG_QA, 181),
            (LABBenchDatasets.TABLE_QA, 244),
            (LABBenchDatasets.LIT_QA2, 199),
            (LABBenchDatasets.FIG_QA2, 101),
            (LABBenchDatasets.TABLE_QA2, 100),
        ],
    )
    @pytest.mark.asyncio
    async def test_get_images(
        self, dataset: str | LABBenchDatasets, expected_len: int
    ) -> None:
        task_dataset = TextQATaskDataset(dataset=dataset, read_data_kwargs={"seed": 42})
        # Yes this len assertion duplicates others, but it helps make sure we
        # have constructed the right dataset
        assert len(task_dataset) == expected_len, "Expecting the right dataset"

        # Confirm a UUID not in the dataset is not matched
        with pytest.raises(ValueError, match="0 rows"):
            await task_dataset.get_images(uuid4())

        # Now check we can get images if they're present
        env = task_dataset.get_new_env_by_idx(0)
        if dataset in {LABBenchDatasets.FIG_QA, LABBenchDatasets.TABLE_QA}:
            assert isinstance(await task_dataset.get_images(env), bytes | list)
        else:
            with pytest.raises(ValueError, match="no images"):
                await task_dataset.get_images(env)


class TestImageQATaskDataset:
    @pytest.mark.asyncio
    async def test_figqa(self) -> None:
        task_dataset: PaperQATaskDataset = TaskDataset.from_name(  # type: ignore[assignment]
            LABBenchDatasets.FIG_QA.value.lower(), read_data_kwargs={"seed": 42}
        )
        assert len(task_dataset) == 181
        MeanMetricsCallback(eval_dataset=task_dataset)  # Confirm we could use this
        # Confirm we auto-opted into no-embedding settings
        assert task_dataset._settings.parsing.defer_embedding
        assert not task_dataset._settings.answer.evidence_retrieval

        env = task_dataset.get_new_env_by_idx(0)
        (first_obs,), tools = await env.reset()
        assert first_obs == Message(
            content=(
                "Use the tools to answer the question: Q: Expression of LAM genes most"
                " significantly decreased in which group?\n\nOptions:\nA) GRN"
                " homozygous mutant\nB) LPS\nC) GRN knockout\nD) Aging\nE) Insufficient"
                " information to answer this question\n\nWhen the answer looks"
                " sufficient, you can terminate by calling the complete tool. If the"
                " answer does not look sufficient, and you have already tried to answer"
                " several times with different evidence, terminate by calling the"
                " complete tool. The current status of evidence/papers/cost is Status:"
                " Paper Count=1 | Relevant Papers=0 | Current Evidence=0 | Current"
                " Cost=$0.0000"
            )
        )
        expected_question_id = "792c87ca-0925-434b-b1e5-d8198ec6bac1"
        assert str(env.state.session.id) == expected_question_id
        (added_doc,) = env._docs.docs.values()
        assert added_doc.citation == f"Row ID {expected_question_id} filename 058.jpg"
        (added_text,) = env._docs.texts
        assert added_text.doc == added_doc
        (added_media,) = added_text.media
        assert isinstance(added_media.data, bytes)
        assert added_media.info["suffix"] == ".jpg"

    @pytest.mark.asyncio
    async def test_tableqa(self) -> None:
        task_dataset: PaperQATaskDataset = TaskDataset.from_name(  # type: ignore[assignment]
            LABBenchDatasets.TABLE_QA.value.lower(),
            dataset=LABBenchDatasets.TABLE_QA,
            read_data_kwargs={"seed": 42},
        )
        assert len(task_dataset) == 244
        MeanMetricsCallback(eval_dataset=task_dataset)  # Confirm we could use this
        # Confirm we auto-opted into no-embedding settings
        assert task_dataset._settings.parsing.defer_embedding
        assert not task_dataset._settings.answer.evidence_retrieval

        env = task_dataset.get_new_env_by_idx(0)
        (first_obs,), tools = await env.reset()
        assert first_obs == Message(
            content=(
                "Use the tools to answer the question: Q: How many LNA antisense"
                " oligonucleotides are in Phase I trials?\n\nOptions:\nA) 2\nB) 4\nC)"
                " 5\nD) 6\nE) 1\nF) Insufficient information to answer this"
                " question\n\nWhen the answer looks sufficient, you can terminate by"
                " calling the complete tool. If the answer does not look sufficient,"
                " and you have already tried to answer several times with different"
                " evidence, terminate by calling the complete tool. The current status"
                " of evidence/papers/cost is Status: Paper Count=2 | Relevant Papers=0"
                " | Current Evidence=0 | Current Cost=$0.0000"
            )
        )
        expected_question_id = "1fbf9336-f853-4cdd-a1ec-5a757b9b9128"
        assert str(env.state.session.id) == expected_question_id
        (added_doc1, added_doc2) = env._docs.docs.values()
        added_docs = {added_doc1, added_doc2}
        assert added_doc1.citation == f"Row ID {expected_question_id} filename 271.png"
        assert added_doc2.citation == f"Row ID {expected_question_id} filename 272.png"
        assert len(env._docs.texts) == 2
        added_text1, added_text2 = env._docs.texts
        assert added_text1.doc != added_text2.doc
        for text in env._docs.texts:
            assert text.doc in added_docs
            (added_media,) = text.media
            assert isinstance(added_media.data, bytes)
            assert added_media.info["suffix"] == ".png"

    @pytest.mark.asyncio
    async def test_figqa2(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            TaskDataset.from_name(
                LABBenchDatasets.FIG_QA2.value.lower(),
                dataset=LABBenchDatasets.FIG_QA2,
                read_data_kwargs={"seed": 42},
            )

    @pytest.mark.asyncio
    async def test_tableqa2(self) -> None:
        with pytest.raises(ValueError, match="not supported"):
            TaskDataset.from_name(
                LABBenchDatasets.TABLE_QA2.value.lower(),
                dataset=LABBenchDatasets.TABLE_QA2,
                read_data_kwargs={"seed": 42},
            )
