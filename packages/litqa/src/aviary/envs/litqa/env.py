import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence

from aviary.utils import (
    DEFAULT_EVAL_MODEL_NAME,
    MultipleChoiceEvaluation,
    MultipleChoiceQuestion,
)

if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar  # For TypeVar.default backport

import logging
import re
from abc import ABC
from copy import deepcopy
from enum import StrEnum
from typing import Any, Generic, Self, assert_never, cast
from uuid import UUID

import pandas as pd
from ldp.data_structures import Trajectory
from llmclient import EmbeddingModel, LiteLLMModel, LLMModel
from paperqa.agents.env import POPULATE_FROM_SETTINGS, PaperQAEnvironment
from paperqa.agents.search import SearchIndex, maybe_get_manifest
from paperqa.agents.tools import Complete
from paperqa.docs import Docs
from paperqa.settings import Settings
from paperqa.types import DocDetails

from aviary.core import (
    TASK_DATASET_REGISTRY,
    Messages,
    TaskDataset,
    ToolRequestMessage,
    ToolResponseMessage,
)

logger = logging.getLogger(__name__)
TEvaluation = TypeVar("TEvaluation", default=MultipleChoiceEvaluation)


DEFAULT_REWARD_MAPPING = {"correct": 1.0, "unsure": 0.1, "incorrect": -1.0}
DEFAULT_LABBENCH_HF_HUB_NAME = "futurehouse/lab-bench"
# Test split from Aviary paper's section 4.3: https://doi.org/10.48550/arXiv.2412.21154
DEFAULT_AVIARY_PAPER_HF_HUB_NAME = "futurehouse/aviary-paper-data"


def discounted_returns(
    rewards: list[float], terminated: list[bool], discount: float = 1.0
) -> list[float]:
    # NOTE: This is a _ldq_shim from pqa. How should we handle this?
    r"""
    Calculate the discounted returns for a list of rewards, considering termination flags and a discount factor.

    The discounted return represents the future discounted rewards from each time step onwards, taking into account
    whether an episode has terminated at each step.

    The discounted return \( G_t \) is given by:

    .. math::
        G_t = \sum_{k=1}^{\infty} \gamma^{k-1} R_{t+k}

        where:
        - \( G_t \) is the discounted return starting from time step \( t \).
        - \( \gamma \) is the discount factor.
        - \( R_{t+k} \) is the reward received at time step \( t+k \).

    NOTE: this could live in ldp.alg, but it's here to avoid circular imports.

    Args:
        rewards: A list of rewards at each time step.
        terminated: A list of boolean flags indicating whether the episode terminated at each time step.
        discount: Discount factor to apply to future rewards. Defaults to 1.0 which means no discounting is applied.

    Returns:
        A list of discounted returns (rewards to go), with each element representing the
            total discounted reward from that step onwards.

    Example:
        >>> rewards = [1.0, 2.0, 3.0]
        >>> terminated = [False, False, True]
        >>> discounted_returns(rewards, terminated, discount=0.9)
        [5.23, 4.7, 3.0]
    """
    returns = []
    r = 0.0
    for reward, term in zip(reversed(rewards), reversed(terminated), strict=False):
        # 1 - term is 0 if the episode has terminated
        r = reward + discount * r * (1 - term)
        returns.append(r)
    returns.reverse()
    return returns


def make_discounted_returns(
    evaluation: MultipleChoiceEvaluation,
    num_steps: int,
    rewards: Mapping[str, float] = DEFAULT_REWARD_MAPPING,
    discount: float = 1.0,
) -> list[float]:
    try:
        return discounted_returns(
            # paper-qa has no intermediary rewards
            [0] * (num_steps - 1) + [rewards[evaluation.value]],
            terminated=[False] * (num_steps - 1) + [True],
            discount=discount,
        )
    except TypeError as exc:
        raise ImportError(
            "Making discounted returns requires the 'ldp' extra for 'ldp'. Please:"
            " `pip install paper-qa[ldp]`."
        ) from exc


class ComputeTrajectoryMetricsMixin:  # Avoid importing from private paperqa._ldp_shims
    """Placeholder parent class for when ldp isn't installed."""


class GradablePaperQAEnvironment(PaperQAEnvironment, Generic[TEvaluation]):
    """Extended environment that can grade answers."""

    def __init__(
        self,
        query: str | MultipleChoiceQuestion,
        settings: Settings,
        docs: Docs,
        llm_model: LiteLLMModel | None = POPULATE_FROM_SETTINGS,
        summary_llm_model: LiteLLMModel | None = POPULATE_FROM_SETTINGS,
        embedding_model: EmbeddingModel | None = POPULATE_FROM_SETTINGS,
        session_id: UUID | None = None,
        sources: str | list[str] | None = None,
        rewards: Mapping[str, float] = DEFAULT_REWARD_MAPPING,
        evaluation_callback: Callable[[TEvaluation], Awaitable] | None = None,
        **env_kwargs,
    ):
        super().__init__(
            query,
            settings,
            docs,
            llm_model,
            summary_llm_model,
            embedding_model,
            session_id,
            **env_kwargs,
        )
        # Enables checking an Index has the right DOI(s)
        self.sources: list[str] | None = (
            [sources] if isinstance(sources, str) else sources
        )
        self._evaluation_callback = evaluation_callback
        self._rewards = rewards

    async def validate_sources(
        self, manifest_or_index: dict[str, dict[str, Any]] | SearchIndex | None = None
    ) -> None:
        """Validate the sources can be found in the input manifest or index."""
        if not self.sources:
            return
        if manifest_or_index is None:  # Let's try to load in the manifest
            manifest_or_index = await maybe_get_manifest(
                filename=await self._settings.agent.index.finalize_manifest_file()
            )
        if isinstance(manifest_or_index, SearchIndex):
            entity: str = "index"
            file_names: set[str] = {k for k in await manifest_or_index.index_files if k}
            lowercased_dois: set[str] = set()
        else:
            entity = "manifest"
            file_names = {k for k in manifest_or_index if k}
            lowercased_dois = {
                v["doi"].lower() for v in manifest_or_index.values() if v["doi"]
            }
        if not file_names:  # File names being empty means something's wrong
            logger.warning(
                f"Can't validate sources {self.sources} without a correctly specified"
                f" {entity}."
            )
            return
        not_found = [
            s
            for s in self.sources
            if s not in file_names and s.lower() not in lowercased_dois
        ]
        if not_found:
            question = (
                self._query
                if isinstance(self._query, str)
                else self._query.question_prompt
            )
            raise ValueError(
                f"Sources {not_found} of {self.sources} not found in the {entity},"
                f" the corresponding query was {question!r}."
            )

    async def _evaluate_answer(self) -> TEvaluation:
        # If the ensuring evaluation fails (e.g. due to OpenAI being down), we can:
        # - Suppress the exception and declare the evaluation as incorrect, which can
        #   negatively reward what otherwise was a good trajectory containing a correct
        #   answer. We don't want "bad" offline data, so it's not what we do.
        # - Suppress the exception and just give super()'s reward, but again this could
        #   incorrectly reward what otherwise was a good trajectory.
        # - Don't suppress the exception, which leads to the trajectory failing, and
        #   removes it from the learnable pool. This is the only safe default behavior.
        evaluation, self.state.session.graded_answer = await cast(
            MultipleChoiceQuestion, self._query
        ).grade(self.state.session.answer)
        return evaluation

    async def step(
        self, action: ToolRequestMessage
    ) -> tuple[Messages, float, bool, bool]:
        messages, reward, done, truncated = await super().step(action)
        if not done or not isinstance(self._query, MultipleChoiceQuestion):
            return messages, reward, done, truncated
        evaluation = await self._evaluate_answer()
        if evaluation_callback := self._evaluation_callback:
            await evaluation_callback(evaluation)

        return (
            messages,
            reward + self._rewards[cast(MultipleChoiceEvaluation, evaluation).value],
            done,
            truncated,
        )

    def __deepcopy__(self, memo) -> Self:
        copy_state = deepcopy(self.state, memo)
        # We don't know the side effects of deep copying a litellm.Router,
        # so we force a shallow copy of these LiteLLMModels
        env_model_kwargs: dict[str, Any] = {
            name: model if model is None else type(model)(**model.model_dump())
            for name, model in (
                ("llm_model", self._llm_model),
                ("summary_llm_model", self._summary_llm_model),
                ("embedding_model", self._embedding_model),
            )
        }
        copy_self = type(self)(
            query=self._query,  # No need to copy since we read only
            settings=deepcopy(self._settings, memo),  # Deepcopy just to be safe
            docs=copy_state.docs,
            sources=self.sources,
            rewards=self._rewards,
            evaluation_callback=self._evaluation_callback,
            **env_model_kwargs,
        )
        copy_self.state = copy_state
        # Because we shallow copied the LiteLLMModels, we need to re-make the
        # tool functions within the tools
        copy_self.tools = copy_self.make_tools()
        return copy_self


def read_litqa_v2_from_hub(
    train_eval_dataset: str = DEFAULT_LABBENCH_HF_HUB_NAME,
    test_dataset: str = DEFAULT_AVIARY_PAPER_HF_HUB_NAME,
    randomize: bool = True,
    seed: int | None = None,
    train_eval_split: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Read LitQA v2 JSONL into train, eval, and test DataFrames.

    Args:
        train_eval_dataset: Hugging Face Hub dataset's name corresponding with train
            and eval splits.
        test_dataset: Hugging Face Hub dataset's name corresponding with a test split.
        randomize: Opt-out flag to shuffle the dataset after loading in by question.
        seed: Random seed to use for the shuffling.
        train_eval_split: Train/eval split fraction, default is 80% train 20% eval.

    Raises:
        DatasetNotFoundError: If any of the datasets are not found, or the
            user is unauthenticated.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "Reading in LitQA2 requires the 'datasets' extra for 'datasets'. Please:"
            " `pip install paper-qa[datasets]`."
        ) from exc

    train_eval = load_dataset(train_eval_dataset, "LitQA2")["train"].to_pandas()
    test = load_dataset(test_dataset, "LitQA2")["test"].to_pandas()
    # Convert to list so it's not unexpectedly a numpy array
    train_eval["distractors"] = train_eval["distractors"].apply(list)
    test["distractors"] = test["distractors"].apply(list)
    # Let downstream usage in the TaskDataset's environment factories check for the
    # presence of other DataFrame columns
    if randomize:
        train_eval = train_eval.sample(frac=1, random_state=seed)
        test = test.sample(frac=1, random_state=seed)
    num_train = int(len(train_eval) * train_eval_split)
    return train_eval[:num_train], train_eval[num_train:], test


class LitQATaskDataset(
    TaskDataset[GradablePaperQAEnvironment], ComputeTrajectoryMetricsMixin, ABC
):
    """
    Abstract base class for a task dataset of LitQA v1 or v2 questions.

    This is an ABC because it's non-specific to a LitQA version.
    Examples include LitQA v1, v2, or a test stub version of LitQA.
    """

    def __init__(
        self,
        settings: Settings | dict | None = None,
        base_docs: Docs | dict | None = None,
        rewards: Mapping[str, float] = DEFAULT_REWARD_MAPPING,
        question_kwargs: Mapping[str, Any] | None = None,
        eval_model: LLMModel | str = DEFAULT_EVAL_MODEL_NAME,
        **env_kwargs,
    ):
        if settings is None:
            settings = Settings()
        if isinstance(settings, dict):
            settings = Settings(**settings)
        self._settings = settings
        if base_docs is None:
            base_docs = Docs()
        if isinstance(base_docs, dict):
            base_docs = Docs(**base_docs)
        self._base_docs = base_docs
        self._rewards = rewards
        self._question_kwargs = question_kwargs
        self._eval_model = eval_model
        self._env_kwargs = env_kwargs

    def _make_gradable_environment(
        self,
        ideal_answer: str,
        distractors: str | list[str],
        question: str,
        sources: str | list[str] | None = None,
    ) -> GradablePaperQAEnvironment:
        mc_question = MultipleChoiceQuestion(
            question=question,
            options=(
                distractors
                if isinstance(distractors, list)
                else MultipleChoiceQuestion.split_options(distractors)
            ),
            ideal_answer=ideal_answer,
            **(self._question_kwargs or {}),
        )
        return GradablePaperQAEnvironment(
            query=mc_question,
            settings=self._settings,
            docs=self._base_docs.model_copy(),
            sources=sources,
            rewards=self._rewards,
            **self._env_kwargs,
        )

    def compute_trajectory_metrics(
        self, trajectories: "Sequence[Trajectory]"
    ) -> dict[str, list[float]]:
        total_paper_count: list[float] = []
        relevant_paper_count: list[float] = []
        evidence_count: list[float] = []
        for t in trajectories:
            split_certainties = [
                split_certainty
                for split_certainty in (
                    re.split(
                        pattern=Complete.CERTAINTY_SPLIT_REGEX_PATTERN,
                        string=obs.content,
                        maxsplit=1,
                    )
                    for obs in t.steps[-1].next_observation
                    if (
                        isinstance(obs, ToolResponseMessage)
                        and obs.name == Complete.TOOL_FN_NAME
                    )
                )
                # Filter for places where the regex split succeeded
                if len(split_certainty) >= 4  # noqa: PLR2004
            ]
            for i, metric_list in enumerate(
                (total_paper_count, relevant_paper_count, evidence_count),
                start=1,  # Regex extraction of status starts after has_successful_answer
            ):
                # NOTE: we use mean to not break if there's 2+ complete calls (which
                # we're prompted not to do). If it happens, they should all have the
                # same status, so the mean value should equal the individual values
                metric_list.append(
                    sum(int(sa[i]) for sa in split_certainties) / len(split_certainties)
                    if split_certainties  # Avoid div0 (when complete wasn't called)
                    else 0
                )
        return super().compute_trajectory_metrics(trajectories) | {
            "total_paper_count": total_paper_count,
            "relevant_paper_count": relevant_paper_count,
            "evidence_count": evidence_count,
            "correct": [
                int(t.steps[-1].reward == self._rewards["correct"])
                for t in trajectories
            ],
            "correct_unsure": [
                int(
                    t.steps[-1].reward
                    in {self._rewards["correct"], self._rewards["unsure"]}
                )
                for t in trajectories
            ],
        }


class LitQAv2TaskSplit(StrEnum):
    TRAIN = "train"
    EVAL = "eval"
    TEST = "test"

    def get_index(self) -> int:
        """
        Get the index of the train (0), eval (1), or test (2) split.

        NOTE: the value matches the index in read_litqa_v2_from_hub's returned splits.
        """
        if self == self.TRAIN:
            return 0
        if self == self.EVAL:
            return 1
        if self == self.TEST:
            return 2
        assert_never(self)  # type: ignore[arg-type]


class LitQAv2TaskDataset(LitQATaskDataset):
    """Task dataset of LitQA v2 questions."""

    def __init__(
        self,
        *args,
        train_eval_dataset: str = DEFAULT_LABBENCH_HF_HUB_NAME,
        test_dataset: str = DEFAULT_AVIARY_PAPER_HF_HUB_NAME,
        read_data_kwargs: Mapping[str, Any] | None = None,
        split: str | LitQAv2TaskSplit = LitQAv2TaskSplit.EVAL,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        split_dfs = read_litqa_v2_from_hub(
            train_eval_dataset, test_dataset, **(read_data_kwargs or {})
        )
        self.data = split_dfs[LitQAv2TaskSplit(split).get_index()]

    def get_new_env_by_idx(self, idx: int) -> GradablePaperQAEnvironment:
        sources = []
        for s in self.data.iloc[idx].sources:
            try:
                (doi,) = (
                    s.split(substr, maxsplit=1)[1]
                    for substr in DocDetails.DOI_URL_FORMATS
                    if substr in s
                )
            except ValueError as exc:
                raise NotImplementedError(
                    f"Didn't handle DOI extraction from source {s!r}."
                ) from exc
            sources.append(doi)
        return self._make_gradable_environment(
            ideal_answer=self.data.iloc[idx].ideal,
            distractors=self.data.iloc[idx].distractors,
            question=self.data.iloc[idx].question,
            sources=sources,
        )

    def __len__(self) -> int:
        return len(self.data)


TASK_DATASET_NAME = "litqa-v2"
TASK_DATASET_REGISTRY[TASK_DATASET_NAME] = (
    LitQAv2TaskDataset.__module__,
    LitQAv2TaskDataset.__name__,
)
