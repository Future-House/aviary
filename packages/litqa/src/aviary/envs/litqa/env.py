import logging
import sys
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from typing import Any, Generic, Self, cast
from uuid import UUID

from ldp.utils import discounted_returns
from llmclient import EmbeddingModel, LiteLLMModel
from paperqa.agents.env import POPULATE_FROM_SETTINGS, PaperQAEnvironment
from paperqa.agents.search import SearchIndex, maybe_get_manifest
from paperqa.docs import Docs
from paperqa.settings import Settings

from aviary.core import (
    Messages,
    MultipleChoiceEvaluation,
    MultipleChoiceQuestion,
    ToolRequestMessage,
)

if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar  # For TypeVar.default backport

logger = logging.getLogger(__name__)
TEvaluation = TypeVar("TEvaluation", default=MultipleChoiceEvaluation)

DEFAULT_REWARD_MAPPING = {"correct": 1.0, "unsure": 0.1, "incorrect": -1.0}


def make_discounted_returns(
    evaluation: MultipleChoiceEvaluation,
    num_steps: int,
    rewards: Mapping[str, float] = DEFAULT_REWARD_MAPPING,
    discount: float = 1.0,
) -> list[float]:
    return discounted_returns(
        # paper-qa has no intermediary rewards
        [0] * (num_steps - 1) + [rewards[evaluation.value]],
        terminated=[False] * (num_steps - 1) + [True],
        discount=discount,
    )


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
        return evaluation  # type: ignore[return-value]

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
