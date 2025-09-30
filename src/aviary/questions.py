import random
import string
from abc import ABC, abstractmethod
from ast import literal_eval
from collections.abc import Sequence
from typing import Annotated, Any, ClassVar, Generic, Literal, Self, TypeVar, cast
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

import aviary.version
from aviary.evaluators import (
    BaseEvaluator,
    CorrectnessEvaluation,
    LLMCorrectnessEvaluator,
)
from aviary.utils import RandomAnnotation, partial_format, shuffle

TGrade = TypeVar("TGrade")


class Question(BaseModel, ABC, Generic[TGrade]):
    """
    Base class for question variants.

    All questions should be JSON serializable via `.model_dump(mode="json")`.
    """

    QUESTION_PROMPT_TEMPLATE: ClassVar[str] = "{question_id}: {question}"

    question_id: str | UUID = Field(
        default="Q", description="Optional question identifier used in the prompt."
    )
    question: str = Field(
        description=(
            "Question to answer, without modifiers such as unsure instructions"
            " or multiple-choice options."
        )
    )
    evaluator: BaseEvaluator[TGrade] = Field(
        default=None,  # type: ignore[arg-type]
        description=(
            "Optional evaluator to use as a grader,"
            " if unspecified then the default evaluation method is used."
        ),
    )
    metadata: dict[str, JsonValue] = Field(
        default_factory=lambda: cast(
            dict[str, JsonValue],
            {
                # Use aviary.version.__version__ to work around fhaviary awkwardness
                "aviary_version": aviary.version.__version__
            },
        ),
        description="Metadata about the question, for traceability.",
    )

    @model_validator(mode="after")
    def set_default_evaluator(self) -> Self:
        if self.evaluator is None:
            self.evaluator = self._make_default_evaluator()  # type: ignore[unreachable]
        return self

    @abstractmethod
    def _make_default_evaluator(self) -> BaseEvaluator[TGrade]:
        """Construct the default evaluation method this question type uses."""

    async def grade(self, answer: str) -> tuple[TGrade, dict[str, str | None]]:
        """Grade a raw answer using the stored evaluation method.

        Returns:
            Two-tuple of evaluation and any intermediary calculations.
        """
        context: dict[str, Any] = {}
        evaluation = await self.evaluator(answer, context)
        return evaluation, context


_CAPITAL_A_INDEX = ord("A")


class MultipleChoiceQuestion(Question[CorrectnessEvaluation]):
    model_config = ConfigDict(extra="forbid")

    MC_QUESTION_PROMPT_TEMPLATE: ClassVar[str] = "\n\n".join((
        Question.QUESTION_PROMPT_TEMPLATE,
        "Options:\n{options}",
    ))
    DEFAULT_UNSURE_OPTION: ClassVar[str] = (
        "Insufficient information to answer this question"
    )
    SEED_USING_QUESTION: ClassVar[Literal["SEED_USING_QUESTION"]] = (
        "SEED_USING_QUESTION"
    )

    prompt_without_id: bool = Field(
        default=False,
        description=(
            "Opt-in flag to exclude question_id from the question_prompt,"
            " if worried about the model memorizing question IDs."
        ),
    )
    prompt_without_options: bool = Field(
        default=False,
        description=(
            "Opt-in flag to exclude options from the question_prompt, effectively"
            " making the prompt be open answer."
        ),
    )
    prompt_template: str | None = Field(
        default=None,
        description=(
            "Optional manual prompt template. If left unspecified,"
            " the class variable default prompt template will be used."
        ),
    )
    options: Sequence[str] = Field(description="All multiple choice options.")
    ideal_answer: str = Field(
        description=(
            "Desired ideal answer. If not one of the provided options, it will be"
            " automatically added."
        )
    )
    unsure_answer: str | None = Field(
        default=DEFAULT_UNSURE_OPTION,
        description=(
            "Unsure answer text. If not one of the provided options, it will be"
            " automatically added."
        ),
    )
    shuffle_seed: (
        int
        | Annotated[random.Random, RandomAnnotation()]
        | Literal["SEED_USING_QUESTION"]
        | None
    ) = Field(
        default=None,
        description=(
            "Optional seed or random number generator to use in randomization of"
            " options, where seeding is not global (e.g. no `random.seed`). Optionally"
            " pass in the string literal 'SEED_USING_QUESTION' to hash the question as"
            " the seed. If making many questions with the same count of options and"
            " sharing a seed across all instantiations, take care to either specify a"
            " different seed per question (e.g. using 'SEED_USING_QUESTION') or specify"
            " a random number generator, to avoid placing the ideal option being"
            " shuffled into the same index for every question."
        ),
    )

    @model_validator(mode="after")
    def add_answers_and_shuffle(self) -> Self:
        if self.ideal_answer not in self.options:
            self.options = [*self.options, self.ideal_answer]
        if self.unsure_answer and self.unsure_answer not in self.options:
            self.options = [*self.options, self.unsure_answer]
        if len(self.options) > len(string.ascii_lowercase):
            raise NotImplementedError(
                "Didn't handle more multiple choice options than letters, options were"
                f" {self.options}."
            )
        if self.shuffle_seed == self.SEED_USING_QUESTION:
            self.shuffle_seed = hash(self.question)
        if self.shuffle_seed is not None:
            self.options = shuffle(self.options, seed=self.shuffle_seed)
            # Ensure deserialization doesn't re-shuffle
            self.shuffle_seed = None
        return self

    @property
    def ideal_answer_index(self) -> int:
        return self.options.index(self.ideal_answer)

    @property
    def ideal_answer_letter(self) -> str:
        return chr(_CAPITAL_A_INDEX + self.ideal_answer_index)

    @property
    def unsure_answer_index(self) -> int | None:
        if self.unsure_answer is None:
            return None
        return self.options.index(self.unsure_answer)

    @property
    def unsure_answer_letter(self) -> str | None:
        if self.unsure_answer_index is None:
            return None
        return chr(_CAPITAL_A_INDEX + self.unsure_answer_index)

    @property
    def question_prompt(self) -> str:
        template_vars = {
            "question": self.question,
            "question_id": (
                type(self).model_fields["question_id"].default
                if self.prompt_without_id
                else self.question_id
            ),
        }
        if self.prompt_without_options:
            return partial_format(
                self.prompt_template or self.QUESTION_PROMPT_TEMPLATE, **template_vars
            )
        return partial_format(
            self.prompt_template or self.MC_QUESTION_PROMPT_TEMPLATE,
            options="\n".join([
                f"{_CAPITAL_A_INDEX + i:c}) {o}" for i, o in enumerate(self.options)
            ]),
            **template_vars,
        )

    @staticmethod
    def split_options(options: str) -> list[str]:
        """Split options string into a list of options.

        Examples:
            >>> MultipleChoiceQuestion.split_options("apples, mangos")
            ['apples', 'mangos']
        """
        try:
            split_options = literal_eval(options)
            if not isinstance(split_options, list):
                raise TypeError("Need split_options to be a list.")  # noqa: TRY301
        except (ValueError, SyntaxError, TypeError):
            split_options = [d.strip("'[ ]\"") for d in options.split(",")]
        return split_options

    def _make_default_evaluator(self):
        return LLMCorrectnessEvaluator.make_llm_extract(
            options=self.options,
            ideal_answer=self.ideal_answer,
            unsure_answer=self.unsure_answer,
        )


class OpenAnswerQuestion(Question[CorrectnessEvaluation]):
    model_config = ConfigDict(extra="forbid")

    OA_QUESTION_PROMPT_TEMPLATE: ClassVar[str] = (
        f"{Question.QUESTION_PROMPT_TEMPLATE}\n\n{{unsure_instruction}}"
    )
    UNSURE_INSTRUCTION_TEMPLATE: ClassVar[str] = (
        "If unable to confidently answer, please answer with '{unsure_answer}'."
    )
    DEFAULT_UNSURE_OPTION: ClassVar[str] = (
        "Insufficient information to answer this question"
    )

    prompt_without_id: bool = Field(
        default=False,
        description=(
            "Opt-in flag to exclude question_id from the question_prompt,"
            " if worried about the model memorizing question IDs."
        ),
    )
    prompt_template: str | None = Field(
        default=None,
        description=(
            "Optional manual prompt template. If left unspecified,"
            " the class variable default prompt template will be used."
        ),
    )
    ideal_answer: str = Field(description="Desired ideal answer.")
    unsure_answer: str | None = Field(
        default=DEFAULT_UNSURE_OPTION,
        description=(
            "Optional unsure answer text. If set to None,"
            " the unsure instruction is unspecified."
        ),
    )

    @property
    def question_prompt(self) -> str:
        template_vars = {
            "question": self.question,
            "question_id": (
                type(self).model_fields["question_id"].default
                if self.prompt_without_id
                else self.question_id
            ),
            "unsure_instruction": (
                self.UNSURE_INSTRUCTION_TEMPLATE.format(
                    unsure_answer=self.unsure_answer
                )
                if self.unsure_answer
                else ""
            ),
        }
        return partial_format(
            self.prompt_template or self.OA_QUESTION_PROMPT_TEMPLATE,
            **template_vars,
        ).strip()  # Strip for empty unsure_instruction

    def _make_default_evaluator(self):
        return LLMCorrectnessEvaluator.make_llm_judge(
            question=self.question_prompt,
            correct_answer=self.ideal_answer,
            unsure_answer=self.unsure_answer,
        )
