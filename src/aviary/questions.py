import random
import string
from ast import literal_eval
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, ClassVar, Generic, Literal, Self, TypeVar
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from aviary.evaluators import CorrectnessEvaluation
from aviary.utils import RandomAnnotation, extract_answer, shuffle

TGrade = TypeVar("TGrade")


class Question(BaseModel, Generic[TGrade]):
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

    async def grade(
        self,
        answer: str,
        method: (
            Callable[[str], Awaitable[tuple[TGrade, dict[str, str | None]]]] | None
        ) = None,
    ) -> tuple[TGrade, dict[str, str | None]]:
        """Grade a raw answer according to the specified method.

        Raises:
            ValueError: If no grading method is specified,
                and the subclass defines no default.

        Returns:
            Two-tuple of evaluation and any intermediary calculations.
        """
        if method is None:
            raise ValueError(
                f"{type(self).__name__} didn't define a default grading method,"
                " please specify one."
            )
        return await method(answer)


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
            return self.QUESTION_PROMPT_TEMPLATE.format(**template_vars)
        return self.MC_QUESTION_PROMPT_TEMPLATE.format(
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

    async def extract_then_exact_match(
        self, answer: str
    ) -> tuple[CorrectnessEvaluation, dict[str, str | None]]:
        """Grade by first extracting an answer, then exact matching it to an option."""
        extracted_answer = await extract_answer(
            proposed_answer=answer, options=self.options
        )
        metadata = {"extracted_answer": extracted_answer}
        if extracted_answer is None:
            return CorrectnessEvaluation.INCORRECT, metadata
        # From here, if we don't match either the ideal or the unsure multiple choice
        # options then we declare the answer as incorrect.
        if extracted_answer == self.ideal_answer:
            return CorrectnessEvaluation.CORRECT, metadata
        if self.unsure_answer and extracted_answer == self.unsure_answer:
            return CorrectnessEvaluation.UNSURE, metadata
        return CorrectnessEvaluation.INCORRECT, metadata

    async def grade(
        self,
        answer: str,
        method: (
            Callable[
                [str], Awaitable[tuple[CorrectnessEvaluation, dict[str, str | None]]]
            ]
            | None
        ) = None,
    ) -> tuple[CorrectnessEvaluation, dict[str, str | None]]:
        return await super().grade(
            answer, method=method or self.extract_then_exact_match
        )
