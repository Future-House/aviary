import sys
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from enum import StrEnum
from typing import Any, Generic, Self

if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar  # For TypeVar.default backport

from pydantic import BaseModel, Field

from aviary.utils import run_prompt

TEvaluation = TypeVar("TEvaluation", default=float)


class BaseEvaluator(BaseModel, ABC, Generic[TEvaluation]):
    @abstractmethod
    async def __call__(
        self, value: str, context: dict[str, Any] | None = None
    ) -> TEvaluation:
        """Evaluate the input value and return an evaluation (e.g. score in [0-1])."""


class CorrectnessEvaluation(StrEnum):
    """Evaluation for binary correctness, with an unsure failover."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    UNSURE = "unsure"  # May be irrelevant if no unsure option provided

    @classmethod
    def calculate_accuracy_precision(
        cls, evaluations: Sequence[Self | str]
    ) -> tuple[float, float]:
        """
        Calculate QA-specific accuracy and precision metrics upon evaluations.

        Raises:
            ZeroDivisionError: if an empty input.

        Returns:
            Two-tuple of accuracy = (num correct) / (num questions) and
                precision = (num correct) / ((num questions) - (num unsure)).
        """  # noqa: DOC502
        evaluations = [e if isinstance(e, cls) else cls(e) for e in evaluations]
        num_correct = sum(e == cls.CORRECT for e in evaluations)
        accuracy = num_correct / len(evaluations)
        precision = num_correct / sum(
            e in {cls.CORRECT, cls.INCORRECT} for e in evaluations
        )
        return accuracy, precision

    @classmethod
    def from_answers(
        cls, answer: str | None, ideal_answer: str, unsure_answer: str | None = None
    ) -> "CorrectnessEvaluation":
        """Use exact match grading to go from answers to correctness."""
        if answer is None:  # An answer can be empty string, so just check None
            return cls.INCORRECT
        # From here, if we don't match either the ideal or the unsure multiple choice
        # options then we declare the answer as incorrect.
        if answer == ideal_answer:
            return cls.CORRECT
        if unsure_answer and answer == unsure_answer:
            return cls.UNSURE
        return cls.INCORRECT


class ExactMatchEvaluator(BaseEvaluator):
    answer: str
    with_strip: bool = True
    case_insensitive: bool = True

    async def __call__(
        self, value: str, context: dict[str, Any] | None = None
    ) -> float:
        answer = self.answer
        if self.with_strip:
            value = value.strip()
            answer = answer.strip()
        if self.case_insensitive:
            value = value.lower()
            answer = answer.lower()
        return float(value == answer)


class LLMCorrectnessEvaluator(BaseEvaluator[CorrectnessEvaluation]):
    prompt_runner: Callable[[str], Awaitable[str]] = Field(
        default=run_prompt,
        description="Function to prompt an LLM, could be for judging or extraction.",
    )
    prompt_template: str = Field(
        description="Prompt template to create the LLM prompt."
    )
    response_evaluator: (
        Callable[[str], CorrectnessEvaluation]
        | Callable[[str], Awaitable[CorrectnessEvaluation]]
    ) = Field(description="Function to evaluate the LLM's response.")

    async def __call__(
        self, value: str, context: dict[str, Any] | None = None
    ) -> CorrectnessEvaluation:
        response = await self.prompt_runner(
            self.prompt_template.format(proposed_answer=value)
        )
        if context is not None:
            context["llm_response"] = response
        evaluated_response = self.response_evaluator(response)
        if isinstance(evaluated_response, Awaitable):
            evaluated_response = await evaluated_response
        return evaluated_response

    @classmethod
    def make_llm_extract(
        cls,
        options: Sequence[str],
        ideal_answer: str,
        unsure_answer: str | None = None,
        prompt_runner: Callable[[str], Awaitable[str]] = run_prompt,
    ) -> Self:
        return cls(
            prompt_runner=prompt_runner,
            prompt_template=(
                "You are evaluating answers for a test which has fixed options. "
                "Repeat back which option the proposed answer matches. "
                "GIVE ONLY THE VERBATIM TEXT OF A FIXED OPTION. "
                "If the proposed answer is empty, invalid, or ambiguous, "
                "return an empty string."
                f"\n\nOptions:\n{options}"
                "\n\nProposed answer: {proposed_answer}"
            ),
            response_evaluator=lambda x: CorrectnessEvaluation.from_answers(
                x, ideal_answer, unsure_answer
            ),
        )

    @classmethod
    def make_llm_judge(
        cls,
        question: str,
        correct_answer: str,
        unsure_answer: str | None = None,
        prompt_runner: Callable[[str], Awaitable[str]] = run_prompt,
    ) -> Self:
        prompt_template = (
            "Here is a question, the correct answer to the question,"
            " and a proposed answer to the question."
        )
        if not unsure_answer:
            prompt_template += (
                " Please tell me if the proposed answer is correct,"
                " given the correct answer."
                " ONLY SAY 'YES' OR 'NO'."
                " No other output is permitted."
                f"\n\nQuestion: {question}"
                f"\n\nCorrect answer: {correct_answer}"
            )

            def evaluator(value: str) -> CorrectnessEvaluation:
                return CorrectnessEvaluation.from_answers(
                    value.strip().lower(), ideal_answer="yes"
                )

        else:
            prompt_template += (
                " Please tell me if the proposed answer is correct or unsure,"
                " given the correct and unsure answers."
                " ONLY SAY 'CORRECT', 'UNSURE', or 'INCORRECT'."
                " No other output is permitted."
                f"\n\nQuestion: {question}"
                f"\n\nCorrect answer: {correct_answer}"
                f"\n\nUnsure answer: {unsure_answer}"
            )

            def evaluator(value: str) -> CorrectnessEvaluation:
                return CorrectnessEvaluation(value.lower())

        return cls(
            prompt_runner=prompt_runner,
            prompt_template=(
                f"{prompt_template}\n\nProposed answer: {{proposed_answer}}"
            ),
            response_evaluator=evaluator,
        )


class LLMJudgeScoredEvaluator(BaseEvaluator):
    question: str = Field(description="Question being evaluated against.")
    rubric: str = Field(description="Rubric to use in evaluating the question.")
    prompt_runner: Callable[[str], Awaitable[str]] = Field(
        default=run_prompt, description="Function to prompt an LLM judge."
    )
    prompt_template: str = Field(
        default=(
            "Here is a question, the correct answer to the question,"
            " and a rubric for evaluating the question."
            " Judge the proposed answer based on the given rubric."
            " Give a score from 0 to 10. No other output is permitted."
            "\n\nQuestion: {question}"
            "\n\nRubric: {rubric}"
            "\n\nProposed answer: {proposed_answer}"
        ),
    )
    response_evaluator: Callable[[str], float] | Callable[[str], Awaitable[float]] = (
        Field(
            default=lambda value: float(value.strip()) / 10,
            description=(
                "Optional function to evaluate the LLM judge's response."
                " If unspecified, the LLM response is divided by 10."
            ),
        )
    )

    async def __call__(
        self, value: str, context: dict[str, Any] | None = None
    ) -> float:
        response = await self.prompt_runner(
            self.prompt_template.format(
                question=self.question,
                rubric=self.rubric,
                proposed_answer=value,
            )
        )
        if context is not None:
            context["llm_response"] = response
        evaluated_response = self.response_evaluator(response)
        if isinstance(evaluated_response, Awaitable):
            evaluated_response = await evaluated_response
        return evaluated_response


class RangeEvaluator(BaseEvaluator):
    lower_bound: float
    inclusive_lower_bound: bool = True
    upper_bound: float
    inclusive_upper_bound: bool = True
    value_extractor: Callable[[str], float] | Callable[[str], Awaitable[float]] = Field(
        default=lambda value: float(value.strip()),
        description=(
            "Optional function to extract a numeric value from the input string."
            " If unspecified, the input string is stripped and  converted to a float."
        ),
    )

    async def __call__(
        self, value: str, context: dict[str, Any] | None = None
    ) -> float:
        numeric_value = self.value_extractor(value)
        if isinstance(numeric_value, Awaitable):
            numeric_value = await numeric_value
        if context is not None:
            context["extracted_value"] = numeric_value
        lower_check = (
            (numeric_value > self.lower_bound)
            if not self.inclusive_lower_bound
            else (numeric_value >= self.lower_bound)
        )
        upper_check = (
            (numeric_value < self.upper_bound)
            if not self.inclusive_upper_bound
            else (numeric_value <= self.upper_bound)
        )
        return float(lower_check and upper_check)
