import sys
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from enum import StrEnum
from typing import Generic, Self

if sys.version_info >= (3, 13):
    from typing import TypeVar
else:
    from typing_extensions import TypeVar  # For TypeVar.default backport

from pydantic import BaseModel, Field

TEvaluation = TypeVar("TEvaluation", default=float)


class BaseEvaluator(BaseModel, ABC, Generic[TEvaluation]):
    @abstractmethod
    async def __call__(self, value: str) -> TEvaluation:
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


class ExactMatchEvaluator(BaseEvaluator):
    answer: str
    with_strip: bool = True
    case_insensitive: bool = True

    async def __call__(self, value: str) -> float:
        answer = self.answer
        if self.with_strip:
            value = value.strip()
            answer = answer.strip()
        if self.case_insensitive:
            value = value.lower()
            answer = answer.lower()
        return float(value == answer)


class LLMJudgeCorrectnessEvaluator(BaseEvaluator):
    question: str = Field(description="Question being evaluated against.")
    answer: str = Field(description="Correct answer to the question.")
    prompt_runner: Callable[[str], Awaitable[str]] = Field(
        description="Function to prompt an LLM judge."
    )
    prompt_template: str = Field(
        default=(
            "Here is a question, the correct answer to the question,"
            " and a proposed answer to the question."
            " Please tell me if the proposed answer is correct,"
            " given the correct answer."
            " ONLY SAY 'YES' OR 'NO'. No other output is permitted."
            "\n\nQuestion: {question}"
            "\n\nCorrect answer: {correct_answer}"
            "\n\nProposed answer: {proposed_answer}"
        ),
    )
    response_evaluator: Callable[[str], Awaitable[float]] | None = Field(
        default=None,
        description=(
            "Optional function to evaluate the LLM judge's response."
            " If unspecified, it's a case insensitive exact-match with 'yes'."
        ),
    )

    async def __call__(self, value: str) -> float:
        response = await self.prompt_runner(
            self.prompt_template.format(
                question=self.question,
                correct_answer=self.answer,
                proposed_answer=value,
            )
        )
        if self.response_evaluator:
            return await self.response_evaluator(response)
        return float(response.strip().lower() == "yes")


class LLMJudgeScoredEvaluator(BaseEvaluator):
    question: str = Field(description="Question being evaluated against.")
    rubric: str = Field(description="Rubric to use in evaluating the question.")
    prompt_runner: Callable[[str], Awaitable[str]] = Field(
        description="Function to prompt an LLM judge."
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
    response_evaluator: Callable[[str], Awaitable[float]] | None = Field(
        default=None,
        description=(
            "Optional function to evaluate the LLM judge's response."
            " If unspecified, the LLM response is divided by 10."
        ),
    )

    async def __call__(self, value: str) -> float:
        response = await self.prompt_runner(
            self.prompt_template.format(
                question=self.question,
                rubric=self.rubric,
                proposed_answer=value,
            )
        )
        if self.response_evaluator:
            return await self.response_evaluator(response)
        return float(response.strip()) / 10


class RangeEvaluator(BaseEvaluator):
    lower_bound: float
    inclusive_lower_bound: bool = True
    upper_bound: float
    inclusive_upper_bound: bool = True
    value_extractor: Callable[[str], Awaitable[float]] | None = Field(
        default=None,
        description=(
            "Optional function to extract a numeric value from the input string."
            " If unspecified, the input string is stripped and  converted to a float."
        ),
    )

    async def __call__(self, value: str) -> float:
        if self.value_extractor:
            numeric_value = await self.value_extractor(value)
        else:
            numeric_value = float(value.strip())
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
