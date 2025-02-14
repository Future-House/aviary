import base64
import contextlib
import inspect
import io
import random
import string
from ast import literal_eval
from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Self, TypeVar, cast

from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler, model_validator
from pydantic_core import core_schema as cs

try:
    from litellm import acompletion
except ImportError:
    acompletion = None

if TYPE_CHECKING:
    import numpy as np


DEFAULT_EVAL_MODEL_NAME = "gpt-4o-mini"
LLM_BOOL_EVAL_CONFIG: dict[str, Any] = {
    "prompt": (
        "Here is a question, the correct answer to the question, and a proposed answer"
        " to the question. Please tell me if the proposed answer is correct, given the"
        " correct answer. ONLY SAY 'YES' OR 'NO'. No other output is permitted."
        "\n\nQuestion: {question}"
        "\n\nCorrect answer: {correct_answer}"
        "\n\nProposed answer: {proposed_answer}"
    ),
    "model": DEFAULT_EVAL_MODEL_NAME,
    "temperature": 0,
}

LLM_EXTRACT_CONFIG = LLM_BOOL_EVAL_CONFIG | {
    "prompt": (
        "You are evaluating answers for a test which has fixed options. "
        "Repeat back which option the proposed answer matches. "
        "GIVE ONLY THE VERBATIM TEXT OF A FIXED OPTION. "
        "If the proposed answer is empty, invalid, or ambiguous, "
        "return an empty string."
        "\n\nOptions:\n{options}"
        "\n\nProposed answer: {proposed_answer}"
    )
}

LLM_SCORE_EVAL_CONFIG = LLM_BOOL_EVAL_CONFIG | {
    "prompt": (
        "Here is a question, the correct answer to the question, and a rubric for"
        " evaluating the question. Judge the proposed answer based on the given rubric."
        " Give a score from 0 to 10. No other output is permitted."
        "\n\nQuestion: {question}"
        "\n\nRubric: {correct_answer}"
        "\n\nProposed answer: {proposed_answer}"
    ),
    "max_score": 10,
}


class EvalAnswerMode(StrEnum):
    EXACT = "exact"  # strings must match exactly
    CONTAINS = "contains"  # the correct answer is contained in the supplied answer
    LLM = "llm"  # Ask an LLM to evaluate
    LLM_SCORE = "llm-score"  # Ask an LLM to evaluate and return the score (normalized)

    def get_default_config(self) -> dict[str, Any]:
        if self == EvalAnswerMode.LLM:
            return LLM_BOOL_EVAL_CONFIG
        if self == EvalAnswerMode.LLM_SCORE:
            return LLM_SCORE_EVAL_CONFIG
        return {}


def partial_format(value: str, **formats: dict[str, Any]) -> str:
    """Partially format a string given a variable amount of formats."""
    for template_key, template_value in formats.items():
        with contextlib.suppress(KeyError):
            value = value.format(**{template_key: template_value})
    return value


def encode_image_to_base64(img: "np.ndarray") -> str:
    """Encode an image to a base64 string, to be included as an image_url in a Message."""
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError(
            "Image processing requires the 'image' extra for 'Pillow'. Please:"
            " `pip install aviary[image]`."
        ) from e

    image = Image.fromarray(img)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return (
        f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
    )


def validate_base64_image(image: str) -> str:
    """Validate if the input string is a valid base64 encoded image and if it is, return the image."""
    try:
        # Support for inclusion of the data:image/ url prefix
        test_image = image.split(",")[1] if image.startswith("data:image/") else image
        base64.b64decode(test_image)
    except Exception as err:
        raise ValueError("Invalid base64 encoded image") from err
    return image


def is_coroutine_callable(obj) -> bool:
    """Get if the input object is awaitable."""
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        return inspect.iscoroutinefunction(obj)
    if callable(obj):
        return inspect.iscoroutinefunction(obj.__call__)
    return False


async def run_prompt(
    prompt: str, model: str = DEFAULT_EVAL_MODEL_NAME, temperature: float | None = None
) -> str:
    try:
        response = await acompletion(
            model=model,
            temperature=temperature,
            messages=[{"content": prompt, "role": "user"}],
        )
    except TypeError:
        raise ImportError(
            "Answer evaluation requires the 'llm' extra for 'litellm'. Please:"
            " `pip install aviary[llm]`."
        ) from None
    return response.choices[0].message.content or ""


async def eval_answer(
    proposed: str,
    correct: str,
    question: str | None = None,
    eval_mode: str | EvalAnswerMode = EvalAnswerMode.CONTAINS,
    llm_eval_config: dict | None = None,
) -> float:
    """Evaluate a proposed answer against a correct answer.

    Will return 0 or 1, except for llm-score which should be between 0 and 1
    """
    eval_mode = EvalAnswerMode(eval_mode)
    if eval_mode in {EvalAnswerMode.LLM, EvalAnswerMode.LLM_SCORE}:
        if question is None:
            raise ValueError("Question must be provided for LLM evaluation mode.")
        default_config = eval_mode.get_default_config()
        config = llm_eval_config or default_config
        prompt = cast(str, config.get("prompt", default_config["prompt"])).format(
            question=question,
            correct_answer=correct,
            proposed_answer=proposed,
        )
        response_msg = await run_prompt(
            prompt,
            model=config.get("model", default_config["model"]),
            temperature=config.get("temperature", default_config["temperature"]),
        )
        if eval_mode == EvalAnswerMode.LLM:
            return await eval_answer(
                response_msg.strip().casefold(), "yes", eval_mode=EvalAnswerMode.EXACT
            )
        try:
            return float(response_msg.strip()) / float(
                config.get("max_score", default_config["max_score"])
            )
        except ValueError:
            return 0

    gt = correct.strip().casefold()
    pred = proposed.strip().casefold()

    if eval_mode == EvalAnswerMode.EXACT:
        return float(pred == gt)

    if eval_mode == EvalAnswerMode.CONTAINS:
        return float(gt in pred)

    raise RuntimeError(f"Invalid evaluation mode: {eval_mode}")


async def extract_answer(
    proposed_answer: str, options: Sequence[str], llm_eval_config: dict | None = None
) -> str | None:
    """Extract the answer matching a proposal from a list of options using an LLM."""
    for option in options:
        if proposed_answer.strip().casefold() == option.strip().casefold():
            return option

    default_config = LLM_EXTRACT_CONFIG
    config = llm_eval_config or default_config
    response_msg = await run_prompt(
        prompt=config.get("prompt", default_config["prompt"]).format(
            options="\n".join(options),
            proposed_answer=proposed_answer,
        ),
        model=config.get("model", default_config["model"]),
        temperature=config.get("temperature", default_config["temperature"]),
    )
    answer = response_msg.strip().casefold()  # noqa: FURB184
    for option in options:
        if answer == option.strip().casefold():
            return option
    return None


class RandomAnnotation:
    """Enable Pydantic annotation for random.Random instances."""

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: type[random.Random], handler: GetCoreSchemaHandler
    ) -> cs.CoreSchema:
        def val_func(state: list) -> random.Random:
            random_inst = source()
            # `Random.setstate()` raises `ValueError`s if the state is invalid,
            # so no need to handle validation on our own. But we do need to
            # cast the internal_state to a tuple
            version, internal_state, gauss_next = state
            random_inst.setstate((version, tuple(internal_state), gauss_next))
            return random_inst

        plain_val_schema = cs.no_info_plain_validator_function(val_func)
        plain_val_schema_json = plain_val_schema.copy() | {
            "serialization": (
                cs.plain_serializer_function_ser_schema(lambda inst: inst.getstate())
            )
        }
        return cs.json_or_python_schema(
            python_schema=cs.union_schema(
                choices=[cs.is_instance_schema(source), plain_val_schema],
                serialization=cs.plain_serializer_function_ser_schema(
                    lambda inst: inst.getstate(), when_used="json"
                ),
            ),
            json_schema=plain_val_schema_json,
        )


T = TypeVar("T")


def shuffle(
    value: Sequence[T], seed: "int | random.Random | np.random.Generator | None" = None
) -> Sequence[T]:
    """Shuffle a non-mutable sequence."""
    # Since most shuffle fn's are in-place, we employ sampling without replacement
    if isinstance(seed, int):
        return random.Random(seed).sample(value, k=len(value))
    if isinstance(seed, random.Random):
        return seed.sample(value, k=len(value))
    if seed is None:
        return random.sample(value, k=len(value))
    # Numpy RNG. Note this will have a type error for sequences like str, but oh well
    return seed.choice(value, size=len(value), replace=False)  # type: ignore[arg-type,return-value]


_CAPITAL_A_INDEX = ord("A")


class MultipleChoiceQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    OPEN_ANSWER_PROMPT_TEMPLATE: ClassVar[str] = "{question_id}: {question}"
    MC_QUESTION_PROMPT_TEMPLATE: ClassVar[str] = "\n\n".join((
        OPEN_ANSWER_PROMPT_TEMPLATE,
        "Options:\n{options}",
    ))
    DEFAULT_UNSURE_OPTION: ClassVar[str] = (
        "Insufficient information to answer this question"
    )
    SEED_USING_QUESTION: ClassVar[Literal["SEED_USING_QUESTION"]] = (
        "SEED_USING_QUESTION"
    )

    question: str = Field(
        description="Question to answer (without multiple choice options)."
    )

    question_id: str = Field(
        default="Q", description="Question identifier used in the prompt."
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
        if self.prompt_without_options:
            return self.OPEN_ANSWER_PROMPT_TEMPLATE.format(
                question=self.question,
                question_id=self.question_id,
            )
        return self.MC_QUESTION_PROMPT_TEMPLATE.format(
            question=self.question,
            question_id=self.question_id,
            options="\n".join([
                f"{_CAPITAL_A_INDEX + i:c}) {o}" for i, o in enumerate(self.options)
            ]),
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

    async def grade(
        self, proposed_answer: str
    ) -> "tuple[MultipleChoiceEvaluation, str | None]":
        extracted_answer = await extract_answer(
            proposed_answer=proposed_answer, options=self.options
        )
        return MultipleChoiceEvaluation.from_answer(
            extracted_answer, self
        ), extracted_answer


class MultipleChoiceEvaluation(StrEnum):
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
        """
        evaluations = [e if isinstance(e, cls) else cls(e) for e in evaluations]
        num_correct = sum(e == cls.CORRECT for e in evaluations)
        accuracy = num_correct / len(evaluations)
        precision = num_correct / sum(
            e in {cls.CORRECT, cls.INCORRECT} for e in evaluations
        )
        return accuracy, precision

    @classmethod
    def from_answer(
        cls, extracted_answer: str | None, question: MultipleChoiceQuestion
    ) -> "MultipleChoiceEvaluation":
        """Make an evaluation from the input answer and multiple choice question.

        Returns:
            Evaluation corresponding to the parsed answer.
        """
        if extracted_answer is None:
            return MultipleChoiceEvaluation.INCORRECT
        # From here, if we don't match either the ideal or the unsure multiple choice
        # options then we declare the answer as incorrect.
        if extracted_answer == question.ideal_answer:
            return MultipleChoiceEvaluation.CORRECT
        if question.unsure_answer and extracted_answer == question.unsure_answer:
            return MultipleChoiceEvaluation.UNSURE
        return MultipleChoiceEvaluation.INCORRECT
