import base64
import inspect
import io
import random
from collections import UserDict
from collections.abc import Awaitable, Callable, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, cast, overload

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema as cs

try:
    from litellm import acompletion
except ImportError:
    acompletion = None

if TYPE_CHECKING:
    import numpy as np

# Work around super weird bug where np.random.Generator in quotes
# is not being respected as a forward reference
try:
    SeedTypes: TypeAlias = "int | random.Random | np.random.Generator | None"
except ImportError:  # NumPy isn't installed
    SeedTypes = int | random.Random | None  # type: ignore[misc]


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


def partial_format(value: str, **formats) -> str:
    """Partially format a string given a variable amount of formats."""

    class PartialDict(UserDict):
        def __missing__(self, key: str) -> str:
            return f"{{{key}}}"

    return value.format_map(PartialDict(formats))


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
            " `pip install fhaviary[llm]`."
        ) from None
    return response.choices[0].message.content or ""


async def eval_answer(
    proposed: str,
    correct: str,
    question: str | None = None,
    eval_mode: str | EvalAnswerMode = EvalAnswerMode.CONTAINS,
    llm_eval_config: dict | None = None,
    prompt_runner: Callable[[str], Awaitable[str]] | None = None,
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
        prompt = cast("str", config.get("prompt", default_config["prompt"])).format(
            question=question,
            correct_answer=correct,
            proposed_answer=proposed,
        )
        if prompt_runner:
            response_msg = await prompt_runner(prompt)
        else:
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
    proposed_answer: str,
    options: Sequence[str],
    llm_eval_config: dict[str, Any] | None = None,
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
        def val_func(
            state: Any,  # Any enables Pydantic validations can fail over on errors
        ) -> random.Random:
            random_inst = source()
            # `Random.setstate()` raises `ValueError`s if the state is invalid,
            # so no need to handle validation on our own. But we do need to
            # cast the internal_state to a tuple
            version, internal_state, gauss_next = state
            random_inst.setstate((version, tuple(internal_state), gauss_next))
            return random_inst

        plain_val_schema = cs.no_info_plain_validator_function(val_func)
        plain_val_schema_json = plain_val_schema.copy() | {
            "serialization": cs.plain_serializer_function_ser_schema(
                lambda inst: inst.getstate()
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


@overload
def shuffle(value: "np.ndarray", seed: SeedTypes = None) -> "np.ndarray": ...


@overload
def shuffle(value: Sequence[T], seed: SeedTypes = None) -> Sequence[T]: ...


def shuffle(value, seed: SeedTypes = None):
    """Shuffle a non-mutable sequence."""
    # Since most shuffle fn's are in-place, we employ sampling without replacement
    if isinstance(seed, int):
        return random.Random(seed).sample(value, k=len(value))
    if isinstance(seed, random.Random):
        return seed.sample(value, k=len(value))
    if seed is None:
        return random.sample(value, k=len(value))
    # Numpy RNG. Note this will have a type error for sequences like str, but oh well
    return seed.choice(value, size=len(value), replace=False)


def format_exc(exc: BaseException) -> str:
    """Format an exception to be friendly for concise and human-readable logs."""
    if isinstance(exc, ExceptionGroup):  # Expand sub-exceptions
        return (
            f"{exc}, where sub-exceptions are:"
            f" {', '.join(repr(e) for e in exc.exceptions)}"
        )
    return repr(exc)
