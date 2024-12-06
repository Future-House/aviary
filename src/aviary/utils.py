import base64
import contextlib
import inspect
import io
from enum import StrEnum
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    import numpy as np


LLM_EVAL_CONFIG = {
    "prompt": (
        "Here is a question, the correct answer to the question, and a proposed answer"
        " to the question. Please tell me if the proposed answer is correct, given the"
        " correct answer. ONLY SAY 'YES' OR 'NO'. No other output is permitted."
        "\n\nQuestion: {question}"
        "\n\nCorrect answer: {correct_answer}"
        "\n\nProposed answer: {proposed_answer}"
    ),
    "model": "gpt-4o-mini",
    "temperature": 0,
}

LLM_SCORE_EVAL_CONFIG = {
    "prompt": (
        "Here is a question, the correct answer to the question, and a rubric for"
        " evaluating the question. Judge the proposed answer based on the given rubric."
        " Give a score from 0 to 10. No other output is permitted."
        "\n\nQuestion: {question}"
        "\n\nRubric: {correct_answer}"
        "\n\nProposed answer: {proposed_answer}"
    ),
    "model": "gpt-4o-mini",
    "temperature": 0,
    "max_score": 10,
}


class EvalAnswerMode(StrEnum):
    EXACT = "exact"  # strings must match exactly
    CONTAINS = "contains"  # the correct answer is contained in the supplied answer
    LLM = "llm"  # Ask an LLM to evaluate
    LLM_SCORE = "llm-score"  # Ask an LLM to evaluate and return the score (normalized)


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


def is_coroutine_callable(obj) -> bool:
    """Get if the input object is awaitable."""
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        return inspect.iscoroutinefunction(obj)
    if callable(obj):
        return inspect.iscoroutinefunction(obj.__call__)
    return False


async def eval_answer(
    proposed: str,
    correct: str,
    question: str | None = None,
    eval_mode: EvalAnswerMode = EvalAnswerMode.CONTAINS,
    llm_eval_config: dict | None = None,
) -> float:
    """Evaluate a proposed answer against a correct answer.

    Will return 0 or 1, except for llm-score which should be between 0 and 1
    """
    if eval_mode in {EvalAnswerMode.LLM, EvalAnswerMode.LLM_SCORE}:
        try:
            from litellm import acompletion
        except ImportError as e:
            raise ImportError(
                "eval_answer requires the 'llm' extra for 'litellm'. Please:"
                " `pip install aviary[llm]`."
            ) from e
        if question is None:
            raise ValueError("Question must be provided for LLM evaluation mode.")
        default_config = (
            LLM_EVAL_CONFIG
            if eval_mode == EvalAnswerMode.LLM
            else LLM_SCORE_EVAL_CONFIG
        )
        config = llm_eval_config or default_config
        prompt = cast(str, config.get("prompt", default_config["prompt"])).format(
            question=question,
            correct_answer=correct,
            proposed_answer=proposed,
        )
        response = await acompletion(
            model=config.get("model", default_config["model"]),
            temperature=config.get("temperature", default_config["temperature"]),
            messages=[{"content": prompt, "role": "user"}],
        )
        if eval_mode == EvalAnswerMode.LLM:
            return await eval_answer(
                response.choices[0].message.content.strip().casefold(),
                "yes",
                eval_mode=EvalAnswerMode.EXACT,
            )
        try:
            return float(response.choices[0].content.strip()) / float(
                config.get("max_score", default_config["max_score"])  # type: ignore[arg-type]
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
