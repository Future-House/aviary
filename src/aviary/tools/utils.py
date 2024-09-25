from collections.abc import Callable
from enum import StrEnum
from functools import partial
from typing import TYPE_CHECKING, cast

from pydantic import BaseModel, Field

from aviary.message import MalformedMessageError, Message

from .base import (
    MessagesAdapter,
    Tool,
    ToolRequestMessage,
    ToolResponseMessage,
    ToolsAdapter,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from litellm import ModelResponse


class EvalAnswerMode(StrEnum):
    EXACT = "exact"  # strings must match exactly
    CONTAINS = "contains"  # the correct answer is contained in the supplied answer
    LLM = "llm"  # Ask an LLM (default: GPT-4o-mini) to evaluate


LLM_EVAL_CONFIG = {
    "prompt": (
        "Here is a question, the correct answer to the question, and a proposed answer to the question. "
        "Please tell me if the proposed answer is correct, given the correct answer. "
        "ONLY SAY 'YES' OR 'NO'. No other output is permitted.\n\n"
        "Question: {question} \n\n"
        "Correct answer: {correct_answer} \n\n"
        "Proposed answer: {proposed_answer}"
    ),
    "model": "gpt-4o-mini",
    "temperature": 0,
}


async def eval_answer(
    proposed: str,
    correct: str,
    question: str | None = None,
    eval_mode: EvalAnswerMode = EvalAnswerMode.CONTAINS,
    llm_eval_config: dict | None = None,
) -> bool:
    if eval_mode == EvalAnswerMode.LLM:
        try:
            from litellm import acompletion
        except ImportError as e:
            raise ImportError(
                "eval_answer requires the 'llm' extra for 'litellm'. Please:"
                " `pip install aviary[llm]`."
            ) from e
        if question is None:
            raise ValueError("Question must be provided for LLM evaluation mode.")
        config = llm_eval_config or LLM_EVAL_CONFIG
        prompt = cast(str, config.get("prompt", LLM_EVAL_CONFIG["prompt"])).format(
            question=question,
            correct_answer=correct,
            proposed_answer=proposed,
        )
        response = await acompletion(
            model=config.get("model", LLM_EVAL_CONFIG["model"]),
            temperature=config.get("temperature", LLM_EVAL_CONFIG["temperature"]),
            messages=[{"content": prompt, "role": "user"}],
        )
        return await eval_answer(
            response.choices[0].message.content.strip().casefold(),
            "yes",
            eval_mode=EvalAnswerMode.EXACT,
        )

    gt = correct.strip().casefold()
    pred = proposed.strip().casefold()

    if eval_mode == EvalAnswerMode.EXACT:
        return pred == gt

    if eval_mode == EvalAnswerMode.CONTAINS:
        return gt in pred

    raise RuntimeError(f"Invalid evaluation mode: {eval_mode}")


class ToolSelector:
    """Simple entity to select a tool based on messages."""

    def __init__(
        self,
        model_name: str = "gpt-4o",
        acompletion: "Callable[..., Awaitable[ModelResponse]] | None" = None,
    ):
        """Initialize.

        Args:
            model_name: Name of the model to select a tool with.
            acompletion: Optional async completion function to use, leaving as the
                default of None will use LiteLLM's acompletion. Alternately, specify
                LiteLLM's Router.acompletion function for centralized rate limiting.
        """
        if acompletion is None:
            try:
                from litellm import acompletion
            except ImportError as e:
                raise ImportError(
                    f"{type(self).__name__} requires the 'llm' extra for 'litellm'. Please:"
                    " `pip install aviary[llm]`."
                ) from e
        self._model_name = model_name
        self._bound_acompletion = partial(cast(Callable, acompletion), model_name)

    async def __call__(
        self, messages: list[Message], tools: list[Tool]
    ) -> ToolRequestMessage:
        """Run a completion that selects a tool in tools given the messages."""
        model_response = await self._bound_acompletion(
            messages=MessagesAdapter.dump_python(
                messages, exclude_none=True, by_alias=True
            ),
            tools=ToolsAdapter.dump_python(tools, exclude_none=True, by_alias=True),
        )
        if (
            len(model_response.choices) != 1
            or model_response.choices[0].finish_reason != "tool_calls"
        ):
            raise MalformedMessageError(
                f"Unexpected shape of LiteLLM model response {model_response}."
            )
        usage = model_response.usage
        return ToolRequestMessage(
            **model_response.choices[0].message.model_dump(),
            info={
                "usage": (usage.prompt_tokens, usage.completion_tokens),
                "model": self._model_name,
            },
        )


class ToolSelectorLedger(BaseModel):
    """Simple ledger to record tools and messages."""

    tools: list[Tool] = Field(default_factory=list)
    messages: list[ToolRequestMessage | ToolResponseMessage | Message] = Field(
        default_factory=list
    )
