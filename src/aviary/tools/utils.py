from collections.abc import Callable
from enum import StrEnum
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar, cast

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
    LLM_SCORE = "llm-score"  # Ask an LLM (default: GPT-4o-mini) to evaluate and return the score (normalized)


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

LLM_SCORE_EVAL_CONFIG = {
    "prompt": (
        "Here is a question, the correct answer to the question, and a rubric for evaluating the question. "
        "Judge the proposed answer based on the given rubric. "
        "Give a score from 0 to 10. No other output is permitted.\n\n"
        "Question: {question} \n\n"
        "Rubric: {correct_answer} \n\n"
        "Proposed answer: {proposed_answer}"
    ),
    "model": "gpt-4o-mini",
    "temperature": 0,
    "max_score": 10,
}


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

    # SEE: https://platform.openai.com/docs/api-reference/chat/create#chat-create-tool_choice
    # > `required` means the model must call one or more tools.
    TOOL_CHOICE_REQUIRED: ClassVar[str] = "required"

    async def __call__(
        self,
        messages: list[Message],
        tools: list[Tool],
        tool_choice: Tool | str | None = TOOL_CHOICE_REQUIRED,
    ) -> ToolRequestMessage:
        """Run a completion that selects a tool in tools given the messages."""
        completion_kwargs: dict[str, Any] = {}
        # SEE: https://platform.openai.com/docs/guides/function-calling/configuring-function-calling-behavior-using-the-tool_choice-parameter
        expected_finish_reason: set[str] = {"tool_calls"}
        if isinstance(tool_choice, Tool):
            completion_kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice.info.name},
            }
            expected_finish_reason = {"stop"}  # TODO: should this be .add("stop") too?
        elif tool_choice is not None:
            completion_kwargs["tool_choice"] = tool_choice
            if tool_choice == self.TOOL_CHOICE_REQUIRED:
                # Even though docs say it should be just 'stop',
                # in practice 'tool_calls' shows up too
                expected_finish_reason.add("stop")

        model_response = await self._bound_acompletion(
            messages=MessagesAdapter.dump_python(
                messages, exclude_none=True, by_alias=True
            ),
            tools=ToolsAdapter.dump_python(tools, exclude_none=True, by_alias=True),
            **completion_kwargs,
        )

        if (num_choices := len(model_response.choices)) != 1:
            raise MalformedMessageError(
                f"Expected one choice in LiteLLM model response, got {num_choices}"
                f" choices, full response was {model_response}."
            )
        choice = model_response.choices[0]
        if choice.finish_reason not in expected_finish_reason:
            raise MalformedMessageError(
                f"Expected a finish reason in {expected_finish_reason} in LiteLLM"
                f" model response, got finish reason {choice.finish_reason!r}, full"
                f" response was {model_response} and tool choice was {tool_choice}."
            )
        usage = model_response.usage
        return ToolRequestMessage(
            **choice.message.model_dump(),
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
