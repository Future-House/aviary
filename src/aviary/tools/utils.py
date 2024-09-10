from functools import partial

from pydantic import BaseModel, Field

from aviary.message import Message

from .base import (
    MessagesAdapter,
    Tool,
    ToolRequestMessage,
    ToolResponseMessage,
    ToolsAdapter,
)


class ToolSelector:
    """Simple entity to select a tool based on messages."""

    def __init__(self, model: str = "gpt-4o"):
        try:
            from litellm import acompletion
        except ImportError as e:
            raise ImportError(
                f"{type(self).__name__} requires the 'llm' extra for 'litellm'. Please:"
                " `pip install aviary[llm]`."
            ) from e
        self._acompletion = partial(acompletion, model)

    async def __call__(
        self, messages: list[Message], tools: list[Tool]
    ) -> ToolRequestMessage:
        """Run a completion that selects a tool in tools given the messages."""
        model_response = await self._acompletion(
            messages=MessagesAdapter.dump_python(
                messages, exclude_none=True, by_alias=True
            ),
            tools=ToolsAdapter.dump_python(tools, exclude_none=True, by_alias=True),
        )
        if (
            len(model_response.choices) != 1
            or model_response.choices[0].finish_reason != "tool_calls"
        ):
            raise NotImplementedError(
                f"Unexpected shape of LiteLLM model response {model_response}."
            )
        return ToolRequestMessage(**model_response.choices[0].message.model_dump())


class ToolSelectorLedger(BaseModel):
    """Simple ledger to record tools and messages."""

    tools: list[Tool] = Field(default_factory=list)
    messages: list[ToolRequestMessage | ToolResponseMessage | Message] = Field(
        default_factory=list
    )
