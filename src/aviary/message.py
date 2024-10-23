from __future__ import annotations

import json
from collections.abc import Iterable
from typing import TYPE_CHECKING, ClassVar, Self

from pydantic import BaseModel, Field, field_validator, model_validator

from aviary.utils import encode_image_to_base64

if TYPE_CHECKING:
    from logging import LogRecord

    import numpy as np


class Message(BaseModel):
    DEFAULT_ROLE: ClassVar[str] = "user"
    VALID_ROLES: ClassVar[set[str]] = {
        DEFAULT_ROLE,
        "system",
        "tool",
        "assistant",
        "function",  # Prefer 'tool'
    }

    role: str = Field(
        default=DEFAULT_ROLE,
        description="Message role matching OpenAI's role conventions.",
    )
    content: str | None = Field(
        default=None,
        description=(
            "Optional message content. Can be a string or a dictionary or None. "
            "If a dictionary (for multimodal content), it will be JSON serialized. "
            "None is a sentinel value for the absence of content "
            "(different than empty string)."
        ),
    )
    content_is_json_str: bool = Field(
        default=False,
        description="Whether the content is JSON-serialized (e.g., for multiple modalities).",
        exclude=True,
        repr=False,
    )

    info: dict | None = Field(
        default=None,
        description="Optional metadata about the message.",
        exclude=True,
        repr=False,
    )

    @field_validator("role")
    @classmethod
    def check_role(cls, v: str) -> str:
        if v not in cls.VALID_ROLES:
            raise ValueError(f"Role {v} was not in {cls.VALID_ROLES}.")
        return v

    @model_validator(mode="before")
    @classmethod
    def serialize_content(cls, data):
        if isinstance(data, dict) and "content" in data:
            content = data["content"]
            if content is not None and not isinstance(content, str):
                try:
                    data["content"] = json.dumps(content)
                    data["content_is_json_str"] = True
                except TypeError as e:
                    raise ValueError(
                        "Content must be a string or JSON-serializable."
                    ) from e
        return data

    def __str__(self) -> str:
        return self.content or ""

    def model_dump(self, *args, **kwargs) -> dict:
        dump = super().model_dump(*args, **kwargs)
        if self.content_is_json_str:
            dump["content"] = json.loads(dump["content"])
        return dump

    def append_text(
        self, text: str, delim: str = "\n", inplace: bool = True
    ) -> Message:
        """Append text to the content.

        Args:
            text: The text to append.
            delim: The delimiter to use when concatenating strings.
            inplace: Whether to modify the message in place.

        Returns:
            The modified message. Note that the original message is modified and returned
            if `inplace=True` and a new message is returned otherwise.
        """
        if not self.content:
            new_content = text
        elif self.content_is_json_str:
            try:
                content_list = json.loads(self.content)
                if not isinstance(content_list, list):
                    raise TypeError("JSON content is not a list.")
                content_list.append({"type": "text", "text": text})
                new_content = json.dumps(content_list)
            except json.JSONDecodeError as e:
                raise ValueError("Content is not valid JSON.") from e
        else:
            new_content = f"{self.content}{delim}{text}"
        if inplace:
            self.content = new_content
            return self
        return self.model_copy(update={"content": new_content}, deep=True)

    @classmethod
    def create_message(
        cls,
        role: str = DEFAULT_ROLE,
        text: str | None = None,
        image: np.ndarray | None = None,
    ) -> Self:
        # Assume no image, and update to image if present
        content: str | list[dict] | None = text
        if image is not None:
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image_to_base64(image)},
                }
            ]
            if text is not None:
                content.append({"type": "text", "text": text})
        return cls(role=role, content=content)


def join(
    msgs: Iterable[Message], delimiter: str = "\n", include_roles: bool = True
) -> str:
    return delimiter.join(
        f"{f'{m.role}: ' if include_roles else ''}{m.content or ''}" for m in msgs
    )


class MalformedMessageError(ValueError):
    """Error to throw if some aspect of a Message variant is malformed."""

    @classmethod
    def common_retryable_errors_log_filter(cls, record: LogRecord) -> bool:
        """
        Filter out common parsing failures not worth looking into from logs.

        Returns:
            False if the LogRecord should be filtered out, otherwise True to keep it.
        """
        # NOTE: match both this Exception type's name and its content, to be robust
        return not all(x in record.msg for x in (cls.__name__, EMPTY_CONTENT_BASE_MSG))


class EnvStateMessage(Message):
    """A message that contains the current state of the environment."""


# Define separately so we can filter out this message type
EMPTY_CONTENT_BASE_MSG = "No content in message"
