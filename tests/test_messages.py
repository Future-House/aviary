import json

import numpy as np
import pytest

from aviary.message import Message
from aviary.tools import (
    ToolCall,
    ToolCallFunction,
    ToolRequestMessage,
    ToolResponseMessage,
)


class TestMessage:
    def test_roles(self) -> None:
        # make sure it rejects invalid roles
        with pytest.raises(ValueError):  # noqa: PT011
            Message(role="invalid", content="Hello, how are you?")
        # make sure it accepts valid roles
        Message(role="system", content="Respond with single words.")

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            (Message(), ""),
            (Message(content="stub"), "stub"),
            (Message(role="system", content="stub"), "stub"),
            (ToolRequestMessage(), ""),
            (ToolRequestMessage(content="stub"), "stub"),
            (
                ToolRequestMessage(
                    content="stub",
                    tool_calls=[
                        ToolCall(
                            id="1",
                            function=ToolCallFunction(name="name", arguments={"hi": 5}),
                        )
                    ],
                ),
                "Tool request message 'stub' for tool calls: name(hi='5') [id=1]",
            ),
            (
                ToolRequestMessage(
                    tool_calls=[
                        ToolCall(
                            id="1",
                            function=ToolCallFunction(
                                name="name1", arguments={"hi": 5}
                            ),
                        ),
                        ToolCall(
                            id="2",
                            function=ToolCallFunction(name="name2", arguments={}),
                        ),
                    ],
                ),
                (
                    "Tool request message '' for tool calls: "
                    "name1(hi='5') [id=1]; name2() [id=2]"
                ),
            ),
            (
                ToolResponseMessage(content="stub", name="name", tool_call_id="1"),
                "Tool response message 'stub' for tool call ID 1 of tool 'name'",
            ),
            (
                Message(
                    content=[
                        {"type": "text", "text": "stub"},
                        {"type": "image_url", "image_url": {"url": "stub_url"}},
                    ]
                ),
                '[{"type": "text", "text": "stub"}, {"type": "image_url", "image_url": {"url": "stub_url"}}]',
            ),
        ],
    )
    def test_str(self, message: Message, expected: str) -> None:
        assert str(message) == expected

    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            (Message(), {"role": "user"}),
            (Message(content="stub"), {"role": "user", "content": "stub"}),
            (
                Message(
                    content=[
                        {"type": "text", "text": "stub"},
                        {"type": "image_url", "image_url": {"url": "stub_url"}},
                    ]
                ),
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "stub"},
                        {"type": "image_url", "image_url": {"url": "stub_url"}},
                    ],
                },
            ),
        ],
    )
    def test_dump(self, message: Message, expected: dict) -> None:
        assert message.model_dump(exclude_none=True) == expected

    def test_image_message(self) -> None:
        # An RGB image of a red square
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[:] = [255, 0, 0]  # (255 red, 0 green, 0 blue) is maximum red in RGB
        message_text = "What color is this square? Respond only with the color name."
        message_with_image = Message.create_message(text=message_text, image=image)
        assert message_with_image.content
        specialized_content = json.loads(message_with_image.content)
        assert len(specialized_content) == 2
        text_idx, image_idx = (
            (0, 1) if specialized_content[0]["type"] == "text" else (1, 0)
        )
        assert specialized_content[text_idx]["text"] == message_text
        assert "image_url" in specialized_content[image_idx]


def test_from_request() -> None:
    trm = ToolRequestMessage(
        content="stub",
        tool_calls=[
            ToolCall(
                id="1",
                function=ToolCallFunction(name="name1", arguments={"hi": 5}),
            ),
            ToolCall(id="2", function=ToolCallFunction(name="name2", arguments={})),
        ],
    )
    assert ToolResponseMessage.from_request(trm, ("stub1", "stub2")) == [
        ToolResponseMessage(content="stub1", name="name1", tool_call_id="1"),
        ToolResponseMessage(content="stub2", name="name2", tool_call_id="2"),
    ]
