import json
import pathlib

import numpy as np
import pytest

from aviary.core import (
    Message,
    ToolCall,
    ToolCallFunction,
    ToolRequestMessage,
    ToolResponseMessage,
)

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures" / "test_images"


def load_base64_image(filename: str) -> str:
    return (FIXTURES_DIR / filename).read_text().strip()


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
                            function=ToolCallFunction(name="foo1", arguments={"hi": 5}),
                        ),
                        ToolCall(
                            id="2",
                            function=ToolCallFunction(name="foo2", arguments={}),
                        ),
                        ToolCall(
                            id="3",
                            function=ToolCallFunction(name="foo1", arguments=""),
                        ),
                        ToolCall(
                            id="4",
                            function=ToolCallFunction(name="foo2", arguments=None),
                        ),
                    ],
                ),
                (
                    "Tool request message '' for tool calls: "
                    "foo1(hi='5') [id=1]; foo2() [id=2]; foo1() [id=3]; foo2() [id=4]"
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
                (
                    '[{"type": "text", "text": "stub"}, {"type": "image_url",'
                    ' "image_url": {"url": "stub_url"}}]'
                ),
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

    @pytest.mark.parametrize(
        ("images", "message_text", "expected_error", "expected_content_length"),
        [
            # Case 1: Invalid base64 image should raise error
            (
                [
                    np.zeros((32, 32, 3), dtype=np.uint8),  # red square
                    "data:image/jpeg;base64,fake_base64_content",  # invalid base64
                ],
                "What color are these squares? List each color.",
                "Invalid base64 encoded image",
                None,
            ),
            # Case 2: Valid images should work
            (
                [
                    np.zeros((32, 32, 3), dtype=np.uint8),  # red square
                    load_base64_image("sample_jpeg_image.b64"),
                ],
                "What color are these squares? List each color.",
                None,
                3,  # 2 images + 1 text
            ),
            # Case 3: A numpy array in non-list formatshould be converted to a base64 encoded image
            (
                np.zeros((32, 32, 3), dtype=np.uint8),  # red square
                "What color is this square?",
                None,
                2,  # 1 image + 1 text
            ),
            # Case 4: A string should be converted to a base64 encoded image
            (
                load_base64_image("sample_jpeg_image.b64"),
                "What color is this square?",
                None,
                2,  # 1 image + 1 text
            ),
            # Case 5: A PNG image should be converted to a base64 encoded image
            (
                load_base64_image("sample_png_image.b64"),
                "What color is this square?",
                None,
                2,  # 1 image + 1 text
            ),
        ],
    )
    def test_image_message(
        self,
        images: list[np.ndarray | str] | np.ndarray | str,
        message_text: str,
        expected_error: str | None,
        expected_content_length: int | None,
    ) -> None:
        # Set red color for numpy array if present
        for img in images:
            if isinstance(img, np.ndarray):
                img[:] = [255, 0, 0]  # (255 red, 0 green, 0 blue) is maximum red in RGB

        if expected_error:
            with pytest.raises(ValueError, match=expected_error):
                Message.create_message(text=message_text, images=images)
            return

        message_with_images = Message.create_message(text=message_text, images=images)
        assert message_with_images.content
        specialized_content = json.loads(message_with_images.content)
        assert len(specialized_content) == expected_content_length

        # Find indices of each content type
        image_indices = []
        text_idx = None
        for i, content in enumerate(specialized_content):
            if content["type"] == "image_url":
                image_indices.append(i)
            else:
                text_idx = i

        if isinstance(images, list):
            assert len(image_indices) == len(images)
        else:
            assert len(image_indices) == 1
        assert text_idx is not None
        assert specialized_content[text_idx]["text"] == message_text

        # Check both images are properly formatted
        for idx in image_indices:
            assert "image_url" in specialized_content[idx]
            assert "url" in specialized_content[idx]["image_url"]
            # Both images should be base64 encoded
            assert specialized_content[idx]["image_url"]["url"].startswith(
                "data:image/"
            )


class TestToolRequestMessage:
    def test_from_request(self) -> None:
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

    def test_append_text(self) -> None:
        trm = ToolRequestMessage(
            content="stub", tool_calls=[ToolCall.from_name("stub_name")]
        )
        trm_inplace = trm.append_text("text")
        assert trm.content == trm_inplace.content == "stub\ntext"
        # Check append performs an in-place change by default
        assert trm.tool_calls[0] is trm_inplace.tool_calls[0]

        trm_copy = trm.append_text("text", inplace=False)
        assert trm_copy.content == "stub\ntext\ntext"
        # Check append performs a deep copy when not inplace
        assert trm.content == "stub\ntext"
        assert trm.tool_calls[0] is not trm_copy.tool_calls[0]
