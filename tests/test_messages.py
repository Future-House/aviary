import json

import numpy as np
import pytest

from aviary.core import (
    Message,
    ToolCall,
    ToolCallFunction,
    ToolRequestMessage,
    ToolResponseMessage,
)
from tests import TEST_IMAGES_DIR


def load_base64_image(filename: str) -> str:
    return (TEST_IMAGES_DIR / filename).read_text().strip()


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
        ("message", "dump_kwargs", "expected"),
        [
            (Message(), {}, {"role": "user"}),
            (Message(content="stub"), {}, {"role": "user", "content": "stub"}),
            (
                Message(
                    content=[
                        {"type": "text", "text": "stub"},
                        {"type": "image_url", "image_url": {"url": "stub_url"}},
                    ]
                ),
                {},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "stub"},
                        {"type": "image_url", "image_url": {"url": "stub_url"}},
                    ],
                },
            ),
            (
                Message(content="stub", info={"foo": "bar"}),
                {"context": {"include_info": True}},
                {"role": "user", "content": "stub", "info": {"foo": "bar"}},
            ),
        ],
    )
    def test_dump(self, message: Message, dump_kwargs: dict, expected: dict) -> None:
        assert message.model_dump(exclude_none=True, **dump_kwargs) == expected

    @pytest.mark.parametrize(
        ("message", "dump_kwargs", "expected"),
        [
            (Message(), {}, {"role": "user"}),
            (Message(content="stub"), {}, {"role": "user", "content": "stub"}),
            (
                Message(
                    content=[
                        {"type": "text", "text": "stub"},
                        {"type": "image_url", "image_url": {"url": "stub_url"}},
                    ]
                ),
                {},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "stub"},
                        {"type": "image_url", "image_url": {"url": "stub_url"}},
                    ],
                },
            ),
            (
                Message(content="stub", info={"foo": "bar"}),
                {"context": {"include_info": True}},
                {"role": "user", "content": "stub", "info": {"foo": "bar"}},
            ),
        ],
    )
    def test_dump_json(
        self, message: Message, dump_kwargs: dict, expected: dict
    ) -> None:
        assert (
            json.loads(message.model_dump_json(exclude_none=True, **dump_kwargs))
            == expected
        )

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
            # Case 6: A bytes image should be converted to a base64 encoded image
            (
                (TEST_IMAGES_DIR / "sample_image.png").read_bytes(),
                "What is this image of?",
                None,
                2,  # 1 image + 1 text
            ),
        ],
    )
    def test_image_message(
        self,
        images: list[np.ndarray | str | bytes] | np.ndarray | str | bytes,
        message_text: str,
        expected_error: str | None,
        expected_content_length: int | None,
    ) -> None:
        # Set red color for numpy array if present
        for img in images if isinstance(images, list) else [images]:
            if isinstance(img, np.ndarray):
                img[:] = [255, 0, 0]  # (255 red, 0 green, 0 blue) is maximum red in RGB

        if expected_error:
            with pytest.raises(ValueError, match=expected_error):
                Message.create_message(text=message_text, images=images)
            return

        message_with_images = Message.create_message(
            text=message_text, images=images, info={"foo": "bar"}
        )
        assert message_with_images.content
        assert message_with_images.info == {"foo": "bar"}
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

    def test_append_text(self, subtests) -> None:
        with subtests.test("text-only content"):
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

        with subtests.test("multimodal content with image"):
            trm = ToolRequestMessage(
                content=[
                    {"type": "text", "text": "stub"},
                    {"type": "image_url", "image_url": {"url": "stub_url"}},
                ],
                tool_calls=[ToolCall.from_name("stub_name")],
            )
            trm_inplace = trm.append_text("text")

            # For multimodal content, verify the text was added to the JSON list
            assert trm.content is not None
            content_list = json.loads(trm.content)
            assert len(content_list) == 3  # original text + image + new text
            assert content_list[0] == {"type": "text", "text": "stub"}
            assert content_list[1]["type"] == "image_url"
            assert content_list[2] == {"type": "text", "text": "text"}
            assert trm_inplace.content == trm.content
            # Check append performs an in-place change by default
            assert trm.tool_calls[0] is trm_inplace.tool_calls[0]

            trm_copy = trm.append_text("text", inplace=False)

            # For multimodal content, verify another text was added
            assert trm_copy.content is not None
            content_list_copy = json.loads(trm_copy.content)
            assert len(content_list_copy) == 4  # original text + image + 2 new texts
            assert trm.content is not None
            content_list_original = json.loads(trm.content)
            assert len(content_list_original) == 3  # unchanged original
            # Check append performs a deep copy when not inplace
            assert trm.tool_calls[0] is not trm_copy.tool_calls[0]

    def test_prepend_text(self, subtests) -> None:
        with subtests.test("text-only content"):
            trm = ToolRequestMessage(
                content="stub", tool_calls=[ToolCall.from_name("stub_name")]
            )
            trm_inplace = trm.prepend_text("text")
            assert trm.content == trm_inplace.content == "text\nstub"
            assert trm.tool_calls[0] is trm_inplace.tool_calls[0]

            trm_copy = trm.prepend_text("text", inplace=False)
            assert trm_copy.content == "text\ntext\nstub"
            assert trm.content == "text\nstub"

        with subtests.test("multimodal content with image"):
            trm = ToolRequestMessage(
                content=[
                    {"type": "text", "text": "stub"},
                    {"type": "image_url", "image_url": {"url": "stub_url"}},
                ],
                tool_calls=[ToolCall.from_name("stub_name")],
            )
            trm_inplace = trm.prepend_text("prepended text")

            # For multimodal content, verify the text was prepended to the JSON list
            assert trm.content is not None  # Ensure typing is correct for mypy
            content_list = json.loads(trm.content)
            assert len(content_list) == 3  # new text + original text + image
            assert content_list[0] == {"type": "text", "text": "prepended text"}
            assert content_list[1] == {"type": "text", "text": "stub"}
            assert content_list[2]["type"] == "image_url"
            assert trm_inplace.content == trm.content
            assert trm.tool_calls[0] is trm_inplace.tool_calls[0]

            trm_copy = trm.prepend_text("prepended text", inplace=False)

            # For multimodal content, verify another text was prepended
            assert trm_copy.content is not None
            content_list_copy = json.loads(trm_copy.content)
            assert len(content_list_copy) == 4
            assert content_list_copy[0] == {"type": "text", "text": "prepended text"}
            assert trm.content is not None
            content_list_original = json.loads(trm.content)
            assert len(content_list_original) == 3


class TestCacheBreakpoint:
    def test_default_is_false(self) -> None:
        msg = Message(content="test")
        assert not msg.cache_breakpoint

    def test_set_cache_breakpoint_returns_self(self) -> None:
        msg = Message(content="test")
        result = msg.set_cache_breakpoint()
        assert result is msg
        assert msg.cache_breakpoint

    def test_set_cache_breakpoint_can_disable(self) -> None:
        msg = Message(content="test").set_cache_breakpoint().set_cache_breakpoint(False)
        assert not msg.cache_breakpoint

    def test_serialization_without_cache_breakpoint(self) -> None:
        data = Message(content="test").model_dump(exclude_none=True)
        assert data == {"role": "user", "content": "test"}

    def test_serialization_with_cache_breakpoint_string_content(self) -> None:
        data = (
            Message(content="test").set_cache_breakpoint().model_dump(exclude_none=True)
        )
        assert data == {
            "role": "user",
            "content": [
                {"type": "text", "text": "test", "cache_control": {"type": "ephemeral"}}
            ],
        }

    def test_serialization_with_cache_breakpoint_multimodal_content(self) -> None:
        data = (
            Message(
                content=[
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": "second"},
                ]
            )
            .set_cache_breakpoint()
            .model_dump(exclude_none=True)
        )
        # cache_control should be on the last block
        assert data["content"][0] == {"type": "text", "text": "first"}
        assert data["content"][1] == {
            "type": "text",
            "text": "second",
            "cache_control": {"type": "ephemeral"},
        }

    def test_serialization_with_cache_breakpoint_empty_content(self) -> None:
        data = (
            Message(content=None).set_cache_breakpoint().model_dump(exclude_none=True)
        )
        # Should not crash, content stays None
        assert data == {"role": "user"}

    def test_cache_breakpoint_excluded_from_dump(self) -> None:
        data = Message(content="test").set_cache_breakpoint().model_dump()
        assert "cache_breakpoint" not in data

    def test_cache_breakpoint_with_image_content(self) -> None:
        data = (
            Message
            .create_message(
                text="Describe this image",
                images=[
                    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                ],
            )
            .set_cache_breakpoint()
            .model_dump(exclude_none=True)
        )
        # cache_control should be on the last block (the text block)
        assert len(data["content"]) == 2
        assert data["content"][0]["type"] == "image_url"
        assert "cache_control" not in data["content"][0]
        assert data["content"][1]["type"] == "text"
        assert data["content"][1]["cache_control"] == {"type": "ephemeral"}

    def test_cache_breakpoint_skipped_when_deserialize_content_false(self) -> None:
        data = (
            Message(content="test")
            .set_cache_breakpoint()
            .model_dump(context={"deserialize_content": False})
        )
        # Content should remain a string, cache_breakpoint not applied
        assert data["content"] == "test"

    def test_cache_breakpoint_logs_warning_when_skipped(self, caplog) -> None:
        import logging

        msg = Message(content="test").set_cache_breakpoint()
        with caplog.at_level(logging.WARNING):
            msg.model_dump(context={"deserialize_content": False})
        assert "cache_breakpoint ignored" in caplog.text


def _make_long_content(prefix: str, num_items: int = 300) -> str:
    """Generate long content for cache testing (>1024 tokens for Anthropic)."""
    return prefix + " ".join(f"item_{i}" for i in range(num_items))


@pytest.mark.asyncio
async def test_cache_breakpoint_live() -> None:
    """Verify cache breakpoint causes upstream content to be cached.

    When cache_breakpoint is set on a user message, all content up to and
    including that message should be cached, even content in prior messages
    that don't have cache_breakpoint set.
    """
    from lmi import LiteLLMModel

    # System message - NOT marked for caching, but will be cached
    # because it's upstream of the breakpoint
    system_msg = Message(role="system", content=_make_long_content("System: "))

    # User context message - marked for caching
    # This caches everything up to and including this message
    user_context = Message(role="user", content=_make_long_content("Context: "))
    user_context.set_cache_breakpoint()

    # Simulated assistant acknowledgment
    assistant_msg = Message(role="assistant", content="Acknowledged.")

    # New user question (not cached)
    user_question = Message(role="user", content="Summarize.")

    messages = [system_msg, user_context, assistant_msg, user_question]

    llm = LiteLLMModel(name="claude-3-5-haiku-20241022")

    # First request - may create cache or hit existing cache
    result1 = await llm.call_single(messages)
    cache_active = (result1.cache_creation_tokens or 0) > 0 or (
        result1.cache_read_tokens or 0
    ) > 0
    assert cache_active, "Expected cache creation or cache read on first request"

    # Second request - should hit cache
    result2 = await llm.call_single(messages)
    assert (result2.cache_read_tokens or 0) > 0, "Expected cache hit on second request"
    # Cached content includes both system and user context (~600 items = ~1200+ tokens)
    assert (result2.cache_read_tokens or 0) > 500, (
        f"Expected >500 cached tokens, got {result2.cache_read_tokens}"
    )
