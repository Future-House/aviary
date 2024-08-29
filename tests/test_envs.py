import asyncio
import json
import os
import pathlib
import tempfile
from typing import ClassVar
from unittest.mock import patch

import pytest
from httpx import AsyncClient
from pydantic import BaseModel

from aviary.api import EnvDBClient, make_environment_db_server
from aviary.env import DummyEnv, DummyEnvState, Environment, Frame, TaskDataset
from aviary.message import Message
from aviary.render import Renderer
from aviary.tools import (
    Tool,
    ToolCall,
    ToolRequestMessage,
    ToolResponseMessage,
    ToolsAdapter,
)
from tests import CILLMModelNames
from tests.conftest import ENV_BACKENDS


class TestDummyEnv:
    @pytest.mark.asyncio
    async def test_dummyenv(self, dummy_env: DummyEnv) -> None:
        async def my_policy(obs: list[Message]) -> ToolRequestMessage:  # noqa: ARG001
            # For testing purposes, we hardcoded the policy
            return ToolRequestMessage(
                tool_calls=[
                    ToolCall.from_name("print_story", story="Once upon a time done")
                ],
            )

        obs, _ = await dummy_env.reset()
        assert isinstance(obs, list)
        assert len(obs) == 1

        action = await my_policy(obs)
        _, reward, done, _ = await dummy_env.step(action)
        assert reward > 0
        assert done

    @pytest.mark.asyncio
    async def test_tool_signatures(self, dummy_env: DummyEnv) -> None:
        _, tools = await dummy_env.reset()
        assert ToolsAdapter.dump_python(tools, exclude_none=True) == [
            {
                "type": "function",
                "info": {
                    "name": "print_story",
                    "description": "Print a story.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "story": {
                                "type": "string",
                                "title": "Story",
                                "description": "Story to print.",
                            }
                        },
                        "required": ["story"],
                    },
                },
            },
            {
                "info": {
                    "description": "Cast the input argument x to a float.",
                    "name": "cast_float",
                    "parameters": {
                        "properties": {"x": {"type": "string", "title": "X"}},
                        "required": ["x"],
                        "type": "object",
                    },
                },
                "type": "function",
            },
            {
                "info": {
                    "description": "Cast the input argument x to an integer.",
                    "name": "cast_int",
                    "parameters": {
                        "properties": {"x": {"type": "number", "title": "X"}},
                        "required": ["x"],
                        "type": "object",
                    },
                },
                "type": "function",
            },
        ]

    def test_loading_from_name(self):
        env: DummyEnv = Environment.from_name("dummy")
        assert isinstance(env, DummyEnv)

        dataset = TaskDataset.from_name("dummy")
        batch = next(iter(dataset.iter_batches(1)))
        assert len(batch) == 1
        assert isinstance(batch[0], DummyEnv)

    @pytest.mark.parametrize(
        "model_name", ["gpt-3.5-turbo", CILLMModelNames.ANTHROPIC.value]
    )
    @pytest.mark.asyncio
    async def test_tool_calling(self, dummy_env: DummyEnv, model_name: str) -> None:
        def get_todo_list(n: int) -> str:
            """Get todo list for today.

            Args:
                n: number of items to return
            """
            return "\n".join(["Go for a walk", "Read a book", "Call a friend"][:n])

        tool = Tool.from_function(get_todo_list)
        dummy_env.tools = [tool]
        tool_request_message = ToolRequestMessage(
            tool_calls=[ToolCall.from_name("get_todo_list", n=3)]
        )
        new_messages = await dummy_env.exec_tool_calls(tool_request_message)
        (new_message,) = new_messages
        assert new_message.content == "Go for a walk\nRead a book\nCall a friend"
        assert new_message.tool_call_id == tool_request_message.tool_calls[0].id

        def get_todo_list_no_args() -> str:
            """Get todo list for today."""
            return "\n".join(["Go for a walk", "Read a book", "Call a friend"])

        tool = Tool.from_function(get_todo_list_no_args)
        dummy_env.tools = [tool]
        tool_request_message = ToolRequestMessage(
            tool_calls=[ToolCall.from_name("get_todo_list_no_args")]
        )
        new_messages = await dummy_env.exec_tool_calls(tool_request_message)
        (new_message,) = new_messages
        assert new_message.content == "Go for a walk\nRead a book\nCall a friend"
        assert new_message.tool_call_id == tool_request_message.tool_calls[0].id

        # ok now try with multiple functions

        def get_calendar() -> str:
            """Get text version of calendar for today."""
            return "9:00am Wake-up\n10:00pm Go to bed\n"

        tool2 = Tool.from_function(get_calendar)
        dummy_env.tools = [tool, tool2]
        tool_request_message = ToolRequestMessage(
            tool_calls=[
                ToolCall.from_name("get_todo_list_no_args"),
                ToolCall.from_name("get_calendar"),
            ],
        )
        new_messages = await dummy_env.exec_tool_calls(tool_request_message)
        if model_name.startswith("claude"):
            # Anthropic not always so smart
            assert 1 <= len(new_messages) <= 2
        else:
            assert len(new_messages) == 2


@pytest.mark.asyncio
async def test_multiple_calls(dummy_env: DummyEnv) -> None:
    obs, tools = await dummy_env.reset()
    calls = [
        ToolCall.from_name(tools[0].info.name, story="Hello, how are you?"),
        ToolCall.from_name(tools[0].info.name, story="Hello, how are you?"),
        ToolCall.from_name(tools[0].info.name, story="Hello, how are you?"),
    ]
    action = ToolRequestMessage(tool_calls=calls)
    obs, reward, done, truncated = await dummy_env.step(action)
    assert reward > 0
    assert done


class TestRendering:
    class SomeState(BaseModel):
        field: int

    @pytest.mark.parametrize(
        ("state", "serialized"),
        [
            (5, 5),
            (5.6, 5.6),
            ("hi", "hi"),
            (True, True),
            (["hi"], ["hi"]),
            ({"hi": 5}, {"hi": 5}),
            (SomeState(field=5), {"field": 5}),
        ],
    )
    def test_serialization(self, state, serialized) -> None:
        assert Frame(state=state).model_dump()["state"] == serialized

    def test_frame_mutability(self) -> None:
        # make a nested list - so shallow copy won't catch it
        mutable_state = [["foo"]]
        non_deep_copy = Frame(state=mutable_state, deepcopy=False)
        mutable_state[0].append("bar")
        assert non_deep_copy.model_dump()["state"] == [["foo", "bar"]]

        mutable_state = [["foo"]]
        deep_copy = Frame(state=mutable_state)
        mutable_state[0].append("bar")
        assert deep_copy.model_dump()["state"] == [["foo"]]

    def test_rendering(self, dummy_env: DummyEnv) -> None:
        # Reset to add state
        asyncio.run(dummy_env.reset())

        renderer = Renderer(name="Name", prefix="test")
        renderer.append(dummy_env.export_frame())
        with tempfile.TemporaryDirectory() as tmpdir:
            build_dir = pathlib.Path(tmpdir)
            renderer.build(build_dir)
            file_paths = list(build_dir.glob("*.json"))
            assert len(file_paths) == 2, "Expected manifest and one object"
            frame_path = file_paths[
                file_paths[0].name.removeprefix("test_").startswith("info")
            ]
            with frame_path.open() as f:
                rehydrated = json.load(f)
        assert rehydrated["state"]["messages"] == [
            "Write a 5 word story via print_story"
        ]


class ParallelizedDummyEnv(DummyEnv):
    def __init__(self, right_hand_broken: bool = False):
        super().__init__()
        self.right_hand_broken = right_hand_broken

    RIGHT_HAND_BROKEN_MESSAGE: ClassVar[str] = "Right hand is broken."

    async def reset(self) -> tuple[list[Message], list[Tool]]:
        def move_right_hand(
            distance: int,  # noqa: ARG001
            state: DummyEnvState,
        ) -> None:
            """
            Move your right hand forward or backward.

            Args:
                distance: Integer distance to move (mm), where forward is positive.
                state: Current state.
            """
            if self.right_hand_broken:  # Use this to test tool errors
                raise RuntimeError(self.RIGHT_HAND_BROKEN_MESSAGE)
            state.reward += 1

        def move_left_hand(
            distance: int,  # noqa: ARG001
            state: DummyEnvState,
        ) -> None:
            """
            Move your left hand forward or backward.

            Args:
                distance: Integer distance to move (mm), where forward is positive.
                state: Current state.
            """
            state.reward += 1

        def smile_and_wave(state: DummyEnvState) -> None:
            """
            Smile and wave.

            Args:
                state: Current state.
            """
            state.reward = 10
            state.done = True

        self.tools = [
            Tool.from_function(move_left_hand),
            Tool.from_function(move_right_hand),
            Tool.from_function(smile_and_wave),
        ]
        self.state = type(self).State(
            messages=[
                Message(
                    role="user",
                    content=(
                        "You are the president of the United States of America."
                        " Please move both hands at the same time, and then smile"
                        " and wave."
                    ),
                )
            ]
        )
        return self.state.messages, self.tools


class TestParallelism:
    @pytest.mark.parametrize(
        "model_name", [CILLMModelNames.ANTHROPIC.value, "gpt-4-turbo"]
    )
    @pytest.mark.asyncio
    async def test_exec_tool_calls_handling(self, model_name: str) -> None:
        env = ParallelizedDummyEnv(right_hand_broken=True)
        obs, tools = await env.reset()
        right_hand_tool = tools[1]

        # 1. Let's DIY create a ToolRequestMessage for test determinism
        request_msg = ToolRequestMessage(
            tool_calls=[ToolCall.from_tool(right_hand_tool, distance=5)]
        )

        # 2. Okay, our hand was broken, let's handle it DIY-style
        try:
            obs, *_ = await env.step(action=request_msg)
        except RuntimeError as exc:
            obs = [
                Message(
                    content=f"Failed to execute tools with message:\n{exc}", role="tool"
                )
            ]
        else:
            raise AssertionError("Should have blown up per the test logic.")

        # 2. Now that we have confirmed that, let's make sure exec_tool_calls
        #    can automate this for us
        obs = await env.exec_tool_calls(
            message=request_msg, state=env.state, handle_tool_exc=True
        )
        (failure_tool_response,) = obs
        assert isinstance(failure_tool_response, ToolResponseMessage)
        assert env.RIGHT_HAND_BROKEN_MESSAGE in failure_tool_response.content


class TestEnvDBServerClient:
    @pytest.mark.parametrize("clean_db_backend", ENV_BACKENDS, indirect=True)
    @pytest.mark.asyncio
    async def test_env_write_and_read(self, clean_db_backend: str) -> None:
        base_server_url = "http://testserver"
        test_env_name = "test_env"
        with patch.dict(
            os.environ, {"AUTH_TOKEN": "stub", "ENV_DB_URI": clean_db_backend}
        ):
            async with AsyncClient(
                app=make_environment_db_server(),
                base_url=base_server_url,
            ) as async_client:
                response = await async_client.post(
                    "/environment_instance",
                    headers={"Authorization": "Bearer stub"},
                    params={"env_name": test_env_name},
                )
                response.raise_for_status()

            custom_client = AsyncClient(
                app=make_environment_db_server(),
                base_url=base_server_url,
            )

            env_db_client = EnvDBClient(
                server_url="",
                request_headers={"Authorization": "Bearer stub"},
            )

            with patch("httpx.AsyncClient.get", custom_client.get):
                all_results = await env_db_client.get_environment_instances(
                    name=test_env_name
                )
                assert response.json() == all_results[0].id

    @pytest.mark.parametrize("clean_db_backend", ENV_BACKENDS, indirect=True)
    @pytest.mark.asyncio
    async def test_env_frame_write_and_read(self, clean_db_backend: str) -> None:
        base_server_url = "http://testserver"
        test_env_name = "test_env"
        with patch.dict(
            os.environ, {"AUTH_TOKEN": "stub", "ENV_DB_URI": clean_db_backend}
        ):
            custom_client = AsyncClient(
                app=make_environment_db_server(),
                base_url=base_server_url,
            )

            env_db_client = EnvDBClient(
                server_url="",
                request_headers={"Authorization": "Bearer stub"},
                # custom_client_func=custom_client_func,
            )
            with (
                patch("httpx.AsyncClient.post", custom_client.post),
                patch("httpx.AsyncClient.get", custom_client.get),
            ):
                environment_id = await env_db_client.write_environment_instance(
                    name=test_env_name
                )
                environment_ids = await env_db_client.get_environment_instances(
                    name=test_env_name
                )

                assert (
                    len(environment_ids) == 1
                ), "There should only be 1 instance present"

                frame = Frame(state={"messages": ["Hello, world!"]})
                frame_id_1 = await env_db_client.write_environment_frame(
                    environment_id=environment_id, frame=frame
                )
                frame_id_2 = await env_db_client.write_environment_frame(
                    environment_id=environment_id, frame=frame
                )
                frames = await env_db_client.get_environment_frames(
                    environment_id=environment_id
                )

                assert len(frames) == 2, "There should be 2 frames present"
                assert (
                    frames[0].id == frame_id_1
                ), "Frame ID was not assigned as expected"
                assert (
                    frames[1].id == frame_id_2
                ), "Frame ID was not assigned as expected"
