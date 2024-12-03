import asyncio

import pytest

from aviary.core import ToolCall, ToolRequestMessage, fenv
from aviary.functional import FunctionalEnvironment


@pytest.fixture
def basic_env():
    @fenv.start()
    def test_env(param: str):
        return f"Test {param}", {"param": param}

    @test_env.tool()
    def test_tool(x: str, state) -> str:  # noqa: ARG001
        """Test tool that processes input."""
        return f"Processed {x}"

    return test_env


@pytest.fixture
def async_env():
    @fenv.start()
    async def test_env(param: str):
        await asyncio.sleep(0.1)
        return f"Async Test {param}", {"param": param}

    @test_env.tool()
    async def test_tool(x: str, state) -> str:  # noqa: ARG001
        """Test tool that processes input."""
        await asyncio.sleep(0.1)
        return f"Async Processed {x}"

    return test_env


def test_environment_creation(basic_env):
    env = basic_env("test")
    assert isinstance(env, FunctionalEnvironment)
    assert len(env.tools) == 1
    assert env.tools[0].info.name == "test_tool"


@pytest.mark.asyncio
async def test_basic_env_reset(basic_env):
    env = basic_env("test_param")
    obs, tools = await env.reset()

    assert len(obs) == 1
    assert obs[0].content == "Test test_param"
    assert obs[0].role == "user"
    assert len(tools) == 1


@pytest.mark.asyncio
async def test_async_env_reset(async_env):
    env = async_env("test_param")
    obs, tools = await env.reset()

    assert len(obs) == 1
    assert obs[0].content == "Async Test test_param"
    assert obs[0].role == "user"
    assert len(tools) == 1


# Test State Management
@pytest.mark.asyncio
async def test_state_management():
    @fenv.start()
    def test_env():
        return "Test", {"custom_value": 42}

    @test_env.tool()
    def modify_state(state) -> None:
        """Modify the state."""
        state.reward = 1.0
        state.done = True
        state.extras["new_value"] = 100

    env = test_env()
    await env.reset()

    assert env.state.reward == 0
    assert env.state.done is False
    assert env.state.extras["custom_value"] == 42

    tool_call = ToolCall.from_name("modify_state")
    action = ToolRequestMessage(tool_calls=[tool_call])
    msgs, reward, done, truncated = await env.step(action)

    assert reward == 1.0
    assert done is True
    assert env.state.extras["new_value"] == 100


@pytest.mark.asyncio
async def test_invalid_start_function():
    @fenv.start()
    def invalid_env():
        return "Only one value"  # Should return tuple

    env = invalid_env()
    with pytest.raises(TypeError):
        await env.reset()


async def test_invalid_state_dict():
    @fenv.start()
    def invalid_env():
        return "Test", "not_a_dict"  # second arg should be a dict

    env = invalid_env()
    with pytest.raises(TypeError):
        await env.reset()


@pytest.mark.asyncio
async def test_tool_execution(basic_env):
    env = basic_env("test")
    await env.reset()

    tool_call = ToolCall.from_name("test_tool", x="data")
    action = ToolRequestMessage(tool_calls=[tool_call])
    msgs, reward, done, truncated = await env.step(action)
    assert len(msgs) == 1
    assert msgs[0].content == "Processed data"


@pytest.mark.asyncio
async def test_frame_export(basic_env):
    env = basic_env("test")
    await env.reset()

    frame = env.export_frame()
    assert isinstance(frame.state, dict)
    assert isinstance(frame.info, dict)
    assert "tool_names" in frame.info
    assert frame.info["tool_names"] == ["test_tool"]


@pytest.mark.asyncio
async def test_state_attribute_access():
    @fenv.start()
    def test_env():
        return "Test", {"custom_attr": "value"}

    @test_env.tool()
    def access_state(state) -> str:
        """Access a custom attribute."""
        return state.custom_attr

    env = test_env()
    await env.reset()

    tool_call = ToolCall.from_name("access_state")
    action = ToolRequestMessage(tool_calls=[tool_call])
    msgs, _, _, _ = await env.step(action)
    assert msgs[0].content == "value"


@pytest.mark.asyncio
async def test_environment_reuse(basic_env):
    env1 = basic_env("test1")
    env2 = basic_env("test2")

    obs1, _ = await env1.reset()
    obs2, _ = await env2.reset()

    assert obs1[0].content == "Test test1"
    assert obs2[0].content == "Test test2"
    assert env1.state.extras["param"] == "test1"
    assert env2.state.extras["param"] == "test2"
