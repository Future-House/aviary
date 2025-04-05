import pytest
from cloning.dataset import CloningDataset, CloningSubDataset

from aviary.core import ToolCall, ToolRequestMessage


@pytest.mark.asyncio
async def test_cloning_env_can_start() -> None:
    env = CloningDataset().get_new_env()
    first_obs, tools = await env.reset()
    assert env.tools == tools
    # assume first tool can take a sequence
    tool_call = ToolCall.from_tool(env.tools[0], "start")
    action = ToolRequestMessage(tool_calls=[tool_call])
    obs, *_ = await env.step(action)
    assert "Failed" not in obs[0].content


@pytest.mark.asyncio
async def test_cloning_env_can_start_with_cloning():
    env = CloningDataset(
        split="small", subset_name=CloningSubDataset.CLONING_SCENARIOS
    ).get_new_env()
    first_obs = await env.reset()
    # assume first tool can take a sequence
    tool_call = ToolCall.from_tool(env.tools[0], "start")
    action = ToolRequestMessage(tool_calls=[tool_call])
    obs, *_ = await env.step(action)
    assert "Failed" not in obs[0].content
