from abc import ABC, abstractmethod
from typing import Any

import httpx

from aviary.env import Environment, TEnvState
from aviary.message import Message
from aviary.tools import MessagesAdapter, Tool, ToolRequestMessage, ToolsAdapter


class EnvironmentClient(Environment[TEnvState], ABC):
    def __init__(
        self,
        reset_endpoint_url: str,
        step_endpoint_url: str,
        request_params: httpx._types.QueryParamTypes | None = None,
        request_headers: httpx._types.HeaderTypes | None = None,
        request_timeout: float | None = None,
    ):
        self._reset_request_url = reset_endpoint_url
        self._step_request_url = step_endpoint_url
        self._request_params = request_params
        self._request_headers = request_headers
        self._request_timeout = request_timeout

    async def reset(self) -> tuple[list[Message], list[Tool]]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._reset_request_url,
                json=self._make_post_json(self.state),
                params=self._request_params,
                headers=self._request_headers,
                timeout=self._request_timeout,
            )
            response.raise_for_status()
            msgs, tools = response.json()
            return MessagesAdapter.validate_python(msgs), ToolsAdapter.validate_python(
                tools
            )

    async def step(
        self, action: ToolRequestMessage
    ) -> tuple[list[Message], float, bool, bool]:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._step_request_url,
                json=self._make_post_json(self.state) | {"action": action.model_dump()},
                params=self._request_params,
                headers=self._request_headers,
                timeout=self._request_timeout,
            )
            response.raise_for_status()
            messages, reward, done, truncated = response.json()
            return MessagesAdapter.validate_python(messages), reward, done, truncated

    @abstractmethod
    def _make_post_json(self, state: TEnvState) -> dict[str, Any]:
        """Extract values from state to sent as JSON for all reset/step POSTs."""
