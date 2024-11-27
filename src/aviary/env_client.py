import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, Self, TypeVar, cast

import httpx
from pydantic import BaseModel

from aviary.env import Environment, TaskDataset
from aviary.message import Message
from aviary.tools import MessagesAdapter, Tool, ToolRequestMessage, ToolsAdapter

logger = logging.getLogger(__name__)

# Not sure why, but mypy complains if we use the TEnvState in aviary.env, so redefine here
TEnvState = TypeVar("TEnvState")


class EnvironmentClient(Environment[TEnvState], ABC, Generic[TEnvState]):
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


class TaskEnvClientState(BaseModel):
    env_id: str


class TaskEnvironmentClient(EnvironmentClient[TaskEnvClientState]):
    """An environment client for environments created by a TaskDatasetServer."""

    def __init__(self, idx: int | None, base_url: str, **kwargs):
        self._idx = idx
        self._start_request_url = base_url + "/start"
        self._close_request_url = base_url + "/close"

        kwargs = {
            "reset_endpoint_url": base_url + "/reset",
            "step_endpoint_url": base_url + "/step",
        } | kwargs

        super().__init__(**kwargs)

    async def _start_remote_env(self) -> str:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._start_request_url,
                json={"task_idx": self._idx},
                timeout=self._request_timeout,
            )
            response.raise_for_status()
            return response.json()["env_id"]

    async def reset(self) -> tuple[list[Message], list[Tool]]:
        # defer starting to reset so we can make it async and set the state
        env_id = await self._start_remote_env()
        self.state = TaskEnvClientState(env_id=env_id)

        return await super().reset()

    async def close(self) -> None:
        if not hasattr(self, "state"):
            logger.warning("Attempting to close an environment that was never started.")
            return None

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self._close_request_url,
                json={"env_id": self.state.env_id},
                timeout=self._request_timeout,
            )
            response.raise_for_status()
            return response.json()

    def _make_post_json(self, state: TaskEnvClientState) -> dict[str, Any]:
        return {"env_id": state.env_id}


UNSET_LEN = object()  # sentinel


class TaskDatasetClient(TaskDataset[TaskEnvironmentClient]):
    def __init__(
        self,
        server_url: str,
        request_timeout: float | None = None,
    ):
        self.server_url = server_url
        self.request_timeout = request_timeout
        self._len: int | object | None = UNSET_LEN

    @classmethod
    async def create(cls, *args, **kwargs) -> Self:
        # We need to make an async request to the server to get the dataset size,
        # so provide this classmethod to instantiate the client & get the size.
        client = cls(*args, **kwargs)
        async with httpx.AsyncClient(base_url=client.server_url) as http_client:
            client._len = (await http_client.get("/info")).json()["dataset_size"]
        return client

    def get_new_env_by_idx(self, idx: int) -> TaskEnvironmentClient:
        return self._make_env_client(idx)

    def get_new_env(self) -> TaskEnvironmentClient:
        return self._make_env_client(None)

    def _make_env_client(self, idx: int | None) -> TaskEnvironmentClient:
        return TaskEnvironmentClient(
            idx=idx, base_url=self.server_url, request_timeout=self.request_timeout
        )

    def __len__(self) -> int:
        if self._len is UNSET_LEN:
            raise RuntimeError(
                "Dataset should be created using `TaskDatasetClient.create()`"
            )
        if self._len is None:
            raise TypeError("Server did not define dataset length.")

        return cast(int, self._len)
