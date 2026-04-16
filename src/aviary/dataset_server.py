import asyncio
import logging
import secrets
import time
import traceback
import uuid
from contextlib import contextmanager
from itertools import starmap
from typing import Any, Generic

from pydantic import BaseModel, Field, model_validator

from aviary.env import TaskDataset, TEnvironment
from aviary.message import Message
from aviary.tools import (
    MessagesAdapter,
    ToolRequestMessage,
    ToolResponseMessage,
    ToolsAdapter,
)

try:
    import uvicorn
    from fastapi import APIRouter, Depends, FastAPI, HTTPException, Security
    from fastapi.security import APIKeyHeader

    missing_dependencies = False
except ImportError:
    # We will raise if a TaskDatasetServer is instantiated but FastAPI/uvicorn are not available
    missing_dependencies = True

logger = logging.getLogger(__name__)


class StartRequest(BaseModel):
    task_idx: int | None = Field(
        default=None,
        description=(
            "Optional index of the dataset to start. If provided, will call"
            " TaskDataset.get_new_env_by_idx(); otherwise, TaskDataset.get_new_env()."
            " Mutually exclusive with task_kwargs."
        ),
    )
    task_kwargs: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Optional keyword arguments passed to TaskDataset.get_new_env_by_args()."
            " Mutually exclusive with task_idx."
        ),
    )

    @model_validator(mode="after")
    def _check_mutually_exclusive(self) -> "StartRequest":
        if self.task_idx is not None and self.task_kwargs is not None:
            raise ValueError(
                "task_idx and task_kwargs are mutually exclusive; specify at most one."
            )
        return self

    def make_env(self, dataset: TaskDataset[TEnvironment]) -> TEnvironment:
        if self.task_kwargs is not None:
            return dataset.get_new_env_by_args(**self.task_kwargs)
        if self.task_idx is not None:
            return dataset.get_new_env_by_idx(self.task_idx)
        return dataset.get_new_env()


class EnvRequest(BaseModel):
    env_id: str = Field(description="Maps to a running env ID")


class StepRequest(BaseModel):
    env_id: str = Field(description="Maps to a running env ID")
    action: ToolRequestMessage | ToolResponseMessage | Message


class FlushRequest(BaseModel):
    last_used: float = Field(default=3600, description="Seconds since last use")


DEFAULT_SERVER_PORT = 8041
BIND_ALL_HOST = "0.0.0.0"  # noqa: S104


class TaskDatasetServer(Generic[TEnvironment]):
    def __init__(
        self,
        dataset: TaskDataset[TEnvironment],
        host: str = BIND_ALL_HOST,
        port: int = DEFAULT_SERVER_PORT,
        api_key: str | None = None,
        router: "APIRouter | None" = None,
    ):
        if missing_dependencies:
            raise ImportError(
                "FastAPI and Uvicorn are required to run a TaskDatasetServer. "
                "Please `pip install fhaviary[server]`."
            )

        self.dataset = dataset
        self.host = host
        self.port = port
        self.api_key = api_key

        # env ID -> (env, last used timestamp)
        self.envs: dict[str, tuple[TEnvironment, float]] = {}
        self.lock = asyncio.Lock()

        self.router = router if router is not None else APIRouter()
        self._setup_routes()

        if router is None:  # Standalone mode: build a default FastAPI app
            self.app: FastAPI | None = FastAPI()
            self.app.include_router(self.router)
        else:  # Mounted mode: caller mounts self.router onto their own app
            self.app = None

    def _get_env(self, env_id: str) -> TEnvironment:
        try:
            env, _ = self.envs[env_id]
        except KeyError:
            raise HTTPException(
                status_code=404, detail=f"Environment {env_id} not found"
            ) from None
        self.envs[env_id] = (env, time.time())
        return env

    async def close(self):
        await asyncio.gather(*[
            self._close(env_id) for env_id in list(self.envs.keys())
        ])

    async def _close(self, env_id: str):
        try:
            env, _ = self.envs[env_id]
            await env.close()
        except Exception:
            logger.exception(f"Failed to close env {env_id}")

    def _setup_routes(self):
        api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

        def verify_api_key(api_key: str | None = Security(api_key_header)):
            if self.api_key and (
                api_key is None or not secrets.compare_digest(api_key, self.api_key)
            ):
                raise HTTPException(
                    status_code=403, detail="Invalid or missing API key"
                )

        @self.router.post("/start", dependencies=[Depends(verify_api_key)])
        async def start(req: StartRequest):
            with handle_exc_as_http_exc():
                env = await asyncio.to_thread(req.make_env, self.dataset)

            async with self.lock:
                env_id = str(uuid.uuid4())
                self.envs[env_id] = (env, time.time())
                return {"env_id": env_id}

        @self.router.post("/reset", dependencies=[Depends(verify_api_key)])
        async def reset(req: EnvRequest):
            async with self.lock:
                env = self._get_env(req.env_id)

            with handle_exc_as_http_exc():
                obs, tools = await env.reset()

            return (
                # Include info so the client receives the full message
                MessagesAdapter.dump_python(obs, context={"include_info": True}),
                ToolsAdapter.dump_python(tools, exclude_none=True, by_alias=True),
            )

        @self.router.post("/step", dependencies=[Depends(verify_api_key)])
        async def step(req: StepRequest):
            async with self.lock:
                env = self._get_env(req.env_id)

            with handle_exc_as_http_exc():
                obs, *reward_done_trunc = await env.step(req.action)

            # Include info so the client receives the full message
            obs_serialized = MessagesAdapter.dump_python(
                obs, context={"include_info": True}
            )
            return obs_serialized, *reward_done_trunc

        @self.router.post("/close", dependencies=[Depends(verify_api_key)])
        async def close(req: EnvRequest):
            async with self.lock:
                env = self._get_env(req.env_id)
                # Even if env.close() fails, untrack this. It is more likely that
                # the env crashed, not that we somehow lost track of it.
                del self.envs[req.env_id]

            with handle_exc_as_http_exc():
                await env.close()

            return {"env_id": req.env_id}

        @self.router.post("/close_old_envs", dependencies=[Depends(verify_api_key)])
        async def close_old_envs(req: FlushRequest):
            """Endpoint to close environments that have not been used in a while.

            Useful for cleaning up dangling environments en masse.
            """
            now = time.time()

            async def close(env_id: str, env: TEnvironment) -> str | None:
                try:
                    await env.close()
                except Exception:
                    logger.exception(f"Failed to close env {env_id}")
                    return None
                return env_id

            # Pop stale envs from self.envs under lock protection so no other handler
            # can reach them mid-close
            to_close: list[tuple[str, TEnvironment]] = []
            async with self.lock:
                for env_id, (env, last_used) in list(self.envs.items()):
                    if now - last_used > req.last_used:
                        to_close.append((env_id, env))
                for env_id, _ in to_close:
                    del self.envs[env_id]
            # Release the lock before awaiting env.close() so the cleanup doesn't
            # block other endpoints.
            closed = await asyncio.gather(*list(starmap(close, to_close)))

            return {
                "closed_env_ids": [env_id for env_id in closed if env_id is not None]
            }

        @self.router.get("/info", dependencies=[Depends(verify_api_key)])
        def info():
            try:
                dataset_len: int | None = len(self.dataset)
            except TypeError:
                dataset_len = None
            return {
                "dataset_size": dataset_len,
                "running_env_ids": list(self.envs.keys()),
            }

    def start(self) -> None:
        if self.app is None:
            raise RuntimeError(
                f"{type(self).__name__} was constructed with an external router; "
                "mount self.router on your own FastAPI app and run uvicorn there."
            )
        uvicorn.run(self.app, host=self.host, port=self.port, log_level="debug")

    async def astart(self) -> None:
        """Async equivalent of start()."""
        if self.app is None:
            raise RuntimeError(
                f"{type(self).__name__} was constructed with an external router; "
                "mount self.router on your own FastAPI app and run uvicorn there."
            )
        config = uvicorn.Config(
            self.app, host=self.host, port=self.port, log_level="debug"
        )
        server = uvicorn.Server(config)
        await server.serve()


@contextmanager
def handle_exc_as_http_exc():
    # If an environment fails, we don't want to tear down the whole server
    try:
        yield
    except Exception as e:
        logger.exception("Exception in environment.")
        raise HTTPException(
            status_code=500, detail=traceback.format_exc() + "\n" + repr(e)
        ) from None
