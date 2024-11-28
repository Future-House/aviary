import asyncio
import logging
import time
import traceback
import uuid
from contextlib import contextmanager
from itertools import starmap
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from aviary.env import Environment, TaskDataset
from aviary.tools import MessagesAdapter, ToolRequestMessage, ToolsAdapter

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException
except ImportError:
    # We will raise if a TaskDatasetServer is instantiated but FastAPI/uvicorn are not available
    uvicorn = FastAPI = HTTPException = None  # type: ignore[misc,assignment]

logger = logging.getLogger(__name__)


class StartRequest(BaseModel):
    task_idx: int | None = Field(
        default=None,
        description="Index of the dataset to start. "
        "If provided, will call TaskDataset.get_new_env_by_idx(); "
        "otherwise, TaskDataset.get_new_env().",
    )


class EnvRequest(BaseModel):
    env_id: str = Field(description="Maps to a running env ID")


class StepRequest(BaseModel):
    env_id: str = Field(description="Maps to a running env ID")
    action: ToolRequestMessage


class FlushRequest(BaseModel):
    last_used: float = Field(default=3600, description="Seconds since last use")


DEFAULT_SERVER_PORT = 8041
BIND_ALL_HOST = "0.0.0.0"  # noqa: S104


# Not sure why, but mypy complains if we use the TEnvironment in aviary.env, so redefine here
TEnvironment = TypeVar("TEnvironment", bound=Environment)


class TaskDatasetServer(Generic[TEnvironment]):
    def __init__(
        self,
        dataset: TaskDataset[TEnvironment],
        host: str = BIND_ALL_HOST,
        port: int = DEFAULT_SERVER_PORT,
    ):
        if FastAPI is None:
            raise ImportError(
                "FastAPI and Uvicorn are required to run a TaskDatasetServer. "
                "Please `pip install fhaviary[server]`."
            )

        self.dataset = dataset
        self.host = host
        self.port = port

        self.app = FastAPI()

        # env ID -> (env, last used timestamp)
        self.envs: dict[str, tuple[TEnvironment, float]] = {}
        self.lock = asyncio.Lock()
        self._setup_routes()

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
        @self.app.post("/start")
        async def start(req: StartRequest):
            with handle_exc_as_http_exc():
                if req.task_idx is None:
                    env = await asyncio.to_thread(self.dataset.get_new_env)
                else:
                    env = await asyncio.to_thread(
                        self.dataset.get_new_env_by_idx, req.task_idx
                    )

            async with self.lock:
                env_id = str(uuid.uuid4())
                self.envs[env_id] = (env, time.time())
                return {"env_id": env_id}

        @self.app.post("/reset")
        async def reset(req: EnvRequest):
            async with self.lock:
                env = self._get_env(req.env_id)

            with handle_exc_as_http_exc():
                obs, tools = await env.reset()

            return (
                MessagesAdapter.dump_python(obs),
                ToolsAdapter.dump_python(tools, exclude_none=True, by_alias=True),
            )

        @self.app.post("/step")
        async def step(req: StepRequest):
            async with self.lock:
                env = self._get_env(req.env_id)

            with handle_exc_as_http_exc():
                obs, *reward_done_trunc = await env.step(req.action)

            obs_serialized = MessagesAdapter.dump_python(obs)
            return obs_serialized, *reward_done_trunc

        @self.app.post("/close")
        async def close(req: EnvRequest):
            async with self.lock:
                env = self._get_env(req.env_id)
                # Even if env.close() fails, untrack this. It is more likely that
                # the env crashed, not that we somehow lost track of it.
                del self.envs[req.env_id]

                with handle_exc_as_http_exc():
                    await env.close()

            return {"env_id": req.env_id}

        @self.app.post("/close_old_envs")
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
                else:
                    del self.envs[env_id]
                    return env_id

            to_close: list[tuple[str, TEnvironment]] = []
            async with self.lock:
                for env_id, (env, last_used) in list(self.envs.items()):
                    if now - last_used > req.last_used:
                        to_close.append((env_id, env))

                closed = await asyncio.gather(*list(starmap(close, to_close)))

            return {
                "closed_env_ids": [env_id for env_id in closed if env_id is not None]
            }

        @self.app.get("/info")
        def info():
            try:
                dataset_len: int | None = len(self.dataset)
            except TypeError:
                dataset_len = None
            return {
                "dataset_size": dataset_len,
                "running_env_ids": list(self.envs.keys()),
            }

    def start(self):
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="debug",
        )

    async def astart(self):
        """Async equivalent of start()."""
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
        raise HTTPException(
            status_code=500, detail=traceback.format_exc() + "\n" + repr(e)
        ) from None
