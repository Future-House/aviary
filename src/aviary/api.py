import os
import secrets
import uuid
from typing import TYPE_CHECKING, Annotated, Any

import httpx

from aviary.db import (
    EnvironmentBase,
    EnvironmentDB,
    EnvironmentDBBackend,
    EnvironmentDBSchema,
    FrameDB,
    FrameDBSchema,
)
from aviary.render import Frame

if TYPE_CHECKING:
    from fastapi import FastAPI


class EnvDBClient:
    """Interact with an Environment DB via REST API."""

    def __init__(
        self,
        server_url: str,
        request_headers: httpx._types.HeaderTypes | None = None,
        request_timeout: float | None = None,
    ):
        self._request_url = server_url
        self._request_headers = request_headers
        self._request_timeout = request_timeout

    async def write_environment_instance(self, name: str) -> uuid.UUID:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self._request_url}/environment_instance",
                params={
                    "env_name": name,
                },
                headers=self._request_headers,
                timeout=self._request_timeout,
            )
            response.raise_for_status()
            return response.json()

    async def write_environment_frame(
        self,
        environment_id: uuid.UUID,
        frame: Frame,
    ) -> uuid.UUID:
        async with httpx.AsyncClient() as client:
            frame_data = frame.model_dump()
            response = await client.post(
                f"{self._request_url}/environment_frame",
                params={"environment_id": str(environment_id)},
                json={
                    "state": frame_data.get("state"),
                    "metadata": frame_data.get("info"),
                },
                headers=self._request_headers,
                timeout=self._request_timeout,
            )
            response.raise_for_status()
            return response.json()

    async def get_environment_instances(
        self,
        name: str | None = None,
        environment_id: uuid.UUID | None = None,
    ) -> list[EnvironmentDB]:
        async with httpx.AsyncClient() as client:
            params = (
                {"env_name": name} if name else {"environment_id": str(environment_id)}
            )
            response = await client.get(
                f"{self._request_url}/environment_instance",
                params=params,
                headers=self._request_headers,
                timeout=self._request_timeout,
            )
            response.raise_for_status()
            return [EnvironmentDB(**obj) for obj in response.json()]

    async def get_environment_frames(
        self,
        environment_id: uuid.UUID | None = None,
        frame_id: uuid.UUID | None = None,
    ) -> list[FrameDB]:
        async with httpx.AsyncClient() as client:
            params = (
                {"environment_id": str(environment_id)}
                if environment_id
                else {"frame_id": str(frame_id)}
            )
            response = await client.get(
                f"{self._request_url}/environment_frame",
                params=params,
                headers=self._request_headers,
                timeout=self._request_timeout,
            )
            response.raise_for_status()
            return [FrameDB(**obj) for obj in response.json()]


def make_environment_db_server(render_docs: bool = False) -> "FastAPI":  # noqa: C901
    """Make a FastAPI app, an interface for the Environment DB."""
    try:
        from fastapi import Depends, FastAPI, HTTPException, status
        from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Environment DB server requires the 'server' extra for 'fastapi'. Please:"
            " `pip install aviary[server]`."
        ) from exc

    backend = EnvironmentDBBackend()

    async def ensure_session():
        if not backend.Session:
            await backend.populate_session(
                base=EnvironmentBase, uri=os.environ["ENV_DB_URI"]
            )
        yield

    asgi_app = FastAPI(
        title="Aviary Environment DB API",
        description="CRUD operations for the Aviary Environment DB",
        # Only render Swagger docs if local since we don't have a login here
        docs_url="/docs" if render_docs else None,
        redoc_url="/redoc" if render_docs else None,
    )
    auth_scheme = HTTPBearer()

    async def validate_token(
        token: Annotated[HTTPAuthorizationCredentials, Depends(auth_scheme)],
    ) -> HTTPAuthorizationCredentials:
        # NOTE: don't use os.environ.get() to avoid possible empty string matches, and
        # to have clearer server failures if the AUTH_TOKEN env var isn't present
        if not secrets.compare_digest(token.credentials, os.environ["AUTH_TOKEN"]):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return token

    @asgi_app.post("/environment_instance")
    async def add_environment_instance(
        env_name: str,
        _: Annotated[HTTPAuthorizationCredentials, Depends(validate_token)],
        _db: Annotated[None, Depends(ensure_session)],
    ) -> uuid.UUID:
        return await backend.add_environment_instance(name=env_name)

    @asgi_app.post("/environment_frame")
    async def add_environment_frame(
        environment_id: uuid.UUID,
        state: dict[str, Any],
        _: Annotated[HTTPAuthorizationCredentials, Depends(validate_token)],
        _db: Annotated[None, Depends(ensure_session)],
        metadata: dict[str, Any] | None = None,
    ) -> uuid.UUID:
        return await backend.add_environment_frame(
            environment_id=environment_id, state=state, metadata=metadata
        )

    @asgi_app.get("/environment_instance")
    async def get_environment_instances(
        _: Annotated[HTTPAuthorizationCredentials, Depends(validate_token)],
        _db: Annotated[None, Depends(ensure_session)],
        env_name: str | None = None,
        environment_id: uuid.UUID | None = None,
    ) -> list[EnvironmentDBSchema]:
        if not (env_name or environment_id):
            raise HTTPException(400, "Must provide either a name or environment_id")
        return await backend.get_environment_instances(
            name=env_name, environment_id=environment_id
        )

    @asgi_app.get("/environment_frame")
    async def get_environment_frames(
        _: Annotated[HTTPAuthorizationCredentials, Depends(validate_token)],
        _db: Annotated[None, Depends(ensure_session)],
        environment_id: uuid.UUID | None = None,
        frame_id: uuid.UUID | None = None,
    ) -> list[FrameDBSchema]:
        if not (frame_id or environment_id):
            raise HTTPException(400, "Must provide either a frame_id or environment_id")
        return await backend.get_environment_frames(
            environment_id=environment_id, frame_id=frame_id
        )

    return asgi_app
