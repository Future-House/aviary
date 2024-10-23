import os
import secrets
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, create_model

from aviary.tools.base import Tool, ToolCall, ToolRequestMessage, reverse_type_map


async def make_tool_server(  # noqa: C901, PLR0915
    environment_factory: Callable,
    name: str = "Aviary Tool Server",
    env_path: Path | None = None,
):
    """Create a FastAPI server for the provided environment.

    This function exposes one endpoint per tool and endpoints to create/view/delete environments.
    In contrast to other environment servers that expose an action endpoint, this one exposes all tools individually.

    This is only for debugging tools and not intended as a strategy for working with environments.
    Most environments have side-effects from using tools that occur in the step function. This
    bypasses that and allows you to call tools directly.

    Args:
        environment_factory: A callable that returns an environment instance.
        name: The name of the server. Defaults to Aviary Tool Server.
        env_path: The path to the directory to store environments
    """
    try:
        import cloudpickle as pickle
        from fastapi import Depends, FastAPI, HTTPException, status
        from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Please install aviary with the 'server' extra like so:"
            " `pip install aviary[server]`."
        ) from exc

    if not env_path:
        env_path = Path(tempfile.gettempdir())
    auth_scheme = HTTPBearer()

    async def validate_token(
        credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),  # noqa: B008
    ) -> str:
        # NOTE: don't use os.environ.get() to avoid possible empty string matches, and
        # to have clearer server failures if the AUTH_TOKEN env var isn't present
        if not secrets.compare_digest(
            credentials.credentials, os.environ["AUTH_TOKEN"]
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect bearer token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return credentials.credentials

    # these seem useful in other contexts, but from what I read
    # it is discouraged to save/load so leaving it defined here
    def save_environment(environment, tools, environment_id):
        # make sure we force all tools to pickle
        for tool in tools:
            tool._force_pickle_fn = True
        with open(env_path / f"{environment_id}.pkl", "wb") as f:
            pickle.dump((environment, tools), f)

    def load_environment(environment_id):
        if not (env_path / f"{environment_id}.pkl").exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Environment {environment_id} not found",
            )
        with open(env_path / f"{environment_id}.pkl", "rb") as f:
            return pickle.load(f)

    def make_environment_id():
        return f"env{str(uuid4())[:8].replace('-', '')}"

    def create_request_model_from_tool(tool: Tool) -> BaseModel:
        fields = {}
        for pname, info in tool.info.parameters.properties.items():
            if pname == "type":
                continue
            # we just assume it exists
            ptype = reverse_type_map[info["type"]] if "type" in info else Any

            # decipher optional description, optional default, and type
            if pname in tool.info.parameters.required:
                if "description" in info:
                    fields[pname] = (ptype, Field(description=info["description"]))
                else:
                    fields[pname] = (ptype, ...)
            elif "description" in info:
                fields[pname] = (
                    ptype | None,
                    Field(description=info["description"], default=None),
                )
            else:
                fields[pname] = (ptype | None, None)

        return create_model(f"{tool.info.name.capitalize()}Params", **fields)  # type: ignore[call-overload]

    web_app = FastAPI(
        title=name,
        description="API Server for Aviary Environment Tools",
        dependencies=[Depends(validate_token)],
    )

    # make a starting environment to save tools
    env = environment_factory()
    _, tools = await env.reset()

    # Dynamically create routes for each tool
    for tool in (t for t in tools if hasattr(t, "_tool_fn")):
        tool_name = tool.info.name
        tool_description = tool.info.description
        RequestModel = create_request_model_from_tool(tool)

        # ensure the this will be in fast api scope
        # because fastapi will barf on a request model that isn't in scope
        # close your eyes PR reviewers
        # also fuck your IDE tools
        RequestModel.__module__ = sys._getframe(1).f_globals.get("__name__", "__main__")

        def create_tool_handler(tool_name, RequestModel, tool_description):
            async def _tool_handler(
                data: RequestModel,  # type: ignore[valid-type]
                environment_id: str = "",
            ):
                if environment_id:
                    env, env_tools = load_environment(environment_id)
                else:
                    env = environment_factory()
                    _, env_tools = await env.reset()
                    environment_id = make_environment_id()

                # ok now find the tool_fn to call it with
                # that came from the env I just loaded
                msg = ToolRequestMessage(
                    tool_calls=[ToolCall.from_name(tool_name, **data.model_dump())]  # type: ignore[attr-defined]
                )
                try:
                    result_msgs, done, *_ = await env.step(msg)
                except Exception as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
                    ) from e

                if done:
                    _, env_tools = await env.reset()

                save_environment(env, env_tools, environment_id)
                return {
                    "result": "\n\n".join([
                        str(msg.content) for msg in result_msgs if msg.content
                    ]),
                    "environment_id": environment_id,
                }

            _tool_handler.__doc__ = tool_description
            return _tool_handler

        tool_handler = create_tool_handler(
            tool.info.name, RequestModel, tool_description
        )

        # Add a POST route so we can invoke the tool function
        web_app.post(
            f"/{tool_name}",
            summary=tool_name,
            name=tool_name,
            description=tool_description,
        )(tool_handler)

        # Add environment endpoints
        @web_app.get(
            "/env/create",
            summary="Create Environment",
            description="Create a new environment",
        )
        async def create_environment_endpoint():
            env = environment_factory()
            _, tools = await env.reset()
            environment_id = make_environment_id()
            save_environment(env, tools, environment_id)
            return environment_id

        @web_app.get(
            "/env/delete/{environment_id}",
            summary="Delete Environment",
            description="Delete an environment",
        )
        async def delete_environment_endpoint(environment_id: str):
            if (env_path / f"{environment_id}.pkl").exists():
                (env_path / f"{environment_id}.pkl").unlink()
            return environment_id

        @web_app.get(
            "/env/view/{environment_id}",
            summary="View Environment",
            description="View an environment",
        )
        async def view_environment_endpoint(environment_id: str):
            if not (env_path / f"{environment_id}.pkl").exists():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Environment {environment_id} not found",
                )
            with (env_path / f"{environment_id}.pkl").open("rb") as f:
                env, _ = pickle.load(f)

            return env.state

    return web_app
