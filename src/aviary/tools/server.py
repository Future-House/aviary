import os
import secrets
import sys
from inspect import signature

from pydantic import Field, create_model

from aviary.tools.base import Tool, reverse_type_map
from aviary.utils import is_coroutine_callable


def make_tool_server(tools: list[Tool], name: str | None = None):  # noqa: C901
    try:
        from fastapi import Depends, FastAPI, HTTPException, status
        from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Please install aviary with the 'server' extra like so:"
            " `pip install aviary[server]`."
        ) from exc

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

    def create_request_model_from_tool(tool):
        fields = {}
        for pname, info in tool.info.parameters.properties.items():
            if pname == "type":
                continue
            # we just assume it exists
            ptype = reverse_type_map[info["type"]]

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
        title=name or "Aviary Tool Server",
        description="API Server for Aviary Environment Tools",
        dependencies=[Depends(validate_token)],
    )

    # filter only for tools that are executable
    tools = [tool for tool in tools if hasattr(tool, "_tool_fn")]

    # Dynamically create routes for each tool
    for tool in tools:
        tool_name = tool.info.name
        tool_description = tool.info.description
        RequestModel = create_request_model_from_tool(tool)
        return_type = signature(tool._tool_fn).return_annotation

        # ensure the this will be in fast api scope
        # close your eyes PR reviewers
        # also fuck your IDE tools
        RequestModel.__module__ = sys._getframe(1).f_globals.get("__name__", "__main__")

        def create_tool_handler(tool_fn, RequestModel, tool_description):
            async def _tool_handler(
                data: RequestModel,  # type: ignore[valid-type]
            ):
                try:
                    # Call the tool function with the provided arguments
                    if is_coroutine_callable(tool_fn):
                        return await tool_fn(**data.model_dump())  # type: ignore[attr-defined]
                    return tool_fn(**data.model_dump())  # type: ignore[attr-defined]
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e)) from e

            _tool_handler.__doc__ = tool_description
            return _tool_handler

        tool_handler = create_tool_handler(
            tool._tool_fn, RequestModel, tool_description
        )

        # Add a POST route for the tool
        web_app.post(
            f"/{tool_name}",
            summary=tool_name,
            name=tool_name,
            response_model=return_type,
            description=tool_description,
        )(tool_handler)

    return web_app
