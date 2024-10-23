import asyncio
import os

import click
import uvicorn

from aviary.env import Environment
from aviary.tools.server import make_tool_server


@click.group()
def cli():
    pass


@cli.command()
@click.argument("env")
@click.option("--host", default="localhost")
@click.option("--port", default=8000)
@click.option("--token", default="secret")
def tools(env, host, port, token):
    if not os.environ.get("AUTH_TOKEN"):
        os.environ["AUTH_TOKEN"] = token

    # use empty task to trigger
    # an empty task/no problem
    def env_factory():
        return Environment.from_name(env, task="")

    app = asyncio.run(make_tool_server(env_factory))
    click.echo(
        f"View tools at http://{host}:{port}/docs and log in with token {token!r}"
    )
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()
