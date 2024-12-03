from collections.abc import Callable
from functools import update_wrapper
from typing import Any

from pydantic import BaseModel, Field

from aviary.env import Environment, Frame
from aviary.message import Message
from aviary.tools import Messages, Tool, ToolRequestMessage
from aviary.utils import is_coroutine_callable


class DynamicState(BaseModel):
    """Dynamic env state model that adapts to provided extras."""

    reward: float = 0
    done: bool = False
    # we do not use ConfigDict extras because we cannot get getattr
    # which is nice.
    extras: dict[str, Any] = Field(
        default_factory=dict,
        description="Extra state variables to persist between steps",
    )

    def __getattr__(self, name: str):
        """Allow direct access to extras as attributes."""
        if name in self.extras:
            return self.extras[name]
        raise AttributeError(f"'State' has no attribute {name!r}")


class FunctionalEnvironment(Environment[DynamicState]):
    """Environment class for function-based environments.

    See @fenv.start() decorator for complete usage details.
    """

    def __init__(
        self,
        start_fn: Callable,
        tools: list[Tool],
        allow_concurrency: bool,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.start_fn = start_fn
        self.tools = tools
        self.args = args
        self.kwargs = kwargs
        self.state = DynamicState()
        self.allow_concurrency = allow_concurrency

    async def _call_start_fn(self) -> tuple[str, DynamicState]:
        """Call start_fn handling both sync and async cases."""
        if is_coroutine_callable(self.start_fn):
            result = await self.start_fn(*self.args, **self.kwargs)
        else:
            result = self.start_fn(*self.args, **self.kwargs)

        if not isinstance(result, tuple) or len(result) != 2:  # noqa: PLR2004
            raise TypeError("Start function must return (observation, state_dict)")

        obs, state_dict = result
        if not isinstance(state_dict, dict):
            raise TypeError("State must be a dictionary")

        return obs, DynamicState(extras=state_dict)

    async def reset(self) -> tuple[Messages, list[Tool]]:
        obs, self.state = await self._call_start_fn()
        return [Message(content=obs)], self.tools

    async def step(
        self, action: ToolRequestMessage
    ) -> tuple[Messages, float, bool, bool]:
        msgs = await self.exec_tool_calls(
            action,
            state=self.state,
            concurrency=self.allow_concurrency,
            handle_tool_exc=True,
        )
        return msgs, self.state.reward, self.state.done, False

    def export_frame(self) -> Frame:
        """Export the current state of the environment."""
        return Frame(
            state={
                "done": self.state.done,
                "reward": self.state.reward,
                "extras": self.state.extras,
            },
            info={"tool_names": [t.info.name for t in self.tools]},
        )


class EnvironmentBuilder:
    """Builder class for constructing functional environments."""

    def __init__(
        self,
        start_fn: Callable[..., tuple[str, dict[str, Any]]],
        allow_concurrency: bool,
    ):
        self.start_fn = start_fn
        self.tools: list[Tool] = []
        self.allow_concurrency = allow_concurrency
        update_wrapper(self, start_fn)

    def __call__(self, *args, **kwargs):
        """Create a new environment instance."""
        return FunctionalEnvironment(
            self.start_fn, self.tools, self.allow_concurrency, *args, **kwargs
        )

    def tool(self, **tool_kwargs):
        """Decorator to add tools to the environment."""

        def decorator(func: Callable):
            tool = Tool.from_function(
                func, **tool_kwargs, allow_empty_param_descriptions=True
            )
            # Check for name conflicts
            if any(t.info.name == tool.info.name for t in self.tools):
                raise RuntimeError(f"Tool with name '{tool.info.name}' already exists")
            self.tools.append(tool)
            return func

        return decorator


class fenv:  # noqa: N801
    """Factory class for creating functional environments."""

    @staticmethod
    def start(allow_concurrency: bool = False):
        """Initialize a new functional environment definition.

        This decorator marks the starting point for defining a functional environment.
        It should be applied to a function that sets up the initial state and first
        observation for the environment. The decorated function will serve as the
        base for adding tools via the .tool() decorator.

        The decorated function should accept any desired parameters that configure
        the environment instance. It must return a tuple containing:
        1. The initial observation message (str)
        2. A dict of any state variables to persist between steps

        Args:
            allow_concurrency (optional): Whether to allow concurrent tool calls.
            Defaults to False.

        Example:
            @fenv.start()
            def my_env(topic: str):
                return f"Write a story about {topic}", {"chosen_topic": topic}

            @my_env.tool()
            def print_story(story: str, state) -> None:
                '''Print the story and complete the task'''
                print(story)
                state.reward = 1
                state.done = True

            # Usage
            env = my_env(topic="space")
            obs, tools = await env.reset()

        Returns:
            callable: A decorator function that transforms the decorated function into
            an environment builder. The returned environment builder provides a .tool()
            decorator for adding tools to the environment.

        Notes:
            - The state dict returned by the decorated function will automatically include
            'reward' (float) and 'done' (bool) fields that can be modified by tools
            - Tools can access the state dict via an optional 'state' parameter
            - The environment should be done/conclude by setting 'state.done = True'
            - The environment implements the standard Aviary interface with async
            reset() and step() methods
        """

        def decorator(func: Callable) -> EnvironmentBuilder:
            if not callable(func):
                raise TypeError("Decorator must be applied to a function")
            return EnvironmentBuilder(func, allow_concurrency=allow_concurrency)

        return decorator
