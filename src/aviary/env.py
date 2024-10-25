from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import logging
import random
from abc import ABC, abstractmethod
from collections.abc import Iterator
from copy import deepcopy
from typing import Annotated, Generic, Self, TypeAlias, TypeVar, cast

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationInfo,
    WrapSerializer,
    field_validator,
)

from aviary.message import Message
from aviary.tools import Tool, ToolCall, ToolRequestMessage, ToolResponseMessage
from aviary.utils import is_coroutine_callable

logger = logging.getLogger(__name__)

# TODO: make TypeVar after https://github.com/pydantic/pydantic/milestone/13
# NOTE: can't use pydantic.JsonValue here because it will deep copy all the way
# down JSON, and we want to support shallow copying capability
Serializable: TypeAlias = dict | list | int | float | str | bool | BaseModel


class Frame(BaseModel):
    """A frame is a snapshot at a given timestep. The name comes from video frame."""

    deepcopy: bool = Field(
        default=True,
        description=(
            "Whether to deepcopy the state and info fields. "
            "Disable if you're sure they're immutable or desire mutability."
        ),
    )

    @staticmethod
    def _custom_serializer(value: Serializable, handler, info):  # noqa: ARG004
        if isinstance(value, BaseModel):
            return value.model_dump()
        return handler(value)

    state: Annotated[Serializable | None, WrapSerializer(_custom_serializer)] = Field(
        default=None,
        description=(
            "Either entire (or a subset of) the current state. Leave as default of None"
            " if state is irrelevant."
        ),
    )
    info: Annotated[Serializable | None, WrapSerializer(_custom_serializer)] = Field(
        default=None, description="Optional metadata that doesn't vary with state."
    )

    @field_validator("state", "info")
    @classmethod
    def make_deepcopy(cls, v: Serializable, info: ValidationInfo) -> Serializable:
        if info.data["deepcopy"]:
            return deepcopy(v)
        return v


# NOTE: setting to None means there is no state
TEnvState = TypeVar("TEnvState")


class Environment(ABC, Generic[TEnvState]):
    """
    An environment is a stateful place where agents use tools and make observations.

    Tools are housed in the environment because they can interact with the environment.

    Environments (and their contained tools) are not trainable.
    """

    tools: list[Tool]
    state: TEnvState

    @abstractmethod
    async def step(
        self, action: ToolRequestMessage
    ) -> tuple[list[Message], float, bool, bool]:
        """Take a step in the environment.

        Args:
            action: Action to take.

        Returns:
            Four-tuple of new observations, instantaneous reward for this action, a flag
                symbolizing if the episode is done, and a flag symbolizing if the
                episode was truncated (e.g. via early stopping).
        """

    @abstractmethod
    async def reset(self) -> tuple[list[Message], list[Tool]]:
        """
        Reset the environment and collect initial observation(s).

        Possible observations could be instructions on how tools are related,
        or the goal of the environment.

        Returns:
            Two-tuple of initial observations and tools.
        """

    def filter_invalid_tool_calls(
        self, message: ToolRequestMessage
    ) -> tuple[ToolRequestMessage, ToolRequestMessage]:
        """Split a list of tool calls into valid and invalid subsets.

        Args:
            message: Tool request message containing tool calls.

        Returns:
            Two-tuple of ToolRequestMessage containing valid messages and
                ToolRequestMessage containing invalid messages
        """
        valid, invalid = [], []
        for tool_call in message.tool_calls:
            tool_used_in_tool_call = next(
                (t for t in self.tools if t.info.name == tool_call.function.name), None
            )
            if tool_used_in_tool_call is not None:
                valid.append(tool_call)
            else:
                invalid.append(tool_call)
        return cast(
            tuple[ToolRequestMessage, ToolRequestMessage],
            tuple(
                ToolRequestMessage(
                    role=message.role,
                    content=message.content,
                    function_call=message.function_call,
                    tool_calls=x,
                )
                for x in (valid, invalid)
            ),
        )

    async def exec_tool_calls(
        self,
        message: ToolRequestMessage,
        ordered: bool = False,
        handle_tool_exc: bool = False,
        **function_kwargs,
    ) -> list[ToolResponseMessage]:
        """
        Execute an ordered list of tool calls.

        Args:
            message: ToolRequestMessage containing the tool calls.
            ordered: Opt-in flag for forcing sequential execution (according to order
                in the above message), otherwise tool calls are made concurrently.
            handle_tool_exc: Opt-in flag to suppress Exceptions and return them as a
                ToolResponseMessage.
            **function_kwargs: Keyword arguments to pass to all tool functions.

        Returns:
            Ordered list of ToolResponseMessages, order matches the order of tool calls
                in the input message.
        """

        async def _exec_tool_call(tool_call: ToolCall) -> ToolResponseMessage:
            try:
                tool = next(
                    t for t in self.tools if t.info.name == tool_call.function.name
                )
            except StopIteration as exc:
                raise ValueError(
                    f"{tool_call.function.name!r} not a valid name in"
                    f" { {t.info.name for t in self.tools} }."
                ) from exc
            # we do a special convenience to make
            # state be optional in the function signature
            need_to_filter = (
                "state" in function_kwargs
                and "state" not in inspect.signature(tool._tool_fn).parameters
                and not hasattr(tool._tool_fn, "requires_state")
            )
            filtered_kwargs = (
                {k: v for k, v in function_kwargs.items() if k != "state"}
                if need_to_filter
                else function_kwargs
            )
            tool_exc: Exception | None = None
            try:
                if is_coroutine_callable(tool._tool_fn):
                    content = await tool._tool_fn(
                        **tool_call.function.arguments, **filtered_kwargs
                    )
                else:
                    # If the function is synchronous, run on a thread
                    content = await asyncio.to_thread(
                        tool._tool_fn,
                        **tool_call.function.arguments,
                        **filtered_kwargs,
                    )
            except Exception as exc:
                if not handle_tool_exc:
                    raise
                logger_msg = f"Encountered exception during tool call for tool {tool.info.name}: {exc!r}"
                # logger.exception is just too verbose and clogs up console logging. This is a
                # more human-friendly version: log a readable error message and emit the exception
                # at DEBUG level.
                logger.error(logger_msg)  # noqa: TRY400
                logger.debug(str(exc), exc_info=True)
                tool_exc = exc
            if tool_exc:
                # No need to mention tool.info.name here, since it'll get wrapped in a ToolResponseMessage
                s_content = f"Encountered exception during tool call: {tool_exc}"
            elif isinstance(content, str):
                s_content = content
            elif isinstance(content, BaseModel):
                s_content = content.model_dump_json(exclude_none=True, by_alias=True)
            else:  # Fallback when content is another type, or None
                s_content = json.dumps(content)
            return ToolResponseMessage.from_call(tool_call, content=s_content)

        if not ordered:
            return await asyncio.gather(
                *(_exec_tool_call(tool_call) for tool_call in message.tool_calls)
            )
        return [await _exec_tool_call(tool_call) for tool_call in message.tool_calls]

    @abstractmethod
    def export_frame(self) -> Frame:
        """
        Export the environment as a Frame.

        If you are not sure what to put in the Frame, just give it the entire state.
        Read Field descriptions in Frame for more information.
        """

    async def close(self) -> None:
        """
        Shutdown the environment.

        If this is unimplemented, __del__ will manage cleanup.
        """

    @classmethod
    def from_task(cls, task: str) -> Self:
        """Create an environment from a task description.

        A task is meant to be closer to a user prompt - like what you would expect
        in calling an LLM. This is how the environment should be used after training
        and in deployment. We don't take config here, because the default environment config
        should be general for arbitrary tasks.

        For example, with GSM8k/calculator: "What is 18 * (number of legs on a cat) / moons of mars?"
        """
        raise NotImplementedError(f"{cls.__name__} does not implement from_task")

    @classmethod
    def from_name(cls, name: str, task: str | None = None, **env_kwargs) -> Self:
        """Create an environment from the name of the class. Call `Environment.available()` to see list."""
        new_cls = _get_cls_from_name(ENV_REGISTRY, name)
        if task is not None:
            if env_kwargs:
                raise ValueError("Cannot pass both a task and environment kwargs.")
            return new_cls.from_task(task)
        return new_cls(**env_kwargs)

    @classmethod
    def available(cls) -> set[str]:
        """See list of available environment classes for `from_name`.

        This is not exhaustive, because some may be importable and so you should just
        try to call `from_name`. This is more for logging/debugging purposes.
        """
        return set(ENV_REGISTRY.keys())


# Maps baseline environment names to their module and class names
ENV_REGISTRY: dict[str, tuple[str, str]] = {
    "dummy": ("aviary.env", "DummyEnv"),
    "calculator": ("aviary.envs.gsm8k.env", "CalculatorEnv"),
    "hotpotqa": ("aviary.envs.hotpotqa.env", "HotPotQAEnv"),
}

TEnvironment = TypeVar("TEnvironment", bound=Environment)


class TaskDataset(ABC, Generic[TEnvironment]):
    """A base class for a dataset of tasks as environments.

    Examples of task datasets: GSM8k, HotPotQA, etc.
    These are related environments instances with different problem
    specifications and reward conditions.
    """

    @classmethod
    def from_name(cls, name: str, **env_kwargs) -> TaskDataset:
        return _get_cls_from_name(TASK_DATASET_REGISTRY, name)(**env_kwargs)

    def __len__(self) -> int:
        raise TypeError(f'"Object of type {self.__class__.__name__}" has no len()')

    def get_new_env_by_idx(self, idx: int) -> TEnvironment:
        """Get an env from a finite dataset."""
        raise NotImplementedError(
            f'"{self.__class__.__name__}" does not implement get_new_env_by_idx'
        )

    def get_new_env(self) -> TEnvironment:
        """Get an env from a non-indexable dataset."""
        raise NotImplementedError(
            f'"{self.__class__.__name__}" does not implement get_new_env'
        )

    def iter_batches(
        self, batch_size: int, shuffle: bool = False
    ) -> Iterator[list[TEnvironment]]:
        """Construct batches from this dataset.

        Args:
            batch_size: Size of each batch.
                Note that if this dataset's size is finite and isn't evenly divisible by
                this value, the last yielded batch will be smaller than batch_size.
            shuffle: Opt-in flag to shuffle without replacement.

        Yields:
            An iterator over batches of environments.
        """
        try:
            n = len(self)
        except TypeError:
            # not a finite-length dataset, so construct an infinite iter
            while True:
                yield [self.get_new_env() for _ in range(batch_size)]
        else:
            # finite-length dataset
            idcs = list(range(n))
            if shuffle:
                random.shuffle(idcs)

            while idcs:
                batch_idcs = idcs[:batch_size]
                idcs = idcs[batch_size:]
                yield [self.get_new_env_by_idx(idx) for idx in batch_idcs]


# Maps baseline task dataset names to their module and class names
TASK_DATASET_REGISTRY: dict[str, tuple[str, str]] = {
    "dummy": ("aviary.env", "DummyTaskDataset"),
    "gsm8k": ("aviary.envs.gsm8k.env", "GSM8kDataset"),
    "hotpotqa": ("aviary.envs.hotpotqa.env", "HotPotQADataset"),
}


class TaskConfig(BaseModel):
    """Convenience for making a config file entry for a TaskDataset."""

    model_config = ConfigDict(extra="forbid")

    name: str
    task_kwargs: dict[str, BaseModel | JsonValue] = Field(
        default_factory=dict, description="Arguments to pass to TaskDataset.from_name()"
    )
    train_kwargs: dict[str, BaseModel | JsonValue] = Field(
        default_factory=dict, description="Additional arguments for the training split."
    )
    eval_kwargs: dict[str, BaseModel | JsonValue] = Field(
        default_factory=dict,
        description="Additional arguments for the evaluation split.",
    )
    test_kwargs: dict[str, BaseModel | JsonValue] = Field(
        default_factory=dict, description="Additional arguments for the test split."
    )

    def make_dataset(self, split: str) -> TaskDataset:
        if split == "train":
            split_kw = self.task_kwargs | self.train_kwargs
        elif split == "eval":
            split_kw = self.task_kwargs | self.eval_kwargs
        elif split == "test":
            split_kw = self.task_kwargs | self.test_kwargs
        else:
            raise NotImplementedError(f"Didn't handle split {split!r}.")
        return TaskDataset.from_name(self.name, **split_kw)


class DummyEnvState(BaseModel):
    messages: list[Message]
    reward: float = 0
    done: bool = False


class DummyEnv(Environment[DummyEnvState]):
    """Simple Environment with basic functionality and no network usage."""

    State = DummyEnvState

    def __init__(self, task: str | None = None, end_immediately: bool = True):
        self.end_immediately = end_immediately
        self.task = task

    @classmethod
    def from_task(cls, task: str) -> DummyEnv:
        return cls(task=task)

    async def step(
        self, action: ToolRequestMessage
    ) -> tuple[list[Message], float, bool, bool]:
        msgs: list[Message] = await self.exec_tool_calls(action, state=self.state)
        self.state.messages.extend(msgs)
        return msgs, self.state.reward, self.state.done, False

    async def reset(self) -> tuple[list[Message], list[Tool]]:
        def print_story(story: str, state: DummyEnvState) -> None:  # noqa: ARG001
            """Print a story.

            Args:
                story: Story to print.
                state: Environment state.
            """
            state.reward = 1
            state.done = self.end_immediately

        def cast_float(x: str) -> float:
            """Cast the input argument x to a float."""
            return float(x)

        def cast_int(x: float) -> int:
            """Cast the input argument x to an integer."""
            return int(x)

        self.tools = [
            Tool.from_function(print_story),
            Tool.from_function(cast_float, allow_empty_param_descriptions=True),
            Tool.from_function(cast_int, allow_empty_param_descriptions=True),
        ]
        self.state = type(self).State(
            messages=[
                Message(
                    content="Write a 5 word story via print_story"
                    + (f" about {self.task}" if self.task else "")
                )
            ],
        )
        return self.state.messages, self.tools

    def export_frame(self) -> Frame:
        return Frame(
            state={"messages": [m.content for m in self.state.messages]},
            info={
                "tool_names": [t.info.name for t in self.tools],
                "done": self.state.done,
                "reward": self.state.reward,
            },
        )


class DummyTaskDataset(TaskDataset[DummyEnv]):
    """A dummy task of infinite DummyEnvs."""

    def get_new_env(self) -> DummyEnv:
        return DummyEnv()

    def __bool__(self) -> bool:
        return True


def _get_cls_from_name(registry: dict[str, tuple[str, str]], name: str):
    try:
        module_name, cls_name = registry[name]
    except KeyError:
        raise ValueError(f"Unknown environment name: {name}") from None

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        # TODO: before release: add install instructions per env?
        raise ImportError(
            f"Could not import env from {module_name}; you need to install it."
        ) from None

    return getattr(module, cls_name)
