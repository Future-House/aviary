# aviary

![PyPI Version](https://img.shields.io/pypi/v/fhaviary)
![PyPI Python Versions](https://img.shields.io/pypi/pyversions/fhaviary)
![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
![Tests](https://github.com/Future-House/aviary/actions/workflows/tests.yml/badge.svg)

Gymnasium framework for training language model agents on constructive tasks.

<!--TOC-->

- [Installation](#installation)
  - [Google Colab](#google-colab)
  - [Developer Installation](#developer-installation)
- [Messages](#messages)
- [Environment](#environment)
- [Functional Environments](#functional-environments)
- [Subclass Environments](#subclass-environments)
  - [Common environments](#common-environments)
  - [Tool](#tool)
    - [Advanced tool descriptions](#advanced-tool-descriptions)
  - [Environment `reset` method](#environment-reset-method)
  - [Environment `step` method](#environment-step-method)
  - [Environment `export_frame` method](#environment-export_frame-method)
  - [View Environment Tools](#view-environment-tools)
- [Environments](#environments)

<!--TOC-->

## Installation

To install aviary (note `fh` stands for FutureHouse):

```bash
pip install fhaviary
```

To install aviary with the bundled environments,
please see the [Environments section below](#environments).

### Google Colab

As of 10/25/2024, unfortunately Google Colab does not yet support Python 3.11 or 3.12
([issue](https://github.com/googlecolab/colabtools/issues/3190)).

Thus, as a workaround, you will need to install Python 3.11 into your notebook.
Here is a sample notebook showing how to do this:
https://colab.research.google.com/drive/1mejZ5cxgKZrMpYEe0iRoanaGGQ0Cr6WI?usp=sharing

Also, note that `async` code works in Google Colab.

### Developer Installation

For local development, please see the [CONTRIBUTING.md](CONTRIBUTING.md).

## Messages

Communication between the agent and environment is done through messages.
Messages have two attributes:

```py
msg = Message(content="Hello, world!", role="assistant")
```

For the meaning of role, see the table below.
You can change around roles as desired,
except for `tool` which has a special meaning in aviary.

| Role      | Host                                             | Example(s)                                                       |
| --------- | ------------------------------------------------ | ---------------------------------------------------------------- |
| assistant | Agent                                            | A tool selector agent's tool selection message                   |
| system    | Agent system prompt                              | "You are an agent."                                              |
| user      | Environment system prompt or emitted observation | HotPotQA problem to solve, or details of an internal env failure |
| tool      | Result of tool run in the environment            | Some number crunching program's output                           |

The `content` is a string that can be anything, or a null value.

## Environment

An environment should have two functions:

```py
obs_msgs, tools = await env.reset()
new_obs_msgs, reward, done, truncated = await env.step(action_msg)
```

where messages are how communication is passed. The `action_msg` should be `ToolRequestMessage` which is 1 or more calls
to tools provided by the `reset`. The `obs_msgs` returned from the environment are `ToolResponseMessage` or other
general messages that are observations. The `reward` is a scalar value. The `done` is a boolean value. The `truncated`
is a boolean value.

## Functional Environments

The easiest way to create an environment is using the functional interface, which just uses functions and decorators to define environments. First, let's define what the environment looks like by defining its `start` function:

```py
from aviary import fenv


@fenv.start()
def my_env(topic):
    # return first observation, and the starting environment state
    # (empty in this case)
    return f"Write a story about {topic}", {}
```

Note that the decorator is a call (`start()`). The `start` decorator starts the definition of an environment. The function, `my_env`, can take whatever you would like and should return a tuple containing the first observation and anything you would like to store about the state of the environment (used to persist/share things between tools). The state will always automatically have an optional `reward` and a boolean `done` that indicates if the environment is complete.

Now we can define some tools:

```py
@my_env.tool()
def multiply(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


@my_env.tool()
def print_story(story: str | bytes, state) -> None:
    """Print a story to user and complete task."""
    print(story)
    state.reward = 1
    state.done = True
```

The tools will be converted into things visible for LLMs using the type hints and the variable descriptions. Thus, the type hinting can be valuable for the agent using it correctly. The docstrings are also passed to the LLM, and is the primary way (along with function name) for communicating about intended tool usage.

You can access the `state` variable in tools, which will have any fields you passed in the return tuple of `start()`. For example, if you returned `{'foo': 'bar'}`, then you could access `state.foo` in the tools.

Stop an environment or set a reward via the `state` variable as shown the second tool. If the reward is not set, it is treated as zero.

Now we can use our environment:

```python
env = my_env(topic="foo")
obs, tools = await env.reset()
```

## Subclass Environments

If you need more control over Environments and tools, you'll want to subclass the `Environment`

First we define an environment by subclassing the `Environment` and defining a `state`. The `state` is all variables
that change per step and we want to keep together. It will be accessible in your tools, so you can use it to store
information that you want to persist between steps and between tools.

```py
from pydantic import BaseModel
from aviary.core import Environment


class ExampleState(BaseModel):
    reward: float = 0
    done: bool = False


class ExampleEnv(Environment[ExampleState]):
    state: ExampleState
```

We do not have other variables aside from `state` for this environment. We could have things like configuration, a name,
tasks, etc. attached to it.

### Common environments

We expose a simple interface to some commonly-used environments that are included in the aviary codebase. You can instantiate one by referring to its name and passing keyword arguments:

```py
from aviary.core import Environment

env = Environment.from_name(
    "calculator",
    problem_id="example-problem",
    problem="What is 2+3?",
    answer=5,
)
```

Included with some environments are collections of problems that define training or evaluation datasets.
We refer to these as `TaskDataset`s, and expose them with a similar interface:

```py
from aviary.core import TaskDataset

dataset = TaskDataset.from_name("hotpotqa", split="dev")
```

### Tool

Now let's define our functions that will make up our tools. We'll just have one tool. Tools can optionally have their
last argument be `state` which is the environment state. This is how you can access the state. This argument will not be
exposed to the agent as a possible parameter and will be injected by the environment (if part of the function
signature).

```py
def print_story(story: str, state: ExampleState):
    """Print a story.

    Args:
        story: Story to print.
        state: Environment state (hidden from agent - can put this string to shutup linter).
    """
    print(story)
    state.reward = 1
    state.done = True
```

There is special syntax we use for defining a tool. The tool is built from the following parts of the function: its
name, its arguments names, the arguments types, and the docstring. The docstring is parsed to get a description of the
function and its arguments, so match the syntax carefully.

Setting the `state.done = True` is how we indicate completion. This example terminates immediately. You can use other
ways to decide to terminate.

You can make the function `async` - the environment will account for that when the tool is called.

#### Advanced tool descriptions

We support more sophisticated signatures, for those who want to use them:

- Multiline docstrings
- Non-primitive type hints (e.g. type unions)
- Default values
- Exclusion of info below `\f` (see below)

If you have summary-level information that belongs in the docstring,
but you don't want it part of the `Tool.info.description`,
add a `r` prefix to the docstring
and inject `\f` before the summary information to exclude.
This convention was created by FastAPI ([docs][1]).

[1]: https://fastapi.tiangolo.com/advanced/path-operation-advanced-configuration/#advanced-description-from-docstring

```python
def print_story(story: str | bytes, state: ExampleState):
    r"""Print a story.

    Extra information that is part of the tool description.

    \f

    This sentence is excluded because it's an implementation detail.

    Args:
        story: Story to print, either as a string or bytes.
        state: Environment state.
    """
    print(story)
    state.reward = 1
    state.done = True
```

### Environment `reset` method

Now we'll define the `reset` function which should set-up the tools,
and return one or more initial observations and the tools.
The `reset` function is `async` to allow for database interactions or HTTP requests.

```py
from aviary.core import Message, Tool


async def reset(self):
    self.tools = [Tool.from_function(ExampleEnv.print_story)]
    start = Message(content="Write a 5 word story and call print")
    return [start], self.tools
```

### Environment `step` method

Now we can define the `step` function which should take an action and return the next observation, reward, done, and if
the episode was truncated.

```py
from aviary.core import Message


async def step(self, action: Message):
    msgs = await self.exec_tool_calls(action, state=self.state)
    return msgs, self.state.reward, self.state.done, False
```

You will probably often use this specific syntax for calling the tools - calling `exec_tool_calls` with the action.

### Environment `export_frame` method

Optionally, we can define a function to export a snapshot of the environment
and its state for visualization or debugging purposes.

```py
from aviary.core import Frame


def export_frame(self):
    return Frame(
        state={"done": self.state.done, "reward": self.state.reward},
        info={"tool_names": [t.info.name for t in self.tools]},
    )
```

### View Environment Tools

If an environment can be instantiated without anything other than a task (i.e., it implements `from_task`), you can start a server to view its tools:

```sh
pip install fhaviary[server]
aviary tools [env name]
```

This will start a server that allows you to view the tools and call them, viewing the descriptions/types and output that an agent would see when using the tools.

## Environments

Here are a few environments implemented with aviary:

| Environment | PyPI                                                           | Extra                | README                                                               |     |
| ----------- | -------------------------------------------------------------- | -------------------- | -------------------------------------------------------------------- | --- |
| GSM8k       | [`aviary.gsm8k`](https://pypi.org/project/aviary.gsm8k/)       | `fhaviary[gsm8k]`    | [`README.md`](packages/gsm8k/README.md#installation)                 |     |
| HotPotQA    | [`aviary.hotpotqa`](https://pypi.org/project/aviary.hotpotqa/) | `fhaviary[hotpotqa]` | [`README.md`](packages/hotpotqa/README.md#installation)              |     |
| PaperQA     | [`paper-qa`](https://pypi.org/project/paper-qa/)               | `fhaviary[paperqa]`  | [`README.md`](https://github.com/Future-House/paper-qa#installation) |     |
