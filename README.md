# aviary

Gymnasium framework for training language model agents on constructive tasks.

## Installation

To install aviary:

```bash
pip install -e .
```

To install aviary and the provided environments:

```bash
pip install -e . -e packages/gsm8k -e packages/hotpotqa
```

To run test suites you will need to set the `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`
environment variables. In `~/.bashrc` you can add:

```bash
export OPENAI_API_KEY=your_openai_api_key
export ANTHROPIC_API_KEY=your_anthropic_api_key
```

## Messages

Communication between the agent and environment is done through messages.
Messages have two attributes:

```py
msg = Message(content="Hello, world!", role="assistant")
```

For the meaning of role, see the table below.
You can change around roles as desired,
except for `tool` which has a special meaning in aviary.

| Role      | Host                    | Example                              |
| --------- | ----------------------- | ------------------------------------ |
| assistant | AI                      | ChatGPT                              |
| system    | AI system prompt        | You are an AI assistant              |
| user      | User                    | You, using ChatGPT                   |
| tool      | Tool in the environment | Some custom number crunching program |

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

Let's see a complete example for building an environment.

### Environment subclass and state

First we define an environment by subclassing the `Environment` and defining a `state`. The `state` is all variables
that change per step and we want to keep together. It will be accessible in your tools, so you can use it to store
information that you want to persist between steps and between tools.

```py
from pydantic import BaseModel
from aviary.env import Environment

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
from aviary.env import Environment

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
from aviary.env import TaskDataset

dataset = TaskDataset.from_name("hotpotqa", split="dev")
```

### Tool

Now let's define our functions that will make up our tools. We'll just have one tool. Tools can optionally have their
last argument be `state` which is the environment state. This is how you can access the state. This argument will not be
exposed to the agent as a possible parameter and will be injected by the environment (if part of the function
signature).

```py
def print_story(story: str, state: ExampleState) -> None:
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
def print_story(story: str | bytes, state: ExampleState) -> None:
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

Now we'll define the `reset` function which should set-up the tools and return one or more observations and the tools.

```py
from aviary.message import Message
from aviary.tools import Tool

def reset(self) -> tuple[list[Message], list[Tool]]:
    self.tools = [Tool.from_function(ExampleEnv.print_story)]

    start = Message(content="Write a 5 word story and call print")

    return [start], self.tools
```

### Environment `step` method

Now we can define the `step` function which should take an action and return the next observation, reward, done, and if
the episode was truncated.

```py
from aviary.message import Message

async def step(self, action: Message) -> tuple[list[Message], float, bool, bool]:
    msgs: list[Message] = await self.exec_tool_calls(action, state=self.state)
    return msgs, self.state.reward, self.state.done, False
```

You will probably often use this specific syntax for calling the tools - calling `exec_tool_calls` with the action.

### Environment `export_frame` method

Lastly, we can define a function to export the state for visualization or debugging purposes. This is optional.

```py
from aviary.env import Frame

def export_frame(self) -> Frame:
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

### GSM8k Environment

#### What it does

The GSM8k environment allows agents to solve math word problems from the GSM8k dataset.

#### Installation

To install the GSM8k environment, run the following command:

```bash
pip install fhaviary[gsm8k]
```

### HotPotQA Environment

#### What it does

The HotPotQA environment allows agents to perform multi-hop question answering on the HotPotQA dataset.

#### Installation

To install the HotPotQA environment, run the following command:

```bash
pip install fhaviary[hotpotqa]
```

### PaperQA Environment

#### What it does

The PaperQA environment allows agents to perform question answering on the PaperQA dataset.

#### Installation

To install the PaperQA environment, follow the instructions in the [PaperQA repository](https://github.com/Future-House/paper-qa).
