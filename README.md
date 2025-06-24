# Aviary

[![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Future-House/aviary)
[![Project Status: Active](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)
![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)
[![Docs](https://assets.readthedocs.org/static/projects/badges/passing-flat.svg)](https://aviary.bio/)
[![PyPI version](https://badge.fury.io/py/fhaviary.svg)](https://badge.fury.io/py/fhaviary)
[![tests](https://github.com/Future-House/aviary/actions/workflows/tests.yml/badge.svg)](https://github.com/Future-House/aviary)
[![CodeFactor](https://www.codefactor.io/repository/github/future-house/aviary/badge/main)](https://www.codefactor.io/repository/github/future-house/aviary/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue?style=flat&logo=python&logoColor=white)](https://www.python.org)

<p align="left">
  <a href="https://arxiv.org/abs/2212.04450">
    <img src="docs/assets/aviary_art.png" width="500" alt="Crows in a gym" />
  </a>
</p>

**Aviary** [^1] is a gymnasium for defining custom language agent RL environments.
The library features pre-existing environments on
math [^2], general knowledge [^3], biological sequences [^4], scientific literature search [^5], and protein stability.
Aviary is designed to work in tandem with its sister library LDP (<https://github.com/Future-House/ldp>)
which enables the user to define custom language agents as Language Decision Processes.
See the following [tutorial][2] for an example of how to run an LDP language agent on an Aviary environment.

[2]: https://github.com/Future-House/aviary/blob/main/tutorials/Building%20a%20GSM8k%20Environment%20in%20Aviary.ipynb

[Overview](#overview)
| [Getting Started](#getting-started)
| [Documentation](https://aviary.bio/)
| [Paper](https://arxiv.org/abs/2412.21154)

## What's New?

- Check out our new [Tutorial](https://github.com/Future-House/ldp/blob/main/tutorials/creating_a_language_agent.ipynb)
  notebook on running an LDP agent in an Aviary environment!
- The Aviary paper has been posted to [arXiv](https://arxiv.org/abs/2412.21154)!
  Further updates forthcoming!

## Overview

<p align="left">
  <a href="https://arxiv.org/abs/2212.04450">
    <img src="docs/assets/Aviary.png" width="800" alt="Aviary and LDP overview from paper" />
  </a>
</p>

A pictorial overview of the five implemented Aviary environments and the language decision process framework.

## Getting Started

To install aviary (note `fh` stands for FutureHouse):

```bash
pip install fhaviary
```

To install aviary together with the incumbent environments:

```bash
pip install 'fhaviary[gsm8k,hotpotqa,litqa,lfrqa]'
```

To run the tutorial notebooks:

```bash
pip install "fhaviary[dev]"
```

### Developer Installation

For local development, please see [CONTRIBUTING.md](CONTRIBUTING.md).

## Tutorial Notebooks

1. [Building a Custom Environment in Aviary](https://github.com/Future-House/aviary/blob/main/tutorials/Building%20a%20Custom%20Environment%20in%20Aviary.ipynb)
2. [Building a GSM8K Environment in Aviary](https://github.com/Future-House/aviary/blob/main/tutorials/Building%20a%20GSM8k%20Environment%20in%20Aviary.ipynb)
3. [Creating Language Agents to Interact with Aviary Environments](https://github.com/Future-House/ldp/blob/main/tutorials/creating_a_language_agent.ipynb)
4. [Evaluate a Llama Agent on GSM8K](https://github.com/Future-House/ldp/blob/main/tutorials/evaluating_a_llama_agent.ipynb)

## Defining a Custom Environment

The example below walks through defining a custom environment in Aviary.
We define a simple environment where an agent takes actions to modify a counter.
The example is also featured in the following [notebook][1].

```python
from collections import namedtuple
from aviary.core import Environment, Message, ToolRequestMessage, Tool

# State in this example is simply a counter
CounterEnvState = namedtuple("CounterEnvState", ["count"])


class CounterEnv(Environment[CounterEnvState]):
    """A simple environment that allows an agent to modify a counter."""

    async def reset(self):
        """Initialize the environment with a counter set to 0. Goal is to count to 10"""
        self.state = CounterEnvState(count=0)

        # Target count
        self.target = 10

        # Create tools allowing the agent to increment and decrement counter
        self.tools = [
            Tool.from_function(self.incr),
            Tool.from_function(self.decr),
        ]

        # Return an observation message with the counter and available tools
        return [Message(content=f"Count to 10. counter={self.state.count}")], self.tools

    async def step(self, action: ToolRequestMessage):
        """Executes the tool call requested by the agent."""
        obs = self.exec_tool_calls(action)

        # The reward is the square of the current count
        reward = int(self.state.count == self.target)

        # Returns observations, reward, done, truncated
        return obs, reward, reward == 1, False

    def incr(self):
        """Increment the counter."""
        self.state.count += 1
        return f"counter={self.state.count}"

    def decr(self):
        """Decrement the counter."""
        self.state.count -= 1
        return f"counter={self.state.count}"
```

## Evaluating an Agent on the Environment

Following the definition of our custom environment,
we can now evaluate a language agent on the environment using
Aviary's sister library LDP (<https://github.com/Future-House/ldp>).

```python
from ldp.agent import Agent
from ldp.graph import LLMCallOp
from ldp.alg import RolloutManager


class AgentState:
    """A container for maintaining agent state across interactions."""

    def __init__(self, messages, tools):
        self.messages = messages
        self.tools = tools


class SimpleAgent(Agent):
    def __init__(self, **kwargs):
        self._llm_call_op = LLMCallOp(**kwargs)

    async def init_state(self, tools):
        return AgentState([], tools)

    async def get_asv(self, agent_state, obs):
        """Take an action, observe new state, return value"""
        action = await self._llm_call_op(
            config={"name": "gpt-4o", "temperature": 0.1},
            msgs=agent_state.messages + obs,
            tools=agent_state.tools,
        )
        new_state = AgentState(
            messages=agent_state.messages + obs + [action],
            tools=agent_state.tools,
        )
        # Return action, state, value
        return action, new_state, 0.0


# Create a simple agent and perform rollouts on the environment

# Endpoint can be model identifier e.g. "claude-3-opus" depending on service
agent = SimpleAgent(config={"model": "my_llm_endpoint"})

runner = RolloutManager(agent=agent)

trajectories = await runner.sample_trajectories(
    environment_factory=CounterEnv,
    batch_size=2,
)
```

Below we expand on some of the core components of the Aviary library together with more advanced usage examples.

### Environment

An environment should have two methods, `env.reset` and `env.step`:

```py
obs_msgs, tools = await env.reset()
new_obs_msgs, reward, done, truncated = await env.step(action_msg)
```

Communication is achieved through messages.

The `action_msg` is an instance of `ToolRequestMessage` which comprises one or more calls
to the `tools` returned by `env.reset` method.

The `obs_msgs` are either general obseravation messages
or instances of `ToolResponseMessage` returned from the environment.
while `reward` is a scalar value, and `done` and `truncated`
are Boolean values.

We explain the message formalism in further detail below.

### Messages

Communication between the agent and environment is achieved via messages.
We follow the [OpenAI](https://platform.openai.com/docs/api-reference/messages/createMessage) standard.
Messages have two attributes:

```py
msg = Message(content="Hello, world!", role="assistant")
```

The `content` attribute can be a string but can also comprise objects such as [images][3].
For example, the `create_message` method can be used to create a message with images:

[3]: https://platform.openai.com/docs/guides/vision?lang=node#uploading-base64-encoded-images

```py
from PIL import Image
import numpy as np

img = Image.open("your_image.jpg")
img_array = np.array(img)

msg = Message.create_message(role="user", text="Hello, world!", images=[img_array])
```

In this case, `content` will be a list of dictionaries with the keys `text` and `image_url`.

```py
{
    {"type": "text", "text": "Hello World!"},
    {"text": "image_url", "image_url": "data:image/png;base64,{base64_image}"},
}
```

The role, see the table below.
You can change around roles as desired,
except for `tool` which has a special meaning in aviary.

| Role      | Host                                             | Example(s)                                                       |
| --------- | ------------------------------------------------ | ---------------------------------------------------------------- |
| assistant | Agent                                            | An agent's tool selection message                                |
| system    | Agent system prompt                              | "You are an agent."                                              |
| user      | Environment system prompt or emitted observation | HotPotQA problem to solve, or details of an internal env failure |
| tool      | Result of a tool run in the environment          | The output of the calculator tool for a GSM8K question           |

The `Message` class is extended in `ToolRequestMessage` and `ToolResponseMessage`
to include the relevant tool name and arguments.

### Subclassing Environments

If you need more control over Environments and tools, you may wish to subclass `Environment`. We illustrate this
with an example environment in which an agent is tasked to write a story.

We subclass `Environment` and define a `state`. The `state` consists of all variables
that change per step that we wish to bundle together. It will be accessible in tools, so you can use `state` to store
information you want to persist between steps and tool calls.

```py
from pydantic import BaseModel
from aviary.core import Environment


class ExampleState(BaseModel):
    reward: float = 0
    done: bool = False


class ExampleEnv(Environment[ExampleState]):
    state: ExampleState
```

We do not have other variables aside from `state` for this environment,
although we could also have variables like configuration, a name,
tasks, etc. attached to it.

### Defining Tools

We will define a single tool that prints a story. Tools may optionally take a final argument
`state` which is the environment state. This argument will not be
exposed to the agent as a parameter but will be injected by the environment
(if part of the function signature).

```py
def print_story(story: str, state: ExampleState):
    """Print a story.

    Args:
        story: Story to print.
        state: Environment state (hidden from agent).
    """
    print(story)
    state.reward = 1
    state.done = True
```

The tool is built from the following parts of the function: its
name, its argument's names, the arguments types, and the docstring.
The docstring is parsed to obtain a description of the
function and its arguments, so be sure to match the syntax carefully.

Environment episode completion is indicated by setting `state.done = True`.
This example terminates immediately - other
termination conditions are also possible.

It is also possible make the function `async` - the environment will account for that when the tool is called.

### Advanced Tool Descriptions

Aviary also supports more sophisticated signatures:

- Multiline docstrings
- Non-primitive type hints (e.g. type unions)
- Default values
- Exclusion of info below `\f` (see below)

If you have summary-level information that belongs in the docstring,
but you don't want it to be part of the `Tool.info.description`,
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

### The Environment `reset` Method

Next we define the `reset` function which initializes the tools
and returns one or more initial observations as well as the tools.
The `reset` function is `async` to allow for database interactions or HTTP requests.

```py
from aviary.core import Message, Tool


async def reset(self):
    self.tools = [Tool.from_function(ExampleEnv.print_story)]
    start = Message(content="Write a 5 word story and call print")
    return [start], self.tools
```

### The Environment `step` Method

Next we define the `step` function which takes an action and returns
the next observation, reward, done, and whether the episode was truncated.

```py
from aviary.core import Message


async def step(self, action: Message):
    msgs = await self.exec_tool_calls(action, state=self.state)
    return msgs, self.state.reward, self.state.done, False
```

You will probably often use this specific syntax for calling the tools - calling `exec_tool_calls` with the action.

### Environment `export_frame` Method

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

### Viewing Environment Tools

If an environment can be instantiated without anything other than the task
(i.e., it implements `from_task`), you can start a server to view its tools:

```sh
pip install fhaviary[server]
aviary tools [env name]
```

This will start a server that allows you to view the tools and call them,
viewing the descriptions/types and output that an agent would see when using the tools.

### Incumbent Environments

Below we list some pre-existing environments implemented in Aviary:

| Environment       | PyPI                                                                 | Extra                         | README                                                           |     |
| ----------------- | -------------------------------------------------------------------- | ----------------------------- | ---------------------------------------------------------------- | --- |
| GSM8k             | [`aviary.gsm8k`](https://pypi.org/project/aviary.gsm8k/)             | `fhaviary[gsm8k]`             | [`README.md`](packages/gsm8k/README.md#installation)             |     |
| HotPotQA          | [`aviary.hotpotqa`](https://pypi.org/project/aviary.hotpotqa/)       | `fhaviary[hotpotqa]`          | [`README.md`](packages/hotpotqa/README.md#installation)          |     |
| LitQA             | [`aviary.litqa`](https://pypi.org/project/aviary.litqa/)             | `fhaviary[litqa]`             | [`README.md`](packages/litqa/README.md#installation)             |     |
| Cloning           | [`aviary.cloning`](https://pypi.org/project/aviary.litqa/)           | `fhaviary[cloning]`           | [`README.md`](packages/cloning/README.md#installation)           |     |
| Protein Stability | [`aviary.protein_stability`](https://pypi.org/project/aviary.litqa/) | `fhaviary[protein_stability]` | [`README.md`](packages/protein_stability/README.md#installation) |     |
| LFRQA             | [`aviary.lfrqa`](https://pypi.org/project/aviary.lfrqa/)             | `fhaviary[lfrqa]`             | [`README.md`](packages/lfrqa/README.md#installation)             |     |

### Task Datasets

Included with some environments are collections of problems that define training or evaluation datasets.
We refer to these as `TaskDataset`s, e.g. for the `HotpotQADataset` subclass of `TaskDataset`:

```py
from aviary.envs.hotpotqa import HotPotQADataset

dataset = HotPotQADataset(split="dev")
```

### Functional Environments

An alternative way to create an environment is using the functional interface,
which uses functions and decorators to define environments.
Let's define an environment that requires an agent to write a story
about a particular topic by implementing its `start` function:

```python
from aviary.core import fenv


@fenv.start()
def my_env(topic):
    # return the first observation and starting environment state
    # (empty in this case)
    return f"Write a story about {topic}", {}
```

The `start` decorator begins the definition of an environment.

The function, `my_env`,
takes an arbitrary input and returns a tuple containing the first observation
and any information you wish to store about the environment state
(used to persist/share information between tools).

The state will always have an optional `reward` and a Boolean `done` that indicate
if the environment episode is complete.
Next we define some tools:

```python
@my_env.tool()
def multiply(x: float, y: float) -> float:
    """Multiply two numbers."""
    return x * y


@my_env.tool()
def print_story(story: str | bytes, state) -> None:
    """Print a story to the user and complete episode."""
    print(story)
    state.reward = 1
    state.done = True
```

The tools will be converted into objects visible for LLMs using the type hints and the variable descriptions.
Thus, the type hinting can be valuable for an agent that uses it correctly.
The docstrings are also passed to the LLM and is the primary means
(along with the function name) for communicating the intended tool usage.

You can access the `state` variable in tools,
which will have any fields you passed in the return tuple of `start()`.
For example, if you returned `{'foo': 'bar'}`,
then you could access `state.foo` in the tools.

You may stop an environment or set a reward via the `state` variable
as shown in the second `print_story` tool.
If the reward is not set, it is treated as zero.
Next we illustrate how to use our environment:

```python
env = my_env(topic="foo")
obs, tools = await env.reset()
```

## Citing Aviary

If Aviary is useful for your work please consider citing the following paper:

```bibtex
@article{Narayanan_Aviary_training_language_2024,
  title   = {{Aviary: training language agents on challenging scientific tasks}},
  author  = {
    Narayanan, Siddharth and Braza, James D. and Griffiths, Ryan-Rhys and
    Ponnapati, Manvitha and Bou, Albert and Laurent, Jon and Kabeli, Ori and
    Wellawatte, Geemi and Cox, Sam and Rodriques, Samuel G. and White, Andrew
    D.
  },
  year    = 2024,
  month   = dec,
  journal = {preprint},
  doi     = {10.48550/arXiv.2412.21154},
  url     = {https://arxiv.org/abs/2412.21154}
}
```

## References

[^1]: Narayanan, S., Braza, J.D., Griffiths, R.R., Ponnapati, M., Bou, A., Laurent, J., Kabeli, O., Wellawatte, G., Cox, S., Rodriques, S.G. and White, A.D., 2024. [Aviary: training language agents on challenging scientific tasks.](https://arxiv.org/abs/2412.21154) arXiv preprint arXiv:2412.21154.

[^2]: Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R. and Hesse, C., 2021. [Training verifiers to solve math word problems.](https://arxiv.org/abs/2110.14168) arXiv preprint arXiv:2110.14168.

[^3]: Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W., Salakhutdinov, R. and Manning, C.D., 2018. [HotpotQA: A dataset for diverse, explainable multi-hop question answering.](https://aclanthology.org/D18-1259/) EMNLP 2018 (pp. 2369-2380).

[^4]: Laurent, J.M., Janizek, J.D., Ruzo, M., Hinks, M.M., Hammerling, M.J., Narayanan, S., Ponnapati, M., White, A.D. and Rodriques, S.G., 2024. [Lab-Bench: Measuring capabilities of language models for biology research.](https://arxiv.org/abs/2407.10362) arXiv preprint arXiv:2407.10362.

[^5]: Skarlinski, M.D., Cox, S., Laurent, J.M., Braza, J.D., Hinks, M., Hammerling, M.J., Ponnapati, M., Rodriques, S.G. and White, A.D., 2024. [Language agents achieve superhuman synthesis of scientific knowledge.](https://arxiv.org/abs/2409.13740) arXiv preprint arXiv:2409.13740.
