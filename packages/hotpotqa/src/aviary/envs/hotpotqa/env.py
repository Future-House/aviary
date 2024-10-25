"""HotPotQA environment for aviary agents.

Implements the HotPotQA multihop question-answering environment from:

Yao et al., ReAct: Synergizing Reasoning and Acting in Language Models.
In The Eleventh International Conference on Learning Representations. 2023

Agents in the HotPotQA environment can perform searches and lookups to retrieve specific information.
The environment supports 3 tools:

Search[entity]: Search for a specific entity on Wikipedia and return relevant information.
Lookup[keyword]: Find and return the next sentence containing the keyword in the current passage.
Finish[answer]: Submit the final answer to the question and conclude the task.
"""

import logging
import random
import re
import string
from collections.abc import Callable
from enum import StrEnum
from typing import Any, ClassVar, cast

import httpx
from bs4 import BeautifulSoup
from datasets import load_dataset
from pydantic import BaseModel, ConfigDict, Field
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from aviary.env import Environment, Frame, TaskDataset
from aviary.message import Message
from aviary.tools import Tool, ToolRequestMessage, ToolResponseMessage

logger = logging.getLogger(__name__)


# Jitter in case we have lots of requests at once, to space them out slightly
@retry(stop=stop_after_attempt(2), wait=wait_exponential_jitter(initial=1, max=4))
async def fetch_with_retry(
    client: httpx.AsyncClient, url: str, **kwargs
) -> httpx.Response:
    response = await client.get(url, **kwargs)
    if response.status_code == httpx.codes.TOO_MANY_REQUESTS:
        response.raise_for_status()
    return response


class HotPotQAEnvState(BaseModel):
    """State of the HotPotQA environment."""

    done: bool = Field(
        default=False,
        description="Flag for being done, automatically set true after many steps.",
    )

    steps: int = Field(default=0, description="Count of environment steps.")

    reward: float = Field(
        default=0.0, description="Current reward value, reset each environment step."
    )

    answer: str | None = Field(
        default=None,
        description="The answer to the question, or None if not yet answered.",
    )

    last_lookup: str | None = Field(
        default=None, description="The last lookup keyword."
    )
    lookup_results: list[str] = Field(
        default_factory=list, description="Results of the last lookup."
    )
    lookup_index: int = Field(
        default=0, description="Index of the last retrieved lookup result."
    )
    page: str | None = Field(default=None, description="The current Wikipedia page.")


def create_tool(function: Callable, name: str) -> Tool:
    """Create a Tool object from a function and set its name.

    Args:
        function: The function to be wrapped by the Tool.
        name: The name to assign to the Tool.

    Returns:
        A Tool object with the specified function and name.
    """
    tool = Tool.from_function(function)
    tool.info.name = name
    return tool


def clean_str(p: str) -> str:
    r"""Clean and normalize a given string by encoding and decoding it.

    Args:
        p: The input string to be cleaned and normalized.

    Returns:
        The cleaned and normalized UTF-8 encoded string.

    Examples:
        >>> clean_str("This is a test string with unicode escape: \\u00e9")
        'This is a test string with unicode escape: é'
        >>> clean_str(r"SBN IT\ICCU\UBO\2771748.")  # SEE: https://en.wikipedia.org/wiki/Tricolour_Day
        'SBN IT\\ICCU\\UBO\\2771748.'
    """
    # Original source:
    # https://github.com/ysymyth/ReAct/blob/6bdb3a1fd38b8188fc7ba4102969fe483df8fdc9/wikienv.py#L10-L11
    try:
        # unicode-escape codec can interpret backslash escapes (e.g., `\n`, `\u1234`)
        return (
            p.encode("latin1").decode("unicode-escape").encode("utf-8").decode("utf-8")
        )
    except (UnicodeEncodeError, UnicodeDecodeError):
        logger.debug(f"Replacing bad characters in input string {p!r}.")
        return p.encode(errors="replace").decode("utf-8", errors="replace")


def normalize_answer(s: str | float) -> str:
    """Normalize the given answer string by applying several text processing steps.

    This function processes the input string through a series of transformations to
    normalize it, which includes:
        1. Convert all characters to lowercase.
        2. Remove punctuation from the text.
        3. Remove articles ('a', 'an', 'the') from the text.
        4. Fix extra whitespace by reducing multiple spaces to a single space.

    Args:
        s: The input answer string to be normalized.

    Returns:
        The normalized answer string.

    Example:
        >>> normalize_answer("The Quick, Brown Fox!")
        'quick brown fox'
    """
    # if s is not a string, convert it to a string
    if not isinstance(s, str):
        return str(s)

    def remove_articles(text: str) -> str:
        """Remove articles ('a', 'an', 'the') from the text."""
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        """Reduce multiple spaces to a single space."""
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        """Remove all punctuation from the text."""
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        """Convert all characters to lowercase."""
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


class HotPotQAEnv(Environment[HotPotQAEnvState]):
    State: ClassVar = HotPotQAEnvState
    wiki_cache: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        question: str,
        correct_answer: str | float,
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,
        tool_failure_reward: float = 0.0,
        proxy: str | None = None,
    ):
        super().__init__()
        self.question = question
        # Normalize the correct answer once here as a minor performance optimization
        self.normalized_correct_answer = normalize_answer(correct_answer)
        self.correct_reward = correct_reward
        self.incorrect_reward = incorrect_reward
        self.tool_failure_reward = tool_failure_reward
        self.proxy = proxy

        # Title case tool names to match third party demonstration data
        self.tools = [
            create_tool(self.search, "Search"),
            create_tool(self.construct_lookup_list, "Lookup"),
            create_tool(self.finish, "Finish"),
        ]

    @classmethod
    def from_task(cls, task: str) -> "HotPotQAEnv":
        return cls(question=task, correct_answer=0.0)

    def calculate_reward(self, answer: str | None) -> float:
        """Calculate the reward based on the agent's answer.

        Returns:
            The correct reward if the agent's answer is correct (prediction exactly
                matches ground truth), otherwise the incorrect reward.
        """
        if answer is None:
            return self.incorrect_reward
        pred, gt = normalize_answer(answer), self.normalized_correct_answer
        return self.correct_reward if pred == gt else self.incorrect_reward

    async def reset(self) -> tuple[list[Message], list[Tool]]:
        """Reset the HotPotQA environment to an initial state.

        This method resets the environment to its initial state, setting up necessary variables and tools
        for the agent to interact with. It prepares the environment for a new episode by initializing
        various attributes and returning the initial observation and tools available for the agent.

        Returns:
            tuple: A tuple containing:
                - list[Message]: The initial observation wrapped in a Message object.
                - list[Tool]: A list of tools (Search, Lookup, and Finish) available for the agent.

        Example:
            >>> env = HotPotQAEnv()
            >>> initial_obs, tools = env.reset(seed=42, idx=5)
            >>> print(initial_obs)
            [Message(content='Question: <question_text>')]
            >>> print(tools)
            [<Tool: Search>, <Tool: Lookup>, <Tool: Finish>]
        """
        self.state = self.State()
        return [Message(content=f"Question: {self.question}")], self.tools

    async def step(
        self, action: ToolRequestMessage
    ) -> tuple[list[Message], float, bool, bool]:
        """Take a step in the environment. Assume only one tool at a time can be called for HotpotQA.

        This method processes an action message, which can be a tool request, a finish request, or an error message.
        It updates the environment state accordingly and returns the observation and done status.

        Args:
            action: Action to take.

        Returns:
            Tuple[List[ToolResponseMessage], bool]: A tuple containing:
                - list[ToolResponseMessage]: The response message(s) from the executed tool.
                - bool: The done status indicating whether the episode is finished.

        Example:
            >>> env = HotPotQAEnv()
            >>> action = ToolRequestMessage(
            ...     tool_calls=[
            ...         ToolCall(
            ...             function=ToolCallFunction(name="Search"),
            ...             arguments={"entity": "Python"},
            ...         )
            ...     ]
            ... )
            >>> obs, done = await env.step(action)
            >>> print(obs, done)
            [ToolResponseMessage(name='Search', tool_call_id='tool_call_id', content='...')], False
        """
        self.state.steps += 1
        if not action.tool_calls:
            return (
                [
                    Message(
                        content=(
                            f"Must call one of the provided tools"
                            f" { {t.info.name for t in self.tools} }."
                        )
                    )
                ],
                self.tool_failure_reward,
                self.state.done,
                False,
            )

        # We accumulate reward across all tool calls in a step
        self.state.reward = 0.0
        valid_action, invalid_action = self.filter_invalid_tool_calls(action)
        # NOTE: valid_action or invalid_action may have an empty list of tool calls
        response_messages = cast(
            list[Message],
            # Ordered since things like search -> lookup need to be run in order.
            # NOTE: Handling tool exceptions here keeps the trajectory going, but I don't
            # think the returned message is useful to the agent/learning. Disabling for now.
            await self.exec_tool_calls(
                valid_action, ordered=True, handle_tool_exc=False
            )
            + [
                ToolResponseMessage.from_call(tool_call, content="Invalid tool call.")
                for tool_call in invalid_action.tool_calls
            ],
        )
        return response_messages, self.state.reward, self.state.done, False

    def export_frame(self) -> Frame:
        """Export the current state of the environment as a Frame object.

        This method creates and returns a Frame object that captures the current state and additional
        information of the HotPotQA environment. This can be useful for logging, debugging, or visualization
        purposes.

        Returns:
            Frame: A Frame object containing the current state and additional information of the environment.

        The returned Frame object includes the following information:
            - state (dict): An empty dictionary representing the current state of the environment.
            - info (dict): A dictionary containing:
                - "steps" (int): The number of steps taken in the current episode.
                - "done" (bool): A flag indicating whether the episode is finished.
                - "reward" (float): The accumulated reward in the current episode.
                - "answer" (Optional[str]): The answer provided by the agent, if any.

        Example:
            >>> env = HotPotQAEnv()
            >>> frame = env.export_frame()
            >>> print(frame.info)
            {'steps': 0, 'done': False, 'reward': 0.0, 'answer': None}
        """
        return Frame(
            info={
                "steps": self.state.steps,
                "done": self.state.done,
                "reward": self.state.reward,
                "answer": self.state.answer,
            }
        )

    def finish(self, answer: str) -> str:
        """Finish the episode.

        Args:
            answer: The answer to the question.
        """
        self.state.done = True
        if not answer:
            return "Finish failed. No answer provided."

        self.state.answer = answer
        self.state.reward += self.calculate_reward(answer)
        return "Finished."

    async def search(self, entity: str) -> str:
        """Searches Wikipedia for the given entity and processes the results.

        Args:
            entity: The entity to search for on Wikipedia.

        Functionality:
            - Constructs and sends a search query to Wikipedia.
            - Parses the search results.
            - If similar results are found, returns a list of similar titles.
            - If the entity page is found, returns a summary of the page content.
            - Handles disambiguation pages by recursively searching with modified query.
        """
        if not entity:
            self.state.done = True
            return "Search failed. No entity provided."

        # In case the searched entity is e.g. a year
        search_entity = (
            str(entity) if isinstance(entity, int) else entity.replace(" ", "+")  # type: ignore[redundant-expr,unreachable]  # noqa: FURB123
        )
        try:
            # Access from cache if we previously searched for search_entity
            response_text = self.wiki_cache[search_entity]
        except KeyError:
            # follow_redirects=True because wikipedia frequently redirects to the correct page
            async with httpx.AsyncClient(
                follow_redirects=True, proxy=self.proxy
            ) as client:
                response = await fetch_with_retry(
                    client,
                    f"https://en.wikipedia.org/w/index.php?search={search_entity}",
                    timeout=15,
                )
                response.raise_for_status()  # Raise an HTTPError for bad responses
                # Cache for subsequent tool calls
                self.wiki_cache[search_entity] = response_text = response.text
        soup = BeautifulSoup(response_text, features="html.parser")
        result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
        if result_divs:  # mismatch
            result_titles = [clean_str(div.get_text().strip()) for div in result_divs]
            self.state.page = None
            return f"Could not find {entity}. Similar: {result_titles[:5]}."

        page = [p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")]
        if any("may refer to:" in p for p in page):  # Recurse
            return await self.search(entity="[" + entity + "]")
        # Clean up any unicode
        self.state.page = ""
        for p in page:
            if len(p.split(" ")) > 2:  # noqa: PLR2004
                self.state.page += clean_str(p)
                if not p.endswith("\n"):
                    self.state.page += "\n"
        # Extract and concatenate the first five sentences from the provided page content.
        obs_list = [
            s.strip() + "."
            for p in self.state.page.split("\n")
            if p.strip()
            for s in p.split(". ")
            if s.strip()
        ]
        return " ".join(obs_list[:5])

    def construct_lookup_list(self, keyword: str) -> str:
        """Construct a list of sentences from the given page content that contain the specified keyword.

        Args:
            keyword: The keyword to search for within the sentences.

        Returns:
            A list of sentences containing the keyword. If `page` is None or empty, an empty list is returned.

        Example:
            page_content = ("Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series
            The Simpsons voiced by Pamela Hayden and created by Matt Groening. Milhouse is Bart Simpson's best friend in
            Mrs. Krabappel's fourth grade class at Springfield Elementary School. He is an insecure, gullible, and
            less popular child than Bart who is often led into trouble by Bart, who takes advantage of his friend's naivety.
            Milhouse is a regular target for school bully Nelson Muntz and his friends Jimbo Jones, Dolph Starbeam and
            Kearney Zzyzwicz. Milhouse has a crush on Bart's sister, Lisa, a common plot element.")

            keyword = "Milhouse"
            result = construct_lookup_list_from_page(keyword, page_content)
            # Output: ['Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The
            Simpsons.',
            'Milhouse is Bart Simpson's best friend in Mrs. Krabappel's fourth grade class at Springfield Elementary
            School.',
            'Milhouse is a regular target for school bully Nelson Muntz and his friends Jimbo Jones, Dolph Starbeam and
            Kearney Zzyzwicz.',
            'Milhouse has a crush on Bart's sister, Lisa, a common plot element.']

        """
        if not keyword:
            self.state.done = True
            return "Lookup failed. No keyword provided"

        if not self.state.page:
            return "Lookup failed. You have not specified a Wikipedia page yet."

        if self.state.last_lookup != keyword:
            self.state.last_lookup = keyword
            self.state.lookup_results = [
                s.strip() + "."
                for s in self.state.page.split(". ")
                if s.strip() and keyword.lower() in s.lower()
            ]
            self.state.lookup_index = 0

        if self.state.lookup_index >= len(self.state.lookup_results):
            return "No more results."

        obs = f"(Result {self.state.lookup_index + 1} / {len(self.state.lookup_results)}) {self.state.lookup_results[self.state.lookup_index]}"
        self.state.lookup_index += 1
        return obs


class HotPotQADifficultyLevel(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class HotPotQAEnvConfig(BaseModel):
    """Configuration model for the HotPotQA environment.

    This defines the configuration parameters for setting up the HotPotQA environment,
    including the path to the data repository and the specific path to the prompt file.
    """

    model_config = ConfigDict(extra="forbid")

    shuffle_data: bool = Field(
        default=False, description="Set True to shuffle the dataset after loading."
    )
    data_fraction: float = Field(
        default=1.0,
        description=(
            "Fraction of data to use, default is full dataset. If set, data is"
            " subsampled after shuffling (if enabled)."
        ),
        gt=0.0,
        le=1.0,
    )
    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    tool_failure_reward: float = 0.0
    difficulty_level: set[HotPotQADifficultyLevel] | None = Field(
        None,
        description="The HotPotQA difficulty level to filter on. "
        "Can be any subset of `{'easy', 'medium', 'hard'}`. "
        "If None, we will not filter any task instances. "
        "Note that not all splits have a 'level' field.",
    )
    proxy: str | None = None


class HotPotQADataset(TaskDataset[HotPotQAEnv]):
    # SEE: https://huggingface.co/datasets/hotpotqa/hotpot_qa
    HOTPOTQA_HUGGING_FACE_DATASET = "hotpotqa/hotpot_qa"

    def get_data_from_hugging_face(
        self, split: str, hf_dataset: str = HOTPOTQA_HUGGING_FACE_DATASET
    ) -> list[tuple[str, str]]:
        """Convert a local file and split to a list of (question, answer) tuples."""
        if split in {"dev", "eval", "val"}:  # Map common aliases
            split = "validation"
        all_datasets = load_dataset(hf_dataset, name="fullwiki", trust_remote_code=True)
        try:
            data = all_datasets[split].select_columns(
                column_names=["question", "answer", "level"]
            )
        except KeyError as exc:
            raise ValueError(
                f"Split {split!r} was invalid for Hugging Face dataset {hf_dataset},"
                f" please specify a split from {set(all_datasets.keys())}."
            ) from exc

        if not all(  # Making up for datasets not being typed: https://github.com/huggingface/datasets/issues/3841
            isinstance(d["question"], str) and isinstance(d["answer"], str | float)
            for d in data
        ):
            raise ValueError(
                f"Dataset {hf_dataset!r} and split {split!r} contains invalid"
                " question(s) or answer(s)."
            )
        return [(d["question"], d["answer"]) for d in data if self._filter_task(d)]

    def __init__(
        self, split: str, config: HotPotQAEnvConfig | dict | None = None, **kwargs
    ):
        super().__init__()
        if isinstance(config, dict):  # Serialized config
            config = HotPotQAEnvConfig(**(config | kwargs))
        elif config is None:
            config = HotPotQAEnvConfig(**kwargs)
        self.config = config
        raw_data = self.get_data_from_hugging_face(split)
        if self.config.shuffle_data:
            random.shuffle(raw_data)
        if self.config.data_fraction < 1.0:
            raw_data = raw_data[: int(self.config.data_fraction * len(raw_data))]
        self.data = raw_data

    def _filter_task(self, task: dict[str, Any]) -> bool:
        """Decide whether to keep a task instance based on configuration options."""
        if self.config.difficulty_level is None:
            return True

        try:
            return task["level"] in self.config.difficulty_level
        except KeyError as e:
            raise RuntimeError(
                "Attempting to filter difficulty level, "
                "but this split does not have a 'level' field."
            ) from e

    def get_new_env_by_idx(self, idx: int) -> HotPotQAEnv:
        return HotPotQAEnv(
            *self.data[idx],
            correct_reward=self.config.correct_reward,
            incorrect_reward=self.config.incorrect_reward,
            tool_failure_reward=self.config.tool_failure_reward,
            proxy=self.config.proxy,
        )

    def __len__(self) -> int:
        return len(self.data)
