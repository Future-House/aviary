import contextlib
import json
from enum import StrEnum
from logging import getLogger
from typing import TYPE_CHECKING, ClassVar, Literal

import datasets
from pydantic import BaseModel, ConfigDict

from aviary.env import Environment, Frame, TaskDataset
from aviary.message import Message
from aviary.tools import Tool, ToolRequestMessage, ToolResponseMessage

if TYPE_CHECKING:
    import pandas as pd

logger = getLogger(__name__)


class CalculatorEnvConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    tool_failure_reward: float = -1.0
    tool_success_reward: float = 0.0
    rel_tol: float = 1e-4

    done_on_failure: bool = True


class CalculatorEnv(Environment[None]):
    def __init__(
        self,
        problem_id: str,
        problem: str,
        answer: float,
        config: CalculatorEnvConfig | None = None,
    ):
        # The problem is not part of the state because it is always the same.
        # Putting it in the state would imply it is somehow affected by .step()
        # or re-initialized by .reset().
        self.problem_id = problem_id
        self.problem = problem
        self.answer = float(answer)  # If passed in as a 0d tensor  # noqa: FURB123

        self.config = config if config is not None else CalculatorEnvConfig()

        self.calc_tool = Tool.from_function(self.calculator)
        self.check_tool = Tool.from_function(self.check_answer)
        self.tools = [self.calc_tool, self.check_tool]

    @classmethod
    def from_task(cls, task: str) -> "CalculatorEnv":
        return cls(problem_id="task", problem=task, answer=0.0)

    async def reset(self) -> tuple[list[Message], list[Tool]]:
        self.state = None  # this environment is effectively stateless
        return [Message(content=self.problem)], self.tools

    async def step(
        self, action: ToolRequestMessage
    ) -> tuple[list[Message], float, bool, bool]:
        if not action.tool_calls:
            return (
                [
                    Message(
                        content="Must call one of the provided tools (calculator or check_answer)."
                    )
                ],
                self.config.tool_failure_reward,
                self.config.done_on_failure,
                False,
            )

        valid_action, invalid_action = self.filter_invalid_tool_calls(action)

        invalid_response_msgs = [
            ToolResponseMessage.from_call(tool_call, content="")
            for tool_call in invalid_action.tool_calls
        ]

        if valid_action.tool_calls:
            results = await self.exec_tool_calls(valid_action)
            response_msgs = []
            total_reward = 0.0
            any_done = False

            for tool_call, result in zip(valid_action.tool_calls, results, strict=True):
                response, reward, done = json.loads(result.content)

                response_msgs.append(
                    ToolResponseMessage.from_call(tool_call, content=str(response))
                )

                total_reward += reward
                any_done |= done

            return response_msgs + invalid_response_msgs, total_reward, any_done, False

        return (
            invalid_response_msgs,
            self.config.tool_failure_reward * len(invalid_response_msgs),
            self.config.done_on_failure,
            False,
        )

    def check_answer(self, answer: str) -> tuple[bool, float, Literal[True]]:
        """Check if the proposed answer is correct.

        Args:
            answer: Proposed answer.

        Returns:
            Three-tuple of if correct, associated reward (correct_reward if correct,
                tool_failure_reward if tool failure, otherwise incorrect_reward), and
                True indicating done.
        """
        try:
            correct: bool = (
                abs(float(answer) - self.answer)
                / (abs(self.answer) + self.config.rel_tol)
                < self.config.rel_tol
            )
            reward = (
                self.config.correct_reward if correct else self.config.incorrect_reward
            )
        except ValueError:
            return False, self.config.tool_failure_reward, True
        else:
            return correct, reward, True

    def calculator(self, expr: str) -> tuple[float | str | None, float, bool]:
        """Calculate a mathematical expression.

        Args:
            expr (str): A valid python expression

        Returns:
            float: Result of the expression
        """
        try:
            expr = expr.strip()
            result = eval(expr)  # noqa: S307  # pylint: disable=eval-used
            # logger.debug(f"{expr} = {result}")
            with contextlib.suppress(ValueError):
                if int(result) == result:
                    result = int(result)

        except Exception:
            # msg = f"ERROR: ({expr}) -> {repr(e)}"
            # logger.error(msg)
            return (
                "Error using calculator",
                self.config.tool_failure_reward,
                self.config.done_on_failure,
            )
        return result, self.config.tool_success_reward, False

    def export_frame(self) -> Frame:
        return Frame(
            state={
                "problem_id": self.problem_id,
                "problem": self.problem,
                "answer": self.answer,
            }
        )


# SEE: https://huggingface.co/datasets/openai/gsm8k
GSM8K_PUBLIC_SOURCE = "openai/gsm8k"


class GSM8kDatasetSplit(StrEnum):
    train_full = "train_full"  # full training set from OpenAI
    train = "train"  # 80% of train_full (idx%5 != 0)
    val = "val"  # 20% of train_full (idx%5 == 0)
    test = "test"

    def get_df_from_hf(
        self, hf_source: str, add_metadata: bool = True
    ) -> "pd.DataFrame":
        # All non-test splits are derived from train
        hf_split = "test" if self == self.test else "train"

        kw = {}
        if hf_source == GSM8K_PUBLIC_SOURCE:
            kw["name"] = "main"  # as opposed to "socratic"

        src_df = (
            datasets.load_dataset(hf_source, split=hf_split, **kw)
            .to_pandas()
            .reset_index(drop=True)
        )
        if self == self.train:
            src_df = src_df[src_df.index % 5 != 0]
        elif self == self.val:
            src_df = src_df[src_df.index % 5 == 0]
        if add_metadata:
            # Assign problem ID for the env
            src_df["problem_id"] = self.value + "_" + src_df.index.astype(str)

            # Attempt to extract a numerical answer
            try:
                src_df["answer_num"] = src_df["answer"].apply(
                    # answer is formatted as: <some text>\n#### <answer_num>
                    lambda a: float(a.split("#### ")[1].replace(",", ""))
                )
            except Exception as e:
                raise RuntimeError(
                    "Failed to extract numerical answer from 'answer' column"
                ) from e
        return src_df


class GSM8kDataset(TaskDataset):
    Split: ClassVar = GSM8kDatasetSplit

    def __init__(
        self,
        split: GSM8kDatasetSplit | str,
        config: CalculatorEnvConfig | dict | None = None,
        hf_source: str = GSM8K_PUBLIC_SOURCE,
    ):
        if isinstance(config, dict):  # Serialized config
            config = CalculatorEnvConfig(**config)
        elif config is None:
            config = CalculatorEnvConfig()
        self.config = config
        self.src_df = GSM8kDatasetSplit(split).get_df_from_hf(hf_source)

    def get_new_env_by_idx(self, idx: int) -> CalculatorEnv:
        row = self.src_df.iloc[idx]
        return CalculatorEnv(
            problem_id=row["problem_id"],
            problem=row["question"],
            answer=row["answer_num"],
            config=self.config,
        )

    def __len__(self) -> int:
        return len(self.src_df)
