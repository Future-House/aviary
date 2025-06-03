import json
import logging
import os
from collections.abc import Sequence
from enum import StrEnum, auto
from functools import partial
from pathlib import Path
from typing import ClassVar, cast

import httpx
from aviary.core import (
    Environment,
    Frame,
    Message,
    MultipleChoiceQuestion,
    Tool,
    ToolRequestMessage,
    argref_by_name,
    eval_answer,
    wraps_doc_only,
)
from aviary.env import ENV_REGISTRY
from aviary.message import EnvStateMessage
from pydantic import BaseModel, Field

from .clients.annotator_client import annotate
from .clients.search_client import search_genes, search_plasmids
from .poly_local import LONG_TIMEOUT, enzyme_cut, find_sequence_overlap
from .sequence_models import BioSequence, BioSequences, SequenceType
from .transforms import add, merge, separate, slice_sequence, view_translation

logger = logging.getLogger(__name__)

GO_BINARIES_PATH = Path(__file__).resolve().parent / "bin"
_BINARIES_FOUND = (
    GO_BINARIES_PATH.exists()
    and GO_BINARIES_PATH.is_dir()
    and (GO_BINARIES_PATH / "orf").is_file()
    and (GO_BINARIES_PATH / "synthesize").is_file()
    and (GO_BINARIES_PATH / "clone").is_file()
    and (GO_BINARIES_PATH / "gibson").is_file()
    and (GO_BINARIES_PATH / "primers").is_file()
)

_USE_LOCAL_BINARIES = False
if _BINARIES_FOUND:
    output_path = GO_BINARIES_PATH / "rebase.withref"
    url = "http://rebase.neb.com/rebase/link_withrefm"

    # Define headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.0.0 Safari/537.36",
    }

    # Download the rebase.withref file if it doesn't exist
    if not os.path.exists(output_path):
        with httpx.Client(headers=headers) as client:
            # Send a GET request
            response = client.get(url)
        try:
            response.raise_for_status()  # Check for HTTP errors
            os.makedirs(GO_BINARIES_PATH, exist_ok=True)
            with open(output_path, "wb") as f:  # noqa: FURB103
                f.write(response.content)
            _USE_LOCAL_BINARIES = True
        except httpx.HTTPStatusError:
            _USE_LOCAL_BINARIES = False
    else:
        _USE_LOCAL_BINARIES = True

_USE_LOCAL_BINARIES = _USE_LOCAL_BINARIES and (
    # Allows the user to force Modal
    os.getenv("CLONING_USE_MODAL", "False").lower() not in {"true", "1"}
)


if _USE_LOCAL_BINARIES:
    from .poly_local import (
        design_primers,
        find_orfs,
        gibson,
        goldengate,
        optimize_translation,
        simulate_pcr,
    )
else:
    from .clients.poly_client import (
        design_primers,
        find_orfs,
        gibson,
        goldengate,
        optimize_translation,
        simulate_pcr,
    )


class CloningState(BaseModel):
    refs: dict[str, BioSequences | BioSequence]
    protocol: list[str] = Field(default_factory=list)
    done: bool = False
    reward: float = 0
    proposed_answer: str | None = None

    def old__str__(self):
        if not self.refs:
            return "No saved sequences available."
        s = "Available sequence keys: " + ", ".join(self.refs.keys()) + "\n\n"
        for k, v in self.refs.items():
            # indent each line within
            indent_v = str(v).replace("\n", "\n" + " " * 4)
            s += f"{k} (type={type(v).__name__})\n--------\n{indent_v}\n\n"
        return s

    def __str__(self):
        s = "Available sequence keys: " + ", ".join(self.refs.keys()) + "\n\n"
        s += json.dumps(
            {k: v.make_state_dict() for k, v in self.refs.items()}, indent=4
        )
        return s


EXAMPLE_STATE = CloningState(
    refs={
        "dummy-seq-abc123": BioSequence(
            sequence="ATCGATCG", name="dummy-seq-abc123", type=SequenceType.DNA
        )
    }
)


ENVIRONMENT_SPECIFIC_INSTRUCTIONS = (
    "You are provided with a key-value store of sequences. "
    "I will always inform of you of the state of this store. "
    "Use keys (names) of sequences in all functions, rather than pass them directly. "
    f"For example, if this the current contents of key-value store:\n{EXAMPLE_STATE}\n"
    "Then, you could call `annotate(sequence='dummy-seq-abc123')` or `find_orfs(sequence='dummy-seq-abc123', min_length=35)`. "
    "Tool call outputs will be added to the key-value store.\n"
    "Sometimes sequences are alone as a BioSequence. Other times, they are bundled in a BioSequences object. "
    "You can separate a BioSequences object into individual sequences using the `separate` tool. "
    "Or you can slice into one sequence from a BioSequences object."
    # "The steps you take will be recorded in a protocol."
)


def _get_unique(names: set[str], new_one: str) -> str:
    if new_one not in names:
        return new_one
    i = 1
    while f"{new_one}_{i}" in names:
        i += 1
    return f"{new_one}-{i}"


def format_annotation_for_eval(s: BioSequence, filter_ann: set | None = None) -> str:
    filter_: set = filter_ann or set()
    return "\n".join([
        f"{a.start} - {a.end} | {a.name} | "
        + (a.full_name.replace("\n", "") if a.full_name else "")
        for a in s.get_ordered_annotations()
        if a.name.strip() not in filter_
    ])


class CloningEnvTaskTypes(StrEnum):
    DIRECT = auto()
    SEQUENCE = auto()
    NO_TASK = auto()
    PLASMID = auto()


typed_argref = partial(argref_by_name, type_check=True)
make_tool = partial(Tool.from_function, types_in_param_descriptions=True)


class CloningEnv(Environment[CloningState]):
    # a classvar for how to define plasmid eval
    plasmid_problem_prompt: ClassVar[str] = (
        "Submit a list of annotations of a plasmid in order that should match the desired task."
        " You will be judged based on a specific rubric revealed in the answer."
        " Your score will be based on if the plasmid meets the requirements. "
        "\n\nTask: {task}"
    )

    def __init__(
        self,
        problem: str,
        answer: str | BioSequence | None = None,
        problem_id: str | None = None,
        start: Sequence[BioSequence | BioSequences] | None = None,
        task_type: CloningEnvTaskTypes = CloningEnvTaskTypes.NO_TASK,
        collapse_prior_user_message: bool = False,
        mcq: MultipleChoiceQuestion | None = None,
    ):
        self.problem_id = problem_id
        self.problem = problem
        self.answer = answer
        self.task_type = task_type
        self.start = start
        self.collapse_prior_user_message = collapse_prior_user_message
        self.mcq = mcq  # Use for zero shotting

        logger.debug(self.problem)
        logger.debug(self.start)

        if self.task_type == CloningEnvTaskTypes.PLASMID and not isinstance(
            self.answer, str
        ):
            raise ValueError("For plasmid response tasks, the answer must be a str.")

    async def reset(self) -> tuple[list[Message], list[Tool]]:
        self.state = CloningState(refs={})
        if self.start:
            for i, s in enumerate(self.start):
                if isinstance(s, BioSequences):
                    self.state.refs[f"start-{i}"] = BioSequences(**s.model_dump())
                else:
                    self.state.refs[s.name] = BioSequence(**s.model_dump())

        tools = [
            make_tool(plasmid_search),
            make_tool(gene_search),
            make_tool(typed_argref(prefix="ann-seq")(annotate)),
            make_tool(
                argref_by_name(prefix="slice", args_to_skip={"name", "start", "end"})(
                    slice_sequence
                )
            ),
            make_tool(typed_argref(prefix="separated")(separate)),
            make_tool(
                typed_argref(
                    prefix="pcr",
                    args_to_skip={
                        "target_tm",
                        "forward_overhang_name",
                        "reverse_overhang_name",
                    },
                )(design_primers)
            ),
            make_tool(
                typed_argref(return_direct=True, args_to_skip={"reverse"})(
                    find_sequence_overlap
                )
            ),
            make_tool(
                typed_argref(
                    prefix="amplicon",
                    args_to_skip={"forward_primer_name", "reverse_primer_name"},
                )(simulate_pcr)
            ),
            make_tool(typed_argref(prefix="cut", args_to_skip={"enzyme"})(enzyme_cut)),
            make_tool(
                typed_argref(
                    prefix="orfs", args_to_skip={"min_length", "codon_table", "strand"}
                )(find_orfs)
            ),
            make_tool(typed_argref(prefix="gibson-assembled")(gibson)),
            make_tool(
                typed_argref(prefix="gg-assembled", args_to_skip={"enzyme"})(goldengate)
            ),
            make_tool(
                typed_argref(
                    prefix="opt",
                    args_to_skip={"cg_content", "codon_table", "min_repeat_length"},
                )(optimize_translation)
            ),
            make_tool(typed_argref()(merge)),
            # add returns None, and we don't want to store that in the state, so set return_direct.
            make_tool(typed_argref(return_direct=True)(add)),
            make_tool(
                # we want to let it see the translation, because some tasks require it
                typed_argref(prefix="translate", return_direct=True)(view_translation),
                allow_empty_param_descriptions=True,
            ),
            make_tool(view_sequence_stats, allow_empty_param_descriptions=True),
            make_tool(view_restriction_sites, allow_empty_param_descriptions=True),
            make_tool(view_sequence, allow_empty_param_descriptions=True),
        ]

        @typed_argref(fxn_requires_state=True)
        def submit_final_sequences(final: BioSequence, state: CloningState) -> None:
            """Submit the final single sequence to complete the task."""
            state.refs["final"] = final
            state.done = True
            seq1 = cast(BioSequence, self.answer).sequence
            seq2 = final.sequence

            # we do simple sequence identity comparison
            if len(seq1) != len(seq2):
                state.reward = 0
            state.reward = sum(a == b for a, b in zip(seq1, seq2, strict=True)) / len(
                seq1
            )
            state.proposed_answer = seq2

        @typed_argref(fxn_requires_state=True)
        async def submit_final_plasmid(final: BioSequence, state: CloningState) -> None:
            """Submit the the final plasmid to complete the task."""
            state.refs["final"] = final
            state.done = True
            # make it fresh to remove annotations
            final_seq = BioSequence(
                sequence=final.sequence, name=final.name, type=final.type
            )
            final_seq = await annotate(final_seq)

            # we extract the annotation and order only:
            state.proposed_answer = format_annotation_for_eval(final_seq)

            logger.debug(f"Proposed: {state.proposed_answer}")
            logger.debug(f"Correct: {self.answer}")

            state.reward = await eval_answer(
                proposed=state.proposed_answer,
                correct=str(self.answer),
                question=CloningEnv.plasmid_problem_prompt.format(task=self.problem),
                eval_mode="llm-score",
            )
            logger.debug(f"Reward: {state.reward}")

        async def submit_answer(answer: str, state: CloningState) -> None:
            """Submit the final answer to complete the task. Only provide answer, without reasoning or comments."""
            state.done = True
            state.proposed_answer = answer
            state.reward = await eval_answer(
                proposed=answer,
                correct=str(self.answer),
                question=self.problem,
                eval_mode="llm",
            )

        if self.task_type == CloningEnvTaskTypes.SEQUENCE:
            tools.append(
                make_tool(submit_final_sequences, allow_empty_param_descriptions=True)
            )
        elif self.task_type == CloningEnvTaskTypes.DIRECT:
            tools.append(make_tool(submit_answer, allow_empty_param_descriptions=True))
        elif self.task_type == CloningEnvTaskTypes.PLASMID:
            tools.append(
                make_tool(submit_final_plasmid, allow_empty_param_descriptions=True)
            )
        else:
            # for most task questions, a starting sequence will provided
            # only for open-ended questions, we will allow creating a sequence
            tools.extend([
                make_tool(create_sequence),
                make_tool(finish),
            ])

        self.tools = tools
        return [
            Message(content=self.problem + "\n\n" + ENVIRONMENT_SPECIFIC_INSTRUCTIONS),
            self._get_state_message(),
        ], tools

    def _get_state_message(self) -> EnvStateMessage:
        return EnvStateMessage(
            content="Current contents of key-value store:\n\n" + str(self.state)
        )

    def _print_state(self):
        for k, v in self.state.refs.items():
            print(f"  {k}: {type(v)}")

    async def step(
        self, action: ToolRequestMessage
    ) -> tuple[list[Message], float, bool, bool]:
        msgs = cast(
            list[Message],
            await self.exec_tool_calls(
                action,
                state=self.state,
                handle_tool_exc=True,
                concurrency=False,
                exec_timeout=LONG_TIMEOUT,
            ),
        )
        # add them to protocol
        for a, result in zip(action.tool_calls, msgs, strict=True):
            if not result.content:
                continue
            output = "\n\t".join(result.content.split("\n"))
            self.state.protocol.append(f"{a!s}:\n{output}")

        state_msg = self._get_state_message()
        if self.collapse_prior_user_message:
            # append last message's content to prior tool response message.
            # this avoids intermediate user messages -- to be compatible with Mistral
            # SEE: https://github.com/BerriAI/litellm/issues/6323
            msgs[-1].content = (
                f"{msgs[-1].content}\n\n{state_msg.content}"
                if msgs[-1].content
                else state_msg.content
            )
        else:
            msgs.append(state_msg)

        return msgs, self.state.reward, self.state.done, False

    def export_frame(self) -> Frame:
        return Frame(
            state="\n\n".join(self.state.protocol),
            info={
                "problem_id": self.problem_id,
                "problem": self.problem,
                "ideal": self.answer,
                "answer": self.state.proposed_answer,
            },
        )


ENV_REGISTRY["cloning"] = "cloning.env", "CloningEnv"


def finish(state: CloningState) -> None:
    """Mark the task as complete."""
    state.done = True


# forward the docstring
@wraps_doc_only(search_plasmids)
async def plasmid_search(query: str, state: CloningState) -> str:
    results = await search_plasmids(query)
    msg = "Found the following plasmids:\n"

    # uniquify the titles
    titles: set[str] = set()
    for r in results:
        r.title = _get_unique(set(state.refs.keys()) | titles, r.title)
        titles.add(r.title)
    # now add a few
    for result in results[:5]:
        state.refs[result.title] = result.to_sequence()
        output = "    \n".join(result.body.splitlines())
        msg += f"  {result.title}: {output}\n"
    return msg


@wraps_doc_only(search_genes)
def gene_search(query: str, state: CloningState) -> str:
    seqs = search_genes(query)
    msg = "Found the following genes:\n"

    for seq in seqs.sequences:
        title = _get_unique(set(state.refs.keys()), seq.name)
        state.refs[title] = seq
        msg += f"  {title}: {seq!s}\n"

    return msg


def create_sequence(name: str, sequence: str, state: CloningState) -> str:
    """Create and store a new DNA sequence. Represents ordering a specific sequence from DNA synthesis vendor.

    Args:
        name: The name of the sequence.
        sequence: The sequence
        state: The state object
    """
    s = BioSequence(sequence=sequence, name=name, type=SequenceType.DNA)
    state.refs[name] = s
    return f"Added sequence {name} to sequences."


@typed_argref(return_direct=True)
def view_restriction_sites(seq: BioSequence) -> str:
    """View the restriction sites present in a sequence. Only considers common enzymes."""
    return seq.annotate_restriction_sites()


@typed_argref(return_direct=True)
@wraps_doc_only(BioSequence.view_statistics)
def view_sequence_stats(seq: BioSequence) -> str:
    return seq.view_statistics()


@typed_argref(return_direct=True)
def view_sequence(seq: BioSequence) -> str:
    """View the sequence."""
    return seq.sequence
