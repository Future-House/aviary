import logging
from pathlib import Path
from typing import ClassVar, cast

from aviary_internal.utils.config import ConfigModel

from aviary.core import (
    Environment,
    Frame,
    Message,
    Messages,
    Tool,
    ToolRequestMessage,
)
from protein_stability.rewards import compute_rosetta_ddg_reward
from protein_stability.state import ProteinStabilityState
from protein_stability.tools import (
    complete,
    compute_hydrophobicity_score,
    compute_llr,
    find_conserved_residues_precomputed,
    get_bond_types_between,
    get_distance_between_residues,
    get_mutated_sequence,
    get_residue_at_position,
    get_secondary_structure,
    get_sequence_properties,
    search_literature_about_the_protein,
    search_scientific_literature,
)

logger = logging.getLogger(__name__)


class ProteinStabilityEnvConfig(ConfigModel):
    binary_reward: bool
    find_conserved_residues_tool: bool


class ProteinStabilityEnv(Environment[ProteinStabilityState]):
    state: ProteinStabilityState
    system_prompt_template: ClassVar[
        str
    ] = """You are an expert in protein design, tasked with improving the stability of a protein sequence. Your objective is to analyze the provided protein sequence and structure, utilizing the available tools to propose mutations that increase its stability.

Provide detailed reasoning and evidence to support your proposed mutations and explain how they might enhance stability.

Design at least 3 mutations and a maximum of 7 mutations to the protein sequence {sequence} to improve its stability.

The provided tools have access to protein's sequence and structure."""

    def __init__(
        self,
        workspace_root: Path,
        local_seq_file: Path,
        local_pdb_file: Path,
        aligned_path: Path,
        chain_id: str,
        protein_name: str,
        wt_seq: str,
        config: ProteinStabilityEnvConfig,
    ) -> None:
        self.workspace_root = workspace_root
        self.local_seq_file = local_seq_file
        self.local_pdb_file = local_pdb_file
        self.aligned_path = aligned_path
        self.chain_id = chain_id
        self.protein_name = protein_name
        self.wt_seq = wt_seq
        self.config = config

    async def reset(self) -> tuple[list[Message], list[Tool]]:
        tools = [
            compute_llr,
            get_bond_types_between,
            get_secondary_structure,
            get_sequence_properties,
            get_distance_between_residues,
            compute_hydrophobicity_score,
            complete,
            get_mutated_sequence,
            search_scientific_literature,
            search_literature_about_the_protein,
            get_residue_at_position,
        ]
        if self.config.find_conserved_residues_tool:
            tools.append(find_conserved_residues_precomputed)
        self.tools = [Tool.from_function(tool) for tool in tools]  # type: ignore[arg-type]

        self.state = ProteinStabilityState.factory(
            workspace_root=self.workspace_root,
            src_seq_file=self.local_seq_file,
            src_pdb_file=self.local_pdb_file,
            protein_name=self.protein_name,
            sys_prompt_template=self.system_prompt_template,
            chain_id=self.chain_id,
            protein_seq=self.wt_seq,
            src_aligned_file=self.aligned_path,
        )
        start = Message(content=self.state.sys_prompt)

        return [start], self.tools

    async def step(
        self, action: ToolRequestMessage
    ) -> tuple[Messages, float, bool, bool]:
        if not action.tool_calls:
            # NOTE: we used to return -0.1 here, but I wasn't sure why - complicates learning
            # and the agent doesn't use it. Set to 0.0 for now. [SN]
            return ([Message(content=("Please execute a tool."))], 0.0, False, False)
        logger.info(f"Executing tool calls: {action.tool_calls}")
        msgs = await self.exec_tool_calls(
            action, state=self.state, handle_tool_exc=True, exec_timeout=600
        )

        for msg in msgs:
            if msg.content and "Failed to execute tool call for tool" in msg.content:
                msg.content = (
                    "Failed to execute tool call. This can be due to invalid input or "
                    "a tool failure. Please check the input and try again."
                )
        logger.info(f"Tool responses: {msgs}")
        if self.state.done:
            reward = await compute_rosetta_ddg_reward(
                self.state, binary=self.config.binary_reward
            )
            self.state.reward += float(reward)
        return cast(list[Message], msgs), self.state.reward, self.state.done, False

    def export_frame(self) -> Frame:
        if not hasattr(self, "state"):
            return Frame()
        return Frame(
            state={
                "steps": self.state.steps,
                "done": self.state.done,
                "reward": self.state.reward,
                "prompt": self.state.sys_prompt,
                "workspace": str(self.state.workspace),
            },
            info={},
        )
