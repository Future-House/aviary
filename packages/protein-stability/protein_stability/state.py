import contextlib
import shutil
from pathlib import Path
from typing import Self
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class ProteinStabilityState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workspace: Path
    seq_file: Path
    pdb_file: Path
    aligned_file: Path
    protein_name: str
    sys_prompt: str
    chain_id: str

    reward: float = 0.0
    steps: int = 0
    done: bool = False

    mutations: list[str] = Field(default_factory=list)
    tool_responses: list[dict] = Field(default_factory=list)

    @classmethod
    def factory(
        cls,
        workspace_root: Path,
        src_seq_file: Path,
        src_pdb_file: Path,
        protein_name: str,
        sys_prompt_template: str,
        chain_id: str,
        protein_seq: str,
        src_aligned_file: Path,
    ) -> Self:
        pdb_id = src_pdb_file.stem
        workspace = cls.make_unique_dir(workspace_root, prefix=f"{pdb_id}_{chain_id}")
        seq_file = workspace / src_seq_file.name
        pdb_file = workspace / src_pdb_file.name
        aligned_file = workspace / src_aligned_file.name

        shutil.copy(src_seq_file, seq_file)
        shutil.copy(src_pdb_file, pdb_file)
        with contextlib.suppress(FileNotFoundError):
            # HACK: we are missing some aligned files in the Megascale dataset, but we can
            # skip here b/c the user should configure `find_conserved_residues_tool=False`
            # Megascale problems. In the future, we should (a) remove the files from the CSV and
            # (b) make this argument optional
            shutil.copy(src_aligned_file, aligned_file)

        sys_prompt = sys_prompt_template.format(
            local_seq_file=seq_file,
            local_pdb_file=pdb_file,
            sequence=protein_seq,
        )

        return cls(
            workspace=workspace,
            seq_file=seq_file,
            pdb_file=pdb_file,
            sys_prompt=sys_prompt,
            chain_id=chain_id,
            protein_name=protein_name,
            aligned_file=aligned_file,
        )

    @staticmethod
    def make_unique_dir(workspace_root: Path, prefix: str) -> Path:
        uniq = str(uuid4())[:8]
        workspace = workspace_root / f"{prefix}_{uniq}"
        workspace.mkdir(parents=True, exist_ok=False)
        return workspace
