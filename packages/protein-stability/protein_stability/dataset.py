import logging
from enum import StrEnum, auto
from pathlib import Path

import pandas as pd
from aviary_internal.utils import DataRepo
from ldp.alg import ComputeTrajectoryMetricsMixin

from aviary.core import TaskDataset
from protein_stability.environment import ProteinStabilityEnv, ProteinStabilityEnvConfig

logger = logging.getLogger(__name__)


# See README.txt in repo for provenance.
DEFAULT_DATA_REPO_NAME = "baseline-envs/proteincrow-stability/v1"


class ProteinStabilityDatasetSplit(StrEnum):
    train = auto()
    val = auto()
    test = auto()
    all = auto()
    pcrow = auto()
    pcrow_small = auto()


class ProteinStabilityDataset(
    TaskDataset[ProteinStabilityEnv], ComputeTrajectoryMetricsMixin
):
    """
    Dataset for the Protein Stability task. Protein Stability task is to propose mutations to a protein sequence
    that would improve its stability. The dataset consists of a list of proteins and their corresponding pdb and sequence
    text files. The task is to propose mutations to the protein sequence that would improve its stability.
    """

    def __init__(
        self,
        workspace_root: Path,
        split: ProteinStabilityDatasetSplit,
        env_config: ProteinStabilityEnvConfig,
        source_data: DataRepo | None = None,
    ):
        if source_data is None:
            source_data = DataRepo(name=DEFAULT_DATA_REPO_NAME)
        source_data.pull(progress=True)

        self.workspace_root = workspace_root
        self.env_config = env_config

        self.data_root = Path(source_data.local_path)
        csv_path = self.data_root / f"stability_{split.value}.csv"
        self.data = pd.read_csv(csv_path, header=0, index_col=False)

    def get_new_env_by_idx(self, idx: int) -> ProteinStabilityEnv:
        row = self.data.iloc[idx]
        txt_path = self.data_root / "filtered_txts" / row["local_txt_path"]
        pdb_path = self._fix_pdb_path(
            self.data_root / "filtered_pdbs" / row["local_pdb_path"]
        )
        aligned_path = self.data_root / "clustal_aligned" / row["aligned_file_path"]
        logger.debug(f"{txt_path}, {pdb_path}, {aligned_path}")
        return ProteinStabilityEnv(
            workspace_root=self.workspace_root,
            local_seq_file=txt_path,
            local_pdb_file=pdb_path,
            aligned_path=aligned_path,
            chain_id=row["chain_id"],
            protein_name=row["protein_name"],
            wt_seq=row["wt_seq"],
            config=self.env_config,
        )

    @staticmethod
    def _fix_pdb_path(pdb_path: Path) -> Path:
        # PDB path casing is off - bandage to fix here
        stem = pdb_path.stem
        pdb_id = stem.split("_")[0]
        pdb_id_lower = pdb_id.lower()
        stem_lower = stem.replace(pdb_id, pdb_id_lower)
        return pdb_path.parent / f"{stem_lower}.pdb"

    def __len__(self) -> int:
        return len(self.data)
