from pathlib import Path
from tempfile import mkdtemp

import pytest
from protein_stability.environment import ProteinStabilityEnv
from protein_stability.rewards import compute_rosetta_ddg
from protein_stability.state import ProteinStabilityState
from protein_stability.tools import (
    complete,
    compute_hydrophobicity_score,
    compute_llr,
    find_conserved_residues,
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

from tests.conftest import IN_GITHUB_ACTIONS

MOCK_DATA_DIR = Path(__file__).parent / "mock_data"
MOCK_WORKSPACE_ROOT = Path(mkdtemp())


@pytest.fixture
def protein_crow_state() -> ProteinStabilityState:
    seq_file = MOCK_DATA_DIR / "example.txt"
    return ProteinStabilityState.factory(
        workspace_root=MOCK_WORKSPACE_ROOT,
        src_seq_file=seq_file,
        src_pdb_file=MOCK_DATA_DIR / "example.pdb",
        src_aligned_file=MOCK_DATA_DIR / "example.fasta",
        protein_name="Example",
        sys_prompt_template=ProteinStabilityEnv.system_prompt_template,
        chain_id="A",
        protein_seq=seq_file.read_text(),
    )


@pytest.mark.asyncio
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test takes too long for CI.")
async def test_get_secondary_structure(protein_crow_state: ProteinStabilityState):
    response = await get_secondary_structure(protein_crow_state)
    assert len(response) > 0


@pytest.mark.asyncio
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test takes too long for CI.")
async def test_perplexity(
    protein_crow_state: ProteinStabilityState,
):
    mutations = ["T1G", "E3S"]
    protein_crow_state.mutations = mutations
    response = await compute_llr(mutations, protein_crow_state)
    assert len(response) > 0


@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test takes too long for CI.")
def test_get_sequence_properties(
    protein_crow_state: ProteinStabilityState,
):
    mutations = ["T1G", "E3S"]
    protein_crow_state.mutations = mutations
    sequence_properties = get_sequence_properties(
        mutations=protein_crow_state.mutations, return_wt=True, state=protein_crow_state
    )
    assert len(sequence_properties) > 0


@pytest.mark.asyncio
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test takes too long for CI.")
async def test_bond_types(
    protein_crow_state: ProteinStabilityState,
):
    residues = ["1", "2", "5", "8", "9"]
    bond_types = await get_bond_types_between(
        residues=residues, bond_type="hydrogen_bonds", state=protein_crow_state
    )
    assert len(bond_types) == 0


@pytest.mark.asyncio
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test takes too long for CI.")
async def test_compute_rosetta_ddg(protein_crow_state):
    protein_crow_state.mutations = ["21Q"]
    response = await compute_rosetta_ddg(protein_crow_state)
    assert response != 0.0


@pytest.mark.asyncio
def test_get_distance_between_two_residues(protein_crow_state):
    response = get_distance_between_residues([2, 3], protein_crow_state)
    assert response


@pytest.mark.asyncio
def test_find_conserved_residues_precomputed(protein_crow_state):
    response = find_conserved_residues_precomputed(protein_crow_state)
    assert response


@pytest.mark.asyncio
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test takes too long for CI.")
def test_find_conserved_residues(protein_crow_state):
    response = find_conserved_residues(protein_crow_state)
    assert response


@pytest.mark.asyncio
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test takes too long for CI.")
async def test_compute_hydrophobicity_score(protein_crow_state):
    response = await compute_hydrophobicity_score(protein_crow_state)
    assert response


@pytest.mark.asyncio
def test_complete(protein_crow_state):
    response = complete(mutations=["1G", "3S"], state=protein_crow_state)
    assert "successfully" in response


@pytest.mark.asyncio
def test_get_sequence(protein_crow_state):
    response = get_mutated_sequence(mutations=["1G", "3S"], state=protein_crow_state)
    assert response


@pytest.mark.asyncio
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test takes too long for CI.")
async def test_search_scientific_literature(protein_crow_state):
    response = await search_scientific_literature(
        question="What biochemical factors improve protein stability ?",
        state=protein_crow_state,
    )
    assert response


@pytest.mark.asyncio
@pytest.mark.skipif(IN_GITHUB_ACTIONS, reason="This test takes too long for CI.")
async def test_search_literature_about_the_protein(protein_crow_state):
    response = await search_literature_about_the_protein(state=protein_crow_state)
    assert response


@pytest.mark.asyncio
def test_get_residue_at_position(protein_crow_state):
    response = get_residue_at_position(residues=[1, 4], state=protein_crow_state)
    assert response
