import hashlib
import logging
import operator
import os
import re
import subprocess
import textwrap
from collections import Counter
from pathlib import Path

import biotite.structure as struc
import numpy as np
import pqapi
from aiohttp import ClientSession, ClientTimeout
from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from biotite.structure.io import pdb

from protein_stability.inference_apis import ESM_API, PRODY_API, ROSETTA_API
from protein_stability.state import ProteinStabilityState

logger = logging.getLogger(__name__)

CACHE_UNIPROT_DIR = Path("~/.cache/proteincrow/uniprot").expanduser()
ROSETTA_DDG_TIMEOUT = ClientTimeout(total=60 * 60)


def check_mutation_string(input_string):
    pattern1 = r"^(\d+)[a-zA-Z].*([a-zA-Z])$"
    pattern2 = r"^([a-zA-Z])(\d+)([a-zA-Z])$"
    match1 = re.match(pattern1, input_string)
    match2 = re.match(pattern2, input_string)
    if match1:
        return f"{match1.group(1)}{match1.group(2)}"
    if match2:
        return f"{match2.group(2)}{match2.group(3)}"
    raise ValueError(
        "Invalid input format. Expected formats are '<int><letter1><any_characters><letter2>' or '<letter1><int><letter2>'."
    )


def hash_sequence(input_sequence: str) -> str:
    sha256_hash = hashlib.sha256()
    sha256_hash.update(input_sequence.encode("utf-8"))
    return sha256_hash.hexdigest()


def apply_mutations_to_sequence(
    txt_file_with_seq: Path, mutations: list[str]
) -> tuple[str, str]:
    """Apply the proposed mutations to the sequence."""
    with open(txt_file_with_seq) as f:  # noqa: FURB101
        wt_sequence = f.read()

    # apply mutation to the sequence
    mut_sequence = list(wt_sequence.strip())
    for mut in mutations:
        try:
            mutated_residue = mut[-1]

            def is_int(s):
                return s.isdigit() or (s.startswith("-") and s[1:].isdigit())

            position = mut[0:-1]

            if is_int(position):
                pass
            else:
                return (
                    f"Error in applying mutation {mut} to the sequence. Please check the mutation format. It "
                    f"should list of mutations written as 235C where C is the "
                    f"mutant residue being proposed at position 235. "
                ), "Error"
            if int(position) >= len(mut_sequence):
                return (
                    f"Error in applying mutation {mut} to the sequence. Position {position} is out of range. "
                    f"Sequence is only {len(mut_sequence)} residues long."
                ), "Error"
            mut_sequence[int(position)] = mutated_residue
        except Exception as e:
            return (
                f"Error in applying mutation {mut} to the sequence. Please check the mutation format. It should "
                f"be 235C where C is the mutant residue being proposed at position 235. Only use single letter codes for amino acids."
            ), "Error"
    return wt_sequence, "".join(mut_sequence)


def get_analysis_of_sequence(sequence: str) -> str:
    """Get the properties of the protein sequence."""
    prot_analysis = ProteinAnalysis(sequence)
    molecular_weight = round(prot_analysis.molecular_weight(), 2)
    aromaticity = round(prot_analysis.aromaticity(), 2)
    instability_index = round(prot_analysis.instability_index(), 2)
    isoelectric_point = round(prot_analysis.isoelectric_point(), 2)
    molar_extinction_coefficient = prot_analysis.molar_extinction_coefficient()
    fraction_charged_residues = round(
        fraction_charge_residues(sequence, total_charge=True, pH=7.4), 2
    )
    sequence_charge = round(sequence_charge_definition(sequence), 2)
    secondary_structure_fraction = prot_analysis.secondary_structure_fraction()
    secondary_structure_fraction = [round(x, 2) for x in secondary_structure_fraction]
    grand_average_hydropathy = prot_analysis.gravy()
    return "\n".join([
        f"<seq>{sequence}</seq>",
        f"<length>{len(sequence)}</length>",
        (
            f'<secondary_structure_fraction helix="{secondary_structure_fraction[0]}" '
            f'turn="{secondary_structure_fraction[1]}" '
            f'sheet="{secondary_structure_fraction[2]}"/>'
        ),
        (
            f'<properties aromaticity="{aromaticity}" '
            f'charged_residues_fraction="{fraction_charged_residues}" '
            f'molecular_weight="{molecular_weight}" '
            f'instability_index="{instability_index}" '
            f'isoelectric_point="{isoelectric_point}" '
            f'grand_average_hydropathy="{grand_average_hydropathy}" '
            f'molar_extinction_coefficient="{molar_extinction_coefficient[0]}" '
            f'sequence_charge="{sequence_charge}" '
            "/>"
        ),
    ])


async def get_secondary_structure(state: ProteinStabilityState) -> str:
    """
    Get the secondary structure annotation of the protein. The output contains per-residue DSSP annotations.

    Args:
        state: ProteinStabilityState, the state of the environment
    """
    with open(state.pdb_file) as f:  # noqa: FURB101
        pdb_string = f.read()

    json_data = {"pdb_strings": [pdb_string], "chain_ids": [state.chain_id]}
    headers = {
        "Authorization": f"Bearer {os.environ['MODAL_DEPLOY_TOKEN']}",
        "Content-Type": "application/json",
    }
    residue_annotations_str = "Failed to annotate"
    async with ClientSession() as session:
        async with session.post(
            f"{ROSETTA_API}/rosetta/dssp_with_pdb_string",
            json=json_data,  # Ensure payload is sent as JSON
            headers=headers,  # Pass headers
            timeout=ClientTimeout(3600.0),
        ) as resp:
            resp.raise_for_status()
            # Debugging: Print status and response
            logger.info(f"Response Status: {resp.status}")
            try:
                response = await resp.json()

                description = response[0]["description"]
                residue_annotations = response[0]["residue_annotations"]
                residue_annotations_list = [v for k, v in residue_annotations.items()]
                residue_annotations_str = (
                    f"<seq>{''.join(residue_annotations_list)}</seq>"
                )
            except Exception as e:
                # Note we do not raise from e - we need to package all the info into the RuntimeError
                # so it gets sent back to the agent.
                raise RuntimeError(f"Failed to get secondary structure: {e}") from None

        # TODO: do we really need both the regions and the sequence? We should do an ablation
        # here.
        return description + "\n" + residue_annotations_str


async def compute_llr(mutations: list[str], state: ProteinStabilityState) -> str:
    """
    Use a protein language model to compute the log likelihood ratio between the proposed mutant residues and the
    wild type residues. Defined as `log(P(mutant) / P(wild type))`, where `P` is the language model likelihood.

    Args:
        mutations: list of strings, in the form ['132A', '143S'], where 132 is the zero-indexed position of the mutation and A is the proposed mutation.
        state: ProteinStabilityState, the state of the environment
    """
    wt_sequence, mut_sequence = apply_mutations_to_sequence(state.seq_file, mutations)
    if mut_sequence == "Error":
        return wt_sequence
    # compute perplexity
    async with (
        ClientSession() as session,
        session.post(
            ESM_API + "/esm/probabilities",
            json={
                "mut_sequence": mut_sequence,
                "wt_sequence": wt_sequence,
                "muts": mutations,
            },
            headers={
                "Authorization": f"Bearer {os.environ['MODAL_DEPLOY_TOKEN']}",
                "Content-Type": "application/json",
            },
        ) as resp,
    ):
        resp.raise_for_status()
        llrs: list[tuple[str, float]] = await resp.json()

    return "Log-likelihood ratios:\n" + "\n".join(
        f'<residue mut="{mut}" llr="{llr}"/>' for mut, llr in llrs
    )


def get_residue_at_position(residues: list[int], state: ProteinStabilityState) -> str:
    """
    Returns the residues and residue properties at the provided zero-indexed positions in the sequence.

    Args:
        residues: list of zero-indexed residue positions for which to get residue properties.
        state: ProteinStabilityState, the state of the environment
    """
    with open(state.seq_file) as f:  # noqa: FURB101
        sequence = f.read()

    group_json = {
        "acidic": ["E", "D"],
        "basic": ["K", "R"],
        "charged": ["E", "D", "K", "R"],
        "polar": ["Q", "N", "S", "T", "G", "C", "H"],
        "aliphatic": ["A", "L", "M", "I", "V"],
        "aromatic": ["F", "Y", "W"],
        "proline": ["P"],
    }
    xml_lines: list[str] = []
    overflow: list[int] = []
    for res in residues:
        if res >= len(sequence):
            overflow.append(res)
        else:
            props = ",".join([
                group for group in group_json if sequence[res] in group_json[group]
            ])
            xml_lines.append(
                f'<residue pos="{res}" aa="{sequence[res]}" properties="{props}"/>'
            )

    description = "\n".join(xml_lines)
    if overflow:
        description += (
            f"\nCould not retrieve residues at positions {','.join(map(str, overflow))}. "
            "Sequence is only {len(sequence)} residues long."
        )
    return description


async def search_literature_about_the_protein(state: ProteinStabilityState):
    """
    Searches the literature for information about functional residues in the protein or the effect of known mutations on stability.

    Args:
        state: current state of the environment
    """
    pdb_id = os.path.basename(state.pdb_file).split("_")[0]
    question = (
        f"What are the functional residues or known mutations that effect protein stability of the protein "
        f"{state.protein_name} found in PDB id {pdb_id}?"
    )
    state.tool_responses.append({"tool": "search_literature", "question": question})
    results = await pqapi.async_agent_query(question)
    return results.session.answer


async def search_scientific_literature(question: str, state: ProteinStabilityState):
    """
    Searches the scientific literature for the given question.

    Args:
        question: the question to ask the literature
        state: current state of the environment
    """
    state.tool_responses.append({"tool": "search_literature", "question": question})
    results = await pqapi.async_agent_query(question)
    return results.session.answer


ALLOWED_BOND_TYPES = [
    "hydrogen_bonds",
    "salt_bridges",
    "repulsive_ionic_bonding",
    "pi_stacking",
    "pi_cation",
    "disulfide_bonds",
]


async def get_bond_types_between(
    residues: list[str], bond_type: str, state: ProteinStabilityState
):
    """
    Gets bonds of specified type between residues in the protein structure.

    Returns an XML list of the form <bond type="BOND_TYPE" res1="RES1" atom1="ATOM1" res2="RES2" atom2="ATOM2"/>,
    where the bond is between ATOM1 (in RES1) and ATOM2 (in RES2).

    Args:
        residues: list of zero-indexed residue positions between which we need to get the bond types. Limit to 10 residues at once.
        bond_type: the type of bond to search for. One of 'hydrogen_bonds', 'salt_bridges', 'repulsive_ionic_bonding', 'pi_stacking', 'pi_cation', 'disulfide_bonds'.
        state: ProteinStabilityState, the state of the environment
    """
    # NOTE: we have to write out the bond types explicitly in the docstring b/c __doc__ is not set if we
    # use an f-string/%/.format. TODO: Do something like get_bond_types_between.__doc__ = ...
    if bond_type not in ALLOWED_BOND_TYPES:
        return f"{bond_type} not found in the response. Please only use one of the following: {ALLOWED_BOND_TYPES}"

    state.tool_responses.append({
        "tool": "get_bond_types_between",
        "question": residues,
    })
    with open(state.seq_file) as f:  # noqa: FURB101
        sequence = f.read()
    with open(state.pdb_file) as f:  # noqa: FURB101
        pdb_string = f.read()

    bond_descriptions: list[str] = []
    payload = {
        "pdb_string": pdb_string,
        "sequence": sequence,
    }
    async with (
        ClientSession() as session,
        session.post(
            f"{PRODY_API}/bonds",
            json=payload,
            headers={
                "Authorization": f"Bearer {os.getenv('MODAL_DEPLOY_TOKEN')}",
                "Content-Type": "application/json",
            },
            timeout=ROSETTA_DDG_TIMEOUT,
        ) as resp,
    ):
        response = await resp.json()
        # TODO: why can't we just do response[look_for_bond_type] here?
        for bond_type_key in response:
            if bond_type == bond_type_key:
                bond_type_results = response[bond_type_key]
                for result in bond_type_results:
                    res1 = result[0][3:]
                    atom1 = result[1]
                    res2 = result[3][3:]
                    atom2 = result[4]
                    residues = [str(res) for res in residues]
                    if str(res1) in residues and str(res2) in residues:
                        bond_descriptions.append(
                            f'<bond type="{bond_type_key}" res1="{res1}" atom1="{atom1}" res2="{res2}" atom2="{atom2}"/>'
                        )
                        logger.info(f"TRUE {bond_descriptions}")
                    else:
                        continue
    if not bond_descriptions:
        return f"No {bond_type} found between the residues."

    return (
        "<bonds>\n" + textwrap.indent("\n".join(bond_descriptions), "  ") + "\n</bonds>"
    )


def fraction_charge_residues(
    sequence: str, total_charge: bool, pH: float = 7.4
) -> float:
    """
    Calculate the fraction of charged residues in a protein sequence.
    If not provided assumed neutral, and only R/K/D/E are considered titratable residues. If pH is provided then
    R/K/D/E/C/Y/H are all considered titratable residues.

    Args:
        sequence: A string representing the protein sequence.
        total_charge: A boolean indicating whether to calculate the total charge.
        pH: A float representing the pH value. Default is 7.4
    """
    # Refer to https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5075173/table/Tab4/ for these values
    pka_values = {
        "C": 8.5,
        "Y": 10.1,
        "H": 6.5,
        "E": 4.1,
        "D": 3.9,
        "K": 10.0,
        "R": 12.5,
    }
    charge_sign = 1 if total_charge else -1
    charge_total = 0.0
    for res in sequence:
        if res in {"K", "R", "H"}:
            charge_total += 1 / (1 + np.power(10, pH - pka_values[res]))

        if res in {"E", "D", "Y", "C"}:
            charge_total += charge_sign / (1 + np.power(10, pka_values[res] - pH))
    return charge_total / len(sequence)


def sequence_charge_definition(sequence: str) -> float:
    """
    Sequence Charge Definition metric can be used to find modification sites on a protein sequence
    that can cause maximal changes in protein conformations.

    https://pubs.aip.org/aip/jcp/article/143/8/085101/73623/A-theoretical-method-to-compute-sequence-dependent

    Args:
        sequence: A string representing the protein sequence
    """

    def get_charge(res_code: str):
        """Get the charge of a residue code."""
        if res_code in {"D", "E"}:
            return -1
        if res_code in {"R", "K"}:
            return 1
        return 0

    total_charge = 0.0
    for m in range(2, len(sequence) + 1):
        for n in range(1, m):
            total_charge += (
                float(get_charge(sequence[m - 1]))
                * float(get_charge(sequence[n - 1]))
                * np.power((m - n), 0.5)
            )
    return total_charge / len(sequence)


def get_mutated_sequence(mutations: list[str], state: ProteinStabilityState):
    """
    Provides the original sequence of the protein and the mutated sequence if any mutations have been proposed.

    Args:
        mutations: list of strings, the mutations to the sequence that you want to propose and are confident will improve the stability
        state: ProteinStabilityState, the state of the environment
    """
    with open(state.seq_file) as f:  # noqa: FURB101
        wt_sequence = f.read()
    out = "<original_sequence>" + wt_sequence + "</original_sequence>"
    if len(mutations) == 0:
        return out

    _, mut_sequence = apply_mutations_to_sequence(state.seq_file, mutations)
    out += "\n<mutated_sequence>" + mut_sequence + "</mutated_sequence>"
    return out


def get_sequence_properties(
    mutations: list[str], return_wt: bool, state: ProteinStabilityState
) -> str:
    """
    Get the properties of the sequence when the mutations are applied to the sequence. Each mutation in the list should be in the format of
    235C where C is the mutant residue being proposed at position 235 (zero-indexed).

    Args:
        mutations: list of mutations to the sequence
        return_wt: whether to return the sequence properties of the wild type sequence as well
        state: current state of the environment

    """
    wt_sequence, mut_sequence = apply_mutations_to_sequence(state.seq_file, mutations)
    logger.info("Applied mutations to sequence")
    if mut_sequence == "Error":
        # Why are we doing this? Because apply_mutations returns (error_message, "Error") if there
        # is an error. So in reality, wt_sequence is an error message. TODO: avoid overloading return
        # variables.
        raise RuntimeError(wt_sequence)

    logger.info("Analyze mutations to sequence")
    desc = "\n".join([
        "<mutant>\n",
        textwrap.indent(get_analysis_of_sequence(mut_sequence.strip()), "  "),
        "\n</mutant>",
    ])
    logger.info("Analysis sequence properties complete")
    if return_wt:
        desc += "\n".join([
            "<wild_type>\n",
            textwrap.indent(get_analysis_of_sequence(wt_sequence.strip()), "  "),
            "\n</wild_type>",
        ])
    return desc


def complete(mutations: list[str], state: ProteinStabilityState):
    """
    Complete the experiment and return the final mutated sequence. Your task is not done until you call this function. Only call this function when you are confident that the proposed mutations will improve the stability of the protein.

    Args:
        mutations: list of strings, the mutations to the sequence that you want to propose and are confident will improve stability. Mutations should be in the format of 235C where C is the mutant residue being proposed at position 235 (zero-indexed).
        state: ProteinStabilityState, the state of the environment
    """
    if len(mutations) == 0:
        return "You have not proposed any mutations. Please propose at least 3 mutations to the protein sequence."
    logger.info(f"Proposed mutations: {mutations}")
    state.mutations = mutations
    # write the mutations to the state

    proposed_mut_folder = Path(state.seq_file).parent / "proposed_mutations"
    os.makedirs(proposed_mut_folder, exist_ok=True)
    path_to_mut_file = (
        proposed_mut_folder
        / f"{os.path.basename(state.seq_file).replace('.txt', '_mut.txt')}"
    )
    Path(path_to_mut_file).write_text(f"{mutations}\n")
    state.done = True
    return (
        "You have successfully proposed the mutations to the protein sequence. The proposed mutations are: "
        + ", ".join(mutations)
    )


async def compute_hydrophobicity_score(state: ProteinStabilityState) -> str:
    """
    Get the top 20 residues with the highest spatial aggregation propensity (SAP) score using Rosetta.

    Args:
        state: current state.
    """
    rosetta_description = (
        "Residues with the highest spatial aggregation propensity (SAP) score:\n"
    )
    with open(state.pdb_file) as f:  # noqa: FURB101
        data = f.read()

    async with (
        ClientSession() as session,
        session.post(
            ROSETTA_API + "/rosetta/compute_sap_score",
            json={"pdb_string": data, "chain_id": state.chain_id},
            headers={
                "Authorization": f"Bearer {os.environ['MODAL_DEPLOY_TOKEN']}",
                "Content-Type": "application/json",
            },
        ) as resp,
    ):
        resp.raise_for_status()
        rosetta_scores = await resp.json()
    rosetta_sap_scores = rosetta_scores["data"]["pose_scores"]
    sorted_rosetta_sap_scores = sorted(
        rosetta_sap_scores.items(), key=operator.itemgetter(1), reverse=True
    )[:20]
    xml_lines: list[str] = [
        f'<residue res="{x[0]}" score="{round(x[1], 2)}"/>'
        for x in sorted_rosetta_sap_scores
    ]
    return rosetta_description + "\n".join(xml_lines)


def find_conserved_residues(state: ProteinStabilityState) -> str:
    """
    Finds the conserved residues in the protein sequence by doing a BLAST search against all non-redundant GenBank CDS translations+PDB+SwissProt+PIR+PRF excluding environmental samples from WGS projects.

    Args:
        state: current state.
    """
    # Read the sequence from the local file
    with open(state.seq_file) as f:  # noqa: FURB101
        seq = f.read().strip()

    pdb_hash = hash_sequence(seq)
    cache_file_path = os.path.join(CACHE_UNIPROT_DIR, f"{pdb_hash}.txt")
    conserved_site_description = ""

    # Create cache directory if it doesn't exist
    os.makedirs(CACHE_UNIPROT_DIR, exist_ok=True)

    # Check if the result is already cached
    if os.path.exists(cache_file_path):
        with open(cache_file_path) as f:  # noqa: FURB101
            return f.read()

    # Perform BLAST search
    wait_for_result = NCBIWWW.qblast("blastp", "nr", seq)
    parse_results = NCBIXML.read(wait_for_result)

    temp_blast_file = f"{pdb_hash}_blast_results.fasta"
    output_temp_aln = f"{pdb_hash}_alignment_results.fasta"
    # Temporary files for BLAST and alignment outputs

    # Write BLAST alignment results
    with open(temp_blast_file, "w") as fasta_file:
        # Iterate over alignments in the BLAST results
        for alignment in parse_results.alignments:
            for hsp in alignment.hsps:
                # Create a unique identifier for each sequence
                fasta_header = f">{alignment.hit_id} {alignment.hit_def}"
                sequence = hsp.sbjct  # The subject sequence from the BLAST result

                # Write the header and sequence to the FASTA file
                fasta_file.write(f"{fasta_header}\n{sequence}\n")

    clustal_command = f"{os.environ['CLUSTAL_OMEGA_EXE_PATH']} -i {temp_blast_file} -o {output_temp_aln} --force"

    result = subprocess.run(  # noqa: S602
        clustal_command, capture_output=True, shell=True, check=False
    )
    logger.info("Clustal Omega ran successfully")

    # Parse the alignment results
    sequence_records = list(SeqIO.parse(output_temp_aln, "fasta"))

    # Analyze conserved residues
    residue_ids = [i for i, res in enumerate(seq) if res != "-"]
    for res_num in residue_ids:
        column = [str(record.seq)[res_num] for record in sequence_records]
        counts = Counter(column)
        if not counts:
            continue
        most_common_residue, most_common_count = counts.most_common(1)[0]
        score = most_common_count / len(column)  # Fraction of the most common residue
        conserved_site_description += (
            f"At residue position {res_num}, {most_common_residue} is conserved in "
            f"{score * 100:.2f}% of sequences. "
        )

    # Cleanup temporary files
    os.unlink(temp_blast_file)
    os.unlink(output_temp_aln)

    # Cache the result and return the description
    if conserved_site_description.strip():
        with open(cache_file_path, "w") as f:  # noqa: FURB103
            f.write(conserved_site_description)
    else:
        conserved_site_description = (
            "No conserved residues found or the BLAST search did not return any results. "
            "Residue numbers here are 0-indexed."
        )

    return conserved_site_description + "Residue numbers here are 0-indexed."


def find_conserved_residues_precomputed(state: ProteinStabilityState) -> str:
    """
    Finds the conserved residues in the protein sequence by doing a BLAST search against all non-redundant GenBank CDS translations+PDB+SwissProt+PIR+PRF excluding environmental samples from WGS projects.

    Args:
        state: current state.
    """
    with open(state.seq_file) as f:  # noqa: FURB101
        seq = f.read().strip()
    # check file exists
    if not os.path.exists(state.aligned_file):
        return "Failed to find information about conserved residues"
    # Parse the alignment results
    sequence_records = list(SeqIO.parse(state.aligned_file, "fasta"))
    conserved_site_description = ""
    # Analyze conserved residues
    residue_ids = [i for i, res in enumerate(seq) if res != "-"]
    for res_num in residue_ids:
        column = [str(record.seq)[res_num] for record in sequence_records]
        counts = Counter(column)
        if not counts:
            continue
        most_common_residue, most_common_count = counts.most_common(1)[0]
        score = most_common_count / len(column)  # Fraction of the most common residue
        conserved_site_description += (
            f"At residue position {res_num}, {most_common_residue} is conserved in "
            f"{score * 100:.2f}% of sequences. "
        )
    return conserved_site_description + "Residue numbers here are 0-indexed."


def get_distance_between_residues(
    residues: list[int], state: ProteinStabilityState
) -> str:
    """
    Returns the distances between all pairs of the specified residues in the protein structure.

    Args:
        residues: list of zero-indexed residue positions between which to compute distance. Limit to 10 residues at once.
        state: ProteinStabilityState, the state of the environment
    """
    state.tool_responses.append({
        "tool": "get_distance_between_residues",
        "question": residues,
    })
    pdb_file = pdb.PDBFile.read(state.pdb_file)
    structure = pdb_file.get_structure(model=1)
    atom_array = structure[struc.filter_amino_acids(structure)]
    distance_description = ["Distance between pairs in Angstroms:"]
    for res1 in residues:
        for res2 in residues:
            # TODO: should this be res1 != res2? Is it important to return both?
            if res1 < res2:
                distance = np.linalg.norm(
                    atom_array[res1].coord - atom_array[res2].coord
                )
                distance_description.append(
                    f'<pair residue1="{res1}" residue2="{res2}" distance="{distance}"/>'
                )
    if len(distance_description) > 1:
        return "\n".join(distance_description)
    return "Unable to find distance between the residues."
