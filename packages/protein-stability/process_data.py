import io
import os
from pathlib import Path

import biotite.structure as struc
import pandas as pd
from aiohttp import ClientSession, ClientTimeout
from biotite.sequence import ProteinSequence
from biotite.structure.io import pdb


async def fix_pdb_file(pdb_file_path: str, chain_id: str) -> str:
    # biotite structure
    structure = pdb.PDBFile.read(pdb_file_path).get_structure(model=1)
    atom_array = structure[struc.filter_amino_acids(structure)]
    atom_array = atom_array[atom_array.chain_id == chain_id]
    pdb_file = pdb.PDBFile()
    pdb_file.set_structure(atom_array)
    input_file_path = io.StringIO()
    pdb_file.write(input_file_path)
    input_file_path.seek(0)
    pdb_str = input_file_path.read()
    print("Calling the Modal API")

    async with (
        ClientSession() as session,
        session.post(
            "https://future-house--openmm-make-app.modal.run/fixpdb",
            json={"pdb_string": pdb_str},
            headers={
                "Authorization": "Bearer e0311794-04de-4f0a-8c43-8a9967285d54",
                "Content-Type": "application/json",
            },
            # Adding timeout to 600 seconds, fix pdb is much faster than alphafold and rfdiffusion
            timeout=ClientTimeout(600),
        ) as resp,
    ):
        resp.raise_for_status()
        response = await resp.json()

    if response:
        output_response = response["results"]["data"]
        Path(pdb_file_path.replace(".pdb", f"_{chain_id}.pdb")).write_text(
            output_response
        )
        return response["results"]["data"]
    return ""


local_dir = "./protein_stability/data"
raw_pdbs_file = "./protein_stability/data/raw_pdbs"
megascale_train_file = "mega_train.csv"
megascale_val_file = "mega_val.csv"
megascale_test_file = "mega_test.csv"

megascale_train_local_path = os.path.join(local_dir, megascale_train_file)
megascale_val_local_path = os.path.join(local_dir, megascale_val_file)
megascale_test_local_path = os.path.join(local_dir, megascale_test_file)

train_df = pd.read_csv(megascale_train_local_path)
val_df = pd.read_csv(megascale_val_local_path)
test_df = pd.read_csv(megascale_test_local_path)

unique_proteins_train = list({x.split(".pdb")[0] for x in train_df["name"].unique()})
unique_proteins_val = list({x.split(".pdb")[0] for x in val_df["name"].unique()})
unique_proteins_test = list({x.split(".pdb")[0] for x in test_df["name"].unique()})

LENGTH_OF_PDB_ID = 4
unique_proteins_train_pdbs_only = [
    x for x in unique_proteins_train if len(x) == LENGTH_OF_PDB_ID
]
unique_proteins_val_pdbs_only = [
    x for x in unique_proteins_val if len(x) == LENGTH_OF_PDB_ID
]
unique_proteins_test_pdbs_only = [
    x for x in unique_proteins_test if len(x) == LENGTH_OF_PDB_ID
]


def get_chain_sequences(pdb_file):
    # Read the PDB file
    pdb_file = pdb.PDBFile.read(pdb_file)
    atom_array = pdb.get_structure(pdb_file)[0]
    amino_acids = atom_array[struc.filter_amino_acids(atom_array)]

    # Get the unique chain IDs
    chain_ids = amino_acids.chain_id
    unique_chain_ids = set(chain_ids)
    chain_sequences = {}

    for chain_id in unique_chain_ids:
        # Filter atoms by chain ID
        chain_atoms = amino_acids[amino_acids.chain_id == chain_id]
        ca_atoms = chain_atoms[chain_atoms.atom_name == "CA"]
        ca_1_letter = [
            ProteinSequence.convert_letter_3to1(res_name)
            for res_name in ca_atoms.res_name
        ]
        chain_sequences[chain_id] = "".join(ca_1_letter)
    return chain_sequences
