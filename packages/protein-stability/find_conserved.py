"""
This is a script that uses the Biopython library to perform a BLAST search on a protein sequence and then align the
results using Clustal Omega. The script then identifies conserved residues in the alignment and returns a description of the conserved residues. Used to pre-process data for find_conserved_residues tool.

This requires you set CLUSTAL_OMEGA_EXE_PATH environment variable to the path of the Clustal Omega executable. You can download Clustal Omega from http://www.clustal.org/omega/. And set the path to the executable in the environment variable.
"""

import hashlib
import logging
import os
import subprocess
from collections import Counter
from pathlib import Path

from Bio import SeqIO
from Bio.Blast import NCBIWWW, NCBIXML

os.environ["CLUSTAL_OMEGA_EXE_PATH"] = ""


def hash_sequence(input_sequence: str) -> str:
    sha256_hash = hashlib.sha256()
    sha256_hash.update(input_sequence.encode("utf-8"))
    return sha256_hash.hexdigest()


logger = logging.getLogger(__name__)

CACHE_UNIPROT_DIR = Path("~/.cache/proteincrow/uniprot").expanduser()


def find_conserved_residues(filename) -> str:
    # Read the sequence from the local file
    seq = Path(filename).read_text().strip()
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
    wait_for_result = NCBIWWW.qblast("blastp", "swissprot", seq)
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
    print("Clustal Omega ran successfully")
    print(result.stderr)
    print(clustal_command)
    print(output_temp_aln)

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
        conserved_site_description = "No conserved residues found or the BLAST search did not return any results.Residue numbers here are 0-indexed."

    return conserved_site_description + "Residue numbers here are 0-indexed."
