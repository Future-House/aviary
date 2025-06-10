import asyncio
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

from Bio.Restriction import RestrictionBatch
from Bio.Restriction.Restriction import RestrictionType
from Bio.Seq import Seq

from .enzymes import ENZYMES
from .sequence_models import (
    ORF,
    BioSequence,
    BioSequences,
    OptimizationResult,
    PCRResult,
    SequenceType,
)
from .transforms import sample_nucleotides

# guess it's possible
# but primers usually are 7,
# this would be insane.
MINIMUM_AMPLICON_LENGTH = 7

# Heuristic guesses for now
SHORT_TIMEOUT = 120
LONG_TIMEOUT = 300

# Define constants
GO_BINARIES_PATH = Path(__file__).resolve().parent / "bin"


def convert_fasta(text: str, default_header: str = "Seq") -> tuple[str, bool]:
    """Reads a FASTA or sequence string and a FASTA file as string."""
    fasta_str = text
    if not text.startswith(">"):
        fasta_str = f">{default_header}\n{text}"
    # it must have a newline at the end (after sequence newline)
    if not fasta_str.endswith("\n\n"):
        fasta_str += "\n"
    return fasta_str, "(circular)" in fasta_str


async def find_orfs(
    sequence: BioSequence,
    min_length: int = 30,
    codon_table: int = 0,
    strand: int = 1,
) -> BioSequences:
    """
    Given a sequence, translate and find all ORFs.

    Args:
        sequence: The DNA sequence
        min_length: Minimum nucleotides length of ORFs to return including stop codon (so subtract 3 for min coding length).
        codon_table: Codon table to use (default is 0 - standard)
        strand: Strand to search for ORFs (0 is both, 1 is forward, -1 is reverse)

    Returns:
       A list of DNA ORFs as a BioSequences
    """
    # Note that the defaults codon_table=0 and strand=1 are to be consistent with SeqQA
    fasta_str, _ = convert_fasta(sequence.to_fasta())

    with NamedTemporaryFile() as f:
        fasta_path = Path(f.name)
        fasta_path.write_text(fasta_str)

        command = [
            str(GO_BINARIES_PATH / "orf"),
            "-fasta",
            str(fasta_path),
            "-min-length",
            str(min_length),
            "-codon-table",
            str(codon_table),
            "-strand",
            str(strand),
            # Mirroring the default arguments in `cloning/remote/poly.py`:
            # include_unterminated=False and allow_nested=True
            "-allow-nested",
        ]

        # Run the subprocess asynchronously
        proc = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=SHORT_TIMEOUT
            )
        except TimeoutError as e:
            proc.kill()
            await proc.communicate()
            raise RuntimeError("ORF search timed out") from e

        # Decode the output
        stdout_str = stdout.decode("utf-8")
        stderr_str = stderr.decode("utf-8")

        # Check if the subprocess completed successfully
        if proc.returncode != 0:
            raise RuntimeError(
                f"Subprocess failed with error: {stdout_str}\n{stderr_str}"
            )

    # Process the output
    data = json.loads(stdout_str)
    if not data["orfs"]:
        return BioSequences()
    try:
        orfs = [ORF(**orf) for orf in data["orfs"]]
        return BioSequences(
            sequences=[
                BioSequence(
                    sequence=orf.sequence,
                    type=SequenceType.DNA,
                    name=f"{sequence.name}-ORF-{i}",
                    description=f"ORF {i} | strand={orf.strand}, frame={orf.frame}, index={orf.start}-{orf.end}",
                )
                for i, orf in enumerate(orfs)
            ]
        )
    except Exception as e:
        # Error message is for the agent
        raise ValueError("Failed to find ORFs") from e


async def optimize_translation(
    sequence: BioSequence,
    cg_content: int = 62,
    codon_table: int = 11,
    min_repeat_length: int = 15,
) -> BioSequence:
    """
    Given FASTA or sequence and type, propose optimized DNA sequence. Usually only necessary for de novo protein sequences.

    The defaults are set to optimize for E. coli.

    Args:
        sequence: The DNA or protein sequence to optimize
        cg_content: Target CG content (percentage)
        codon_table: Codon table to use (default is 11 - bacteria) https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi
        min_repeat_length: Min length before it is considered a repeat (must be at least 12)

    Returns:
       An optimized DNA sequence
    """
    # very likely to just fail if it's too low
    min_repeat_length = max(12, int(min_repeat_length))  # noqa: FURB123

    if cg_content < 10 or cg_content > 90:
        raise ValueError("cg_content must an integer between 10 and 90")

    command = [
        str(GO_BINARIES_PATH / "synthesize"),
        "-sequence",
        "".join(sequence.to_fasta().splitlines()[1:]),
        "-type",
        sequence.type.value.lower(),
        "-max-gc",
        str(cg_content),
        "-codon-table",
        str(codon_table),
        "-repeat-length",
        str(min_repeat_length),
    ]

    # Run subprocess asynchronously
    proc = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=SHORT_TIMEOUT
        )
    except TimeoutError as e:
        proc.kill()
        await proc.communicate()
        raise RuntimeError("Optimize translation timed out") from e

    # Decode the output
    stdout_str = stdout.decode("utf-8")
    stderr_str = stderr.decode("utf-8")

    # Check if the process was successful
    if proc.returncode != 0:
        raise RuntimeError(f"Subprocess failed with error: {stdout_str}\n{stderr_str}")

    # Process the output
    try:
        data = OptimizationResult(**json.loads(stdout_str))
        return BioSequence(
            sequence=data.optimized_dna,
            type=sequence.type,
        )
    except Exception as e:
        # Error message is for the agent
        raise ValueError("Could not optimize sequence.") from e


# unused at the moment - may want to revisit
async def enzyme_cut_poly(
    sequence: BioSequence,
    enzyme: str,
) -> BioSequences:
    """
    Given a single sequence and single enzyme, cut the sequence.

    Args:
        sequence: DNA sequence to cut
        enzyme: Name of enzyme to use for cutting

    Returns:
        A BioSequences object containing fragments
    """
    result = await digest_and_ligate(sequence, enzyme, ligate=False)
    if not result.sequences:
        # this means it failed to cut - just return the original sequence
        return BioSequences(sequences=[sequence])
    return result


def enzyme_cut(sequence: BioSequence, enzyme: str) -> BioSequences:
    """
    Given a single sequence and single enzyme, cut the sequence.

    Args:
        sequence: DNA sequence to cut
        enzyme: Name of enzyme to use for cutting

    Returns:
        A BioSequences object containing fragments
    """
    # Initialize a restriction batch with the specified enzyme
    rb = RestrictionBatch([enzyme])

    # Check if the enzyme is recognized
    if enzyme not in rb:
        # Error message is for the agent
        raise ValueError(
            f"Enzyme {enzyme!r} not found in Biopython's Restriction module."
        )

    enz = cast(RestrictionType, rb.get(enzyme))
    fragments = enz.catalyze(Seq(sequence.sequence), linear=not sequence.is_circular)
    return BioSequences(
        sequences=[
            BioSequence(
                sequence=str(fragment),
                type=SequenceType.DNA,
                description=f"Fragment {i}",
            )
            for i, fragment in enumerate(fragments)
        ]
    )


async def goldengate(sequences: BioSequences, enzyme: str = "BsaI") -> BioSequences:
    """
    Given a list of DNA sequences, perform Golden Gate assembly.

    Args:
        sequences: DNA sequences to assemble
        enzyme: Name of enzyme to use for cutting (BsaI by default)

    Returns:
        All possible assemblies
    """
    return await digest_and_ligate(sequences, enzyme, ligate=True)


async def digest_and_ligate(
    sequences: BioSequence | BioSequences, enzymes: str, ligate: bool = False
) -> BioSequences:
    """
    Given a list of DNA sequences and enzymes, digest and ligate the sequences.

    Args:
        sequences: DNA sequences to assemble
        enzymes: Comma separated list of enzymes to use for cutting (e.g., BsaI,EcoRI)
        ligate: Whether to ligate the fragments after cutting.

    Returns:
        All possible assemblies
    """
    fasta_str, _ = convert_fasta(sequences.to_fasta())
    with NamedTemporaryFile() as f:
        fasta_path = Path(f.name)
        fasta_path.write_text(fasta_str)

        command = [
            str(GO_BINARIES_PATH / "clone"),
            "-rebase",
            str(GO_BINARIES_PATH / "rebase.withref"),
            "-fasta",
            str(fasta_path),
            "-enzymes",
            enzymes,
        ]
        if ligate:
            command.append("-ligate")

        # Run the subprocess asynchronously
        proc = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=LONG_TIMEOUT
            )
        except TimeoutError as e:
            proc.kill()
            await proc.communicate()
            raise RuntimeError("Digestion & ligation simulation timed out") from e

        # Decode the output
        stdout_str = stdout.decode("utf-8")
        stderr_str = stderr.decode("utf-8")

        # Check if the subprocess completed successfully
        if proc.returncode != 0:
            raise RuntimeError(
                f"Subprocess failed with error: {stdout_str}\n{stderr_str}"
            )

    try:
        return BioSequences.from_fasta(stdout_str)

    except Exception as e:
        # Error message is for the agent
        raise RuntimeError(
            "Digestion & ligation simulation failed on these inputs."
        ) from e


async def gibson(sequences: BioSequences) -> BioSequences:
    """
    Given a list of DNA sequences, predict Gibson Assembly outcome.

    Args:
        sequences: DNA sequences to assemble

    Returns:
        All possible assemblies
    """
    fasta_str, _ = convert_fasta(sequences.to_fasta())
    with NamedTemporaryFile() as f:
        fasta_path = Path(f.name)
        fasta_path.write_text(fasta_str)
        command = [
            str(GO_BINARIES_PATH / "gibson"),
            "-fasta",
            str(fasta_path),
        ]

        # Run the subprocess asynchronously
        proc = await asyncio.create_subprocess_exec(
            *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=LONG_TIMEOUT
            )
        except TimeoutError as e:
            proc.kill()
            await proc.communicate()
            raise RuntimeError("Gibson assembly simulation timed out") from e

        # Decode the output
        stdout_str = stdout.decode("utf-8")
        stderr_str = stderr.decode("utf-8")

        # Check if the subprocess completed successfully
        if proc.returncode != 0:
            raise RuntimeError(
                f"Subprocess failed with error: {stdout_str}\n{stderr_str}"
            )

    try:
        return BioSequences.from_fasta(stdout_str)
    except Exception as e:
        # Error message is for the agent
        raise RuntimeError("Gibson assembly simulation failed on these inputs.") from e


def maybe_convert_seq(seq: BioSequence | str | None) -> str:
    # have to use ducktyping here - unclear what is wrong
    # if isinstance(seq, BioSequence):
    #     return seq.sequence
    try:
        return seq.sequence  # type: ignore[union-attr]
    except AttributeError:
        return seq or ""  # type: ignore[return-value]


lock = asyncio.Lock()


async def design_primers(
    sequence: BioSequence,
    forward_overhang: BioSequence | None = None,
    forward_overhang_name: str | None = None,
    reverse_overhang: BioSequence | None = None,
    reverse_overhang_name: str | None = None,
    target_tm: float | int = 60.0,  # noqa: PYI041
) -> list[BioSequence]:
    """
    Given a DNA sequence, design PCR primers.

    Args:
        sequence: DNA sequence to design primers for
        forward_overhang: Overhang to add to the forward primer. Can specify at most one of forward_overhang and
            forward_overhang_name (optional)
        forward_overhang_name: Enzyme common name to use for the forward overhang (optional).
        reverse_overhang: Overhang to add to the reverse primer. Can specify at most one of reverse_overhang and
            reverse_overhang_name (optional).
        reverse_overhang_name: Enzyme common name to use for the reverse overhang (optional).
        target_tm: Target melting temperature of the primers
    Returns:
        Forward primer, reverse primer, and amplicon sequences
    """
    if forward_overhang and forward_overhang_name:
        raise ValueError(
            "Should specify at most one of forward_overhang and forward_overhang_name"
        )
    if reverse_overhang and reverse_overhang_name:
        raise ValueError(
            "Should specify at most one of reverse_overhang and reverse_overhang_name"
        )

    fw_str = maybe_convert_seq(forward_overhang or forward_overhang_name)
    bw_str = maybe_convert_seq(reverse_overhang or reverse_overhang_name)
    # Now check if these overhangs refer to enzymes
    # enzymes have mixed case, so we can check
    if fw_str and not (fw_str.islower() or fw_str.isupper()):
        try:
            fw_str = sample_nucleotides(str(ENZYMES[fw_str]["recognition_site"]))
        except KeyError as e:
            raise ValueError(f"Unknown enzyme {fw_str}") from e
    if bw_str and not (bw_str.islower() or bw_str.isupper()):
        try:
            bw_str = sample_nucleotides(str(ENZYMES[bw_str]["recognition_site"]))
        except KeyError as e:
            raise ValueError(f"Unknown enzyme {bw_str}") from e
    fasta_str, circular = convert_fasta(sequence.to_fasta())
    command = [
        str(GO_BINARIES_PATH / "primers"),
        "-sequence",
        "".join(fasta_str.splitlines()[1:]),
        "-forward-overhang",
        fw_str,
        "-reverse-overhang",
        bw_str,
        "-target-tm",
        str(target_tm),
        "-forward-primer",
        "",
        "-reverse-primer",
        "",
    ]

    if circular:
        command.append("-circular")

    # Run the subprocess asynchronously
    proc = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=LONG_TIMEOUT
        )
    except TimeoutError as e:
        proc.kill()
        await proc.communicate()
        raise RuntimeError("Designing primers timed out") from e

    # Decode the output
    stdout_str = stdout.decode("utf-8")
    stderr_str = stderr.decode("utf-8")

    # Check if the subprocess completed successfully
    if proc.returncode != 0:
        raise RuntimeError(f"Subprocess failed with error: {stdout_str}\n{stderr_str}")

    try:
        # Process the output
        data = PCRResult(**json.loads(stdout_str))
    except Exception as e:
        # Error message is for the agent
        raise RuntimeError("Failed to design primers for these inputs.") from e

    # have seen this just fail and return empty amplicon
    if len(data.amplicon_fasta) < 10:
        # Error message is for the agent
        raise ValueError(
            "Unable to design primers: PCR simulation ran successfully, but no amplicon was observed."
        )
    return [
        BioSequence(
            name="Forward Primer",
            sequence=data.forward_primer,
            type=SequenceType.DNA,
        ),
        BioSequence(
            name="Reverse Primer",
            sequence=data.reverse_primer,
            type=SequenceType.DNA,
        ),
        BioSequence.from_fasta(data.amplicon_fasta),
    ]


async def simulate_pcr(
    sequence: BioSequence,
    forward_primer: BioSequence | None = None,
    forward_primer_name: str | None = None,
    reverse_primer: BioSequence | None = None,
    reverse_primer_name: str | None = None,
) -> BioSequence:
    """
    Simulate PCR.

    Args:
        sequence: DNA sequence to design primers for
        forward_primer: Forward primer sequence (as 5'->3'). Must specify exactly one of
            forward_primer and forward_primer_name.
        forward_primer_name: Enzyme common name to use for the forward primer.
        reverse_primer: Reverse primer sequence (as 5'->3'). Must specify exactly one of
            reverse_primer and reverse_primer_name.
        reverse_primer_name: Enzyme common name to use for the reverse primer.

    Returns:
        Amplicon sequence
    """
    if (forward_primer is None) == (forward_primer_name is None):
        raise ValueError(
            "Must specify exactly one of forward_primer and forward_primer_name"
        )
    if (reverse_primer is None) == (reverse_primer_name is None):
        raise ValueError(
            "Must specify exactly one of reverse_primer and reverse_primer_name"
        )

    fasta_str, circular = convert_fasta(sequence.to_fasta())
    fw_str = maybe_convert_seq(forward_primer or forward_primer_name)
    bw_str = maybe_convert_seq(reverse_primer or reverse_primer_name)

    command = [
        str(GO_BINARIES_PATH / "primers"),
        "-sequence",
        "".join(fasta_str.splitlines()[1:]),
        "-forward-overhang",
        "",
        "-reverse-overhang",
        "",
        "-target-tm",
        str(45.0),
        "-forward-primer",
        fw_str,
        "-reverse-primer",
        bw_str,
    ]

    if circular:
        command.append("-circular")

    # Run the subprocess asynchronously
    proc = await asyncio.create_subprocess_exec(
        *command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=LONG_TIMEOUT
        )
    except TimeoutError as e:
        proc.kill()
        await proc.communicate()
        raise RuntimeError("Simulating PCR timed out") from e

    # Decode the output
    stdout_str = stdout.decode("utf-8")
    stderr_str = stderr.decode("utf-8")

    # Check if the subprocess completed successfully
    if proc.returncode != 0:
        raise RuntimeError(f"Subprocess failed with error: {stderr_str}")

    try:
        data = PCRResult(**json.loads(stdout_str))

    except Exception as e:
        raise RuntimeError("Failed to simulate PCR for these inputs.") from e

    if len(data.amplicon_fasta) < MINIMUM_AMPLICON_LENGTH:
        raise ValueError(
            "PCR simulation ran successfully, but no amplicon was observed."
        )
    return BioSequence.from_fasta(data.amplicon_fasta)


def find_sequence_overlap(
    sequence: BioSequence, primer: BioSequence, reverse: bool
) -> tuple[int, int] | None:
    """Searches for the overlap of a primer in a sequence.

    Returns a tuple of the start and end position of the overlap. If an overlap is not
    found, None is returned. Overhangs are supported: if part of the primer hangs off
    the sequence, then only the overlapping interval is returned.

    Args:
        sequence: The DNA sequence to search in
        primer: The primer to search for.
        reverse: Whether to search for the reverse complement of the primer.
    """

    def reverse_complement(seq: str) -> str:
        complement = str.maketrans("ACGT", "TGCA")
        return seq.translate(complement)[::-1]

    # Convert sequence and primer to uppercase strings
    seq_str = sequence.sequence.upper()
    # TODO: support primer lookup by name
    primer_str = maybe_convert_seq(primer).upper()

    # If reverse is True, get the reverse complement of the primer
    if reverse:
        primer_str = reverse_complement(primer_str)

    seq_len = len(seq_str)
    primer_len = len(primer_str)

    # First, see if we have an exact match
    index = seq_str.find(primer_str)
    if index != -1:
        return (index, index + primer_len)

    # No exact match - now check for a match at the beginning or end with overhang
    # Determine the range of possible overlap lengths
    max_overlap = min(primer_len, seq_len)

    overlap_length = 0
    for i in range(1, max_overlap + 1):
        if not reverse:
            # Check if the last i bases of the primer match the first i bases of the sequence
            if primer_str[-i:] == seq_str[:i]:
                overlap_length = i
        elif primer_str[:i] == seq_str[-i:]:
            # Check if the first i bases of the primer match the last i bases of the sequence
            overlap_length = i

    if overlap_length == 0:
        return None

    if not reverse:
        # Overlap is at the start of the sequence
        start = 0
        end = overlap_length
    else:
        # Overlap is at the end of the sequence
        start = seq_len - overlap_length
        end = seq_len

    return (start, end)
