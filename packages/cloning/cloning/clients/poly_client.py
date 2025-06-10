import os

import httpx

from ..enzymes import ENZYMES
from ..sequence_models import (
    API_URL,
    DEFAULT_TIMEOUT,
    ORF,
    BioSequence,
    BioSequences,
    OptimizationResult,
    PCRResult,
    SequenceType,
)
from ..transforms import sample_nucleotides

# guess it's possible
# but primers usually are 7,
# this would be insane.
MINIMUM_AMPLICON_LENGTH = 7


async def find_orfs(
    sequence: BioSequence,
    min_length: int = 30,
    codon_table: int = 0,
    strand: int = 1,
) -> BioSequences:
    """
    Given a sequence, find ORFs.

    Args:
        sequence: The DNA sequence
        min_length: Minimum nucleotides length of ORFs to return including stop codon (so subtract 3 for min coding length).
        codon_table: Codon table to use (default is 0 - standard).
        strand: Strand to search. 1 is coding strand, -1 is template, 0 is both.

    Returns:
       A list of DNA ORFs as a BioSequences
    """
    # Note that the defaults codon_table=0 and strand=1 are to be consistent with SeqQA
    auth_token = os.environ["MODAL_DEPLOY_TOKEN"]

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/find-orfs",
            headers={
                "Authorization": f"Bearer {auth_token}",
            },
            json={
                "fasta": sequence.to_fasta()
            },  # <- this is a "manual" construction of the pydantic object
            # It looks like this because when there is only one post argument, the whole
            # body is the first object.
            params={
                "min_length": str(min_length),
                "codon_table": str(codon_table),
                "strand": str(strand),
            },
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
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
    auth_token = os.environ["MODAL_DEPLOY_TOKEN"]

    # very likely to just fail if it's too low
    min_repeat_length = max(12, int(min_repeat_length))  # noqa: FURB123

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/synthesize",
            headers={
                "Authorization": f"Bearer {auth_token}",
            },
            json={
                "fasta": sequence.to_fasta()
            },  # <- this is a "manual" construction of the pydantic object
            # It looks like this because when there is only one post argument, the whole
            # body is the first object.
            params={
                "seq_type": sequence.type.value,
                "cg_content": str(cg_content),
                "codon_table": str(codon_table),
                "min_repeat_length": str(min_repeat_length),
            },
            timeout=DEFAULT_TIMEOUT,
        )
        if response.status_code == 422:
            # want to convert 422 to an error that is expected/can be caught
            msg = response.text
            if "returned non-zero exit status 1" in msg:
                raise ValueError(msg)
        response.raise_for_status()
        data = response.json()
        try:
            result = OptimizationResult(**data)
            return BioSequence(
                sequence=result.optimized_dna,
                type=sequence.type,
            )
        except Exception as e:
            raise ValueError("Could not optimize sequence.") from e


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
        A cut with list of fragments
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/cut",
            headers={
                "Authorization": f"Bearer {os.environ['MODAL_DEPLOY_TOKEN']}",
            },
            json={
                "fasta": sequence.to_fasta()
            },  # <- this is a "manual" construction of the pydantic object
            # It looks like this because when there is only one post argument, the whole
            # body is the first object.
            params={
                "enzyme": enzyme,
            },
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        result = BioSequences.from_fasta(response.text)
        if not result.sequences:
            # this means it failed to cut - just return the original sequence
            return BioSequences(sequences=[sequence])
        return result


async def goldengate(sequences: BioSequences, enzyme: str = "BsaI") -> BioSequences:
    """
    Given a list of DNA sequences, perform Golden Gate assembly.

    Args:
        sequences: DNA sequences to assemble
        enzyme: Name of enzyme to use for cutting (BsaI by default)

    Returns:
        All possible assemblies
    """
    return await digest_and_ligate(sequences, enzyme)


async def digest_and_ligate(sequences: BioSequences, enzymes: str) -> BioSequences:
    """
    Given a list of DNA sequences and enzymes, digest and ligate the sequences.

    Args:
        sequences: DNA sequences to assemble
        enzymes: Comma separated list of enzymes to use for cutting (e.g., BsaI,EcoRI)

    Returns:
        All possible assemblies
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/digest",
            headers={
                "Authorization": f"Bearer {os.environ['MODAL_DEPLOY_TOKEN']}",
            },
            json={
                "fasta": sequences.to_fasta()
            },  # <- this is a "manual" construction of the pydantic object
            # It looks like this because when there is only one post argument, the whole
            # body is the first object.
            params={
                "enzymes": enzymes,
            },
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        return BioSequences.from_fasta(response.text)


async def gibson(sequences: BioSequences) -> BioSequences:
    """
    Given a list of DNA sequences, predict Gibson Assembly outcome.

    Args:
        sequences: DNA sequences to assemble

    Returns:
        All possible assemblies
    """
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/gibson",
            headers={
                "Authorization": f"Bearer {os.environ['MODAL_DEPLOY_TOKEN']}",
            },
            json={
                "fasta": sequences.to_fasta()
            },  # <- this is a "manual" construction of the pydantic object
            # It looks like this because when there is only one post argument, the whole
            # body is the first object.
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        return BioSequences.from_fasta(response.text)


def maybe_convert_seq(seq: BioSequence | str | None) -> str:
    # have to use ducktyping here - unclear what is wrong
    # if isinstance(seq, BioSequence):
    #     return seq.sequence
    try:
        return seq.sequence  # type: ignore[union-attr]
    except AttributeError:
        return seq or ""  # type: ignore[return-value]


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
        except KeyError:
            raise ValueError(f"Unknown enzyme {fw_str}") from None
    if bw_str and not (bw_str.islower() or bw_str.isupper()):
        try:
            bw_str = sample_nucleotides(str(ENZYMES[bw_str]["recognition_site"]))
        except KeyError:
            raise ValueError(f"Unknown enzyme {bw_str}") from None

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/primers",
            headers={
                "Authorization": f"Bearer {os.environ['MODAL_DEPLOY_TOKEN']}",
            },
            json={
                "fasta": sequence.to_fasta()
            },  # <- this is a "manual" construction of the pydantic object
            # It looks like this because when there is only one post argument, the whole
            # body is the first object.
            params={
                "forward_overhang": fw_str,
                "reverse_overhang": bw_str,
                "target_tm": str(target_tm),
            },
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        result = PCRResult(**data)
        if len(result.amplicon_fasta) < MINIMUM_AMPLICON_LENGTH:
            raise RuntimeError(
                "Failed to design primers - PCR simulation shows no amplicon."
            )
        return [
            BioSequence(
                name="Forward Primer",
                sequence=result.forward_primer,
                type=SequenceType.DNA,
            ),
            BioSequence(
                name="Reverse Primer",
                sequence=result.reverse_primer,
                type=SequenceType.DNA,
            ),
            BioSequence.from_fasta(result.amplicon_fasta),
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
        forward_primer_name: Enzyme name to use for the forward primer.
        reverse_primer: Reverse primer sequence (as 5'->3'). Must specify exactly one of
            reverse_primer and reverse_primer_name.
        reverse_primer_name: Enzyme name to use for the reverse primer.

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

    fw_str = maybe_convert_seq(forward_primer or forward_primer_name)
    bw_str = maybe_convert_seq(reverse_primer or reverse_primer_name)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/pcr",
            headers={
                "Authorization": f"Bearer {os.environ['MODAL_DEPLOY_TOKEN']}",
            },
            json={
                "fasta": sequence.to_fasta()
            },  # <- this is a "manual" construction of the pydantic object
            # It looks like this because when there is only one post argument, the whole
            # body is the first object.
            params={
                "forward_primer": fw_str,
                "reverse_primer": bw_str,
            },
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
        result = PCRResult(**data)
        if len(result.amplicon_fasta) < MINIMUM_AMPLICON_LENGTH:
            raise ValueError("PCR simulation failed - no amplification observed.")
        return BioSequence.from_fasta(result.amplicon_fasta)
