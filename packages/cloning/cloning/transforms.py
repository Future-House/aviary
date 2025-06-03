"""DNA manipulation functions."""

import random

from .sequence_models import BioSequence, BioSequences, SequenceType, make_pretty_id


def slice_sequence(
    seqs: BioSequences, name: str, start: int | None = None, end: int | None = None
) -> BioSequence:
    """Slice a sequence from a BioSequences object. Represents ordering a specific sequence from DNA synthesis vendor.

    Args:
        seqs: A BioSequences object containing multiple sequences.
        name: The name of the sequence to slice.
        start: The start index of the slice - 1 based indexing. If not provided, will start at the beginning of the sequence.
        end: The end index of the slice - 1 based indexing. If not provided, will go to the end.

    Returns:
        The sliced sequence.
    """
    new_seq = ""
    stype = SequenceType.DNA
    # try to convert any mispecified types
    if isinstance(start, str):
        start = int(start)  # type: ignore[unreachable]
    if isinstance(end, str):
        end = int(end)  # type: ignore[unreachable]
    # clean up indices for what an LLM/user would expect
    if start:
        start += 1
    if end:
        end += 1
    if start and end and start == end:
        # because they may ask for slice 5 to 5 and expect 1 base
        end = start + 1
    if isinstance(seqs, BioSequence):  # type: ignore[unreachable]
        new_seq = seqs.sequence[start:end]  # type: ignore[unreachable]
        stype = seqs.type
    else:
        for seq in seqs.sequences:
            if seq.name == name:
                new_seq = seq.sequence[start:end]
                stype = seq.type
        if not new_seq:
            raise TypeError(
                f"Sequence with name {name} not found in BioSequences object. Available sequences: { {seq.name for seq in seqs.sequences} }"
            )
    return BioSequence(
        sequence=new_seq,
        name=make_pretty_id(prefix=f"{name}-slice"),
        type=stype,
        is_circular=False,
    )


def separate(
    seqs: BioSequences | BioSequence,
) -> list[BioSequence]:
    """Separate a BioSequences object into individual BioSequences. Represents electrophoresis gel separation.

    Args:
        seqs: A BioSequences object containing multiple sequences.

    Returns:
        A list of BioSequence objects
    """
    # Agent often tries to split a single BioSequence. Return it as a list.
    if isinstance(seqs, BioSequence):
        return [seqs]
    return seqs.sequences


def mix(seq1: BioSequence, seq2: BioSequence) -> BioSequences:
    """Mix two BioSequence objects into a single BioSequences object.

    Args:
        seq1: BioSequence object 1
        seq2: BioSequence object 2

    Returns:
        BioSequences
    """
    return BioSequences(sequences=[seq1, seq2])


def merge(seqs: list[BioSequence]) -> BioSequences:
    """Merge multiple BioSequence objects into a single BioSequences object.

    Args:
        seqs: A list of BioSequence objects.

    Returns:
        A BioSequences object containing the joined sequences.
    """
    return BioSequences(sequences=seqs)


def add(sequences: BioSequences, new_sequence: BioSequence) -> None:
    """Add a new sequence to an existing BioSequences object in-place.

    Args:
        sequences: A BioSequences object containing multiple sequences.
        new_sequence: The new sequence to add.
    """
    sequences.sequences.append(new_sequence)


def sample_nucleotides(seq: str) -> str:
    """Given a sequence that contains mixed nucleotides, return a random sequence with only one nucleotide type."""
    result = []
    for s in seq:
        s = s.upper()
        if s not in "ATCG":
            result.append(random.choice(mixed_nucleotides[s]))
        else:
            result.append(s)
    return "".join(result)


def reverse_complement(sequence: str) -> str:
    return "".join(complement_base(base) for base in reversed(sequence))


def complement(sequence: str) -> str:
    return "".join(complement_base(base) for base in sequence)


def complement_base(base_pair: str) -> str:
    """
    Accepts a base pair and returns its complement base pair.

    This function expects characters in the range a-z and A-Z and
    will return a space ' ' for characters that are not matched
    to any known base.
    """
    return complement_table.get(base_pair, " ")


mixed_nucleotides = {
    "B": "CGT",
    "D": "AGT",
    "H": "ACT",
    "K": "GT",
    "M": "AC",
    "N": "ACGT",
    "R": "AG",
    "S": "CG",
    "V": "ACG",
    "W": "AT",
    "Y": "CT",
}

# Complement table provides 1:1 mapping between bases and their complements
complement_table = {
    "A": "T",
    "T": "A",
    "C": "G",
    "G": "C",
    "B": "V",
    "V": "B",
    "D": "H",
    "H": "D",
    "K": "M",
    "M": "K",
    "N": "N",
    "R": "Y",
    "S": "S",
    "W": "W",
    "Y": "R",
    "a": "t",
    "t": "a",
    "c": "g",
    "g": "c",
    "b": "v",
    "v": "b",
    "d": "h",
    "h": "d",
    "k": "m",
    "m": "k",
    "n": "n",
    "r": "y",
    "s": "s",
    "w": "w",
    "y": "r",
}


def _chunk_string(s, chunk_size=20):
    # Chunk things up so the llms can count (breaking up tokenization)
    chunks = [s[i : i + chunk_size] for i in range(0, len(s), chunk_size)]
    lines = [
        f"{i * chunk_size + 1:03d}-{min(len(s), (i + 1) * chunk_size):03d}: "
        + " ".join(c)
        for i, c in enumerate(chunks)
    ]
    return "\n".join(lines)


def view_translation(seq: BioSequence, start: int = 1, stop: int | None = None) -> str:
    r"""View the AA translation of a DNA sequence.

    Args:
        seq: The DNA sequence to translate.
        start: Optional start AA position (1-indexed)
        stop: Optional stop AA position (inclusive)
    """
    # lazily import, since this it's only used here
    from Bio.Seq import Seq

    if isinstance(seq, BioSequences):  # type: ignore[unreachable]
        raise TypeError("Only a single sequence can be translated")

    if seq.type != SequenceType.DNA:
        raise ValueError("Sequence must be DNA for translation")
    s = Seq(seq.sequence)
    t = str(s.translate())[start - 1 : stop]

    return _chunk_string(t)
