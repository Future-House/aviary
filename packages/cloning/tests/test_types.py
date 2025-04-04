from textwrap import dedent

import pytest
from envs.cloning.cloning.sequence_models import (
    Annotation,
    BioSequence,
    BioSequences,
    SequenceType,
)


@pytest.fixture
def genbank_content():
    return """LOCUS       TestSequence            36 bp    DNA     circular  01-JAN-1980
DEFINITION  Test sequence for unit testing.
ACCESSION   Unknown
VERSION     Unknown
KEYWORDS    .
SOURCE      Synthetic construct
  ORGANISM  Synthetic construct
            .
FEATURES             Location/Qualifiers
     source          1..36
                     /organism="Synthetic construct"
                     /mol_type="other DNA"
                     /topology="circular"
ORIGIN
        1 atgcgtacgt agctagcgtag ctgctagcta cgatcgatcg
//
"""


@pytest.fixture
def annotations():
    return [
        Annotation(
            start=0, end=10, strand=1, type="gene", name="gene1", full_name="Gene 1"
        ),
        Annotation(
            start=15, end=25, strand=-1, type="CDS", name="cds1", full_name="CDS 1"
        ),
    ]


@pytest.fixture
def circular_sequence(annotations: list[Annotation]):
    return BioSequence(
        sequence="ATGCGTACGTAGCTAGCGTAGCTAGCTAGCTACGATCGATCG",
        type=SequenceType.DNA,
        is_circular=True,
        name="TestSequence",
        description="Test sequence for unit testing",
        annotations=annotations,
    )


def test_genbank_reading(genbank_content: str):
    bio_seq = BioSequence.from_genbank(genbank_content)
    assert bio_seq.is_circular, "Circularity not parsed correctly"
    assert bio_seq.annotations, "Annotations not parsed correctly"
    annotations = bio_seq.annotations
    assert len(annotations) == 1, "Incorrect number of annotations"

    first_ann = annotations[0]
    assert first_ann.start == 0, "Annotation start position incorrect"
    assert first_ann.end == 36, "Annotation end position incorrect"
    assert not first_ann.name, "Annotation name incorrect"


def test_genbank_writing_and_reading(circular_sequence: BioSequence):
    content = circular_sequence.to_genbank()

    bio_seq = BioSequence.from_genbank(content)
    assert bio_seq.is_circular, "Circularity not preserved after writing/reading"
    assert bio_seq.annotations, "Annotations not parsed correctly"
    annotations = bio_seq.annotations
    assert len(annotations) == 2, "Annotations not preserved after writing/reading"

    first_ann = annotations[0]
    assert first_ann.start == 0, (
        "Annotation start position incorrect after writing/reading"
    )
    assert first_ann.end == 10, (
        "Annotation end position incorrect after writing/reading"
    )


def test_to_from_fasta():
    fasta_str = dedent("""
    >Unnamed
    GTAGTAGTAGTACCCCCCCCCTTTCCT
    >Unnamed
    CCGCCAAGCCGAAAAAAAAGTAGTAGTAGTA



    """)
    seqs = BioSequences.from_fasta(fasta_str)
    assert seqs.sequences[0].sequence == "GTAGTAGTAGTACCCCCCCCCTTTCCT"
    assert seqs.sequences[1].sequence == "CCGCCAAGCCGAAAAAAAAGTAGTAGTAGTA"


def test_to_fast_nonewlines():
    seq = BioSequence(
        sequence="GTAGTAGTAGTACCCCCCCCCTTTCCT",
        type=SequenceType.DNA,
        is_circular=False,
        name="Sequence",
        description="A\nSequce\nwith newlines\n\n",
    )
    fasta_str = seq.to_fasta()
    assert len(fasta_str.split("\n")) == 3, "Output too many newlines"


def test_from_hard_fasta():
    fasta_str = "\u003eAmplicon_1\nGCCCAGTTCCGCCCATTCTCCGCCCCATGGCTGACTAATTTTTTTTATTTATGCAGAGGCCGAGGCCGCCTCGGCCTCTGAGCTATTCCAGAAGTAGTGAGGAGGCTTTTTTGGAGGCCTAGGCTTTTGCAAAGATCGATCAAGAGACAGGATGAGGATCGTTTCGCATGATTGAACAAGATGGATTGCACGCAGGTTCTCCGGCCGCTTGGGTGGAGAGGCTATTCGGCTATGACTGGGCACAACAGACAATCGGCTGCTCTGATGCCGCCGTGTTCCGGCTGTCAGCGCAGGGGCGCCCGGTTCTTTTTGTCAAGACCGACCTGTCCGGTGCCCTGAATGAACTGCAAGACG"
    seq = BioSequence.from_fasta(fasta_str)
    assert seq.name == "Amplicon_1", "Name not parsed correctly"


def test_annotate_enzyme_sites(circular_sequence: BioSequence):
    seq = circular_sequence
    cur_str = str(seq)
    seq.annotate_restriction_sites()
    assert seq.annotations, "No annotations added"
    assert len(seq.annotations) > 0, "No annotations added"
    n = len(seq.annotations)
    seq.annotate_restriction_sites()
    assert len(seq.annotations) == n, "Annotations added on second call"

    assert len(str(seq)) == len(cur_str), "Restriction site annotations leaked"
