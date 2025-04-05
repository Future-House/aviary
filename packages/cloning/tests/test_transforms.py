import pytest
from cloning.sequence_models import BioSequence
from cloning.transforms import (
    complement,
    complement_base,
    reverse_complement,
    view_translation,
)


# These are just random sequences I made
@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ("ATCG", "CGAT"),
        ("AAAAACCCCCGGGGGTTTTT", "AAAAACCCCCGGGGGTTTTT"),
        ("ATCGatcg", "cgatCGAT"),
        ("", ""),
        ("ATCG ATCG", "CGAT CGAT"),
    ],
)
def test_reverse_complement(sequence, expected):
    assert reverse_complement(sequence) == expected


# These are just random sequences I made
@pytest.mark.parametrize(
    ("sequence", "expected"),
    [
        ("ATCG", "TAGC"),
        ("AAAAACCCCCGGGGGTTTTT", "TTTTTGGGGGCCCCCAAAAA"),
        ("ATCGatcg", "TAGCtagc"),
        ("", ""),
        ("ATCG ATCG", "TAGC TAGC"),
    ],
)
def test_complement(sequence, expected):
    assert complement(sequence) == expected


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("A", "T"),
        ("T", "A"),
        ("C", "G"),
        ("G", "C"),
        ("a", "t"),
        ("t", "a"),
        ("c", "g"),
        ("g", "c"),
        ("N", "N"),
        ("X", " "),
        (" ", " "),
    ],
)
def test_complement_base(base, expected):
    assert complement_base(base) == expected


@pytest.mark.parametrize("func", [reverse_complement, complement])
def test_empty_string(func):
    assert not func("")


@pytest.mark.parametrize("func", [reverse_complement, complement])
def test_mixed_case(func):
    seq = "ATCGatcg"
    result = func(seq)
    assert result.lower() == func(seq.lower()).lower()


def test_complement_base_invalid():
    assert all(complement_base(c) == " " for c in "Xx!@#$%^&*()")


def test_view_translation():
    # Just a known sequence that translates
    s = BioSequence(sequence="GTGGCCATTGTAA", type="dna")
    assert "VAIV" in view_translation(s).replace(" ", "")
