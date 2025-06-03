import re

import httpx
import pytest
from aviary.core import argref_by_name

from cloning.clients.annotator_client import annotate
from cloning.clients.poly_client import (
    design_primers,
    digest_and_ligate,
    find_orfs,
    gibson,
    goldengate,
    optimize_translation,
    simulate_pcr,
)
from cloning.clients.search_client import search_genes, search_plasmids
from cloning.poly_local import enzyme_cut, find_sequence_overlap
from cloning.sequence_models import BioSequence, BioSequences, SequenceType


@pytest.mark.flaky(reruns=3, only_on=[httpx.ReadTimeout])
@pytest.mark.asyncio
async def test_annotate_sacb():
    # this is sacB and the annotation should label it as sacb
    fasta = ">sacB\ngtggcgagcccgatcttccccatcggtgatgtcggcgatataggcgccagcaaccgcacctgtggcgccggtgatgccggccacgatgcgtccggcgtagaggatcgagatctcgatcccgcgaaattaatacgactcactataggggaattgtgagcggataacaattcccctctagaaataattttgtttaactttaagaaggagatataccatgcatcatcatcatcatcacagcagcggcagagaaaacttgtatttccagggccatatgagttctcctcctgaaagatccataacttcgtatagcatacattatacgaagttatgcggccgcgacgtccacatatacctgccgttcactattatttagtgaaatgagatattatgatattttctgaattgtgattaaaaaggcaactttatgcccatgcaacagaaactataaaaaatacagagaatgaaaagaaacagatagattttttagttctttaggcccgtagtctgcaaatccttttatgattttctatcaaacaaaagaggaaaatagaccagttgcaatccaaacgagagtctaatagaatgaggtcgaaaagtaaatcgcgcgggtttgttactgataaagcaggcaagacctaaaatgtgtaaagggcaaagtgtatactttggcgtcaccccttacatattttaggtctttttttattgtgcgtaactaacttgccatcttcaaacaggagggctggaagaagcagaccgctaacacagtacataaaaaaggagacatgaacgatgaacatcaaaaagtttgcaaaacaagcaacagtattaacctttactaccgcactgctggcaggaggcgcaactcaagcgtttgcgaaagaaacgaaccaaaagccatataaggaaacatacggcatttcccatattacacgccatgatatgctgcaaatccctgaacagcaaaaaaatgaaaaatatcaagttcctgagttcgattcgtccacaattaaaaatatctcttctgcaaaaggcctggacgtttgggacagctggccattacaaaacgctgacggcactgtcgcaaactatcacggctaccacatcgtctttgcattagccggagatcctaaaaatgcggatgacacatcgatttacatgttctatcaaaaagtcggcgaaacttctattgacagctggaaaaacgctggccgcgtctttaaagacagcgacaaattcgatgcaaatgattctatcctaaaagaccaaacacaagaatggtcaggttcagccacatttacatctgacggaaaaatccgtttattctacactgatttctccggtaaacattacggcaaacaaacactgacaactgcacaagttaacgtatcagcatcagacagctctttgaacatcaacggtgtagaggattataaatcaatctttgacggtgacggaaaaacgtatcaaaatgtacagcagttcatcgatgaaggcaactacagctcaggcgacaaccatacgctgagagatcctcactacgtagaagataaaggccacaaatacttagtatttgaagcaaacactggaactgaagatggctaccaaggcgaagaatctttatttaacaaagcatactatggcaaaagcacatcattcttccgtcaagaaagtcaaaaacttctgcaaagcgataaaaaacgcacggctgagttagcaaacggcgctctcggtatgattgagctaaacgatgattacacactgaaaaaagtgatgaaaccgctgattgcatctaacacagtaacagatgaaattgaacgcgcgaacgtctttaaaatgaacggcaaatggtacctgttcactgactcccgcggatcaaaaatgacgattgacggcattacgtctaacgatatttacatgcttggttatgtttctaattctttaactggcccatacaagccgctgaacaaaactggccttgtgttaaaaatggatcttgatcctaacgatgtaacctttacttactcacacttcgctgtacctcaagcgaaaggaaacaatgtcgtgattacaagctatatgacaaacagaggattctacgcagacaaacaatcaacgtttgcgcctagcttcctgctgaacatcaaaggcaagaaaacatctgttgtcaaagacagcatccttgaacaaggacaattaacagttaacaaataaaaacgcaaaagaaaatgccgatatcctattggcattgacg"
    result = await annotate(BioSequence.from_fasta(fasta))
    assert len(result.annotations) > 1
    # at least one of them should have SacB
    assert any("sacB" in feature.name for feature in result.annotations)


@pytest.mark.flaky(reruns=3, only_on=[httpx.ReadTimeout])
@pytest.mark.asyncio
async def test_annotate_puc19_ori():
    """This should find an ORI that crosses the 0 mark on circular plasmid."""
    fasta = ">puc19 (circular)\nGAGATACCTACAGCGTGAGCTATGAGAAAGCGCCACGCTTCCCGAAGGGAGAAAGGCGGACAGGTATCCGGTAAGCGGCAGGGTCGGAACAGGAGAGCGCACGAGGGAGCTTCCAGGGGGAAACGCCTGGTATCTTTATAGTCCTGTCGGGTTTCGCCACCTCTGACTTGAGCGTCGATTTTTGTGATGCTCGTCAGGGGGGCGGAGCCTATGGAAAAACGCCAGCAACGCGGCCTTTTTACGGTTCCTGGCCTTTTGCTGGCCTTTTGCTCACATGTTCTTTCCTGCGTTATCCCCTGATTCTGTGGATAACCGTATTACCGCCTTTGAGTGAGCTGATACCGCTCGCCGCAGCCGAACGACCGAGCGCAGCGAGTCAGTGAGCGAGGAAGCGGAAGAGCGCCCAATACGCAAACCGCCTCTCCCCGCGCGTTGGCCGATTCATTAATGCAGCTGGCACGACAGGTTTCCCGACTGGAAAGCGGGCAGTGAGCGCAACGCAATTAATGTGAGTTAGCTCACTCATTAGGCACCCCAGGCTTTACACTTTATGCTTCCGGCTCGTATGTTGTGTGGAATTGTGAGCGGATAACAATTTCACACAGGAAACAGCTATGACCATGATTACGCCAAGCTTGCATGCCTGCAGGTCGACTCTAGAGGATCCCCGGGTACCGAGCTCGAATTCACTGGCCGTCGTTTTACAACGTCGTGACTGGGAAAACCCTGGCGTTACCCAACTTAATCGCCTTGCAGCACATCCCCCTTTCGCCAGCTGGCGTAATAGCGAAGAGGCCCGCACCGATCGCCCTTCCCAACAGTTGCGCAGCCTGAATGGCGAATGGCGCCTGATGCGGTATTTTCTCCTTACGCATCTGTGCGGTATTTCACACCGCATATGGTGCACTCTCAGTACAATCTGCTCTGATGCCGCATAGTTAAGCCAGCCCCGACACCCGCCAACACCCGCTGACGCGCCCTGACGGGCTTGTCTGCTCCCGGCATCCGCTTACAGACAAGCTGTGACCGTCTCCGGGAGCTGCATGTGTCAGAGGTTTTCACCGTCATCACCGAAACGCGCGAGACGAAAGGGCCTCGTGATACGCCTATTTTTATAGGTTAATGTCATGATAATAATGGTTTCTTAGACGTCAGGTGGCACTTTTCGGGGAAATGTGCGCGGAACCCCTATTTGTTTATTTTTCTAAATACATTCAAATATGTATCCGCTCATGAGACAATAACCCTGATAAATGCTTCAATAATATTGAAAAAGGAAGAGTATGAGTATTCAACATTTCCGTGTCGCCCTTATTCCCTTTTTTGCGGCATTTTGCCTTCCTGTTTTTGCTCACCCAGAAACGCTGGTGAAAGTAAAAGATGCTGAAGATCAGTTGGGTGCACGAGTGGGTTACATCGAACTGGATCTCAACAGCGGTAAGATCCTTGAGAGTTTTCGCCCCGAAGAACGTTTTCCAATGATGAGCACTTTTAAAGTTCTGCTATGTGGCGCGGTATTATCCCGTATTGACGCCGGGCAAGAGCAACTCGGTCGCCGCATACACTATTCTCAGAATGACTTGGTTGAGTACTCACCAGTCACAGAAAAGCATCTTACGGATGGCATGACAGTAAGAGAATTATGCAGTGCTGCCATAACCATGAGTGATAACACTGCGGCCAACTTACTTCTGACAACGATCGGAGGACCGAAGGAGCTAACCGCTTTTTTGCACAACATGGGGGATCATGTAACTCGCCTTGATCGTTGGGAACCGGAGCTGAATGAAGCCATACCAAACGACGAGCGTGACACCACGATGCCTGTAGCAATGGCAACAACGTTGCGCAAACTATTAACTGGCGAACTACTTACTCTAGCTTCCCGGCAACAATTAATAGACTGGATGGAGGCGGATAAAGTTGCAGGACCACTTCTGCGCTCGGCCCTTCCGGCTGGCTGGTTTATTGCTGATAAATCTGGAGCCGGTGAGCGTGGGTCTCGCGGTATCATTGCAGCACTGGGGCCAGATGGTAAGCCCTCCCGTATCGTAGTTATCTACACGACGGGGAGTCAGGCAACTATGGATGAACGAAATAGACAGATCGCTGAGATAGGTGCCTCACTGATTAAGCATTGGTAACTGTCAGACCAAGTTTACTCATATATACTTTAGATTGATTTAAAACTTCATTTTTAATTTAAAAGGATCTAGGTGAAGATCCTTTTTGATAATCTCATGACCAAAATCCCTTAACGTGAGTTTTCGTTCCACTGAGCGTCAGACCCCGTAGAAAAGATCAAAGGATCTTCTTGAGATCCTTTTTTTCTGCGCGTAATCTGCTGCTTGCAAACAAAAAAACCACCGCTACCAGCGGTGGTTTGTTTGCCGGATCAAGAGCTACCAACTCTTTTTCCGAAGGTAACTGGCTTCAGCAGAGCGCAGATACCAAATACTGTTCTTCTAGTGTAGCCGTAGTTAGGCCACCACTTCAAGAACTCTGTAGCACCGCCTACATACCTCGCTCTGCTAATCCTGTTACCAGTGGCTGCTGCCAGTGGCGATAAGTCGTGTCTTACCGGGTTGGACTCAAGACGATAGTTACCGGATAAGGCGCAGCGGTCGGGCTGAACGGGGGGTTCGTGCACACAGCCCAGCTTGGAGCGAACGACCTACACCGAACT"
    result = await annotate(BioSequence.from_fasta(fasta))
    print(result)
    assert len(result.annotations) > 1
    # at least one of them should have SacB
    assert any("ori" in feature.name for feature in result.annotations)


@pytest.mark.flaky(reruns=3, only_on=[httpx.ReadTimeout])
@pytest.mark.asyncio
async def test_search_plasmids():
    result = await search_plasmids("origin of replication")
    assert len(result) > 5

    # we have been getting impossible types in them
    s = result[0].to_sequence()
    assert s.type == SequenceType.DNA
    assert "K" not in s.sequence

    result = await search_plasmids("pUC cloning vector")
    # make sure it goes to specific plasmid, not just
    # search over all plasmids
    assert result[0].docid == "addgene-plasmid-50005-sequence-222046", (
        "Did not find correct plasmid."
    )


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=3, only_on=[httpx.ReadTimeout])
async def test_find_orfs():
    # find_orfs uses defaults amenable for SeqQA, but we used different settings when
    # designing these tests
    find_orfs_kws = {"codon_table": 1, "strand": 0}

    # A random sequence which I then inserted into a reference ORF finder
    # https://www.ncbi.nlm.nih.gov/orffinder/ other inputs: `"ATG" and alternative initiation codons`,
    # Must end with TGA, TAA, TAG to be compatible with https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi?chapter=tgencodes#SG1
    # Note - there are actually 28 orfs, only 9 display (??) but you can get full list on track listing
    seq1 = BioSequence(
        sequence="ATGCCATAGCATTTTTATCCATAAGATTAGCATTTTTATCTCATTGGATGGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATGCATTGACATATATAAACATATATACTTAAATATAAATATATCCATGCTAATCTCATACTAAATCTCATACTAAATCATGTCATGTCATCTCATACATGTCATGTCATCTCATACATGTCATGTCATCTCATACATGTCATGTCATCTCATACATCTCATACA",
        type=SequenceType.DNA,
    )
    result = await find_orfs(
        seq1,
        min_length=30,
        **find_orfs_kws,
    )
    print(result)
    assert (
        len(result.sequences) == 28
    )  # after removing ORFs with end codons [ATA,TAC,CAT,ACA] from `orffinder`
    assert result.sequences[0].type == SequenceType.DNA

    # make sure min_length is respected
    result = await find_orfs(
        seq1,
        min_length=1000,
        **find_orfs_kws,
    )
    assert len(result.sequences) == 0

    # check a reference from the ORF finder
    # https://www.ncbi.nlm.nih.gov/orffinder/
    seq2 = BioSequence(
        sequence="CAGAGCTCAGATGGTTTCGCCATGAGCACGTCCGGTACTCTGATGCCGTGGCTTCGCACTTAGACCAGGCGACTATCCCACGTCTCTTCATTTCCGTTTGGCTCTCGTGGAACGTACGCATCTTTGGGTTACTCCCCG",
        type=SequenceType.DNA,
    )
    result = await find_orfs(
        seq2,
        min_length=10,
        **find_orfs_kws,
    )
    # output from https://www.ncbi.nlm.nih.gov/orffinder/ results without CCC and TCT stop codons
    assert result.sequences[0].sequence == "ATGAGCACGTCCGGTACTCTGATGCCGTGGCTTCGCACTTAG"

    # remove codons with

    # search ORFs only on the reverse strand
    result = await find_orfs(seq1, min_length=30, strand=-1)
    signs = {
        match.group(1)
        for orf in result.sequences
        if (match := re.search(r"strand=([+-])", orf.description))
    }
    assert signs == {"-"}, "All ORFs should be on the reverse strand"

    # search ORFs with "ATG" as the start codon
    result = await find_orfs(seq1, min_length=30, codon_table=0)
    start_codons = {orf.sequence[:3] for orf in result.sequences}
    assert start_codons == {"ATG"}, "Must only contain ATG start codons"


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=3, only_on=[httpx.ReadTimeout])
async def test_synthesize():
    # I made this sequence bad for optimization due to repeats (and high GC content)
    seq = BioSequence(
        sequence="AAAAAAAAAAAADDAAAAALAAAAAEAAAAAAAAAARAADDAAAARNAALREAREAYFKAASAQAAALRAQDPAAGQAHARAAAQALADASNAAAAADEAALAA",
        type=SequenceType.PROTEIN,
    )
    result = await optimize_translation(seq)
    # since this alg is stochastic, we just check that it's a reasonable length
    assert len(result.sequence) > 100


@pytest.mark.parametrize(
    ("sequence", "enzyme", "circular", "expected_fragments"),
    [
        (
            "AAATTTTATCGGTCTCGAATTCGAGA",
            "BsaI",
            False,
            [
                "AAATTTTATCGGTCTCG",
                "AATTCGAGA",
            ],
        ),
        (
            "AAATTTTATCGGTCTCGAATTCGAGA",
            "EcoRI",
            False,
            [
                "AAATTTTATCGGTCTCG",
                "AATTCGAGA",
            ],
        ),
    ],
)
def test_enzyme_cut(
    sequence: str, enzyme: str, circular: bool, expected_fragments: list[str]
) -> None:
    # build a biosequence from info
    seq = BioSequence(sequence=sequence, type=SequenceType.DNA, is_circular=circular)
    result = enzyme_cut(seq, enzyme)
    # Expect the fragments to be ordered left to right
    assert [fragment.sequence for fragment in result.sequences] == expected_fragments


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=3, only_on=[httpx.ReadTimeout])
async def test_goldengate():
    seqs = BioSequences(
        sequences=[
            BioSequence(
                sequence="AAATTTTATCGGTCTCGAATTCGAGA",
                type=SequenceType.DNA,
                is_circular=False,
            ),
            BioSequence(
                sequence="AATATCGGTCTCGAATTCGAGA",
                type=SequenceType.DNA,
                is_circular=True,
            ),
        ]
    )

    result = await goldengate(seqs, "BsaI")

    assert len(result.sequences) > 0, "No sequences returned"


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=3, only_on=[httpx.ReadTimeout])
async def test_digest():
    seqs = BioSequences(
        sequences=[
            BioSequence(
                sequence="AAATTTTATCGGTCTCGAATTCGAGA",
                type=SequenceType.DNA,
                is_circular=False,
            ),
            BioSequence(
                sequence="AATATCGGTCTCGAATTCGAGA",
                type=SequenceType.DNA,
                is_circular=True,
            ),
        ]
    )

    result = await digest_and_ligate(seqs, "SexAI,BsaI")
    assert len(result.sequences) > 0, "No sequences returned"


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=3, only_on=[httpx.ReadTimeout])
async def test_design_primers():
    # this was run through poly directly to get the expected results
    seq = "aataattacaccgagataacacatcatggataaaccgatactcaaagattctatgaagctatttgaggcacttggtacgatcaagtcgcgctcaatgtttggtggcttcggacttttcgctgatgaaacgatgtttgcactggttgtgaatgatcaacttcacatacgagcagaccagcaaacttcatctaacttcgagaagcaagggctaaaaccgtacgtttataaaaagcgtggttttccagtcgttactaagtactacgcgatttccgacgacttgtgggaatccagtgaacgcttgatagaagtagcgaagaagtcgttagaacaagccaatttggaaaaaaagcaacaggcaagtagtaagcccgacaggttgaaagacctgcctaacttacgactagcgactgaacgaatgcttaagaaagctggtataaaatcagttgaacaacttgaagagaaaggtgcattgaatgcttacaaagcgatacgtgactctcactccgcaaaagtaagtattgagctactctgggctttagaaggagcgataaacggcacgcactggagcgtcgttcctcaatctcgcagagaagagctggaaaatgcgctttcttaa"
    s = BioSequence(sequence=seq, type=SequenceType.DNA, is_circular=False)
    result = await design_primers(s, target_tm=55.0)
    assert result[0].sequence == "AATAATTACACCGAGATAACACATCATGG"
    assert result[1].sequence == "TTAAGAAAGCGCATTTTCCAGC"
    assert result[2].sequence == seq.upper()

    # now try with an enzyme specified
    result = await design_primers(s, target_tm=55.0, forward_overhang="BsaJI")
    # recognition sequence is CCNNGG
    # N = any base
    assert re.search("CC[ATCG][ATCG]GG", result[0].sequence), (  # noqa: RUF039
        "Forward overhang missing recognition sequence for BsaJI"
    )


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=3, only_on=[httpx.ReadTimeout])
async def test_simulate_pcr():
    # from poly unit test
    seq = "aataattacaccgagataacacatcatggataaaccgatactcaaagattctatgaagctatttgaggcacttggtacgatcaagtcgcgctcaatgtttggtggcttcggacttttcgctgatgaaacgatgtttgcactggttgtgaatgatcaacttcacatacgagcagaccagcaaacttcatctaacttcgagaagcaagggctaaaaccgtacgtttataaaaagcgtggttttccagtcgttactaagtactacgcgatttccgacgacttgtgggaatccagtgaacgcttgatagaagtagcgaagaagtcgttagaacaagccaatttggaaaaaaagcaacaggcaagtagtaagcccgacaggttgaaagacctgcctaacttacgactagcgactgaacgaatgcttaagaaagctggtataaaatcagttgaacaacttgaagagaaaggtgcattgaatgcttacaaagcgatacgtgactctcactccgcaaaagtaagtattgagctactctgggctttagaaggagcgataaacggcacgcactggagcgtcgttcctcaatctcgcagagaagagctggaaaatgcgctttcttaa"
    s = BioSequence(sequence=seq, type=SequenceType.DNA, is_circular=False)
    primerf = "TTATAGGTCTCATACTAATAATTACACCGAGATAACACATCATGG"
    primerr = "TATATGGTCTCTTCATTTAAGAAAGCGCATTTTCCAGC"
    result = await simulate_pcr(s, forward_primer=primerf, reverse_primer=primerr)
    assert (
        result.sequence
        == "TTATAGGTCTCATACTAATAATTACACCGAGATAACACATCATGGATAAACCGATACTCAAAGATTCTATGAAGCTATTTGAGGCACTTGGTACGATCAAGTCGCGCTCAATGTTTGGTGGCTTCGGACTTTTCGCTGATGAAACGATGTTTGCACTGGTTGTGAATGATCAACTTCACATACGAGCAGACCAGCAAACTTCATCTAACTTCGAGAAGCAAGGGCTAAAACCGTACGTTTATAAAAAGCGTGGTTTTCCAGTCGTTACTAAGTACTACGCGATTTCCGACGACTTGTGGGAATCCAGTGAACGCTTGATAGAAGTAGCGAAGAAGTCGTTAGAACAAGCCAATTTGGAAAAAAAGCAACAGGCAAGTAGTAAGCCCGACAGGTTGAAAGACCTGCCTAACTTACGACTAGCGACTGAACGAATGCTTAAGAAAGCTGGTATAAAATCAGTTGAACAACTTGAAGAGAAAGGTGCATTGAATGCTTACAAAGCGATACGTGACTCTCACTCCGCAAAAGTAAGTATTGAGCTACTCTGGGCTTTAGAAGGAGCGATAAACGGCACGCACTGGAGCGTCGTTCCTCAATCTCGCAGAGAAGAGCTGGAAAATGCGCTTTCTTAAATGAAGAGACCATATA"
    )


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=3, only_on=[httpx.ReadTimeout])
async def test_seqqa_simulate_pcr():
    # from seqqa question
    seq = "CTAAAGGAACGGGGGGTCTAAGCAATGTACTCTATCGTCGACCCTTTTACATGCTCTGTGCGGATATTAAGATTGTTGATAGAGCCAATAGCTGGGAACCTCGTGGTTTGCAACTCGCGCTTGCTCCTCATTAGTCAAATTGAAGCAAGCTAATGGTCTGGCATATAAACATGGTGAAACGATCATGCACAGTTGTCATATCCCCGAGTCCGGCAAGGTGGATGTCTCCCTTCCTAATACGATAATTTGGACCCGCCATCAGTGTAAGTCATTGGCCTGGGTTTTCGTGAGCTGTTGAGTGCGTGGTACCGGACAGCCAGTGAATCCAGCGATAGCTTAAAGTCCCACCGCTGGGGCGGCCCGTCTTCAATCGTGCTCTCAGACGGCATTCCGTTGCCAAAGGTCTGGTTCTATATTATCGGTAACTTTGTAACGACGCCTTTGCGGCTGACTCGATCATATAAAGAATCCAGATCATCGCGTGTGGACCGAGCAAAAGGGTGACTCCGGCAGATACCTGTTCCTCGAAGACCGGCGTAAGGTTATTTATAAATTCTTTTTGCATGTAAGATAAGGAATATCTGAAGGCCTTCAGTACCCTTAATGAAACCCCAATAGCGATCGGGGCGACTCCCTCATATCTGGAATTTTTTGTCTCAACCACAACGACGACATTGGTATCCGGTGCTTCGCACTTTCTATGCTGCAGACCGGTGGTATTCAGTTACGGCGCGTGGTATTGCCGCTCGGGCGCCCAGTGTGTATTGCTGTGAGAGCAGTGCATGTCTCAGAAAAACCTGTGTCAGTAAGAACTCATCAGACCACTGAACAGCCAGTTTATCAGACTAATACGCGTGGTTACCGGCATGCTTTCTCCAGAGAGATCAGTCCATAAGACCTATTGTATGCAGACAATGAAATAGGTATAGCATGATGTATTTTTGAGAGATCTATGATCGGCCCTGGGGACGTTGCACTAGCGCACCATACGGCAATGGGG"
    s = BioSequence(sequence=seq, type=SequenceType.DNA, is_circular=False)
    primerf = "TTATCGGTAACTTTGTAACG"
    primerr = "GTGCGAAGCACCGGATACCAATG"
    result = await simulate_pcr(s, forward_primer=primerf, reverse_primer=primerr)
    assert len(result.sequence) == 280


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=3, only_on=[httpx.ReadTimeout])
async def test_argrefed_pcr():
    # from seqqa question
    class MyState:
        def __init__(self):
            self.refs = {}

    seq = "CTAAAGGAACGGGGGGTCTAAGCAATGTACTCTATCGTCGACCCTTTTACATGCTCTGTGCGGATATTAAGATTGTTGATAGAGCCAATAGCTGGGAACCTCGTGGTTTGCAACTCGCGCTTGCTCCTCATTAGTCAAATTGAAGCAAGCTAATGGTCTGGCATATAAACATGGTGAAACGATCATGCACAGTTGTCATATCCCCGAGTCCGGCAAGGTGGATGTCTCCCTTCCTAATACGATAATTTGGACCCGCCATCAGTGTAAGTCATTGGCCTGGGTTTTCGTGAGCTGTTGAGTGCGTGGTACCGGACAGCCAGTGAATCCAGCGATAGCTTAAAGTCCCACCGCTGGGGCGGCCCGTCTTCAATCGTGCTCTCAGACGGCATTCCGTTGCCAAAGGTCTGGTTCTATATTATCGGTAACTTTGTAACGACGCCTTTGCGGCTGACTCGATCATATAAAGAATCCAGATCATCGCGTGTGGACCGAGCAAAAGGGTGACTCCGGCAGATACCTGTTCCTCGAAGACCGGCGTAAGGTTATTTATAAATTCTTTTTGCATGTAAGATAAGGAATATCTGAAGGCCTTCAGTACCCTTAATGAAACCCCAATAGCGATCGGGGCGACTCCCTCATATCTGGAATTTTTTGTCTCAACCACAACGACGACATTGGTATCCGGTGCTTCGCACTTTCTATGCTGCAGACCGGTGGTATTCAGTTACGGCGCGTGGTATTGCCGCTCGGGCGCCCAGTGTGTATTGCTGTGAGAGCAGTGCATGTCTCAGAAAAACCTGTGTCAGTAAGAACTCATCAGACCACTGAACAGCCAGTTTATCAGACTAATACGCGTGGTTACCGGCATGCTTTCTCCAGAGAGATCAGTCCATAAGACCTATTGTATGCAGACAATGAAATAGGTATAGCATGATGTATTTTTGAGAGATCTATGATCGGCCCTGGGGACGTTGCACTAGCGCACCATACGGCAATGGGG"
    s = BioSequence(sequence=seq, type=SequenceType.DNA, is_circular=False)
    primerf = BioSequence(
        sequence="TTATCGGTAACTTTGTAACG", type=SequenceType.DNA, is_circular=False
    )
    primerr = BioSequence(
        sequence="GTGCGAAGCACCGGATACCAATG", type=SequenceType.DNA, is_circular=False
    )
    state = MyState()
    state.refs["primerf"] = primerf
    state.refs["primerr"] = primerr
    state.refs["s"] = s
    fxn = argref_by_name(return_direct=True)(simulate_pcr)
    result = await fxn(
        "s", forward_primer="primerf", reverse_primer="primerr", state=state
    )
    assert len(result.sequence) == 280

    # make sure it throws an error for invalid ref
    with pytest.raises(KeyError):
        await fxn("s1", "primerf", "primerr", state=state)


@pytest.mark.asyncio
@pytest.mark.flaky(reruns=3, only_on=[httpx.ReadTimeout])
async def test_gibson():
    # I ran these through benchling. They are the shortest possible to get something
    # interesting
    seqs = BioSequences(
        sequences=[
            BioSequence(
                sequence="GTAGTAGTAGTACCCCCCCCCTTTCCT",
                type=SequenceType.DNA,
                is_circular=False,
            ),
            BioSequence(
                sequence="CCGCCAAGCCGAAAAAAAAGTAGTAGTAGTA",
                type=SequenceType.DNA,
                is_circular=False,
            ),
        ]
    )
    result = await gibson(seqs)
    assert len(result.sequences) == 1
    assert (
        result.sequences[0].sequence == "CCGCCAAGCCGAAAAAAAAGTAGTAGTAGTACCCCCCCCCTTTCCT"
    )

    # try swapping order
    seqs = BioSequences(sequences=[seqs.sequences[1], seqs.sequences[0]])
    result2 = await gibson(seqs)
    assert result2.sequences[0].sequence == result.sequences[0].sequence


def test_search_genes():
    query = 'napF[gene] AND "Escherichia coli"[Organism]'
    results = search_genes(query).sequences
    # Ideally we check for 2, but that might change if genes are added to the DB
    assert 0 < len(results) <= 5


@pytest.mark.parametrize(
    ("seq", "primer", "reverse", "expected"),
    [
        # Exact match at start of seq
        ("ATCGGCTA", "ATCG", False, (0, 4)),
        # Exact match in middle of seq
        ("ATCGGCTA", "TCGG", False, (1, 5)),
        # 2 bp overhang off start
        ("ATCGGCTA", "GCAT", False, (0, 2)),
        # Exact match at end of seq
        ("ATCGGCTA", "TAGC", True, (4, 8)),
        # Exact match in middle of seq
        ("ATCGGCTA", "AGCC", True, (3, 7)),
        # 2 bp overhang off end
        ("ATCGGCTA", "GCTA", True, (6, 8)),
    ],
)
def test_find_seq_overlap(
    seq: str, primer: str, reverse: bool, expected: tuple[int, int] | None
):
    result = find_sequence_overlap(
        BioSequence(sequence=seq, type=SequenceType.DNA), primer, reverse
    )
    assert result == expected
