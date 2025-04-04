import json
import subprocess
from enum import StrEnum, auto
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from modal import App, Image
from models import FastaRequest, convert_fasta

app = App("poly-lib")
router = APIRouter()

image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("git", "gcc", "wget")
    .run_commands(
        "wget https://go.dev/dl/go1.22.5.linux-amd64.tar.gz && tar -C /usr/local -xzf go1.22.5.linux-amd64.tar.gz && rm go1.22.5.linux-amd64.tar.gz"
    )
    .env({
        "PATH": "/usr/local/go/bin:$PATH",
        "GOPATH": "/root/go",
        "GOROOT": "/usr/local/go",
    })
    .copy_local_dir("poly_lib", "/root/poly_lib")
    .workdir("/root/poly_lib")
    .run_commands(
        "wget \
        -O rebase.withref \
        https://storage.googleapis.com/fh-modal-artifacts/rebase.ref"
    )
    .run_commands(
        "go mod init poly-lib && \
        go mod edit -replace=github.com/bebop/poly=github.com/whitead/poly@f92359b10f3e57a9712f5fe5b2ccd0d78154fe76 && \
        go get github.com/bebop/poly/synthesis/codon && \
        go get github.com/bebop/poly/transform && \
        go get github.com/bebop/poly/synthesis/fix && \
        go get github.com/bebop/poly/synthesis/codon && \
        go get github.com/bebop/poly/checks && \
        go get github.com/bebop/poly/clone && \
        go get github.com/bebop/poly/primers/pcr && \
        go get github.com/bebop/poly/io/fasta && \
        go build -o orf orf.go && \
        go build -o clone clone.go && \
        go build -o primers primers.go && \
        go build -o gibson gibson.go && \
        go build -o synthesize synthesize.go"
    )
)


class SequenceType(StrEnum):
    DNA = auto()
    PROTEIN = auto()


@app.function(image=image, allow_concurrent_inputs=10, cpu=0.5)
def find_orfs(
    fasta: str,
    min_length: int,
    codon_table: int,
    strand: int,
    include_unterminated: bool,
    allow_nested: bool,
) -> list[dict]:
    fasta_path = Path("seq.fasta")
    fasta_path.write_text(fasta)
    args = [
        "/root/poly_lib/orf",
        "-fasta",
        str(fasta_path),
        "-min-length",
        str(min_length),
        "-codon-table",
        str(codon_table),
        "-strand",
        str(strand),
        "-include-unterminated" if include_unterminated else "",
        "-allow-nested" if allow_nested else "",
    ]
    args = [arg for arg in args if arg]
    result = subprocess.run(  # noqa: S603
        args,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


@router.post("/find-orfs")
def find_orfs_route(
    request: FastaRequest,
    min_length: int = 30,
    codon_table: int = 1,
    strand: int = 1,
    include_unterminated: bool = False,
    allow_nested: bool = True,
) -> list[dict]:
    """
    Identify Open Reading Frames (ORFs) in a DNA sequence.

    Scans the input DNA sequence for ORFs on both forward and reverse strands across all reading frames.
    Translates identified ORFs into amino acid sequences using the specified codon table and outputs
    the results in JSON format.

    ### Implementation Notes:
    - Only ORFs that begin with a valid start codon and end with a stop codon are considered.
    - Unterminated ORFs (those that start with a start codon but do not have a stop codon) are
      included if they meet the minimum length requirement and include_unterminated is True.
    - Both start and stop codons are required for defining the boundaries of an ORF.
    - The program examines all three possible reading frames on each strand.

    ### Args:
    - `request`: A request with a fasta sequence or plain sequence. Should be an object with one key "fasta" whose value is a sequence string or fasta string.
    - `min_length`: The minimum length of ORFs to report, in base pairs.
    - `codon_table`: The codon table index for translation (1 for standard). If 0, will use the standard codon table with only ATG start codon.
    - `strand`: Which strand to search for ORFs. 0 for both, 1 for forward, -1 for reverse.
    - `include_unterminated`: Returns unterminated ORFs. True to get unterminated ORFs
    - `allow_nested`: True to allow nested ORFs.

    ### Returns:
    A JSON object containing the input sequence, parameters, and a list of identified ORFs with
    their positions, strand, frame, nucleotide sequence, and translated amino acids.
    """
    seq, _ = convert_fasta(request.fasta)
    return find_orfs.remote(
        seq, min_length, codon_table, strand, include_unterminated, allow_nested
    )


@app.function(image=image, allow_concurrent_inputs=10, cpu=0.5)
def synthesize(
    sequence: str,
    seq_type: SequenceType,
    cg_content: int,
    codon_table: int,
    min_repeat_length: int,
) -> dict:
    if cg_content < 10 or cg_content > 90:
        raise ValueError("cg_content must an integer between 10 and 90")
    result = subprocess.run(  # noqa: S603
        [
            "/root/poly_lib/synthesize",
            "-sequence",
            sequence,
            "-type",
            seq_type.value.lower(),
            "-max-gc",
            str(cg_content),
            "-codon-table",
            str(codon_table),
            "-repeat-length",
            str(min_repeat_length),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


@router.post("/synthesize")
def synthesize_route(
    request: FastaRequest,
    seq_type: SequenceType,
    cg_content: int = 62,
    codon_table: int = 11,
    min_repeat_length: int = 15,
) -> dict:
    """
    Optimize a DNA sequence for synthesis and ordering.

    This function optimizes a DNA sequence by performing codon substitutions to optimize CG content and homopolymer repeats.

    ### Args:
    - `request`: A request with a fasta sequence or plain sequence. Should be an object with one key 'fasta' whose value is a sequence string or fasta string.
    - `seq_type`: The type of sequence to optimize (DNA or protein).
    - `cg_content`: The target CG content as a percentage.
    - `codon_table`: The codon table index for translation (1 for standard). Uses NCBI genetic code table indices.
    - `min_repeat_length`: The minimum length of homopolymer repeats to avoid (in DNA bases).

    ### Returns:
    A JSON object with the optimized sequence, the original sequence, and notes about the optimization.
    """
    fasta_str, _ = convert_fasta(request.fasta)
    sequence = "".join(fasta_str.splitlines()[1:])
    try:
        return synthesize.remote(
            sequence, seq_type, cg_content, codon_table, min_repeat_length
        )
    except Exception as e:
        # we do exceptions here because it can fail to optimize sequence
        # which is not a bug, and we want to give that detail
        if "returned non-zero exit status 1" in str(e):
            raise HTTPException(status_code=422, detail=str(e)) from e
        raise


@app.function(image=image, allow_concurrent_inputs=10, cpu=0.5)
def digest_and_ligate(fasta_str: str, enzymes: str, ligate: bool) -> str:
    fasta_path = Path("seq.fasta")
    fasta_path.write_text(fasta_str)
    result = subprocess.run(  # noqa: S603
        [
            "/root/poly_lib/clone",
            "-rebase",
            "rebase.withref",
            "-fasta",
            fasta_path,
            "-enzymes",
            enzymes,
            "-ligate" if ligate else "",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


@router.post("/cut", response_class=PlainTextResponse)
def enzyme_cut_route(request: FastaRequest, enzyme: str) -> str:
    """
    Digest a DNA sequence with a restriction enzyme.

    This endpoint simulates the digestion of a DNA sequence using a specified restriction enzyme.
    It cuts the input sequence at recognition sites of the enzyme and returns the resulting fragments
    in FASTA format.

    ### Args:
    - `request`: A request containing a DNA sequence in FASTA format or as a plain sequence. Should be
      an object with one key `"fasta"` whose value is a sequence string or FASTA-formatted string.
    - `enzyme`: The name of the restriction enzyme to use for cutting. The enzyme must be present in
      the REBASE database used by the system.

    ### Returns:
    A FASTA-formatted string containing the fragments resulting from the digestion, each labeled with
    a fragment identifier.

    ### Implementation Notes:
    - The enzyme recognition sites and cut positions are obtained from the REBASE database.
    - The input sequence can be linear or circular.
    - Only one enzyme can be specified per request; to digest with multiple enzymes, use the `/digest`.
    - The fragments are output in the order they appear in the input sequence after digestion.
    """
    fasta_str, _ = convert_fasta(request.fasta)
    return digest_and_ligate.remote(fasta_str, enzyme, ligate=False)


@router.post("/digest", response_class=PlainTextResponse)
def digest_route(request: FastaRequest, enzymes: str) -> str:
    """
    Digest and ligate DNA sequences with restriction enzymes.

    This endpoint simulates the digestion of one or more DNA sequences using specified restriction enzymes, followed by ligation of the resulting fragments. It is useful for modeling cloning experiments where DNA is cut and reassembled using enzymes.

    ### Args:
    - `request`: A request containing DNA sequences in FASTA format or as plain sequences. Should be an
      object with one key `"fasta"` whose value is a sequence string or FASTA-formatted string containing
      one or more sequences.
    - `enzymes`: A comma-separated string of enzyme names to use for cutting. The enzymes must be present
      in the REBASE database used by the system.

    ### Returns:
    A FASTA-formatted string containing the assembled constructs resulting from the digestion and ligation,
    each labeled with a construct identifier.

    ### Implementation Notes:
    - The enzyme recognition sites and cut positions are obtained from the REBASE database.
    - Multiple enzymes can be specified; the sequences will be digested by all specified enzymes in the order provided.
    - The input sequences can be linear or circular.
    - After digestion, the fragments are ligated to form new constructs, simulating the circular ligation process.
    - The ligation step attempts to join compatible fragment ends; fragments with incompatible ends will not be ligated.
    - Infinite loops (constructs that cannot be formed due to incompatible ends) are ignored.
    - The output constructs are unique; duplicate constructs are not included in the output.
    - The constructs are output in the order they are formed after ligation.

    """
    # we need to do this, in case missing newline at the end
    fasta_str, _ = convert_fasta(request.fasta)
    return digest_and_ligate.remote(fasta_str, enzymes, ligate=True)


@app.function(image=image, allow_concurrent_inputs=10, cpu=0.5)
def design_primers(
    sequence: str,
    forward_overhang: str,
    reverse_overhang: str,
    target_tm: float,
    forward_primer: str,
    reverse_primer: str,
    circular: bool,
) -> dict:
    result = subprocess.run(  # noqa: S603
        [
            "/root/poly_lib/primers",
            "-sequence",
            sequence,
            "-forward-overhang",
            forward_overhang,
            "-reverse-overhang",
            reverse_overhang,
            "-target-tm",
            str(target_tm),
            "-forward-primer",
            forward_primer,
            "-reverse-primer",
            reverse_primer,
            "-circular" if circular else "",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return (
            {"error": result.stderr, "out": result.stdout}
            if (result.stderr or result.stdout)
            else {}
        )


@router.post("/primers")
def design_primers_route(
    request: FastaRequest,
    forward_overhang: str = "",
    reverse_overhang: str = "",
    target_tm: float = 60.0,
) -> dict:
    """
    Design primers for a given DNA sequence.

    This function designs primers given the passed in temperature and optional overhangs.

    ### Args:
    - `request`: A request with a fasta sequence or plain sequence. Should be an object with one key "fasta" whose value is a sequence string or fasta string.
    - `forward_overhang`: Optional overhang for the forward primer.
    - `reverse_overhang`: Optional overhang for the reverse primer.
    - `target_tm`: The target melting temperature for the primers.

    ### Returns:
    A JSON object with the forward and reverse primer sequences and amplicon as FASTA strings.
    """
    fasta_str, circular = convert_fasta(request.fasta)
    sequence = "".join(fasta_str.splitlines()[1:])
    return design_primers.remote(
        sequence, forward_overhang, reverse_overhang, target_tm, "", "", circular
    )


@router.post("/pcr")
def simulate_pcr_route(
    request: FastaRequest,
    forward_primer: str = "",
    reverse_primer: str = "",
    target_tm: float = 45.0,
) -> dict:
    """
    Simulate a PCR reaction with a forward and reverse primer.

    This function assesses melting temperature using SantaLucia. The default temperature approximates
    ~55C using salts and a more modern polymerase.

    ### Assumptions:
    - SantaLucia model
    - 500 nM (nanomolar) primer concentration
    - 50 mM (millimolar) sodium concentration
    - 0 mM (millimolar) magnesium concentration

    ### Args:
    - `request`: A request with a fasta sequence or plain sequence. Should be an object with one key "fasta" whose value is a sequence string or fasta string.
    - `forward_primer`: The forward primer sequence.
    - `reverse_primer`: The reverse primer sequence.
    - `target_tm`: The target melting temperature for the primers.

    ### Returns:
    A JSON object with the forward and reverse primer sequences and amplicon as FASTA strings.
    """
    fasta_str, circular = convert_fasta(request.fasta)
    sequence = "".join(fasta_str.splitlines()[1:])
    return design_primers.remote(
        sequence, "", "", target_tm, forward_primer, reverse_primer, circular
    )


@app.function(image=image, allow_concurrent_inputs=10, cpu=0.5)
def gibson_assemble(fasta_str: str) -> str:
    fasta_path = Path("seq.fasta")
    fasta_path.write_text(fasta_str)
    result = subprocess.run(  # noqa: S603
        ["/root/poly_lib/gibson", "-fasta", fasta_path],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


@router.post("/gibson", response_class=PlainTextResponse)
def gibson_assemble_route(request: FastaRequest) -> str:
    """
    Simulate Gibson Assembly of DNA fragments.

    This endpoint simulates the Gibson Assembly method for joining multiple DNA fragments seamlessly.
    It reads DNA sequences from a FASTA-formatted input and assembles them into longer sequences by
    identifying optimal overlaps between fragments (including reverse complements). The output includes
    both linear and circular assemblies in FASTA format.

    ### Args:
    - `request`: A request containing a FASTA-formatted string of DNA sequences to assemble. Should be an
      object with one key `"fasta"` whose value is a valid FASTA string containing one or more sequences.

    ### Returns:
    A FASTA-formatted string containing the assembled sequences, with descriptions indicating the fragments
    used in each assembly and whether the assembly is linear or circular.

    ### Implementation Notes:
    - The input sequences should be linear fragments; circular sequences are not supported.
    - Overlaps are identified based on maximum sequence overlap, including reverse complements.
    - Assemblies are extended recursively until no further overlaps are found.
    - The method detects both linear assemblies (when no circularization occurs) and circular assemblies
      (when the sequence can be circularized based on overlapping ends).
    - The output sequences are unique; duplicate assemblies are not included in the output.
    """
    fasta_str, _ = convert_fasta(request.fasta)
    return gibson_assemble.remote(fasta_str)
