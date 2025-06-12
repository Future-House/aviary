from pydantic import BaseModel


# created this to prevent fasta from being
# URL param
class FastaRequest(BaseModel):
    fasta: str


class SearchResult(BaseModel):
    title: str
    body: str
    docid: str
    featureid: int
    sequence: str
    score: float = 0.0
    genbank: str | None = None


def convert_fasta(text: str, default_header: str = "Seq") -> tuple[str, bool]:
    """Reads a FASTA or sequence string and a FASTA file as string.

    Returns:
        tuple: A tuple containing the FASTA string and a boolean indicating if the sequence is circular.
    """
    fasta_str = text
    if not text.startswith(">"):
        fasta_str = f">{default_header}\n{text}"
    # it must have a newline at the end (after sequence newline)
    if not fasta_str.endswith("\n\n"):
        fasta_str += "\n"
    return fasta_str, "(circular)" in fasta_str
