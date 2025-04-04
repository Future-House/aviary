import contextlib
import logging
import os
from collections.abc import Callable
from io import StringIO
from urllib.error import HTTPError

import httpx
from Bio import Entrez, SeqIO
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from cloning.sequence_models import (
    API_URL,
    DEFAULT_TIMEOUT,
    BioSequence,
    BioSequences,
    SearchResult,
    SequenceType,
)

logger = logging.getLogger(__name__)


async def search_plasmids(
    query: str,
) -> list[SearchResult]:
    """
    Searches from a database of plasmids and returns best matching features or plasmids.

    Args:
        query: Keyword query search. Can negate terms with a minus sign. Supports AND/OR.

    Returns:
        A list of matching plasmids or features.
    """
    auth_token = os.environ["MODAL_DEPLOY_TOKEN"]

    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{API_URL}/search",
            headers={
                "Authorization": f"Bearer {auth_token}",
            },
            params={
                "query": query,
            },
            timeout=DEFAULT_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()
    return [SearchResult(**result) for result in data]


@retry(
    stop=stop_after_attempt(5),
    # jitter so that concurrent retries are less likely to collide
    wait=wait_exponential_jitter(initial=2, max=60, jitter=4),
    retry=retry_if_exception_type(HTTPError),
)
def query_entrez(endpoint: Callable, entrez_read: bool = True, **kwargs):
    with contextlib.closing(endpoint(**kwargs)) as handle:
        return Entrez.read(handle) if entrez_read else handle.read()


def search_genes(query: str) -> BioSequences:
    """Searches the NCBI nucleotide database for genes using the Entrez API and returns up to 5 results.

    Args:
        query: Search query.

    Returns:
        Up to 5 matching genes as a BioSequences object.
    """
    # TODO: rewrite this using httpx.AsyncClient and bypass biopython.
    Entrez.api_key = os.environ.get("NCBI_API_KEY", None)
    # we will handle retries with tenacity, for jittering & exp backoff
    Entrez.max_tries = 1

    # Search the gene database with the query
    search_results = query_entrez(
        Entrez.esearch, db="gene", term=query, retmax=5, idtype="acc"
    )
    ids = search_results["IdList"]

    sequences: list[BioSequence] = []
    for gene_id in ids:
        try:
            summary = query_entrez(Entrez.esummary, db="gene", id=gene_id)
            gene_summary = summary["DocumentSummarySet"]["DocumentSummary"][0]

            gene_name = gene_summary.get("Name", "")
            description = gene_summary.get("Description", "")
            organism = gene_summary.get("Organism", {}).get("ScientificName", "")

            gene_info = gene_summary["GenomicInfo"][0]
            start, stop = int(gene_info["ChrStart"]), int(gene_info["ChrStop"])
            if start < stop:
                strand = "1"
            else:
                strand = "2"
                start, stop = stop + 1, start + 1
            sequence_fasta = query_entrez(
                Entrez.efetch,
                db="nucleotide",
                id=gene_info["ChrAccVer"],
                rettype="fasta",
                strand=strand,
                seq_start=str(start),
                seq_stop=str(stop),
                entrez_read=False,
            )
            nucleotide_sequence = str(SeqIO.read(StringIO(sequence_fasta), "fasta").seq)

            sequences.append(
                BioSequence(
                    sequence=nucleotide_sequence,
                    name=gene_name,
                    description=f"({organism}) {description}",
                    type=SequenceType.DNA,
                )
            )

        except Exception as e:
            logger.exception(f"An error occurred while processing ID {gene_id}: {e!r}.")

    return BioSequences(sequences=sequences)
