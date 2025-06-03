import os

import httpx

from ..sequence_models import API_URL, DEFAULT_TIMEOUT, BioSequence


async def annotate(
    sequence: BioSequence,
) -> BioSequence:
    """
    Annotates the sequence, showing major features (e.g., proteins, ORIs, etc.).

    Args:
        sequence: The FASTA sequence to annotate.

    Returns:
        A new version of the sequence with annotations.
    """
    # Validate fasta here
    auth_token = os.environ["MODAL_DEPLOY_TOKEN"]

    # before going let's just see if it's already annotated
    # and they're not just restriction sites
    if sequence.annotations and any(
        a.type != "restriction site" for a in sequence.annotations
    ):
        return sequence

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{API_URL}/annotate",
            headers={
                "Authorization": f"Bearer {auth_token}",
            },
            json={
                "fasta": sequence.to_fasta(),
            },
            timeout=DEFAULT_TIMEOUT * 3,  # annotate takes longer
        )
        response.raise_for_status()
        return BioSequence.from_genbank(response.text)
