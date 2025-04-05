import logging
import os

from aiohttp import ClientSession
from proteincrow.constants import ROSETTA_DDG_TIMEOUT
from tenacity import retry, stop_after_attempt, wait_exponential_jitter

from protein_stability.inference_apis import CARTDDG_API
from protein_stability.state import ProteinStabilityState
from protein_stability.tools import apply_mutations_to_sequence

logger = logging.getLogger(__name__)


async def compute_rosetta_ddg(state: ProteinStabilityState) -> float:
    logger.info(f"🚀 Starting Rosetta DDG computation for mutations: {state.mutations}")

    # Handle mutations and prepare sequences
    if not state.mutations:
        logger.info("No mutations provided.")
        return 0.0

    wt_sequence, mut_sequence = apply_mutations_to_sequence(
        state.seq_file, state.mutations
    )

    # Convert mutations to Rosetta's expected 1-indexed format
    mutations = [
        f"{wt_sequence[int(mutation[:-1])]}{int(mutation[:-1]) + 1}{mutation[-1]}"
        for mutation in state.mutations
    ]
    logger.info("wt_sequence")
    logger.info(f"🧬 Converted Mutations: {mutations}")

    # Read the PDB file content
    with open(state.pdb_file) as pdb_file:  # noqa: FURB101
        pdb_string = pdb_file.read()

    # Construct the mutation string for Rosetta
    mutation_string = f"total {len(mutations)}\n{len(mutations)}\n" + "\n".join(
        f"{mutation[0]} {mutation[1:-1]} {mutation[-1]}" for mutation in mutations
    )
    logger.info(f"🧬 Formatted Mutation String:\n{mutation_string}")

    # Prepare and send the request to Rosetta API
    headers = {
        "Authorization": f"Bearer {os.getenv('MODAL_DEPLOY_TOKEN')}",
        "Content-Type": "application/json",
    }
    data = {"pdb_string": pdb_string, "mutation_string": mutation_string}

    async with ClientSession() as session:
        try:
            async with session.post(
                f"{CARTDDG_API}/compute/ddg_monomer",
                json=data,
                headers=headers,
                timeout=ROSETTA_DDG_TIMEOUT,
            ) as resp:
                resp.raise_for_status()
                response = await resp.json()

            ddg_score = response["ddg"]
            # if ddg_score > 0:
            #     # Do we need a warning here? This happens 75% of the time with a baseline agent.
            #     # Commenting out for now (SN)
            #     logger.warning(f"Received a non-negative ddG score: {ddg_score}")
            return ddg_score  # noqa: TRY300, RET504
        except Exception as e:
            logger.exception("Failed to compute ddG")
            return 1.0  # default to a large positive ddG in case of failure


@retry(stop=stop_after_attempt(3), wait=wait_exponential_jitter(initial=10))
async def compute_rosetta_ddg_reward(
    state: ProteinStabilityState, binary: bool
) -> float:
    ddg = await compute_rosetta_ddg(state)
    if binary:
        return 1.0 if ddg < 0 else 0.0
    return max(0.0, -ddg)
