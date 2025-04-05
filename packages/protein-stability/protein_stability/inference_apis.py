import os

PRODY_API = os.environ.get(
    "PROTEIN_STABILITY_PRODY_API",
    "https://future-house--stability-prody-endpoint.modal.run",
)
CARTDDG_API = os.environ.get(
    "PROTEIN_STABILITY_CARTDDG_API",
    "https://future-house--stability-rosettaddg-endpoint.modal.run",
)
ROSETTA_API = os.environ.get(
    "PROTEIN_STABILITY_ROSETTA_API",
    "https://future-house--stability-rosetta-endpoint.modal.run",
)
ESM_API = os.environ.get(
    "PROTEIN_STABILITY_ESM_API",
    "https://future-house--stability-esm-endpoint.modal.run",
)
