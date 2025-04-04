import os
import os.path
import subprocess
import tempfile

import modal
from fastapi import Depends, FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from modal import App, Image, asgi_app
from proteincrow.modal.middleware import validate_token

app = App("stability-prody")
web_app = FastAPI(dependencies=[Depends(validate_token)])


@web_app.exception_handler(RequestValidationError)
def validation_exception_handler(exc: RequestValidationError):
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(
        content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )


if modal.is_local():
    local_secret = modal.Secret.from_dict({
        "AUTH_TOKEN": os.environ["MODAL_DEPLOY_TOKEN"]
    })
else:
    local_secret = modal.Secret.from_dict({})

image = (
    Image.debian_slim(python_version="3.9")
    .micromamba()
    .apt_install("wget", "git", "curl", "gcc", "g++")
    .run_commands(
        "git clone https://github.com/prody/ProDy.git; cd ProDy; python setup.py install"
    )
    .micromamba_install(
        "openbabel", channels=["conda-forge", "bioconda"]
    )  # this is the version used by the model
    .run_commands("cd /ProDy/prody/proteins/hpbmodule; ls -l")
    .run_commands("cp /ProDy/prody/proteins/hpbmodule/hpb_Python3.9/hpb.so /usr/lib/")
    .run_commands(
        "cp /ProDy/prody/proteins/hpbmodule/hpb_Python3.9/hpb.so /ProDy/prody/proteins/"
    )
    .run_commands("cp /ProDy/prody/proteins/hpbmodule/hpb_Python3.9/hpb.so .")
    .pip_install("pdb-tools")
)

with image.imports():
    from prody import addMissingAtoms, parsePDB
    from prody.proteins.interactions import Interactions


@app.function(image=image)
def compute_ddg(json_data: dict) -> dict:
    interaction_json = {}
    pdb_str = json_data["pdb_string"]

    interactions = Interactions()
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w", suffix=".pdb", dir="."
    ) as temp_file:
        temp_file.write(pdb_str)
        temp_file.flush()
        # addMissingAtoms(temp_file.name, method="openbabel")
        # base_file_name = os.path.basename(temp_file.name)
        # atoms = parsePDB(temp_file.name).select("protein")
        addMissingAtoms(temp_file.name, method="openbabel")
        base_file_name = os.path.basename(temp_file.name)
        processed_file_name = "addH_" + str(base_file_name)
        renumbered_file_name = "renumbered_" + base_file_name
        with open(renumbered_file_name, "w") as renumbered_file:
            subprocess.run(  # noqa: S603
                ["pdb_reres", "-1", processed_file_name],  # noqa: S607
                stdout=renumbered_file,
                check=True,
            )
        # Parse the renumbered PDB file
        atoms = parsePDB(renumbered_file_name).select("protein")
        interactions.calcProteinInteractions(atoms)
        try:
            interaction_json["hydrogen_bonds"] = interactions.getHydrogenBonds()
        except Exception:
            interaction_json["hydrogen_bonds"] = []
        interaction_json["salt_bridges"] = interactions.getSaltBridges()
        interaction_json["repulsive_ionic_bonding"] = (
            interactions.getRepulsiveIonicBonding()
        )
        interaction_json["pi_stacking"] = interactions.getPiStacking()
        interaction_json["pi_cation"] = interactions.getPiCation()
        interaction_json["disulfide_bonds"] = interactions.getDisulfideBonds()
        return interaction_json


@web_app.post("/bonds")
async def get_bonds(json_data: dict):
    blob = await compute_ddg.remote.aio(json_data)
    return JSONResponse(content=blob)


@web_app.get("/")
async def root() -> dict:
    return {"message": "Prody API is running."}


@app.function(secrets=[local_secret])
@asgi_app()
def endpoint() -> FastAPI:
    return web_app
