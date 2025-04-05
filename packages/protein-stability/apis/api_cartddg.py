"""
This module contains the API for the Rosetta DDG application. The application is used to predict the change in free energy upon mutation of a protein structure. The application can be used to predict the change in free energy
for monomeric protein's wild type sequence and mutated sequence.
"""

import json
import os
import os.path
import subprocess
import uuid
from collections import defaultdict

import modal
from fastapi import Depends, FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from modal import App, Image, asgi_app
from proteincrow.modal.middleware import validate_token

app = App("stability-rosettaddg")
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
        "AUTH_TOKEN": os.environ["MODAL_DEPLOY_TOKEN"],
        "SERVICE_ACCOUNT_JSON": os.environ["SERVICE_ACCOUNT_JSON"],
    })
else:
    local_secret = modal.Secret.from_dict({})

image = (
    Image.debian_slim(python_version="3.9")
    .micromamba()
    .apt_install("wget", "git", "curl")
    .run_commands(
        "wget https://downloads.rosettacommons.org/downloads/academic/3.12/rosetta_bin_linux_3.12_bundle.tgz; tar "
        "-xvzf rosetta_bin_linux_3.12_bundle.tgz"
    )
    .run_commands("git clone https://github.com/ELELAB/RosettaDDGPrediction.git")
    .pip_install(
        "google-cloud-storage",
        "google-auth",
    )
)

rosetta_path = os.path.join(
    os.getcwd(), "rosetta_bin_linux_3.12_bundle", "main/source/bin"
)
path_to_executable = "/rosetta_bin_linux_2020.08.61146_bundle/main/source/bin/cartesian_ddg.static.linuxgccrelease"
path_to_rosetta_scripts = (
    "/rosetta_bin_linux_2020.08.61146_bundle/main/source/bin/rosetta_scripts.static"
    ".linuxgccrelease"
)
# cartDDG monomer protocol params - These should not be changed unless you know what you are doing.
cartddg_ref2015_protocol = (
    "-in:file:s {pdb_file_name} -ddg:mut_file {mut_list_filename} -ddg:iterations 3 "
    "-ddg::score_cutoff 1.0 -ddg:bbnbrs 1 -fa_max_dis 9.0 -score:weights ref2015_cart"
)
with image.imports():
    from google.cloud import storage
    from google.oauth2 import service_account


@app.function(image=image, timeout=60 * 60, concurrency_limit=100)
def compute_ddg_monomer(json_data: dict) -> dict:
    """
    Uses Rosetta to predict the change in free energy upon mutation of a protein structure,
    calculating the stability difference if it is a monomeric protein.
    """

    def parse_ddg_list(lines):
        """Parses the lines to extract scores based on 'WT' and 'MUT' identifiers."""
        scores = defaultdict(list)
        for line in lines:
            if "COMPLEX:" in line:
                parts = line.split()
                try:
                    identifier = parts[2]
                    total_score = float(parts[3])
                    scores[identifier].append(total_score)
                except (IndexError, ValueError) as e:
                    print(
                        f"Warning: Skipping line due to parsing error: {line}. Error: {e}"
                    )
        return scores

    def calculate_average(scores):
        print(scores)
        """Calculates the average of each list of scores for WT and MUT, returning the ddG."""
        wt_scores = [
            sum(scores[key]) / len(scores[key])
            for key in scores
            if key.startswith("WT")
        ]
        mut_scores = [
            sum(scores[key]) / len(scores[key])
            for key in scores
            if key.startswith("MUT")
        ]
        return (
            (sum(mut_scores) / len(mut_scores) - sum(wt_scores) / len(wt_scores))
            if wt_scores and mut_scores
            else 0
        )

    unique_id = str(uuid.uuid4())
    pdb_file_path = f"{unique_id}.pdb"
    mut_list_path = f"{unique_id}.mut_list"

    # Write files to disk. Not using tempfiles due to Subprocess visibility issues
    with open(pdb_file_path, "w") as pdb_file, open(mut_list_path, "w") as mut_file:  # noqa: FURB103
        pdb_file.write(json_data["pdb_string"])
        mut_file.write(json_data["mutation_string"])

    # Run the Rosetta DDG protocol
    print("Running Rosetta DDG protocol")
    cart_ddg_command = f"{path_to_executable} {cartddg_ref2015_protocol.format(pdb_file_name=pdb_file_path, mut_list_filename=mut_list_path)}"
    print(f"Running command: {cart_ddg_command}")
    result = subprocess.run(  # noqa: S602
        cart_ddg_command, capture_output=True, shell=True, check=False
    )
    print("Protocol run complete")
    print("-----")
    print(os.listdir("."))
    try:
        mut_list_path_ddg = f"{mut_list_path.replace('.mut_list', '.ddg')}"
        print("mut_list_path_ddg", mut_list_path_ddg)
        with open(mut_list_path_ddg) as ddg_file:
            ddg = calculate_average(parse_ddg_list(ddg_file.readlines()))

        return {"ddg": ddg, "message": "Protocol completed successfully."}  # noqa: TRY300
    except Exception as e:
        crash_message = "Unknown error occurred."
        if os.path.exists("ROSETTA_CRASH.log"):
            with open("ROSETTA_CRASH.log") as log_file:
                crash_message = "".join(log_file.readlines()[-50:])
                print(crash_message)
        print(f"Error: {e}")
        return {"ddg": 0.0, "message": f"Protocol failed: {crash_message}"}


@app.function(
    image=image,
    secrets=[local_secret, modal.Secret.from_name("gcp-proteincrow")],
    timeout=30000,
)
def get_peptide_derive_peptide(json_data: dict) -> dict:
    gcs_path = json_data["gcs_path"]
    service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info
    )
    client = storage.Client(credentials=credentials)
    bucket = client.get_bucket("proteincrow")
    blob = bucket.blob(gcs_path)

    pep_length = json_data["peptide_length"]
    target_chain_id = json_data["target_chain_id"]
    receptor_chain_id = json_data["receptor_chain_id"]
    xml_string = f"""<ROSETTASCRIPTS>
    <FILTERS>
        <PeptideDeriver
            name="Peptiderive"
            restrict_receptors_to_chains="{target_chain_id}"
            restrict_partners_to_chains="{receptor_chain_id}"
            pep_lengths="{pep_length}"
            dump_peptide_pose="true"
            dump_report_file="true"
            dump_prepared_pose="true"
            dump_cyclic_poses="true"
            skip_zero_isc="true"
            do_minimize="true"
            report_format="markdown"
        />
    </FILTERS>
    <PROTOCOLS>
        <Add filter="Peptiderive" />
    </PROTOCOLS>
</ROSETTASCRIPTS>
"""
    pdb_temp_file = os.path.join(os.getcwd(), str(uuid.uuid4()) + ".pdb")
    xml_temp_file = os.path.join(os.getcwd(), str(uuid.uuid4()) + ".xml")
    xml_temp_file = os.path.join(os.getcwd(), str(uuid.uuid4()) + ".xml")
    blob.download_to_filename(pdb_temp_file)
    pattern = r"receptorE_partnerA_10aa_best_linear_linear_peptide[0-9a-fA-F\-]+"
    with open(xml_temp_file, "w") as f:  # noqa: FURB103
        f.write(xml_string)
    subprocess.run(  # noqa: S602
        f"{path_to_rosetta_scripts} -in:path:database /rosetta_bin_linux_2020.08.61146_bundle/main/database "
        f"-in:file:s {pdb_temp_file} -jd2:delete_old_poses -nstruct 1 -out:chtimestamp -out:no_nstruct_label "
        f"-out:mute core.conformation.Conformation -out:mute protocols.moves.RigidBodyMover "
        f"-run:other_pose_to_scorefile -ex1 -ex2 -randomize_missing_coords -ignore_unrecognized_res "
        f"-override_rsd_type_limit -parser:protocol {xml_temp_file} -scorefile {pdb_temp_file}.score.sc -overwrite",
        capture_output=True,
        shell=True,
        check=False,
    )
    response_json = {}
    os.remove(pdb_temp_file)
    os.remove(xml_temp_file)
    try:
        for filename in os.listdir(os.getcwd()):
            filepath = os.path.join(os.getcwd(), filename)
            if "peptiderive.txt" in filepath:
                with open(filepath) as f:
                    lines = f.readlines()
                peptide = "".join(lines)
                response_json = {
                    "peptide": lines,
                    "message": f"Protocol was run successfully. {peptide}",
                }
            else:
                response_json = {
                    "message": "Protocol failed to run. Peptiderive output file is empty",
                    "peptide": "",
                }
    except Exception:  # If Rosetta fails there will be a crash log
        with open("ROSETTA_CRASH.log") as f:
            lines = f.readlines()[-50:]  # Last fifty lines
        response_json = {
            "message": f"Protocol failed to run. {''.join(lines)}",
            "ddg": "0.0",
        }
    return response_json


@app.function(
    image=image,
    secrets=[local_secret, modal.Secret.from_name("gcp-proteincrow")],
    timeout=30000,
)
def compute_ddg_complex(json_data: dict) -> dict:
    """Uses Rosetta to prediction the change in free energy upon mutation of a protein structure in a complex."""
    gcs_path = json_data["gcs_path"]
    service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info
    )
    client = storage.Client(credentials=credentials)
    bucket = client.get_bucket("proteincrow")
    blob = bucket.blob(gcs_path)

    mut_str = json_data["mutation_string"]
    chains_to_move = json_data["receptor_chain_id"]

    # Write files to disk. Not using tempfiles due to issues with Subprocess seeing them
    pdb_file_name = os.path.join(os.getcwd(), str(uuid.uuid4()) + ".pdb")
    mut_list_filename = os.path.join(os.getcwd(), str(uuid.uuid4()))
    blob.download_to_filename(pdb_file_name)
    with open(mut_list_filename, "w") as g:  # noqa: FURB103
        g.write(mut_str)

    ddgdbfile = os.path.join(os.getcwd(), str(uuid.uuid4()) + "_ddg.db3")
    structdbfile = os.path.join(os.getcwd(), str(uuid.uuid4()) + "_struct.db3")
    path_to_script = (
        "/RosettaDDGPrediction/RosettaDDGPrediction/RosettaScripts/Flex_ddG.xml"
    )
    flex_ddg_args = (
        f"-in:file:s {pdb_file_name} -parser:protocol {path_to_script} "
        f"-parser:script_vars:chainstomove {chains_to_move} -parser:script_vars:resfile {mut_list_filename} "
        f"-parser:script_vars:backrubntrials 35000 -parser:script_vars:backrubtrajstride 7000 "
        f"-parser:script_vars:ddgdbfile {ddgdbfile} -parser:script_vars:structdbfile {structdbfile} "
        f"-out:pdb: True -in:use_database: True"
    )

    subprocess.run(  # noqa: S602
        f"{path_to_rosetta_scripts} {flex_ddg_args}",
        capture_output=True,
        shell=True,
        check=False,
    )
    # If the protocol call was successful, parse the output from the .ddg file

    # Try catch here because rosetta runs can fail for many reasons including timeout if the structure was big.
    try:
        print("file name", f"{mut_list_filename.replace('.mut_list.ddg', '.ddg')}")
        with open(f"{mut_list_filename.replace('.mut_list.ddg', '.ddg')}") as f:
            lines = f.readlines()
        os.remove(f"{mut_list_filename}.ddg")
        os.remove(pdb_file_name)
        os.remove(mut_list_filename)
        os.remove(structdbfile)
        os.remove(ddgdbfile)
        return {"ddg": lines, "message": "Protocol was run successfully."}  # noqa: TRY300
    except Exception:  # If Rosetta fails there will be a crash log
        with open("ROSETTA_CRASH.log") as f:
            lines = f.readlines()[-50:]  # Last fifty lines
        print(lines)
        return {"message": f"Protocol failed to run. {''.join(lines)}", "ddg": 0.0}


@web_app.post("/compute/{protocol}")
async def compute(json_data: dict, protocol: str = ""):
    if protocol == "ddg_monomer":
        blob = await compute_ddg_monomer.remote.aio(json_data)
    elif protocol == "ddg_complex":
        blob = await compute_ddg_complex.remote.aio(json_data)
    elif protocol == "peptidederive":
        blob = await get_peptide_derive_peptide.remote.aio(json_data)
    else:
        blob = {
            "message": "Invalid protocol. Please choose either 'ddg_monomer' or 'ddg_complex'."
        }
    return JSONResponse(content=blob)


@app.function(secrets=[local_secret], timeout=60 * 60)
@asgi_app()
def endpoint() -> FastAPI:
    return web_app
