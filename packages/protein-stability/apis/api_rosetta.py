import hashlib
import json
import os
import tempfile
import textwrap
from typing import Any
from urllib.request import Request

import modal
from fastapi import Depends, FastAPI, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from modal import App, Dict, Image, asgi_app
from proteincrow.modal.middleware import validate_token

persisted_dict = Dict.from_name("rosetta_cache", create_if_missing=True)

app = App("stability-rosetta")
web_app = FastAPI(dependencies=[Depends(validate_token)])


@web_app.exception_handler(RequestValidationError)
def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(request)
    exc_str = f"{exc}".replace("\n", " ").replace("   ", " ")
    print(f"{request}: {exc_str}")
    content = {"status_code": 10422, "message": exc_str, "data": None}
    return JSONResponse(
        content=content, status_code=status.HTTP_422_UNPROCESSABLE_ENTITY
    )


image = (
    Image.debian_slim(python_version="3.9")
    .apt_install(
        "mc",
        "git",
        "bash",
        "zlib1g-dev",
        "build-essential",
        "cmake",
        "ninja-build",
        "clang",
        "clang-tools",
        "clangd",
        "curl",
    )
    .run_commands("git clone https://github.com/RosettaCommons/rosetta.git")
    .run_commands("cd rosetta/source; ./scons.py -j8 mode=release bin")
    .pip_install("pyrosetta-installer")
    .run_commands(
        "python -c 'import pyrosetta_installer; pyrosetta_installer.install_pyrosetta()'"
    )
    .run_commands("git clone https://github.com/nrbennet/dl_binder_design.git")
    .pip_install("google-cloud-storage", "google-auth")
    .apt_install("wget")
    .run_commands(
        "wget https://files.ipd.uw.edu/pub/robust_de_novo_design_minibinders_2021/supplemental_files/scripts_and_main_pdbs.tar.gz; "
        "tar -xvf /scripts_and_main_pdbs.tar.gz"
    )
)

with image.imports():
    import pyrosetta
    from google.cloud import storage
    from google.oauth2 import service_account
    from pyrosetta import pose_from_pdb
    from pyrosetta.rosetta import core
    from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover

    DSSP = pyrosetta.rosetta.protocols.moves.DsspMover
    # TODO: One of the protocols uses these weights - per_sap_score. In the future we might want to not init
    # this way and instead do it in the call but for now since this is the only function we have, it's fine
    # it takes too long to init otherwise
    # these weights are used by the paper so we are using the same rosetta weights.
    pyrosetta.init(options="-corrections:beta_nov16 -renumber_pdb")
    one_letter_amino_acid_alphabet = list("ARNDCQEGHILKMFPSTWYV-")
    three_letter_amino_acid_alphabet = [
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
        "GAP",
    ]
    # python 3.8 doesn't have the strict flag, so we have to use noqa
    aa_1_3 = dict(zip(one_letter_amino_acid_alphabet, three_letter_amino_acid_alphabet))  # noqa: B905

    def hash_pdb(input_string):
        sha256_hash = hashlib.sha256()
        sha256_hash.update(input_string.encode("utf-8"))
        return sha256_hash.hexdigest()


if modal.is_local():
    local_secret = modal.Secret.from_dict({
        "AUTH_TOKEN": os.environ["MODAL_DEPLOY_TOKEN"]
    })
else:
    local_secret = modal.Secret.from_dict({})


@app.function(image=image)
def compute_sap_score(json_data: dict) -> dict:
    """
    Uses Rosetta to predict the per residue SAP score for a pdb.

    Args:
        json_data: dict with the following
            pdb_string: str, pdb structure to predict the per residue SAP score for
            chain_id: str, chain id of the pdb

    Returns:
        dict with the following
            success: bool, if the function ran successfully
            data: dict with the following
                pose_scores: dict, per residue SAP scores
                sequence: str, sequence of the pdb
                pose2pdb: dict, mapping of pose index to pdb index
    """
    pdb_str = json_data["pdb_string"]
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".pdb") as sample_input_pdb:
        sample_input_pdb.write(pdb_str)
        sample_input_pdb.flush()
        pdb_hash = hash_pdb(pdb_str)

        # if pdb_hash not in persisted_dict:
        pose = pyrosetta.pose_from_file(sample_input_pdb.name)
        xml_parser = pyrosetta.rosetta.protocols.rosetta_scripts.RosettaScriptsParser()
        # this path comes with the git clone repo above.
        protocol = xml_parser.generate_mover(
            "/supplemental_files/cao_2021_protocol/per_res_sap.xml"
        )
        protocol.apply(pose)
        pose_scores = {}
        pose_sequence = pose.sequence()
        pose2pdb = {}
        for key in pose.scores:
            if "my_per_res_sap" in key:
                # Returns a json with a key for each residue and its score
                res_idx = int(key.replace("my_per_res_sap_", ""))
                res_type = pose_sequence[res_idx - 1]
                pose_scores[f"{res_type}{res_idx}"] = pose.scores.get(key)
                pose2pdb[res_idx] = pose.pdb_info().pose2pdb(res_idx)

        persisted_dict = {
            "pose_scores": pose_scores,
            "sequence": pose.sequence(),
            "pose2pdb": pose2pdb,
        }
    return {"data": persisted_dict}


@app.function(image=image)
def thread_sequence(json_data: dict) -> dict:
    """
    Uses Rosetta to thread a sequence onto a pdb.

    Args:
        json_data: dict with the following
            pdb_string: str, pdb structure to thread
            binder_sequence: str, sequence to thread onto the pdb
    Returns:
        dict with the following
            success: bool, if the function ran successfully
            data: dict with the following
                threaded_pdb: str, threaded pdb string
    """
    pdb_str = json_data["pdb_string"]
    binder_seq = json_data["binder_sequence"]

    input_pdb_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".pdb")  # noqa: SIM115
    input_pdb_file.write(pdb_str)
    input_pdb_file.flush()

    pose = pyrosetta.pose_from_file(input_pdb_file.name)
    rsd_set = pose.residue_type_set_for_pose(core.chemical.FULL_ATOM_t)
    for res_i, mut_to in enumerate(binder_seq):
        res_name = aa_1_3[mut_to]
        new_res = core.conformation.ResidueFactory.create_residue(
            rsd_set.name_map(res_name)
        )
        bool_true_flag = True
        pose.replace_residue(res_i + 1, new_res, bool_true_flag)
    out_file_path = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".pdb")  # noqa: SIM115
    pose.dump_pdb(out_file_path.name)
    out_file_path.flush()
    with open(out_file_path.name) as f:  # noqa: FURB101
        pdb_str = f.read()

    os.remove(out_file_path.name)
    os.remove(input_pdb_file.name)
    return {"data": {"threaded_pdb": pdb_str}}


@app.function(image=image, secrets=[modal.Secret.from_name("gcp-proteincrow")])
def thread_and_relax_one_seq(
    gcs_path_to_pdb: str, proposed_seq: str, seq_idx: int
) -> str:
    """
    Uses Rosetta to thread a sequence onto a pdb and then relaxes the structure.

    Args:
        gcs_path_to_pdb: path to pdb on gcs
        proposed_seq: proposed sequence to thread onto the pdb
        seq_idx: index of the sequence in the list of sequences sampled
    Returns:
        path to the relaxed pdb folder on gcs
    """
    gcs_path_to_pdb = gcs_path_to_pdb.replace("gs://", "")
    tempfile_path = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".pdb")  # noqa: SIM115
    # AUTH
    service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info
    )
    client = storage.Client(credentials=credentials)
    bucket = client.get_bucket("proteincrow")

    blob = bucket.blob(gcs_path_to_pdb)
    blob.download_to_filename(tempfile_path.name)
    tempfile_path.flush()

    pdb_folder = "/".join(gcs_path_to_pdb.split("/")[:-1])
    pdb_name_path = "".join(gcs_path_to_pdb.split("/")[-1])
    pose = pyrosetta.pose_from_file(tempfile_path.name)

    rsd_set = pose.residue_type_set_for_pose(core.chemical.FULL_ATOM_t)
    for res_i, mut_to in enumerate(proposed_seq):
        res_name = aa_1_3[mut_to]
        new_res = core.conformation.ResidueFactory.create_residue(
            rsd_set.name_map(res_name)
        )
        bool_true_flag = True
        pose.replace_residue(res_i + 1, new_res, bool_true_flag)
    xml_parser = pyrosetta.rosetta.protocols.rosetta_scripts.RosettaScriptsParser()
    # This xml path comes from the git clone in the image setup above.
    protocol = xml_parser.generate_mover(
        "/dl_binder_design/mpnn_fr/RosettaFastRelaxUtil.xml"
    )
    protocol.apply(pose)
    output_file_path = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="w+", delete=False, suffix=".pdb"
    )
    pose.dump_pdb(output_file_path.name)
    gcs_post_relax_path = f"{pdb_folder}/relaxed/{pdb_name_path.replace('.pdb', '')}_relaxed_{seq_idx}.pdb"
    bucket.blob(gcs_post_relax_path).upload_from_filename(output_file_path.name)
    return f"{pdb_folder}/relaxed/"


@app.function(image=image, secrets=[modal.Secret.from_name("gcp-proteincrow")])
def thread_and_relax_opticrow(
    gcs_path_to_pdb: str, proposed_seq: str, upload_to: str
) -> str:
    """
    Uses Rosetta to thread a sequence onto a pdb and then relaxes the structure.

    Args:
        gcs_path_to_pdb: path to pdb on gcs
        proposed_seq: proposed sequence to thread onto the pdb
        upload_to: upload path on gcs
    Returns:
        path to the relaxed pdb folder on gcs
    """
    print("gcs_path_to_pdb", gcs_path_to_pdb)
    print("proposed_seq", proposed_seq)
    print("upload_to", upload_to)

    gcs_path_to_pdb = gcs_path_to_pdb.replace("gs://", "")
    # get path in form of gcs://bucket_name/proteincrow/UUID/pre_relax_mpnn_response.json
    tempfile_path = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".pdb")  # noqa: SIM115
    # AUTH
    service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info
    )
    client = storage.Client(credentials=credentials)
    bucket = client.get_bucket("proteincrow")

    blob = bucket.blob(gcs_path_to_pdb)
    blob.download_to_filename(tempfile_path.name)
    tempfile_path.flush()

    pdb_folder = "/".join(gcs_path_to_pdb.split("/")[:-1])
    pdb_name_path = "".join(gcs_path_to_pdb.split("/")[-1])
    pose = pyrosetta.pose_from_file(tempfile_path.name)

    rsd_set = pose.residue_type_set_for_pose(core.chemical.FULL_ATOM_t)
    for res_i, mut_to in enumerate(proposed_seq):
        res_name = aa_1_3[mut_to]
        new_res = core.conformation.ResidueFactory.create_residue(
            rsd_set.name_map(res_name)
        )
        bool_true_flag = True
        pose.replace_residue(res_i + 1, new_res, bool_true_flag)
    xml_parser = pyrosetta.rosetta.protocols.rosetta_scripts.RosettaScriptsParser()
    # This xml path comes from the git clone in the image setup above.
    protocol = xml_parser.generate_mover(
        "/dl_binder_design/mpnn_fr/RosettaFastRelaxUtil.xml"
    )
    protocol.apply(pose)
    output_file_path = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="w+", delete=False, suffix=".pdb"
    )
    pose.dump_pdb(output_file_path.name)
    gcs_post_relax_path = f"{pdb_folder}/relaxed/{pdb_name_path.replace('.pdb', '')}_relaxed_{upload_to}.pdb"
    bucket.blob(gcs_post_relax_path).upload_from_filename(output_file_path.name)
    return gcs_post_relax_path


@app.function(image=image)
def fast_relax(json_data: dict) -> dict:
    """
    Uses Rosetta to relax a pdb structure.

    Args:
        json_data: dict with the following
            pdb_string: str, pdb structure to relax
    Returns:
        dict with the following
            success: bool, if the function ran successfully
            data: dict with the following
                relaxed_pdb: str, relaxed pdb string
    """
    pdb_str = json_data["pdb_string"]
    input_pdb_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".pdb")  # noqa: SIM115
    input_pdb_file.write(pdb_str)
    input_pdb_file.flush()

    pose = pyrosetta.pose_from_file(input_pdb_file.name)
    xml_parser = pyrosetta.rosetta.protocols.rosetta_scripts.RosettaScriptsParser()
    # This xml path comes from the git clone in the image setup above.
    protocol = xml_parser.generate_mover(
        "/dl_binder_design/mpnn_fr/RosettaFastRelaxUtil.xml"
    )
    protocol.apply(pose)
    output_file_path = tempfile.NamedTemporaryFile(  # noqa: SIM115
        mode="w+", delete=False, suffix=".pdb"
    )
    pose.dump_pdb(output_file_path.name)
    with open(output_file_path.name) as f:  # noqa: FURB101
        pdb_str = f.read()
    os.remove(output_file_path.name)
    os.remove(input_pdb_file.name)
    return {
        "data": {"relaxed_pdb": pdb_str},
    }


@app.function(image=image, secrets=[modal.Secret.from_name("gcp-proteincrow")])
def thread_and_relax(json_data: dict):
    arguments_list: list[tuple] = []
    for gcs_path, proposed_seq in json_data.items():
        arguments_list.append((gcs_path, proposed_seq, len(arguments_list)))
    response = list(thread_and_relax_one_seq.starmap(arguments_list))
    if len(response) > 0:
        return response[0]
    return response


@app.function(image=image, secrets=[modal.Secret.from_name("gcp-proteincrow")])
def interface_analyze(gcs_path: str, sequence_idx: int):
    print(f"Analyzing the interface of the structure at {sequence_idx}- {gcs_path}")

    def interface_data_to_json(interface_data):
        """
        This function takes a PyRosetta InterfaceData object and returns a JSON string
        with relevant properties.
        """
        # Extract the properties from the InterfaceData object
        data_dict = {
            "aromatic_dG_fraction": interface_data.aromatic_dG_fraction,
            "aromatic_dSASA_fraction": interface_data.aromatic_dSASA_fraction,
            "aromatic_nres": interface_data.aromatic_nres,
            "centroid_dG": interface_data.centroid_dG,
            "complex_total_energy": interface_data.complex_total_energy,
            "complexed_interface_score": interface_data.complexed_interface_score,
            "complexed_SASA": interface_data.complexed_SASA,
            "crossterm_interface_energy": interface_data.crossterm_interface_energy,
            "crossterm_interface_energy_dSASA_ratio": interface_data.crossterm_interface_energy_dSASA_ratio,
            "delta_unsat_hbonds": interface_data.delta_unsat_hbonds,
            "dG": interface_data.dG,
            "dG_dSASA_ratio": interface_data.dG_dSASA_ratio,
            "dhSASA": interface_data.dhSASA,
            "dhSASA_rel_by_charge": interface_data.dhSASA_rel_by_charge,
            "dhSASA_sc": interface_data.dhSASA_sc,
            "dSASA": interface_data.dSASA,
            "dSASA_sc": interface_data.dSASA_sc,
            "gly_dG": interface_data.gly_dG,
            "hbond_E_fraction": interface_data.hbond_E_fraction,
            "interface_nres": interface_data.interface_nres,
            "interface_residues": interface_data.interface_residues,
            "interface_to_surface_fraction": interface_data.interface_to_surface_fraction,
            "packstat": interface_data.packstat,
            "pymol_sel_hbond_unsat": interface_data.pymol_sel_hbond_unsat,
            "pymol_sel_interface": interface_data.pymol_sel_interface,
            "pymol_sel_packing": interface_data.pymol_sel_packing,
            "sc_value": interface_data.sc_value,
            "separated_interface_score": interface_data.separated_interface_score,
            "separated_SASA": interface_data.separated_SASA,
            "separated_total_energy": interface_data.separated_total_energy,
            "ss_helix_nres": interface_data.ss_helix_nres,
            "ss_loop_nres": interface_data.ss_loop_nres,
            "ss_sheet_nres": interface_data.ss_sheet_nres,
            "total_hb_E": interface_data.total_hb_E,
        }

        # Mapping and calculations for the requested metrics
        return {
            "dSASA_int": data_dict["dSASA"],
            "dG_separated": data_dict["dG"],
            "dG_separated/dSASAx100": data_dict["dG_dSASA_ratio"] * 100,
            "delta_unsatHbonds": data_dict["delta_unsat_hbonds"],
            "packstat": data_dict["packstat"],
            "dG_cross": data_dict["crossterm_interface_energy"],
            "dG_cross/dSASAx100": data_dict["crossterm_interface_energy_dSASA_ratio"]
            * 100,
            "cen_dG": data_dict["centroid_dG"],
            "nres_int": data_dict["interface_nres"],
            "side1_score": None,  # Not provided in your data
            "side2_score": None,  # Not provided in your data
            "side1_normalized": None,  # Not provided in your data
            "side2_normalized": None,  # Not provided in your data
            "hbonds_int": data_dict["total_hb_E"],
            "hbond_E_fraction": data_dict["hbond_E_fraction"],
        }

    def analyze_interface(pdb_file, chain_A, chain_B):
        """
        Analyze the interface between two chains in a PDB file using PyRosetta's InterfaceAnalyzerMover.

        Args:
            pdb_file (str): Path to the PDB file.
            chain_A (str): Chain identifier for the first chain.
            chain_B (str): Chain identifier for the second chain.

        Returns:
            dict: A dictionary with interface analysis data including SASA, BSA, etc.
        """
        # Load the PDB file into a Pose object
        pose = pose_from_pdb(pdb_file)

        # Define the chains for interface analysis (e.g., "A_B" for chains A and B)
        interface = f"{chain_A}_{chain_B}"

        # Set up the InterfaceAnalyzerMover
        interface_analyzer = InterfaceAnalyzerMover(interface)

        # Apply the analyzer to the pose
        interface_analyzer.apply(pose)
        return interface_analyzer.get_all_data()

        # Gather results from the analysis

    service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info
    )
    client = storage.Client(credentials=credentials)
    bucket = client.get_bucket("proteincrow")
    blob = bucket.blob(gcs_path)
    input_pdb_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".pdb")  # noqa: SIM115
    blob.download_to_filename(input_pdb_file.name)
    input_pdb_file.flush()
    interface_data = analyze_interface(input_pdb_file.name, "A", "B")
    return interface_data_to_json(interface_data)


@app.function(
    image=image,
    timeout=30000,
    secrets=[local_secret, modal.Secret.from_name("gcp-proteincrow")],
)
def analyze_all_interface(json_data):
    gcs_paths = json_data["gcs_paths"]
    arguments_list = []
    for idx, gcs_path in enumerate(gcs_paths):
        arguments_list.append((gcs_path, idx))

    return list(interface_analyze.starmap(arguments_list))


@app.function(image=image, secrets=[modal.Secret.from_name("gcp-proteincrow")])
def get_dssp_of_a_structure(gcs_path: str):
    service_account_info = json.loads(os.environ["SERVICE_ACCOUNT_JSON"])
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info
    )
    client = storage.Client(credentials=credentials)
    bucket = client.get_bucket("proteincrow")
    blob = bucket.blob(gcs_path)
    input_pdb_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".pdb")  # noqa: SIM115
    blob.download_to_filename(input_pdb_file.name)
    input_pdb_file.flush()
    pose = pose_from_pdb(input_pdb_file.name)
    chain_pose = pose.split_by_chain(pose.chain("A"))
    return DSSP.apply(chain_pose)


@app.function(image=image, secrets=[modal.Secret.from_name("gcp-proteincrow")])
def get_dssp_of_a_structure_with_pdb_path(
    pdb_string: str, chain_id: str = "A"
) -> dict[str, Any]:
    print(
        f"Finding DSSP secondary structure of chain {chain_id} in the given PDB file."
    )
    input_pdb_file = tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".pdb")  # noqa: SIM115
    input_pdb_file.write(pdb_string)
    input_pdb_file.flush()
    pose = pose_from_pdb(input_pdb_file.name)
    pyrosetta.rosetta.protocols.moves.DsspMover().apply(pose)
    secondary_structure = pose.secstruct()

    residue_annotations = dict(enumerate(secondary_structure))

    # Process the secondary structure to identify ranges
    description_list = []
    current_ss = secondary_structure[0]
    start_res = 0

    def make_region_tag(dssp_code: str, start: int, end: int) -> str:
        # Represents a secondary structure region in XML format
        return f'<region start="{start}" end="{end}" dssp="{dssp_code}"/>'

    for i, ss in enumerate(secondary_structure[1:], start=1):
        if ss != current_ss:
            description_list.append(make_region_tag(current_ss, start_res, i - 1))
            start_res = i
            current_ss = ss

    # Add the last range
    description_list.append(
        make_region_tag(current_ss, start_res, len(secondary_structure) - 1)
    )

    # TODO: maybe use XMLTree instead of making this string by hand?
    description = (
        "<regions>\n"
        + textwrap.indent("\n".join(description_list), "  ")
        + "\n</regions>"
    )
    return {"residue_annotations": residue_annotations, "description": description}


@app.function(
    image=image,
    timeout=30000,
    secrets=[local_secret, modal.Secret.from_name("gcp-proteincrow")],
)
def all_dssp_with_pdb_string(json_data):
    gcs_paths = json_data["pdb_strings"]
    chain_id = json_data["chain_ids"]
    arguments_list = [(gcs_path, chain_id) for gcs_path in gcs_paths]
    return list(get_dssp_of_a_structure_with_pdb_path.starmap(arguments_list))


@app.function(
    image=image,
    timeout=30000,
    secrets=[local_secret, modal.Secret.from_name("gcp-proteincrow")],
)
def all_dssp(json_data):
    arguments_list = []
    gcs_paths = json_data["gcs_paths"]
    for gcs_path in gcs_paths:
        arguments_list.append(gcs_path)  # noqa: PERF402
    return list(get_dssp_of_a_structure.starmap(arguments_list))


@web_app.post("/rosetta/{protocol}")
async def compute_route(json_data: dict, protocol: str):
    blob = None
    if protocol == "compute_sap_score":
        blob = await compute_sap_score.remote.aio(json_data)
    elif protocol == "thread":
        blob = await thread_sequence.remote.aio(json_data)
    elif protocol == "relax":
        blob = await fast_relax.remote.aio(json_data)
    elif protocol == "thread_and_relax":
        blob = await thread_and_relax.remote.aio(json_data)
    elif protocol == "interface_analyze":
        blob = await analyze_all_interface.remote.aio(json_data)
    elif protocol == "dssp":
        blob = await all_dssp.remote.aio(json_data)
    elif protocol == "dssp_with_pdb_string":
        blob = await all_dssp_with_pdb_string.remote.aio(json_data)
    elif protocol == "thread_and_relax_opticrow":
        raise NotImplementedError("Need to update Modal past v0.64.31.")
        # blob = await thread_and_relax_opticrow.remote.aio(
        #     json_data["gcs_path_to_pdb"],
        #     json_data["proposed_seq"],
        #     json_data["upload_to"],
        # )

    return JSONResponse(content=blob)


@app.function(secrets=[local_secret])
@asgi_app()
def endpoint() -> FastAPI:
    return web_app
