import subprocess
import tempfile
import uuid

from fastapi import APIRouter
from fastapi.responses import Response
from modal import App, Image, Mount
from models import FastaRequest, convert_fasta

app = App("plasmid-annotator")
router = APIRouter()

image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "wget")
    .run_commands(
        "wget https://mmseqs.com/latest/mmseqs-linux-avx2.tar.gz; tar xvfz mmseqs-linux-avx2.tar.gz"
    )
    .env({"MMSEQS": "/mmseqs/bin/mmseqs"})
    .run_commands(
        "wget https://storage.googleapis.com/fh-modal-artifacts/snapgene.fasta"
    )
    .run_commands("wget https://storage.googleapis.com/fh-modal-artifacts/fpbase.fasta")
    .copy_local_file("./databases.sh", "databases.sh")
    .run_commands("chmod +x databases.sh", "./databases.sh /root")
    .pip_install("biopython")
)


mounts = Mount.from_local_dir(
    local_path="annotator_lib",
    remote_path="/root",
)


# size of RAM based on sequence database size
# CPU count - good scaling and choosing cpu=4 or something gives 20 second runtimes
# container_idle_timeout - 1000 seconds to allow for bursty job low startup
@app.function(
    image=image, mounts=[mounts], memory=2**15, cpu=16, container_idle_timeout=1000
)
def annotate(fasta_str: str, circular: bool) -> str:
    # write fasta file
    with tempfile.NamedTemporaryFile("w") as f:
        if circular:
            # we double the sequence to make it circular
            header = fasta_str.split("\n")[0]
            body = fasta_str.split("\n")[1]
            fasta_str = f"{header}\n{body}{body}\n"
        f.write(fasta_str)
        f.flush()
        fasta_path = f.name
        # make a short name with uuid
        name = str(uuid.uuid4())[:3]
        # run annotation pipeline via shell script
        result = subprocess.run(  # noqa: S603
            ["/bin/bash", "./annotate.sh", fasta_path, name],
            capture_output=True,
            text=True,
            env={"MMSEQS": "/mmseqs/bin/mmseqs"},
            check=False,
        )
        # send to stdout to be capture by modal
        print("Annotate stderr:", result.stderr)
        print("Annotate stdout:", result.stdout)
        from annotation2gbk import annotation2genbank

        annotation2genbank(
            f"{name}/all_results.tsv",
            fasta_path,
            f"{name}.gbk",
            e_score_threshold=10**-5,
            tcov_threshold=0.7,
            circular=circular,
        )
    with open(f"{name}.gbk") as f:  # noqa: FURB101
        return f.read()


@router.post("/annotate")
async def annotate_endpoint(request: FastaRequest, circular: bool | None = None):
    """
    Annotate a sequence via BLASTX search against multiple databases.

    This endpoint performs a BLASTX search to annotate the provided sequence using the following databases:
    - Common synthetic biology components
    - Fluorescent Protein Database
    - SwissProt

    ### Args:
    - `request` (FastaRequest): A request object containing a FASTA sequence or plain sequence.
      It should have a key "fasta" whose value is either a sequence string or a FASTA-formatted string.
    - `circular` (bool | None, optional): Indicates whether the sequence is circular.
      If not provided, the function will check for the phrase `(circular)` in the FASTA header.

    ### Returns:
    - `Response`: The annotated sequence in GenBank format.
      - Content-Type: text/plain; charset=utf-8
      - On success: GenBank formatted string
      - On failure: Error message with status code 500

    ### Notes:
    - If the `circular` parameter is not provided, the function will attempt to determine
      the circularity from the FASTA header.
    - The annotation process may take some time depending on the sequence length and complexity.
    """
    fasta, is_circular = convert_fasta(request.fasta)
    if circular is None:
        circular = is_circular
    try:
        gb_str = await annotate.remote.aio(fasta, circular)
    except Exception as e:
        return Response(content=str(e), status_code=500)

    # return as text
    return Response(
        content=gb_str.encode("utf-8"), media_type="text/plain; charset=utf-8"
    )
