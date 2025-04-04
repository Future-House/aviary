import os

import modal
from fastapi import Depends, FastAPI
from fastapi.responses import JSONResponse
from modal import Image, Stub, asgi_app, enter, method
from proteincrow.modal.middleware import validate_token

app = Stub("stability-esm")
web_app = FastAPI(dependencies=[Depends(validate_token)])


image = (
    Image.debian_slim(python_version="3.11")
    .micromamba()
    .apt_install("wget", "git")
    .pip_install(
        "torch",
        "fair-esm",
    )
    .run_commands(
        "wget https://dl.fbaipublicfiles.com/fair-esm/models/esm2_t33_650M_UR50D.pt -P /root/.cache/torch/hub/checkpoints/",
    )
)

if modal.is_local():
    local_secret = modal.Secret.from_dict({
        "AUTH_TOKEN": os.environ["MODAL_DEPLOY_TOKEN"],
        "SERVICE_ACCOUNT_JSON": os.environ["SERVICE_ACCOUNT_JSON"],
    })
else:
    local_secret = modal.Secret.from_dict({})

with image.imports():
    import esm
    import torch


@app.cls(image=image, gpu="a100")
class ESMModel:
    @enter()
    async def load_model(self):
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        self.batch_converter = self.alphabet.get_batch_converter()
        self.model.eval().cuda()
        return self.model

    @method()
    async def compute_perplexity(self, protein_seq: str):
        data = [("sequence", protein_seq)]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        log_probs = []
        for i in range(1, len(protein_seq) - 1):
            batch_tokens_masked = batch_tokens.clone()
            batch_tokens_masked[0, i] = self.alphabet.mask_idx
            with torch.no_grad():
                token_probs = torch.log_softmax(
                    self.model(batch_tokens_masked.cuda())["logits"], dim=-1
                )
            log_probs.append(
                token_probs[0, i, self.alphabet.get_idx(protein_seq[i])].item()
            )
        return {"perplexity": sum(log_probs)}

    @method()
    def get_embeddings(self, sequences: list[tuple]):
        data = [(f"sequence_{idx}", seq) for idx, seq in sequences]
        batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
        with torch.no_grad():
            return self.model(batch_tokens.cuda(), repr_layers=[33])["representations"]

    @method()
    async def get_probabilities(self, json_data: dict) -> list[tuple[str, float]]:
        mutate_sequence = {}
        response = []

        def find_mutations(seq1, seq2):
            # Ensure the sequences are of the same length
            if len(seq1) != len(seq2):
                raise ValueError("Sequences must be of equal length")

            # Find the positions and mutations
            return [
                (i, seq1[i], seq2[i]) for i in range(len(seq1)) if seq1[i] != seq2[i]
            ]

        wt_sequence = json_data["wt_sequence"]
        mut_sequence = json_data["mut_sequence"]
        mutations_found = find_mutations(wt_sequence, mut_sequence)

        for position, wt_res, mut_res in mutations_found:
            data = [("sequence", wt_sequence)]
            batch_labels, batch_strs, batch_tokens = self.batch_converter(data)
            batch_tokens_masked = batch_tokens.clone()
            batch_tokens_masked[0, position + 1] = self.alphabet.mask_idx
            with torch.no_grad():
                logits = torch.log_softmax(
                    self.model(batch_tokens_masked.cuda())["logits"], dim=-1
                )
            probabilities = torch.nn.functional.softmax(logits[0, position], dim=0)
            log_probabilities = torch.log(probabilities)
            wt_residue = batch_tokens[0, position].item()
            log_prob_wt = log_probabilities[wt_residue].item()
            log_prob_mt = log_probabilities[self.alphabet.get_idx(mut_res)].item()
            mutate_sequence[mut_res] = log_prob_mt - log_prob_wt
            response.append((
                f"{wt_res}{position}{mut_res}",
                log_prob_mt - log_prob_wt,
            ))
        return response


@app.function(image=image, gpu="h100", concurrency_limit=100)
def get_embedding_from_sequence(sequences: list, idx: int) -> dict:
    print(f"Running batch {idx}")
    print("Embedding Sequences")
    return ESMModel().get_embeddings.remote(sequences)


@app.function(image=image)
def execute_parallel_embeddings(batches: list) -> list:
    print(f"Parallel Embeddings: {len(batches)} batches")
    arguments = [(batch, idx) for idx, batch in enumerate(batches)]
    return list(get_embedding_from_sequence.starmap(arguments))


@app.function(image=image, gpu="a10g")
def predict(json_data: dict) -> dict:
    mut_sequence = json_data["mut_sequence"]
    wt_sequence = json_data["wt_sequence"]
    mut_perplexity = ESMModel().compute_perplexity.remote(mut_sequence)
    wt_perplexity = ESMModel().compute_perplexity.remote(wt_sequence)
    if mut_perplexity["perplexity"] < wt_perplexity["perplexity"]:
        return {
            "message": "The proposed mutated sequence has lower pseudo perplexity than the wild type sequence."
        }
    return {
        "message": "The proposed mutated sequence has higher pseudo perplexity than the wild type sequence."
    }


@web_app.post("/esm/{name}")
async def compute(json_data: dict, name: str):
    if name == "perplexity":
        blob = await predict.remote.aio(json_data)
    elif name == "probabilities":
        raise NotImplementedError("Need to update Modal past v0.64.31.")
        # blob = ESMModel().get_probabilities.remote(json_data)
    else:
        blob = {"message": "No name found"}
    return JSONResponse(content=blob)


@app.function(secrets=[local_secret])
@asgi_app()
def endpoint():
    return web_app
