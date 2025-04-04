# Molecular Cloning Environment

This environment is designed to provide a set of tools for molecular cloning. It includes the following tools:

1. Annotate FASTA
2. Search for plasmids via keyword search
3. Design PCR primers/ simulate PCR
4. Simulate Gibson Assembly
5. Simulate Golden Gate Assembly
6. Simulate restriction enzyme cloning
7. Annotate restriction sites
8. Various other sequence manipulation tools

## Usage

If you want to just use it w/o reward or task from the command line:

```bash
MODAL_DEPLOY_TOKEN=... OPENAI_API_KEY=... python -m cloning.app "Clone GFP into pUC19"
```

## Annotate FASTA

This uses MMSeqs2 against SwissProt, plasmid features, and fluorescent proteins to annotate a FASTA file. It returns a genbank file with the feature annotations.

This followed this paper: https://academic.oup.com/nar/article/49/W1/W516/6279845 to a large extent, except with the faster/newer mmseqs2 tool. We skipped RNA for now as well.

Some relevant parameter definitions (see https://en.wikipedia.org/wiki/BLAST_(biotechnology)): 1. tcov_threshold - threshold coverage of the target sequence 2. e_score_threshold - E-value threshold for inclusion in the output 3. circular - we're annotating plasmids so we have them as circular. There are two ways to treat this (1) duplicate the sequence and treat it as linear, but then need to latter remove the duplicate annotations (circular=True case) (2) do not duplicate the sequence and just ignore the annotations that cross the origin of replication (circular=False case). We chose the latter for now.

## Human-operated environment

### Installation & Setup

1. `git clone` this repo
2. Install `uv`

```bash
cd aviary-internal
uv python pin 3.12
uv venv
# NOTE: gsm8k provides some experiment utilities
uv pip install -e aviary_internal -e envs/gsm8k -e envs/cloning
```

Make sure to obtain and add these keys to your shell environment:

```bash
export OPENAI_API_KEY=... # by default, we use an LLM to auto-evaluate whether you got the question right
export MODAL_DEPLOY_TOKEN=... # by default, we use cloning tools that are running on Modal
export NCBI_API_KEY=... # optional, but recommended (see below)
```

### Local Tools

If you would like to build local versions of the tools for Linux run the following command,
supplying your OS specifications as command line arguments as required.

| OS           | Command                                 |
| ------------ | --------------------------------------- |
| Linux/Ubuntu | `bash build_poly_tools.sh`              |
| macOS        | `bash build_poly_tools.sh darwin arm64` |

Local tools also require data from [rebase](http://rebase.neb.com/rebase/rebase.files.html),
which will be automatically downloaded on first use.

If local tools are built, the environment will prefer them over Modal-hosted tools.
To override this, specify the environment variable `CLONING_USE_MODAL=1`.

### `MODAL_DEPLOY_TOKEN`

The `search_plasmids` tool is always run using Modal.

Other tools can be run via Modal or locally,
depending if you've built the local tools (see [Local Tools](#local-tools)).

### `NCBI_API_KEY`

We use NCBI's Entrez API to search NCBI's nucleotide database for genes.
Without an API key, you're placed in their shared free tier, which is rate limited.

To make an API key,
following the instructions from https://support.nlm.nih.gov/kbArticle/?pn=KA-05317

1. Go to https://account.ncbi.nlm.nih.gov/
2. Sign in
3. Select "Create a new NCBI account"
4. Go to https://account.ncbi.nlm.nih.gov/settings/
5. Scroll to "API Key Management"
6. Create an API key and store it as `NCBI_API_KEY`

### Configuration

We have a special Agent class (`InteractiveAgent`) that emulates a language agent's behavior, but requests input from a human.
The cloning module provides an experiment (`CloningInteractiveExpt`) that instantiates this agent and runs it against cloning environments, storing trajectories in an output directory.
To use it, create a configuration file (e.g. `interactive.yaml`) with the following contents:

```yaml
expt: cloning.expts.common.CloningInteractiveExpt

env:
  # By default, will run over SeqQA. This restricts us to the test split.
  split: test
  # Uncomment this if you want to use a custom SeqQA dataset; otherwise data will be pulled from HF Hub
  # data_source: /path/to/LAB-Bench-internal/SeqQA

evaluator:
  # Disable this if you want to run through the questions in order
  shuffle: True

output_repo:
  # Outputs will be stored in ~/aviary_data/<name> and pushed to GCS.
  # People typically prefix with their (human) name to avoid clashing namespaces.
  name: sid/seqqa-human/example-run-v0
```

### Running

And finally, use `aviary-internal`'s built-in `run_expt` utility to start the experiment:

```bash
run_expt interactive.yaml
```

You will then see something like:

```text
AVAILABLE TOOLS:
--------------------------------------------------------------------------------
plasmid_search(query: string):
   Searches from a database of plasmids and returns best matching features or plasmids.

   query: Keyword query search. Can negate terms with a minus sign. Supports AND/OR.

[many many tools omitted]

submit_answer(answer: string):
   Submit the final answer to complete the task. Only provide answer, without reasoning or comments.

   answer

--------------------------------------------------------------------------------

OBSERVATIONS:
--------------------------------------------------------------------------------
I want to clone the gpp gene from E. coli into the plasmid pUC19 using restriction-ligation cloning. Which of the following primer pairs should I use to do the cloning with the enzymes EcoRI and SphI?

Options:
seq-e1d4a95f8277,  seq-ef7e945d8eb1
seq-8a7bd9593d70,  seq-87389e48ec3d
seq-49e7542a9d44,  seq-ece5e2c8dc55
seq-10a292a25eb7,  seq-e0d0306dacf9


You are provided with a key-value store of sequences. I will always inform of you of the state of this store. Use keys (names) of sequences in all functions, rather than pass them directly. For example, if this the current contents of key-value store:
Available sequence keys: dummy-seq-abc123

{
    "dummy-seq-abc123": {
        "type": "BioSequence(dna, Linear)",
        "name": "dummy-seq-abc123",
        "sequence": "8 bp: ATCGATCG"
    }
}
Then, you could call `annotate(sequence='dummy-seq-abc123')` or `find_orfs(sequence='dummy-seq-abc123', min_length=35)`. Tool call outputs will be added to the key-value store.
Sometimes sequences are alone as a BioSequence. Other times, they are bundled in a BioSequences object. You can separate a BioSequences object into individual sequences using the `separate` tool. Or you can slice into one sequence from a BioSequences object.

Current contents of key-value store:

Available sequence keys: seq-e1d4a95f8277, seq-ef7e945d8eb1, seq-8a7bd9593d70, seq-87389e48ec3d, seq-49e7542a9d44, seq-ece5e2c8dc55, seq-10a292a25eb7, seq-e0d0306dacf9

{
    "seq-e1d4a95f8277": {
        "type": "BioSequence(dna, Linear)",
        "name": "seq-e1d4a95f8277",
        "sequence": "31 bp: GAATTCATGGGTTCCACCTCGTCGCTGTATG"
    },
    [many sequences omitted]
}

--------------------------------------------------------------------------------
>>> Select tool by name:
```

On the last line, you can now select a tool. You will then be prompted for arguments to the tool. For example:

```text
>>> Select tool by name: enzyme_cut
>>> Enter parameter (sequence: string): seq-8a7bd9593d70
>>> Enter parameter (enzyme: string): EcoRI

OBSERVATIONS:
--------------------------------------------------------------------------------
cut-65057aa0 (BioSequences):
seq-f1d90b4c17f4: Fragment 0
Type: dna, Linear
Sequence (1 bp): G

seq-28fa47971557: Fragment 1
Type: dna, Linear
Sequence (28 bp): AATTCATGGGGTTCCACCTCGTCGCTGT

Current contents of key-value store:

Available sequence keys: seq-e1d4a95f8277, seq-ef7e945d8eb1, seq-8a7bd9593d70, seq-87389e48ec3d, seq-49e7542a9d44, seq-ece5e2c8dc55, seq-10a292a25eb7, seq-e0d0306dacf9, cut-65057aa0

{
    [many sequences omitted]
    "cut-65057aa0": {
        "type": "BioSequences",
        "sequences": [
            {
                "type": "BioSequence(dna, Linear)",
                "name": "seq-f1d90b4c17f4",
                "sequence": "1 bp: G",
                "description": "Fragment 0"
            },
            {
                "type": "BioSequence(dna, Linear)",
                "name": "seq-28fa47971557",
                "sequence": "28 bp: AATTCATGGGGTTCCACCTCGTCGCTGT",
                "description": "Fragment 1"
            }
        ]
    }
}

--------------------------------------------------------------------------------
>>> Select tool by name:
```

Note that after you enter all arguments, the output of the tool and the updated key-value store are displayed.
You may keep calling tools until you have the answer, at which point call the `submit_answer` tool.

You have two special keywords that you can enter at any time:

- `CLEAR`: The current tool call will be reset, and you can start your input over again. This is useful if you are halfway through writing arguments for a tool and decide you'd rather use a different tool.
- `EXIT`: will exit the current problem and skip ahead to the next one.

Finally, press `ctrl+C` to stop at any time.
