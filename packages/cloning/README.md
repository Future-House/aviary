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

## Installation and Modal setup

First, we install the Aviary environment, which depends on `uv` being available:

```bash
git clone https://github.com/Future-House/aviary.git
cd aviary
uv python pin 3.12
uv venv
uv pip install -e .
uv pip install -e envs/cloning
```

Next, we need to deploy tool endpoints for the environment to use.
We use [Modal](https://modal.com/docs) to serve the tools, so you will have to create an account.
Once you have done so, obtain a deploy token and set it as the shell environment variable `MODAL_DEPLOY_TOKEN`.

### Modal environment setup

Create a new Modal environment to encapsulate the cloning tools.
For the rest of this README, we will assume the environment is called `aviary-oss`.

```
modal environment create aviary-oss
```

Create two Modal storage volumes to contain plasmid GenBank files and a search index, respectively (see next section):

```
modal volume create -e aviary-oss genbank-files
modal volume create -e aviary-oss genbank-index
```

## Tool data sources

### Plasmid search index

The environment's plasmid search tool is used to retrieve the DNA sequences of plasmids based on name or keywords.
The tool builds an index over the contents of the `genbank-files` volume created in the previous step; we leave its population up to the user, since the library of usable plasmids is use-case specific.

To populate the volume, you can create a local library and upload it:

```bash
$ ls my_plasmid_library/
plasmid_1.gbk
plasmid_2.gbk
[...]
$ ls my_plasmid_library/* | xargs -n1 modal volume put -e aviary-oss genbank-files
```

All files are expected to end with the `.gbk` extension.

Finally, build the index by calling:

```bash
cd aviary/packages/cloning/remote
modal run -e aviary-oss search.py
```

### Annotation FASTA files

The sequence annotation tool requires two reference FASTA files.
These must be provided as shell environment variables set to downloadable URLS:

- `AVIARY_CLONING_FPBASE_FASTA_URL`: Sequences from the [Fluorescent Protein Database](https://www.fpbase.org/)
- `AVIARY_CLONING_EXTRA_FASTA_URL`: Any other sequences to be used for annotating plasmids, e.g. the Kozak sequence.

## Tool deployment and configuration

To deploy the tool set as a Modal app, execute the following:

```bash
cd aviary/packages/cloning/remote
modal deploy -e aviary-oss deploy.py
```

This may take some time on first execution, as Docker images need to be built and pushed.

### Modal app URL configuration

In the output from `modal deploy`, one of the final lines reports the base URL of the FastAPI endpoints.
For example, when running the above in the FutureHouse Modal instance, we see:

```
✓ Created objects.
├── 🔨 Created mount /home/sid/dev/aviary/packages/cloning/remote/annotator.py
├── 🔨 Created mount /home/sid/dev/aviary/packages/cloning/remote/annotator_lib
├── 🔨 Created mount /home/sid/dev/aviary/packages/cloning/remote/search.py
├── 🔨 Created mount PythonPackage:poly
├── 🔨 Created mount PythonPackage:deploy
├── 🔨 Created mount PythonPackage:models
├── 🔨 Created mount PythonPackage:dependencies
├── 🔨 Created function build_index.
├── 🔨 Created function PlasmidSearch.*.
├── 🔨 Created web function fastapi_app => https://future-house-aviary-oss--cloning-fastapi-app.modal.run
├── 🔨 Created mount /home/sid/dev/aviary/packages/cloning/remote/poly_lib
├── 🔨 Created mount databases.sh
├── 🔨 Created function find_orfs.
├── 🔨 Created function synthesize.
├── 🔨 Created function digest_and_ligate.
├── 🔨 Created function design_primers.
├── 🔨 Created function gibson_assemble.
└── 🔨 Created function annotate.
✓ App deployed in 43.445s! 🎉
```

where `https://future-house-aviary-oss--cloning-fastapi-app.modal.run` is the server host.
Assign that URL as the shell environment variable `AVIARY_CLONING_MODAL_URL`.

### `NCBI_API_KEY`

The environments' gene search tool uses NCBI's Entrez search API.
It is easy to hit the API's rate limits when running many parallel rollouts, but this can be mitigated by acquiring an NCBI API key ([instructions](https://support.nlm.nih.gov/kbArticle/?pn=KA-05317)).
Once obtained, assign it to the shell environment variable `NCBI_API_KEY`.

## Test setup

To run a simple test of the cloning environment, install [`ldp`](https://github.com/Future-House/ldp) and run the following:

```bash
MODAL_DEPLOY_TOKEN=... OPENAI_API_KEY=... python -m cloning.app "Clone GFP into pUC19"
```

You may have to modify the query for a successful rollout, based on the contents of the `genbank-files` plasmid library.

## FASTA Annotation

The sequence annotation tool uses MMSeqs2 against SwissProt, plasmid features, and fluorescent proteins to annotate a FASTA file. It returns a genbank file with the feature annotations.

This followed this paper: https://academic.oup.com/nar/article/49/W1/W516/6279845 to a large extent, except with the faster/newer mmseqs2 tool. We skipped RNA for now as well.

Some relevant parameter definitions (see https://en.wikipedia.org/wiki/BLAST_(biotechnology)): 1. tcov_threshold - threshold coverage of the target sequence 2. e_score_threshold - E-value threshold for inclusion in the output 3. circular - we're annotating plasmids so we have them as circular. There are two ways to treat this (1) duplicate the sequence and treat it as linear, but then need to latter remove the duplicate annotations (circular=True case) (2) do not duplicate the sequence and just ignore the annotations that cross the origin of replication (circular=False case). We chose the latter for now.
