# aviary.notebook

A Jupyter notebook environment.

## Installation

To install the notebook environment, run the following command:

```bash
pip install 'fhaviary[notebook]'
```

To allow the environment to run notebooks in containerized sandboxes (recommended), first build the default image:

```bash
cd docker/
docker build -t aviary-notebook-env -f Dockerfile.pinned .
```

And second, set the environment variable `NB_ENVIRONMENT_USE_DOCKER=true`.
You may use your own Docker image with a different name,
in which case you must override the environment variable `NB_ENVIRONMENT_DOCKER_IMAGE`.
