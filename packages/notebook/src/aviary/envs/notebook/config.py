import os

USE_DOCKER = bool(os.getenv("NB_ENVIRONMENT_USE_DOCKER", "true").lower() == "true")
NB_ENVIRONMENT_DOCKER_IMAGE = os.getenv(
    "NB_ENVIRONMENT_DOCKER_IMAGE", "aviary-notebook-env"
)

# Some R error messages can be 100,000 of characters
NB_OUTPUT_LIMIT = 3000  # chars
# Streams from a docker container. Don't set to `sys.stdout.fileno()`
# because we want to differentiate from file I/O
DOCKER_STREAM_TYPE_STDOUT = 1
DOCKER_STREAM_TYPE_STDERR = 2
