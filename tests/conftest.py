from typing import Any

import pytest

from aviary.env import DummyEnv

from . import CASSETTES_DIR


@pytest.fixture(name="dummy_env")
def fixture_dummy_env() -> DummyEnv:
    return DummyEnv()


OPENAI_API_KEY_HEADER = "authorization"
ANTHROPIC_API_KEY_HEADER = "x-api-key"
# SEE: https://github.com/kevin1024/vcrpy/blob/v6.0.1/vcr/config.py#L43
VCR_DEFAULT_MATCH_ON = "method", "scheme", "host", "port", "path", "query"


@pytest.fixture(scope="session", name="vcr_config")
def fixture_vcr_config() -> dict[str, Any]:
    return {
        "filter_headers": [OPENAI_API_KEY_HEADER, ANTHROPIC_API_KEY_HEADER, "cookie"],
        "record_mode": "once",
        "match_on": ["method", "host", "path", "query"],
        "allow_playback_repeats": True,
        "cassette_library_dir": str(CASSETTES_DIR),
    }
