from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlencode, urlparse

import pytest
import vcr.request

from aviary.core import DummyEnv

from . import CASSETTES_DIR

if TYPE_CHECKING:
    import vcr.request


@pytest.fixture(name="dummy_env")
def fixture_dummy_env() -> DummyEnv:
    return DummyEnv()


OPENAI_API_KEY_HEADER = "authorization"
ANTHROPIC_API_KEY_HEADER = "x-api-key"
# SEE: https://github.com/kevin1024/vcrpy/blob/v6.0.1/vcr/config.py#L43
VCR_DEFAULT_MATCH_ON = "method", "scheme", "host", "port", "path", "query"


def filter_api_keys(request: "vcr.request.Request") -> "vcr.request.Request":
    """Filter out API keys from request URI query parameters."""
    parsed_uri = urlparse(request.uri)
    if parsed_uri.query:  # If there's a query that may contain API keys
        query_params = parse_qs(parsed_uri.query)

        # Filter out the Google Gemini API key, if present
        if "key" in query_params:
            query_params["key"] = ["<FILTERED>"]

        # Rebuild the URI, with filtered parameters
        filtered_query = urlencode(query_params, doseq=True)
        request.uri = parsed_uri._replace(query=filtered_query).geturl()

    return request


@pytest.fixture(scope="session", name="vcr_config")
def fixture_vcr_config() -> dict[str, Any]:
    return {
        "filter_headers": [OPENAI_API_KEY_HEADER, ANTHROPIC_API_KEY_HEADER, "cookie"],
        "before_record_request": filter_api_keys,
        "record_mode": "once",
        "match_on": ["method", "host", "path", "query"],
        "allow_playback_repeats": True,
        "cassette_library_dir": str(CASSETTES_DIR),
    }
