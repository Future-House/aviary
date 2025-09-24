from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx_aiohttp
import litellm.llms.custom_httpx.aiohttp_transport
import pytest
import vcr.stubs.httpx_stubs

from . import CASSETTES_DIR

if TYPE_CHECKING:
    import vcr.request  # noqa: TC004

    from aviary.core import DummyEnv


@pytest.fixture(name="dummy_env")
def fixture_dummy_env() -> "DummyEnv":
    # Lazily import from aviary so typeguard doesn't throw:
    # > /path/to/.venv/lib/python3.12/site-packages/typeguard/_pytest_plugin.py:93:
    # > InstrumentationWarning: typeguard cannot check these packages because they
    # > are already imported: aviary
    from aviary.core import DummyEnv

    return DummyEnv(task="applesauce")


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


class PreReadCompatibleAiohttpResponseStream(
    httpx_aiohttp.transport.AiohttpResponseStream
):
    """aiohttp-backed response stream that works if the response was pre-read."""

    async def __aiter__(self) -> AsyncIterator[bytes]:
        with httpx_aiohttp.transport.map_aiohttp_exceptions():
            if self._aiohttp_response._body is not None:
                # Happens if some intermediary called `await _aiohttp_response.read()`
                # TODO: take into account chunk size
                yield self._aiohttp_response._body
            else:
                async for chunk in self._aiohttp_response.content.iter_chunked(
                    self.CHUNK_SIZE
                ):
                    yield chunk


async def _async_vcr_send(cassette, real_send, *args, **kwargs):  # noqa: ARG001
    """VCR send that only sends, not possibly recording or playing back responses."""
    return await real_send(*args, **kwargs)


# Permanently patch the original response stream,
# to work around https://github.com/karpetrosyan/httpx-aiohttp/issues/23
# and https://github.com/BerriAI/litellm/issues/11724
httpx_aiohttp.transport.AiohttpResponseStream = (  # type: ignore[misc]
    litellm.llms.custom_httpx.aiohttp_transport.AiohttpResponseStream  # type: ignore[misc]
) = PreReadCompatibleAiohttpResponseStream  # type: ignore[assignment]

# Permanently patch vcrpy's async VCR recording functionality,
# to work around https://github.com/kevin1024/vcrpy/issues/944
vcr.stubs.httpx_stubs._async_vcr_send = _async_vcr_send
