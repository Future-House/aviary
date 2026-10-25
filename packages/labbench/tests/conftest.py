import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from lmi.utils import (
    ANTHROPIC_API_KEY_HEADER,
    OPENAI_API_KEY_HEADER,
)
from paperqa import Settings
from vcr import VCR

CASSETTES_DIR = Path(__file__).parent / "cassettes"
UUID_PATTERN = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
)


def pytest_recording_configure(config: pytest.Config, vcr: VCR) -> None:  # noqa: ARG001
    def normalize_request(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: "<id>"
                if key in {"id", "tool_call_id"}
                else normalize_request(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [normalize_request(item) for item in value]
        if isinstance(value, str):
            return UUID_PATTERN.sub("<uuid>", value)
        return value

    def match_openai_json(current, recorded) -> None:
        if current.host != recorded.host or current.path != recorded.path:
            return
        if current.host != "api.openai.com":
            return
        current_data = json.loads(current.body)
        recorded_data = json.loads(recorded.body)
        if current.path.endswith("/embeddings"):
            # Parser output can vary slightly; the batch and document prefix
            # identify the recorded embedding response.
            if (
                len(current_data["input"]) != len(recorded_data["input"])
                or str(current_data["input"][0])[:64]
                != str(recorded_data["input"][0])[:64]
            ):
                raise AssertionError("Embedding batches differ")
        elif normalize_request(current_data) != normalize_request(recorded_data):
            raise AssertionError("OpenAI request bodies differ")

    vcr.register_matcher("openai_json", match_openai_json)


@pytest.fixture(scope="session", name="stub_data_dir")
def fixture_stub_data_dir() -> Path:
    return Path(__file__).parent / "stub_data"


@pytest.fixture
def agent_home_dir(tmpdir) -> Iterator[Any]:
    """Set up a unique temporary folder for the agent module."""
    with patch.dict("os.environ", {"PQA_HOME": str(tmpdir)}):
        yield tmpdir


@pytest.fixture
def agent_index_dir(agent_home_dir: Path) -> Path:
    return agent_home_dir / ".pqa" / "indexes"


@pytest.fixture
def agent_test_settings(agent_index_dir: Path, stub_data_dir: Path) -> Settings:
    # NOTE: originally here we had usage of embedding="sparse", but this was
    # shown to be too crappy of an embedding to get past the Obama article
    settings = Settings()
    settings.agent.index.paper_directory = stub_data_dir
    settings.agent.index.index_directory = agent_index_dir
    settings.parsing.use_doc_details = False
    settings.agent.search_count = 2
    settings.answer.answer_max_sources = 2
    settings.answer.evidence_k = 10
    return settings


@pytest.fixture(name="agent_task_settings")
def fixture_agent_task_settings(agent_test_settings: Settings) -> Settings:
    agent_test_settings.agent.index.manifest_file = "stub_manifest.csv"
    return agent_test_settings


@pytest.fixture(scope="session", name="vcr_config")
def fixture_vcr_config() -> dict[str, Any]:
    return {
        "filter_headers": [OPENAI_API_KEY_HEADER, ANTHROPIC_API_KEY_HEADER, "cookie"],
        "match_on": ["method", "host", "path", "query"],
        "allow_playback_repeats": True,
        "cassette_library_dir": str(CASSETTES_DIR),
        # "drop_unused_requests": True,  # Restore after https://github.com/kevin1024/vcrpy/issues/961
    }
