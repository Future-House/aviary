from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from paperqa import Settings

CASSETTES_DIR = Path(__file__).parent / "cassettes"


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
    settings.agent.search_count = 2
    settings.answer.answer_max_sources = 2
    settings.answer.evidence_k = 10
    return settings


@pytest.fixture(name="agent_task_settings")
def fixture_agent_task_settings(agent_test_settings: Settings) -> Settings:
    agent_test_settings.agent.index.manifest_file = "stub_manifest.csv"
    return agent_test_settings


# TODO: Import from llmclient.utils after next llmclient release
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
