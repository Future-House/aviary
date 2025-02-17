import os
import shutil
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest
from paperqa import Settings


@pytest.fixture(scope="session", name="stub_data_dir")
def fixture_stub_data_dir() -> Path:
    return Path(__file__).parent / "stub_data"


@pytest.fixture
def tmp_path_cleanup(tmp_path: Path) -> Iterator[Path]:
    yield tmp_path
    # Cleanup after the test
    if tmp_path.exists():
        shutil.rmtree(tmp_path, ignore_errors=True)


@pytest.fixture
def agent_home_dir(tmp_path_cleanup: str | os.PathLike) -> Iterator[str | os.PathLike]:
    """Set up a unique temporary folder for the agent module."""
    with patch.dict("os.environ", {"PQA_HOME": str(tmp_path_cleanup)}):
        yield tmp_path_cleanup


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
