import os
from pathlib import Path

import pytest

from aviary.db import EnvironmentDBBackend
from aviary.env import DummyEnv

TEST_OUTPUT_DIR = Path(__file__).parent / "test_outputs"

ENV_BACKENDS = (
    "sqlite+aiosqlite:///:memory:",
    f"sqlite+aiosqlite:///{TEST_OUTPUT_DIR}/envruns.sqlite3",
)


@pytest.fixture(name="dummy_env")
def fixture_dummy_env() -> DummyEnv:
    return DummyEnv()


@pytest.fixture(scope="session", autouse=True)
def _fixture_make_test_outputs_dir():
    TEST_OUTPUT_DIR.mkdir(exist_ok=True)


@pytest.fixture
def clean_db_backend(request):
    """Empty the DB session between test cases and remove the database file."""
    db_backend = request.param

    # Remove the database file if it exists
    if db_backend.startswith("sqlite") and not db_backend.endswith(":memory:"):
        file_path = db_backend.split("///")[-1]
        if os.path.exists(file_path):
            os.remove(file_path)

    yield db_backend

    # Finalizer code to run after the test
    EnvironmentDBBackend.Session = None
    EnvironmentDBBackend.uri = None
