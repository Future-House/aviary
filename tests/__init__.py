import pathlib
from enum import StrEnum


class CILLMModelNames(StrEnum):
    """Models to use for generic CI testing."""

    ANTHROPIC = "claude-3-haiku-20240307"  # Cheap and not Anthropic's cutting edge
    OPENAI = "gpt-4o-mini-2024-07-18"  # Cheap and not OpenAI's cutting edge


TESTS_DIR = pathlib.Path(__file__).parent
CASSETTES_DIR = TESTS_DIR / "cassettes"
