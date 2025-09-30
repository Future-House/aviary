import pathlib
from enum import StrEnum


class CILLMModelNames(StrEnum):
    """Models to use for generic CI testing."""

    ANTHROPIC = "claude-3-5-haiku-20241022"  # Cheap and not Anthropic's cutting edge
    OPENAI = "gpt-5-mini-2025-08-07"  # Cheap and not OpenAI's cutting edge


TESTS_DIR = pathlib.Path(__file__).parent
CASSETTES_DIR = TESTS_DIR / "cassettes"
