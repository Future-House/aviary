from enum import StrEnum


class CILLMModelNames(StrEnum):
    """Models to use for generic CI testing."""

    ANTHROPIC = "claude-3-haiku-20240307"  # Cheap and not Anthropic's cutting edge
