import pytest

from aviary.tools import eval_answer


@pytest.mark.parametrize(
    ("proposed", "correct", "question", "eval_mode", "expected"),
    [
        pytest.param(
            "\n\n250",
            "250",
            None,
            "exact",
            True,
        ),
        pytest.param(
            "Answer:\n\n250",
            "250",
            None,
            "exact",
            False,
        ),
        pytest.param(
            "Answer\n\n: 250",
            "250",
            None,
            "contains",
            True,
        ),
        pytest.param("A)", "A", None, "contains", True),
        pytest.param("The answer is C", "D", None, "contains", False),
        pytest.param(
            "Based on all factors considered, the most compelling answer is Gerald, C",
            "C",
            "Which of the following is most likely true:\n\n A) Piggie, B) Pigeon, C) Gerald\n",
            "llm",
            True,
        ),
    ],
)
@pytest.mark.asyncio
async def test_eval_answer(proposed, correct, question, eval_mode, expected):
    assert await eval_answer(proposed, correct, question, eval_mode) == expected


@pytest.mark.asyncio
async def test_eval_llm_config():
    config = {"temperature": 0.5}
    assert await eval_answer("250", "250", "What is 25 * 10?", "llm", config)
