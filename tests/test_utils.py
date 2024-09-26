import pytest

from aviary.core import eval_answer


@pytest.mark.vcr
@pytest.mark.parametrize(
    ("proposed", "correct", "question", "eval_mode", "expected"),
    [
        pytest.param("\n\n250", "250", None, "exact", True, id="exact"),
        pytest.param(
            "Answer:\n\n250", "250", None, "exact", False, id="exact with noise"
        ),
        pytest.param(
            "Answer\n\n: 250", "250", None, "contains", True, id="contains with noise"
        ),
        pytest.param("A)", "A", None, "contains", True, id="contains multiple choice"),
        pytest.param(
            "The answer is C", "D", None, "contains", False, id="contains wrong answer"
        ),
        pytest.param(
            "Based on all factors considered, the most compelling answer is Gerald, C",
            "C",
            "Which of the following is most likely true:\n\n A) Piggie, B) Pigeon, C) Gerald\n",
            "llm",
            True,
            id="llm basic",
        ),
    ],
)
@pytest.mark.asyncio
async def test_eval_answer(proposed, correct, question, eval_mode, expected):
    assert await eval_answer(proposed, correct, question, eval_mode) == expected


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_eval_llm_config():
    config = {"temperature": 0.5}
    assert await eval_answer("250", "250", "What is 25 * 10?", "llm", config)
