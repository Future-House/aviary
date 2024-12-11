import pytest

from aviary.core import eval_answer, extract_answer_llm


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
            "Which of the following is most likely true:\n\nA) Piggie, B) Pigeon, C) Gerald\n",
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


@pytest.mark.vcr
@pytest.mark.parametrize(
    ("proposed", "options", "expected"),
    [
        pytest.param("A", ["A", "B", "C"], "A", id="exact"),
        pytest.param("a", ["A", "B", "C"], "A", id="exact"),
        pytest.param("F", ["B", "C"], None, id="not in options"),
        pytest.param("A or B", ["A", "B", "C"], None, id="not exact"),
        pytest.param(
            "Based on the context given, Serif et al. (2026) claim that "
            "the overwhelming cause of regime collapse arises from economic factors. "
            "Yet, most other scholars (Gerald and Robinson for example) believe the collapse "
            "was due to social unrest because of the prolonged epidemic of 2025. I tend to agree "
            "with the majority - although I can see both sides. Thus my response "
            "is that the social unrest was the significant factor in the collapse of the regime.",
            ["Economic factors", "Social unrest", "Political corruption"],
            "Social unrest",
            id="complex",
        ),
        pytest.param("", ["A", "B", "C"], None, id="empty proposed"),
    ],
)
@pytest.mark.asyncio
async def test_extract_answer_llm(proposed, options, expected):
    assert await extract_answer_llm(proposed, options) == expected
