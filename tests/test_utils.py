from collections.abc import Iterable, Sequence

import pytest

from aviary.core import eval_answer, extract_answer
from aviary.utils import MultipleChoiceEvaluation, MultipleChoiceQuestion
from tests.conftest import VCR_DEFAULT_MATCH_ON


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
async def test_eval_answer(
    proposed: str, correct: str, question: str | None, eval_mode: str, expected: float
) -> None:
    assert await eval_answer(proposed, correct, question, eval_mode) == expected


@pytest.mark.vcr
@pytest.mark.parametrize(
    ("proposed_answer", "options", "expected"),
    [
        pytest.param("A", ["A", "B", "C"], "A", id="exact-uppercase"),
        pytest.param("a", ["A", "B", "C"], "A", id="exact-lowercase"),
        pytest.param("F", ["B", "C"], None, id="not in options"),
        pytest.param("A or B", ["A", "B", "C"], None, id="gave-two"),
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
        pytest.param("", ["A", "B", "C"], None, id="empty-proposal"),
    ],
)
@pytest.mark.asyncio
async def test_extract_answer(
    proposed_answer: str, options: Sequence[str], expected: str | None
) -> None:
    assert await extract_answer(proposed_answer, options) == expected


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_eval_llm_config():
    config = {"temperature": 0.5}
    assert await eval_answer("250", "250", "What is 25 * 10?", "llm", config)


class TestLitQAEvaluation:
    @staticmethod
    def _assert_prompt_is_valid(
        mc_question: MultipleChoiceQuestion,
        question: str,
        ideal_answer: str,
        distractors: Iterable[str],
    ) -> None:
        question_prompt = mc_question.question_prompt
        for substr in (
            question,
            "Insufficient information",
            ideal_answer,
            *distractors,
        ):
            assert question_prompt.count(substr) == 1

    # Use for general purpose testing
    ZIP_CODE_QUESTION_IDEAL_DISTRACTORS = (
        "What is my office's zip code?",
        "94107",
        ["-8", "94106", "cheesecake"],
    )
    # The following two are used to check we don't leak on the LLM's innate knowledge
    MEANING_OF_LIFE_QUESTION_IDEAL_DISTRACTORS = (
        "What is the meaning of life?",
        "42",
        ["-84", "11", "cheesecake"],
    )
    # Source: https://github.com/Future-House/LAB-Bench/blob/43b2045c67a2da12c233689cf538f1ed5c42f590/LitQA2/litqa-v2-public.jsonl#L130
    LITQA2_QUESTION_IDEAL_DISTRACTORS = (
        (
            "What method was used to demonstrate that the enzyme PafA is stable after"
            " incubation with 4M urea for 14 days?"
        ),
        "circular dichroism",
        ["cryo EM", "x-ray crystallography", "NMR"],
    )

    @pytest.mark.asyncio
    @pytest.mark.vcr(match_on=[*VCR_DEFAULT_MATCH_ON, "body"])
    @pytest.mark.parametrize(
        (
            "question",
            "ideal_answer",
            "distractors",
            "actual_answer",
            "expected_eval",
            "expected_extracted_answer",
        ),
        [
            pytest.param(
                *ZIP_CODE_QUESTION_IDEAL_DISTRACTORS,
                "the answer is 94107",
                MultipleChoiceEvaluation.CORRECT,
                "94107",
                id="matched-correct-option",
            ),
            pytest.param(
                *ZIP_CODE_QUESTION_IDEAL_DISTRACTORS,
                "the answer is 14004",
                MultipleChoiceEvaluation.INCORRECT,
                None,
                id="didnt-match-and-no-llm-innate-knowledge",
            ),
            pytest.param(
                *ZIP_CODE_QUESTION_IDEAL_DISTRACTORS,
                "the answer is 94106",
                MultipleChoiceEvaluation.INCORRECT,
                "94106",
                id="matched-incorrect-option",
            ),
            pytest.param(
                *ZIP_CODE_QUESTION_IDEAL_DISTRACTORS,
                "Insufficient information",
                MultipleChoiceEvaluation.UNSURE,
                MultipleChoiceQuestion.DEFAULT_UNSURE_OPTION,
                id="matched-unsure-option",
            ),
            pytest.param(
                *ZIP_CODE_QUESTION_IDEAL_DISTRACTORS,
                "the answer is 94106 or 94107",
                MultipleChoiceEvaluation.INCORRECT,
                None,
                id="matched-several-options",
            ),
            pytest.param(
                *ZIP_CODE_QUESTION_IDEAL_DISTRACTORS,
                "",
                MultipleChoiceEvaluation.INCORRECT,
                None,
                id="empty-answer1",
            ),
            pytest.param(
                *MEANING_OF_LIFE_QUESTION_IDEAL_DISTRACTORS,
                "14",
                MultipleChoiceEvaluation.INCORRECT,
                None,
                id="didnt-match-and-llm-has-innate-knowledge",
            ),
            pytest.param(
                *MEANING_OF_LIFE_QUESTION_IDEAL_DISTRACTORS,
                "",
                MultipleChoiceEvaluation.INCORRECT,
                None,
                id="empty-answer2",
            ),
            pytest.param(
                *LITQA2_QUESTION_IDEAL_DISTRACTORS,
                "",
                MultipleChoiceEvaluation.INCORRECT,
                None,
                id="empty-answer3",
            ),
        ],
    )
    async def test_grade(
        self,
        question: str,
        ideal_answer: str,
        distractors: str | list[str],
        actual_answer: str,
        expected_eval: MultipleChoiceEvaluation,
        expected_extracted_answer: str | None,
    ) -> None:
        """Tests that we can create a multiple choice question and evaluate answers."""
        mc_question = MultipleChoiceQuestion(
            question=question,
            options=distractors,
            ideal_answer=ideal_answer,
            shuffle_seed=42,  # Seed for VCR cassette
        )
        self._assert_prompt_is_valid(mc_question, question, ideal_answer, distractors)
        evaluation, graded_answer = await mc_question.grade(actual_answer)
        assert evaluation == expected_eval
        if evaluation == MultipleChoiceEvaluation.CORRECT:
            assert graded_answer == ideal_answer
        assert graded_answer == expected_extracted_answer

    def test_consistent_mc_options(self) -> None:
        """Tests that creating multiple evaluations with the same seed results in the same prompt."""
        question, ideal, distractors = self.MEANING_OF_LIFE_QUESTION_IDEAL_DISTRACTORS
        mc_question_1a = MultipleChoiceQuestion(
            question=question, ideal_answer=ideal, options=distractors, shuffle_seed=0
        )
        self._assert_prompt_is_valid(mc_question_1a, question, ideal, distractors)

        mc_question_1b = MultipleChoiceQuestion(
            question=question, ideal_answer=ideal, options=distractors, shuffle_seed=0
        )
        self._assert_prompt_is_valid(mc_question_1b, question, ideal, distractors)
        assert mc_question_1a == mc_question_1b, (
            "Same seeding should lead to same prompts"
        )

        mc_question_1a_copy = MultipleChoiceQuestion(**mc_question_1a.model_dump())
        self._assert_prompt_is_valid(mc_question_1a_copy, question, ideal, distractors)
        assert mc_question_1a == mc_question_1b, (
            "Serialization then deserialization should lead to same prompts"
        )

        mc_question_2a = MultipleChoiceQuestion(
            question=question,
            ideal_answer=ideal,
            options=distractors,
            shuffle_seed=MultipleChoiceQuestion.SEED_USING_QUESTION,
        )
        self._assert_prompt_is_valid(mc_question_2a, question, ideal, distractors)

        mc_question_2b = MultipleChoiceQuestion(
            question=question,
            ideal_answer=ideal,
            options=distractors,
            shuffle_seed=MultipleChoiceQuestion.SEED_USING_QUESTION,
        )
        self._assert_prompt_is_valid(mc_question_2b, question, ideal, distractors)
        assert mc_question_2a == mc_question_2b, (
            "Same seeding strategy should lead to same prompts"
        )
        assert mc_question_2a != mc_question_1a, (
            "Different seeding strategies should lead to different prompts"
        )


class TestMultipleChoiceEvaluation:
    @pytest.mark.parametrize(
        ("evals", "accuracy_precision"),
        [
            (
                [
                    MultipleChoiceEvaluation.CORRECT,
                    MultipleChoiceEvaluation.CORRECT,
                    MultipleChoiceEvaluation.CORRECT,
                ],
                (1, 1),
            ),
            (["correct", "correct", "unsure"], (2 / 3, 1)),
            (
                [
                    MultipleChoiceEvaluation.CORRECT,
                    MultipleChoiceEvaluation.UNSURE,
                    "incorrect",
                ],
                (1 / 3, 1 / 2),
            ),
        ],
    )
    def test_calculate_accuracy_precision(
        self,
        evals: Sequence[MultipleChoiceEvaluation],
        accuracy_precision: tuple[float, float],
    ) -> None:
        assert (
            MultipleChoiceEvaluation.calculate_accuracy_precision(evals)
            == accuracy_precision
        )
