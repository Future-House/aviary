from collections.abc import Iterable, Sequence

import pytest

from aviary.core import CorrectnessEvaluation, MultipleChoiceQuestion
from tests.conftest import VCR_DEFAULT_MATCH_ON


class TestMultipleChoice:
    @staticmethod
    def _assert_prompt_is_valid(
        mc_question: MultipleChoiceQuestion,
        question: str,
        ideal_answer: str,
        distractors: Iterable[str],
        has_no_options: bool = False,
    ) -> None:
        question_prompt = mc_question.question_prompt
        assert question_prompt.count(question) == 1
        for substr in (
            "Options",
            "Insufficient information",
            ideal_answer,
            *distractors,
        ):
            assert question_prompt.count(substr) == (1 if not has_no_options else 0)

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
                CorrectnessEvaluation.CORRECT,
                "94107",
                id="matched-correct-option",
            ),
            pytest.param(
                *ZIP_CODE_QUESTION_IDEAL_DISTRACTORS,
                "the answer is 14004",
                CorrectnessEvaluation.INCORRECT,
                None,
                id="didnt-match-and-no-llm-innate-knowledge",
            ),
            pytest.param(
                *ZIP_CODE_QUESTION_IDEAL_DISTRACTORS,
                "the answer is 94106",
                CorrectnessEvaluation.INCORRECT,
                "94106",
                id="matched-incorrect-option",
            ),
            pytest.param(
                *ZIP_CODE_QUESTION_IDEAL_DISTRACTORS,
                "Insufficient information",
                CorrectnessEvaluation.UNSURE,
                MultipleChoiceQuestion.DEFAULT_UNSURE_OPTION,
                id="matched-unsure-option",
            ),
            pytest.param(
                *ZIP_CODE_QUESTION_IDEAL_DISTRACTORS,
                "the answer is 94106 or 94107",
                CorrectnessEvaluation.INCORRECT,
                None,
                id="matched-several-options",
            ),
            pytest.param(
                *ZIP_CODE_QUESTION_IDEAL_DISTRACTORS,
                "",
                CorrectnessEvaluation.INCORRECT,
                None,
                id="empty-answer1",
            ),
            pytest.param(
                *MEANING_OF_LIFE_QUESTION_IDEAL_DISTRACTORS,
                "14",
                CorrectnessEvaluation.INCORRECT,
                None,
                id="didnt-match-and-llm-has-innate-knowledge",
            ),
            pytest.param(
                *MEANING_OF_LIFE_QUESTION_IDEAL_DISTRACTORS,
                "",
                CorrectnessEvaluation.INCORRECT,
                None,
                id="empty-answer2",
            ),
            pytest.param(
                *LITQA2_QUESTION_IDEAL_DISTRACTORS,
                "",
                CorrectnessEvaluation.INCORRECT,
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
        expected_eval: CorrectnessEvaluation,
        expected_extracted_answer: str | None,
    ) -> None:
        """Tests that we can create a multiple choice question and evaluate answers."""
        mc_question = MultipleChoiceQuestion(
            question=question,
            options=distractors,
            ideal_answer=ideal_answer,
            shuffle_seed=42,  # Seed for VCR cassette
        )
        try:
            self._assert_prompt_is_valid(
                mc_question, question, ideal_answer, distractors
            )
        except AssertionError as exc:
            _ = 0
        evaluation, metadata = await mc_question.grade(actual_answer)
        assert evaluation == expected_eval
        if evaluation == CorrectnessEvaluation.CORRECT:
            assert metadata["extracted_answer"] == ideal_answer
        assert metadata["extracted_answer"] == expected_extracted_answer

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
        assert mc_question_1a == mc_question_1a_copy == mc_question_1b, (
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
            "Question seeding strategy should lead to same prompts"
        )
        assert mc_question_2a != mc_question_1a, (
            "Different seeding strategies should lead to different prompts"
        )

    def test_no_options(self) -> None:
        question, ideal, _ = self.MEANING_OF_LIFE_QUESTION_IDEAL_DISTRACTORS
        mcq = MultipleChoiceQuestion(
            question=question,
            ideal_answer=ideal,
            shuffle_seed=0,
            prompt_without_options=True,
            options=[],
        )
        self._assert_prompt_is_valid(mcq, question, ideal, [], has_no_options=True)

        mcq_copy = MultipleChoiceQuestion(**mcq.model_dump())
        self._assert_prompt_is_valid(mcq_copy, question, ideal, [], has_no_options=True)
        assert mcq == mcq_copy, (
            "Serialization then deserialization should lead to same prompts"
        )

    @pytest.mark.parametrize(
        (
            "options",
            "ideal_answer",
            "unsure_answer",
            "seed",
            "expected_ideal_letter",
            "expected_unsure_letter",
        ),
        [
            # Test cases for ideal and unsure answer letters
            (["A", "B"], "C", "Not sure", 42, "D", "B"),  # With seed 42
            (["X", "Y"], "Z", "Unsure", 0, "D", "A"),  # With seed 0
            (["A", "B", "C"], "B", None, 42, "C", None),  # Ideal answer in options
            (
                ["D", "E", "F"],
                "E",
                MultipleChoiceQuestion.DEFAULT_UNSURE_OPTION,
                0,
                "B",
                "A",
            ),
            (
                ["A", "B", "Not sure"],
                "C",
                "Not sure",
                0,
                "A",
                "D",
            ),  # Unsure answer in options
        ],
    )
    def test_answer_letters(
        self,
        options: list[str],
        ideal_answer: str,
        unsure_answer: str | None,
        seed: int,
        expected_ideal_letter: str,
        expected_unsure_letter: str | None,
    ) -> None:
        """Test that ideal_answer_letter and unsure_answer_letter return correct letters after shuffling."""
        mc_question = MultipleChoiceQuestion(
            question="test question",
            options=options,
            ideal_answer=ideal_answer,
            unsure_answer=unsure_answer,
            shuffle_seed=seed,  # Use specific seeds for predictable shuffling
        )
        # Check ideal answer letter
        assert mc_question.ideal_answer_letter == expected_ideal_letter
        assert ideal_answer in mc_question.options

        # Check unsure answer letter
        assert mc_question.unsure_answer_letter == expected_unsure_letter
        if unsure_answer is not None:
            assert unsure_answer in mc_question.options


class TestCorrectnessEvaluation:
    @pytest.mark.parametrize(
        ("evals", "accuracy_precision"),
        [
            (
                [
                    CorrectnessEvaluation.CORRECT,
                    CorrectnessEvaluation.CORRECT,
                    CorrectnessEvaluation.CORRECT,
                ],
                (1, 1),
            ),
            (["correct", "correct", "unsure"], (2 / 3, 1)),
            (
                [
                    CorrectnessEvaluation.CORRECT,
                    CorrectnessEvaluation.UNSURE,
                    "incorrect",
                ],
                (1 / 3, 1 / 2),
            ),
        ],
    )
    def test_calculate_accuracy_precision(
        self,
        evals: Sequence[CorrectnessEvaluation],
        accuracy_precision: tuple[float, float],
    ) -> None:
        assert (
            CorrectnessEvaluation.calculate_accuracy_precision(evals)
            == accuracy_precision
        )
