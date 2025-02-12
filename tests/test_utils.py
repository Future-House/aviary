import json
import random
from collections.abc import Iterable, Sequence
from copy import deepcopy
from typing import Annotated

import numpy as np
import pytest
from pydantic import BaseModel

from aviary.core import (
    MultipleChoiceEvaluation,
    MultipleChoiceQuestion,
    eval_answer,
    extract_answer,
)
from aviary.utils import RandomAnnotation, T, shuffle
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


@pytest.mark.parametrize(
    ("sequence", "seed", "expected"),
    [
        pytest.param((), None, [], id="empty-sequence"),
        pytest.param((1,), None, [1], id="single-element"),
        pytest.param("12345", 42, ["1", "5", "3", "2", "4"], id="string"),
        pytest.param(
            list(range(10)),
            random.Random(42),
            [1, 0, 4, 9, 6, 5, 8, 2, 3, 7],
            id="random-rng",
        ),
        pytest.param(
            list(range(10)),
            np.random.default_rng(42),
            [2, 9, 1, 6, 3, 8, 5, 7, 4, 0],
            id="numpy-rng",
        ),
    ],
)
def test_shuffle(
    sequence: Sequence[T],
    seed: int | random.Random | np.random.Generator | None,
    expected: Sequence[T],
) -> None:
    deepcopy_sequence = deepcopy(sequence)
    shuffled = shuffle(sequence, seed)
    assert sequence == deepcopy_sequence, "Should not mutate input"
    # Use length then element-wise comparison to work around numpy:
    # > The truth value of an array with more than one element is ambiguous.
    assert len(shuffled) == len(expected)
    assert all(v == e for v, e in zip(shuffled, expected, strict=True))


def test_random_annotation() -> None:
    class SomeModel(BaseModel):
        rng: Annotated[random.Random, RandomAnnotation()]

    model = SomeModel(rng=random.Random(5))
    assert isinstance(model.rng, random.Random)

    # 1. Manually check serialized RNG is expected
    for deserialized in (
        json.loads(model.model_dump_json()),  # JSON str
        model.model_dump(mode="json"),  # JSON dict
    ):
        rng_serialized = deserialized.pop("rng")
        assert not deserialized, "Expected only one key in the serialized model"
        version, internal_state, gauss_next = rng_serialized
        assert isinstance(version, int)
        assert isinstance(internal_state, list)
        assert isinstance(gauss_next, float | None)

    # 2. Check deserialized RNG behaves as original RNG
    for i, deserialized_model in enumerate((
        SomeModel.model_validate_json(model.model_dump_json()),  # JSON str
        SomeModel.model_validate(model.model_dump(mode="json")),  # JSON dict
    )):
        if i == 0:
            # Sample original model once so RNG aligns for both deserialized
            # models in the `for` loop
            sampled_original = model.rng.sample(list(range(10)), k=6)
        sampled_deserialized = deserialized_model.rng.sample(list(range(10)), k=6)
        assert sampled_original == sampled_deserialized, (
            "Deserialization seeding failed"
        )


class TestLitQAEvaluation:
    @staticmethod
    def _assert_prompt_is_valid(
        mc_question: MultipleChoiceQuestion,
        question: str,
        ideal_answer: str,
        distractors: Iterable[str],
        is_open_answer: bool = False,
    ) -> None:
        question_prompt = mc_question.question_prompt
        assert question_prompt.count(question) == 1
        for substr in ("Insufficient information", ideal_answer, *distractors):
            assert question_prompt.count(substr) == (1 if not is_open_answer else 0)

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

    def test_consistent_open_answer(self) -> None:
        question, ideal, distractors = self.MEANING_OF_LIFE_QUESTION_IDEAL_DISTRACTORS
        mc_question_1a = MultipleChoiceQuestion(
            question=question,
            ideal_answer=ideal,
            options=distractors,
            shuffle_seed=0,
            prompt_without_options=True,
        )
        self._assert_prompt_is_valid(
            mc_question_1a, question, ideal, distractors, is_open_answer=True
        )

        mc_question_1b = MultipleChoiceQuestion(
            question=question,
            ideal_answer=ideal,
            options=distractors,
            shuffle_seed=0,
            prompt_without_options=True,
        )
        self._assert_prompt_is_valid(
            mc_question_1b, question, ideal, distractors, is_open_answer=True
        )
        assert mc_question_1a == mc_question_1b, (
            "Same seeding should lead to same prompts"
        )

        mc_question_1a_copy = MultipleChoiceQuestion(**mc_question_1a.model_dump())
        self._assert_prompt_is_valid(
            mc_question_1a_copy, question, ideal, distractors, is_open_answer=True
        )
        assert mc_question_1a == mc_question_1a_copy == mc_question_1b, (
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
