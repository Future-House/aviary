import json
import random
from collections.abc import Sequence
from copy import deepcopy
from typing import Annotated, Any

import numpy as np
import pytest
from pydantic import BaseModel

from aviary.core import eval_answer, extract_answer
from aviary.utils import RandomAnnotation, T, partial_format, shuffle


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
            "Which of the following is most likely true:\n\nA) Piggie, B) Pigeon, C)"
            " Gerald\n",
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
            "Based on the context given, Serif et al. (2026) claim that the"
            " overwhelming cause of regime collapse arises from economic factors. Yet,"
            " most other scholars (Gerald and Robinson for example) believe the"
            " collapse was due to social unrest because of the prolonged epidemic of"
            " 2025. I tend to agree with the majority - although I can see both sides."
            " Thus my response is that the social unrest was the significant factor in"
            " the collapse of the regime.",
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
        # Include str so we can test failing over for non-Random values
        rng: Annotated[random.Random, RandomAnnotation()] | str

    model = SomeModel(rng="SEED_SENTINEL")
    assert model.rng == "SEED_SENTINEL"

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
        assert isinstance(deserialized_model.rng, random.Random)
        sampled_deserialized = deserialized_model.rng.sample(list(range(10)), k=6)
        assert sampled_original == sampled_deserialized, (
            "Deserialization seeding failed"
        )


@pytest.mark.parametrize(
    ("value", "formats", "expected"),
    [
        pytest.param("Hi {name}", {"name": "Alice"}, "Hi Alice", id="single-var"),
        pytest.param("({x}, {y})", {"x": 10, "y": 20}, "(10, 20)", id="two-vars"),
        pytest.param(
            "Hi {fname} {lname}",
            {"fname": "Bob"},
            "Hi Bob {lname}",
            id="one-of-two-vars",
        ),
        pytest.param("String.", {"unused": "value"}, "String.", id="not-a-template"),
        pytest.param(
            "Hi {fname} {mname} {lname}",
            {"fname": "Bob", "lname": "Bobson"},
            "Hi Bob {mname} Bobson",
            id="two-of-three-vars",
        ),
    ],
)
def test_partial_format(value: str, formats: dict[str, Any], expected: str) -> None:
    assert partial_format(value, **formats) == expected
