"""Tests for template filtering and contrastive-example construction (no network)."""

from syco_steering import config
from syco_steering.data import build_contrastive_examples

TEMPLATE = config.WRONG_BELIEF_TEMPLATE


def _record(template, *, correct="Right", incorrect="Wrong", user="Q? hint."):
    return {
        "prompt": [{"type": "human", "content": user}],
        "base": {
            "dataset": "trivia_qa",
            "question": "Q?",
            "answer": [correct],
            "correct_answer": correct,
            "incorrect_answer": incorrect,
        },
        "metadata": {"prompt_template": template},
    }


def test_filters_to_target_template():
    records = [
        _record(TEMPLATE, user="A"),
        _record("{question}", user="B"),
        _record("other template", user="C"),
        _record(TEMPLATE, user="D"),
    ]
    examples = build_contrastive_examples(records, TEMPLATE, n=10, seed=0)
    assert len(examples) == 2
    users = {e["user"] for e in examples}
    assert users == {"A", "D"}


def test_drops_records_missing_answers():
    good = _record(TEMPLATE, user="good")
    missing_incorrect = _record(TEMPLATE, user="bad")
    missing_incorrect["base"]["incorrect_answer"] = ""
    examples = build_contrastive_examples([good, missing_incorrect], TEMPLATE, n=10, seed=0)
    assert len(examples) == 1
    assert examples[0]["user"] == "good"


def test_completions_reference_correct_and_incorrect():
    records = [_record(TEMPLATE, correct="Paris", incorrect="London")]
    [ex] = build_contrastive_examples(records, TEMPLATE, n=1, seed=0)
    assert "London" in ex["syco"] and "Paris" not in ex["syco"]
    assert "Paris" in ex["honest"] and "London" not in ex["honest"]
    # User turn is taken verbatim, not reconstructed.
    assert ex["user"] == "Q? hint."


def test_cap_and_determinism():
    records = [_record(TEMPLATE, user=f"u{i}") for i in range(50)]
    a = build_contrastive_examples(records, TEMPLATE, n=10, seed=0)
    b = build_contrastive_examples(records, TEMPLATE, n=10, seed=0)
    assert len(a) == 10
    assert [e["user"] for e in a] == [e["user"] for e in b]  # same seed -> same order
