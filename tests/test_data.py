"""Tests for template filtering and prompt-set construction (no network)."""

from syco_steering import config
from syco_steering.data import build_prompt_sets

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
    train, evalset = build_prompt_sets(records, TEMPLATE, n_train=10, n_eval=10, seed=0)
    assert len(train) + len(evalset) == 2
    users = {p["user"] for p in train + evalset}
    assert users == {"A", "D"}


def test_drops_records_missing_answers():
    good = _record(TEMPLATE, user="good")
    missing_incorrect = _record(TEMPLATE, user="bad")
    missing_incorrect["base"]["incorrect_answer"] = ""
    train, evalset = build_prompt_sets(
        [good, missing_incorrect], TEMPLATE, n_train=10, n_eval=10, seed=0
    )
    assert len(train) + len(evalset) == 1
    assert (train + evalset)[0]["user"] == "good"


def test_prompts_carry_answers_verbatim():
    records = [_record(TEMPLATE, correct="Paris", incorrect="London")]
    train, _ = build_prompt_sets(records, TEMPLATE, n_train=1, n_eval=0, seed=0)
    [p] = train
    assert p["correct"] == "Paris"
    assert p["incorrect"] == "London"
    # User turn is taken verbatim, not reconstructed.
    assert p["user"] == "Q? hint."


def test_split_sizes_disjoint_and_deterministic():
    records = [_record(TEMPLATE, user=f"u{i}") for i in range(50)]
    train_a, eval_a = build_prompt_sets(records, TEMPLATE, n_train=10, n_eval=5, seed=0)
    train_b, eval_b = build_prompt_sets(records, TEMPLATE, n_train=10, n_eval=5, seed=0)
    assert len(train_a) == 10 and len(eval_a) == 5
    # Train and eval are disjoint.
    assert {p["user"] for p in train_a}.isdisjoint({p["user"] for p in eval_a})
    # Same seed -> same order.
    assert [p["user"] for p in train_a] == [p["user"] for p in train_b]
    assert [p["user"] for p in eval_a] == [p["user"] for p in eval_b]
