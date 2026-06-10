"""Load the sycophancy `answer.jsonl` dataset and build prompt sets.

This module is intentionally torch-free so it can be exercised locally and in
tests without a model or GPU.

The contrastive *texts* are no longer templated completions: following the paper
(arXiv 2604.08169), the probe is trained on the model's own responses generated
under a sycophancy-inducing vs. an honesty-inducing system prompt (see
`activations.generate_responses`). This module only selects and splits the user
prompts those responses are generated for.
"""

from __future__ import annotations

import json
import random
import ssl
import urllib.request


def _ssl_context() -> ssl.SSLContext | None:
    """Use certifi's CA bundle when available (avoids the macOS framework-Python
    'CERTIFICATE_VERIFY_FAILED' quirk). Returns None to fall back to defaults."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return None


def load_records(url: str) -> list[dict]:
    """Download `answer.jsonl` and parse one JSON object per line.

    Top-level keys per record: `prompt`, `base`, `metadata`.
    """
    with urllib.request.urlopen(url, context=_ssl_context()) as resp:
        text = resp.read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def build_prompt_sets(
    records: list[dict],
    template: str,
    n_train: int,
    n_eval: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Filter to the target template and split into train / eval prompt sets.

    Keep only rows whose ``metadata.prompt_template == template`` AND that carry
    both ``base.correct_answer`` and ``base.incorrect_answer``. Shuffle with
    ``seed``, then take the first ``n_train`` rows as the probe-training prompts
    and the next ``n_eval`` rows as the disjoint held-out steering-eval prompts.

    Each returned dict is ``{"user", "correct", "incorrect"}`` where ``user`` is
    the human turn(s) of ``record["prompt"]`` taken verbatim (it already states
    the wrong belief; do not reconstruct it).
    """
    kept = [
        r
        for r in records
        if r.get("metadata", {}).get("prompt_template") == template
        and r.get("base", {}).get("correct_answer")
        and r.get("base", {}).get("incorrect_answer")
    ]
    rng = random.Random(seed)
    rng.shuffle(kept)

    def to_prompt(r: dict) -> dict:
        return {
            "user": "\n\n".join(
                turn["content"] for turn in r["prompt"] if turn.get("type") == "human"
            ),
            "correct": r["base"]["correct_answer"],
            "incorrect": r["base"]["incorrect_answer"],
        }

    train = [to_prompt(r) for r in kept[:n_train]]
    evalset = [to_prompt(r) for r in kept[n_train : n_train + n_eval]]
    return train, evalset
