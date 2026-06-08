"""Load the sycophancy `answer.jsonl` dataset and build contrastive examples.

This module is intentionally torch-free so it can be exercised locally and in
tests without a model or GPU.
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

    Top-level keys per record: `prompt`, `base`, `metadata` (see spec §6).
    """
    with urllib.request.urlopen(url, context=_ssl_context()) as resp:
        text = resp.read().decode("utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def build_contrastive_examples(
    records: list[dict],
    template: str,
    n: int,
    seed: int,
) -> list[dict]:
    """Filter to the target template, then build contrastive completions.

    Keep only rows whose ``metadata.prompt_template == template`` AND that carry
    both ``base.correct_answer`` and ``base.incorrect_answer``. Shuffle with
    ``seed`` and take the first ``n``.

    Returns dicts ``{"user", "syco", "honest"}`` where:
      - ``user``   is the human turn(s) of ``record["prompt"]`` (already states the
                   wrong belief verbatim; do not reconstruct it),
      - ``syco``   affirms the user's incorrect answer,
      - ``honest`` states the correct answer.

    The two completions share an identical frame ("The answer is {x}.") and differ
    only in the answer named. This is deliberate: an earlier version used distinct
    openers ("Yes, you're right ..." vs "Actually, that's not correct ..."), which
    made the class label readable from the surface tokens — the probe then hit 100%
    accuracy at the embedding layer (hidden_states[0]) and the chosen layer failed
    the middle-third guard (spec §9 #5). With a matched frame, the discriminating
    answer token is a different arbitrary entity per example, so separation can only
    come from the *contextualized* representation that relates the stated answer to
    the user's claim — i.e. the mid-network "stance", which is what we want to steer.
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
    kept = kept[:n]

    examples: list[dict] = []
    for r in kept:
        user = "\n\n".join(
            turn["content"] for turn in r["prompt"] if turn.get("type") == "human"
        )
        correct = r["base"]["correct_answer"]
        incorrect = r["base"]["incorrect_answer"]
        examples.append(
            {
                "user": user,
                # Surface-matched frame: same wording, only the answer differs, so
                # the class can't be read off the literal tokens. See the docstring.
                "syco": f"The answer is {incorrect}.",
                "honest": f"The answer is {correct}.",
            }
        )
    return examples
