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
      - ``honest`` corrects the user with the correct answer.
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
                "syco": f"Yes, you're right — the answer is {incorrect}.",
                "honest": f"Actually, that's not correct. The answer is {correct}.",
            }
        )
    return examples
