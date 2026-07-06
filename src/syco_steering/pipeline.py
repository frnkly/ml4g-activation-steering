"""End-to-end orchestration shared by both entrypoints (local and Modal)."""

from __future__ import annotations

import json
import os
import random

import numpy as np

from . import config
from .data import build_prompt_sets, load_records
from .probe import extract_direction, layer_sweep
from .validate import validate


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def run_pipeline(out_dir: str) -> dict:
    """Full flow: load model -> load/filter data -> generate on-policy
    contrastive responses (sycophantic vs. honest system prompt) -> per-token
    activations -> token-level layer sweep (shortlist) -> pooled direction
    extraction (paper Eq. 1) -> validate. Persists artifacts to ``out_dir``
    and returns the metrics dict.

    Artifacts:
      - steering_vector.npz  (v_hat, steering_vector, m, mu_pos, sig_pos,
                              delta_mu, best_layer, model)
      - responses.json       (the generated training responses, for inspection)
      - metrics.json         (accs, best_layer, n_layers, test_acc, auroc,
                              caa_cosine, n_prompts, n_tokens per class)
      - layer_acc.png, projection_hist.png
    """
    _set_seeds(config.SEED)
    os.makedirs(out_dir, exist_ok=True)

    # Imported here (not at module top) so the package stays importable without
    # torch on machines that only run the probe/validate code or the tests.
    from .activations import extract_token_acts, generate_responses, load_model

    model, tok = load_model(config.MODEL)
    records = load_records(config.DATASET_URL)
    train_prompts, _ = build_prompt_sets(
        records,
        config.WRONG_BELIEF_TEMPLATE,
        config.N_TRAIN_PROMPTS,
        config.N_EVAL_PROMPTS,
        config.SEED,
    )
    if not train_prompts:
        raise RuntimeError(
            "No prompts after filtering — check WRONG_BELIEF_TEMPLATE "
            "against the dataset's metadata.prompt_template values."
        )

    resp_syco = generate_responses(
        model, tok, train_prompts, config.SYSTEM_SYCO,
        config.MAX_NEW_TRAIN, config.GEN_BATCH_SIZE,
    )
    resp_honest = generate_responses(
        model, tok, train_prompts, config.SYSTEM_HONEST,
        config.MAX_NEW_TRAIN, config.GEN_BATCH_SIZE,
    )

    X_syco, g_syco = extract_token_acts(
        model, tok, train_prompts, resp_syco, config.SYSTEM_SYCO
    )
    X_honest, g_honest = extract_token_acts(
        model, tok, train_prompts, resp_honest, config.SYSTEM_HONEST
    )

    accs, best_layer = layer_sweep(
        X_honest, X_syco, g_honest, g_syco, config.SEED,
        max_tokens=config.SWEEP_MAX_TOKENS,
    )
    direction = extract_direction(X_honest, X_syco, g_honest, g_syco, best_layer)
    val = validate(
        X_honest,
        X_syco,
        g_honest,
        g_syco,
        best_layer,
        direction,
        out_dir,
        accs=accs,
        sweep_seed=config.SEED,
    )

    n_layers = int(X_honest.shape[1])
    metrics = {
        "model": config.MODEL,
        "n_prompts": len(train_prompts),
        "n_tokens_honest": int(len(X_honest)),
        "n_tokens_syco": int(len(X_syco)),
        "n_layers": n_layers,
        "accs": [float(a) for a in accs],
        "best_layer": int(best_layer),
        "test_acc": val["test_acc"],
        "auroc": val["auroc"],
        "caa_cosine": val["caa_cosine"],
    }

    # Persist artifacts.
    np.savez(
        os.path.join(out_dir, "steering_vector.npz"),
        v_hat=direction["v_hat"],
        steering_vector=direction["steering_vector"],
        m=np.float32(direction["m"]),
        mu_pos=np.float32(direction["mu_pos"]),
        sig_pos=np.float32(direction["sig_pos"]),
        delta_mu=np.float32(direction["delta_mu"]),
        best_layer=np.int64(direction["best_layer"]),
        model=config.MODEL,
    )
    with open(os.path.join(out_dir, "responses.json"), "w") as f:
        json.dump(
            {
                "prompts": train_prompts,
                "syco_responses": resp_syco,
                "honest_responses": resp_honest,
            },
            f,
            indent=2,
        )
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
