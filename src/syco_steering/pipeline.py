"""End-to-end orchestration shared by both entrypoints (local and Modal)."""

from __future__ import annotations

import json
import os
import random

import numpy as np

from . import config
from .data import build_contrastive_examples, load_records
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
    """Full flow: load model -> load/filter data -> extract activations ->
    layer sweep -> extract direction -> validate. Persists artifacts to
    ``out_dir`` and returns the metrics dict.

    Artifacts:
      - steering_vector.npz  (v_hat, steering_vector, m, mu_pos, sig_pos,
                              delta_mu, best_layer, model)
      - metrics.json         (accs, best_layer, n_layers, test_acc, auroc,
                              caa_cosine, n_pairs)
      - layer_acc.png, projection_hist.png
    """
    _set_seeds(config.SEED)
    os.makedirs(out_dir, exist_ok=True)

    # Imported here (not at module top) so the package stays importable without
    # torch on machines that only run the probe/validate code or the tests.
    from .activations import extract_all, load_model

    model, tok = load_model(config.MODEL)
    records = load_records(config.DATASET_URL)
    examples = build_contrastive_examples(
        records, config.WRONG_BELIEF_TEMPLATE, config.N_PAIRS, config.SEED
    )
    if not examples:
        raise RuntimeError(
            "No contrastive examples after filtering — check WRONG_BELIEF_TEMPLATE "
            "against the dataset's metadata.prompt_template values."
        )

    H_syco, H_honest = extract_all(model, tok, examples)

    accs, best_layer = layer_sweep(H_honest, H_syco, config.SEED)
    direction = extract_direction(H_honest, H_syco, best_layer)
    val = validate(
        H_honest,
        H_syco,
        best_layer,
        direction,
        out_dir,
        accs=accs,
        sweep_seed=config.SEED,
    )

    n_layers = int(H_honest.shape[1])
    metrics = {
        "model": config.MODEL,
        "n_pairs": len(examples),
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
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics
