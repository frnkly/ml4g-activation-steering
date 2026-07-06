"""Held-out metrics, projection histograms, and the CAA-cosine consistency check.

torch-free. Uses a non-interactive matplotlib backend so it works headless (Modal).
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402

from .probe import pool_by_response  # noqa: E402


def _caa_direction(E_pos: np.ndarray, E_neg: np.ndarray) -> np.ndarray:
    """Contrastive-activation-addition direction: normalized mean difference
    (honest - syco) over the response-averaged embeddings."""
    d = E_pos.mean(axis=0) - E_neg.mean(axis=0)
    return d / np.linalg.norm(d)


def validate(
    X_honest: np.ndarray,
    X_syco: np.ndarray,
    g_honest: np.ndarray,
    g_syco: np.ndarray,
    layer: int,
    direction: dict,
    out_dir: str,
    accs: list[float] | None = None,
    sweep_seed: int = 0,
) -> dict:
    """Compute held-out metrics for the POOLED probe and save plots to ``out_dir``.

    (a) held-out test accuracy + AUROC at ``layer`` on a 75/25 split of the
        response-averaged embeddings, grouped by PROMPT id so the honest and
        sycophantic response to the same prompt never straddle the split.
        Caveat: the layer itself was selected on this same data, so the
        reported numbers carry a mild selection-bias optimism; treat them as
        sanity checks, not unbiased estimates.
    (b) projection histograms with the boundary ``m`` -> projection_hist.png.
        Two panels: pooled projections (the probe's training geometry, paper
        Eq. 1) and per-token projections (the wider distribution the StTP/StMP
        gate sees at inference). Both straddle the same boundary because
        pooling changes the class variances, not the class means.
    (c) cosine(v_hat, CAA mean-difference direction) -> consistency (paper
        section A.3). Both vectors come from the same activations, so this
        checks probe/CAA agreement only — it cannot detect confounds shared
        by both.

    Also saves the token-level layer-sweep accuracy curve -> layer_acc.png
    (pass ``accs`` in; they are recomputed if omitted).
    """
    os.makedirs(out_dir, exist_ok=True)

    E_pos = pool_by_response(X_honest, g_honest, layer)
    E_neg = pool_by_response(X_syco, g_syco, layer)
    X = np.concatenate([E_pos, E_neg], axis=0)
    y = np.concatenate([np.ones(len(E_pos)), np.zeros(len(E_neg))]).astype(int)
    # Same prompt id for a prompt's honest and syco example -> the pair stays
    # on one side of the split (prompt-content leakage guard).
    prompt_ids = np.concatenate([np.unique(g_honest), np.unique(g_syco)])

    # (a) Held-out split with a seed distinct from the sweep's.
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=sweep_seed + 1)
    tr, te = next(splitter.split(X, y, prompt_ids))
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(X[tr], y[tr])
    test_acc = float(clf.score(X[te], y[te]))
    auroc = float(roc_auc_score(y[te], clf.decision_function(X[te])))

    # (c) CAA cosine consistency check (on the pooled embeddings).
    v_hat = np.asarray(direction["v_hat"], dtype=np.float64)
    caa = _caa_direction(E_pos, E_neg)
    caa_cosine = float(np.dot(v_hat, caa) / (np.linalg.norm(v_hat) * np.linalg.norm(caa)))

    # (b) Pooled + per-token projection histograms with the decision boundary m.
    m = float(direction["m"])
    panels = {
        "response-averaged (probe training)": (E_pos @ v_hat, E_neg @ v_hat),
        "per-token (what the gate sees)": (
            X_honest[:, layer].astype(np.float32) @ v_hat,
            X_syco[:, layer].astype(np.float32) @ v_hat,
        ),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (name, (p_pos, p_neg)) in zip(axes, panels.items()):
        bins = np.linspace(
            min(p_pos.min(), p_neg.min()), max(p_pos.max(), p_neg.max()), 40
        )
        ax.hist(p_neg, bins=bins, alpha=0.6, label="sycophantic (0)", color="tab:red")
        ax.hist(p_pos, bins=bins, alpha=0.6, label="honest (1)", color="tab:blue")
        ax.axvline(m, color="k", linestyle="--", label=f"boundary m = {m:.2f}")
        ax.set_xlabel(r"projection onto $\hat{v}$")
        ax.set_ylabel("count")
        ax.set_title(f"{name}, layer {layer}")
        ax.legend()
    fig.tight_layout()
    proj_path = os.path.join(out_dir, "projection_hist.png")
    fig.savefig(proj_path, dpi=120)
    plt.close(fig)

    # Layer-sweep curve (token-level shortlist heuristic).
    if accs is None:
        from .probe import layer_sweep

        accs, _ = layer_sweep(X_honest, X_syco, g_honest, g_syco, sweep_seed)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(accs)), accs, marker="o")
    ax.axvline(layer, color="tab:green", linestyle="--", label=f"selected layer {layer}")
    ax.set_xlabel("hidden_states layer index")
    ax.set_ylabel("held-out token accuracy")
    ax.set_title("Layer sweep (per-token shortlist accuracy)")
    ax.legend()
    fig.tight_layout()
    layer_path = os.path.join(out_dir, "layer_acc.png")
    fig.savefig(layer_path, dpi=120)
    plt.close(fig)

    return {
        "test_acc": test_acc,
        "auroc": auroc,
        "caa_cosine": caa_cosine,
    }
