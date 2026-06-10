"""Held-out metrics, projection histogram, and the CAA-cosine consistency check.

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

from .probe import _group_split, stack_tokens  # noqa: E402


def _caa_direction(X_honest: np.ndarray, X_syco: np.ndarray, layer: int) -> np.ndarray:
    """Contrastive-activation-addition direction: normalized mean difference
    (honest - syco) over tokens at ``layer``."""
    d = X_honest[:, layer].astype(np.float32).mean(0) - X_syco[:, layer].astype(
        np.float32
    ).mean(0)
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
    """Compute held-out metrics and save plots to ``out_dir``.

    (a) held-out test accuracy + AUROC at ``layer`` on a fresh GROUP-aware split
        with a different seed than the sweep. Caveat: the layer itself was
        selected on this same data, so the reported numbers carry a mild
        selection-bias optimism; treat them as sanity checks, not unbiased
        estimates.
    (b) per-token projection histogram of both classes with boundary ``m``
        -> projection_hist.png,
    (c) cosine(v_hat, CAA mean-difference direction) -> consistency. Both
        vectors come from the same activations, so this checks probe/CAA
        agreement only — it cannot detect confounds shared by both.

    Also saves the layer-sweep accuracy curve -> layer_acc.png (pass ``accs`` in;
    they are recomputed if omitted).
    """
    os.makedirs(out_dir, exist_ok=True)

    # (a) Fresh held-out split with a seed distinct from the sweep's.
    X, y, groups = stack_tokens(X_honest, X_syco, g_honest, g_syco, layer)
    X_tr, X_te, y_tr, y_te = _group_split(X, y, groups, sweep_seed + 1)
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(X_tr, y_tr)
    test_acc = float(clf.score(X_te, y_te))
    auroc = float(roc_auc_score(y_te, clf.decision_function(X_te)))

    # (c) CAA cosine consistency check.
    v_hat = np.asarray(direction["v_hat"], dtype=np.float64)
    caa = _caa_direction(X_honest, X_syco, layer)
    caa_cosine = float(np.dot(v_hat, caa) / (np.linalg.norm(v_hat) * np.linalg.norm(caa)))

    # (b) Per-token projection histogram with the decision boundary m.
    proj_honest = X_honest[:, layer].astype(np.float32) @ v_hat
    proj_syco = X_syco[:, layer].astype(np.float32) @ v_hat
    m = float(direction["m"])
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(
        min(proj_honest.min(), proj_syco.min()),
        max(proj_honest.max(), proj_syco.max()),
        40,
    )
    ax.hist(proj_syco, bins=bins, alpha=0.6, label="sycophantic tokens (0)", color="tab:red")
    ax.hist(proj_honest, bins=bins, alpha=0.6, label="honest tokens (1)", color="tab:blue")
    ax.axvline(m, color="k", linestyle="--", label=f"boundary m = {m:.2f}")
    ax.set_xlabel(r"token projection onto $\hat{v}$")
    ax.set_ylabel("count")
    ax.set_title(f"Per-token projection separation at layer {layer}")
    ax.legend()
    fig.tight_layout()
    proj_path = os.path.join(out_dir, "projection_hist.png")
    fig.savefig(proj_path, dpi=120)
    plt.close(fig)

    # Layer-sweep curve.
    if accs is None:
        from .probe import layer_sweep

        accs, _ = layer_sweep(X_honest, X_syco, g_honest, g_syco, sweep_seed)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(accs)), accs, marker="o")
    ax.axvline(layer, color="tab:green", linestyle="--", label=f"selected layer {layer}")
    ax.set_xlabel("hidden_states layer index")
    ax.set_ylabel("held-out token accuracy")
    ax.set_title("Layer sweep (per-token probe accuracy)")
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
