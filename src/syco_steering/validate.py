"""Held-out metrics, projection histogram, and the CAA-cosine robustness check.

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
from sklearn.model_selection import train_test_split  # noqa: E402

from .probe import _stack_xy  # noqa: E402


def _caa_direction(H_honest: np.ndarray, H_syco: np.ndarray, layer: int) -> np.ndarray:
    """Contrastive-activation-addition direction: normalized mean difference
    (honest - syco) at ``layer``."""
    d = H_honest[:, layer].mean(0) - H_syco[:, layer].mean(0)
    return d / np.linalg.norm(d)


def validate(
    H_honest: np.ndarray,
    H_syco: np.ndarray,
    layer: int,
    direction: dict,
    out_dir: str,
    accs: list[float] | None = None,
    sweep_seed: int = 0,
) -> dict:
    """Compute held-out metrics and save plots to ``out_dir``.

    (a) held-out test accuracy + AUROC at ``layer`` on a fresh stratified split
        with a different seed than the sweep,
    (b) projection histogram of both classes with boundary ``m`` -> projection_hist.png,
    (c) cosine(v_hat, CAA mean-difference direction) -> robustness.

    Also saves the layer-sweep accuracy curve -> layer_acc.png (pass ``accs`` in;
    they are recomputed per layer if omitted).
    """
    os.makedirs(out_dir, exist_ok=True)

    # (a) Fresh held-out split with a seed distinct from the sweep's.
    X, y = _stack_xy(H_honest, H_syco, layer)
    test_seed = sweep_seed + 1
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=test_seed
    )
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(X_tr, y_tr)
    test_acc = float(clf.score(X_te, y_te))
    auroc = float(roc_auc_score(y_te, clf.decision_function(X_te)))

    # (c) CAA cosine robustness check.
    v_hat = np.asarray(direction["v_hat"], dtype=np.float64)
    caa = _caa_direction(H_honest, H_syco, layer)
    caa_cosine = float(np.dot(v_hat, caa) / (np.linalg.norm(v_hat) * np.linalg.norm(caa)))

    # (b) Projection histogram with the decision boundary m.
    proj_honest = H_honest[:, layer] @ v_hat
    proj_syco = H_syco[:, layer] @ v_hat
    m = float(direction["m"])
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(
        min(proj_honest.min(), proj_syco.min()),
        max(proj_honest.max(), proj_syco.max()),
        30,
    )
    ax.hist(proj_syco, bins=bins, alpha=0.6, label="sycophantic (0)", color="tab:red")
    ax.hist(proj_honest, bins=bins, alpha=0.6, label="honest (1)", color="tab:blue")
    ax.axvline(m, color="k", linestyle="--", label=f"boundary m = {m:.2f}")
    ax.set_xlabel(r"projection onto $\hat{v}$")
    ax.set_ylabel("count")
    ax.set_title(f"Projection separation at layer {layer}")
    ax.legend()
    fig.tight_layout()
    proj_path = os.path.join(out_dir, "projection_hist.png")
    fig.savefig(proj_path, dpi=120)
    plt.close(fig)

    # Layer-sweep curve.
    if accs is None:
        from .probe import layer_sweep

        accs, _ = layer_sweep(H_honest, H_syco, sweep_seed)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(range(len(accs)), accs, marker="o")
    ax.axvline(layer, color="tab:green", linestyle="--", label=f"selected layer {layer}")
    ax.set_xlabel("hidden_states layer index")
    ax.set_ylabel("held-out accuracy")
    ax.set_title("Layer sweep (probe accuracy)")
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
