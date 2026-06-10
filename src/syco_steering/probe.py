"""Layer sweep and steering-direction / boundary extraction via logistic regression.

torch-free: operates purely on numpy TOKEN-level activation arrays of shape
``(T, num_layers + 1, hidden)`` so it can be unit-tested on synthetic data.

The probe is fit on individual response-token activations (not pooled examples)
because the inference-time gate in StTP/StMP compares individual token states to
the boundary ``m`` — the boundary and the projection statistics must be
calibrated on the same per-token distribution they are later applied to.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit


def stack_tokens(
    X_honest: np.ndarray,
    X_syco: np.ndarray,
    g_honest: np.ndarray,
    g_syco: np.ndarray,
    layer: int,
):
    """Build (X, y, groups) at a layer. honest = class 1, syco = class 0.

    Group ids of the two classes are kept disjoint (the honest and sycophantic
    response to the same prompt are different texts, but we still never want
    tokens of one response split across train and test).
    """
    X = np.concatenate(
        [X_honest[:, layer].astype(np.float32), X_syco[:, layer].astype(np.float32)],
        axis=0,
    )
    y = np.concatenate([np.ones(len(X_honest)), np.zeros(len(X_syco))]).astype(int)
    offset = int(g_honest.max()) + 1 if len(g_honest) else 0
    groups = np.concatenate([g_honest, g_syco + offset])
    return X, y, groups


def _group_split(X, y, groups, seed):
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed)
    tr, te = next(splitter.split(X, y, groups))
    return X[tr], X[te], y[tr], y[te]


def _subsample(X, y, groups, max_tokens, seed):
    if max_tokens is None or len(X) <= max_tokens:
        return X, y, groups
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=max_tokens, replace=False)
    return X[idx], y[idx], groups[idx]


def layer_sweep(
    X_honest: np.ndarray,
    X_syco: np.ndarray,
    g_honest: np.ndarray,
    g_syco: np.ndarray,
    seed: int,
    max_tokens: int | None = None,
) -> tuple[list[float], int]:
    """Fit a probe per layer on a 75/25 GROUP-aware split (groups = response id,
    so tokens of one response never straddle the split); return per-layer
    held-out accuracies and the index of the best layer (honest=1, syco=0).

    ``max_tokens`` caps the number of tokens per fit (random subsample) to keep
    the sweep fast; the final direction is refit on all tokens.
    """
    n_layers = X_honest.shape[1]
    accs: list[float] = []
    for layer in range(n_layers):
        X, y, groups = stack_tokens(X_honest, X_syco, g_honest, g_syco, layer)
        X, y, groups = _subsample(X, y, groups, max_tokens, seed)
        X_tr, X_te, y_tr, y_te = _group_split(X, y, groups, seed)
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(X_tr, y_tr)
        accs.append(float(clf.score(X_te, y_te)))
    best_index = int(np.argmax(accs))
    return accs, best_index


def extract_direction(
    X_honest: np.ndarray, X_syco: np.ndarray, layer: int
) -> dict:
    """Refit the probe on ALL tokens at ``layer`` and extract the steering geometry.

    Returns a dict with:
      - ``v_hat``         unit direction toward honest (class 1),
      - ``m``             decision boundary along v_hat = -b / ||w||,
      - ``mu_pos``        mean honest TOKEN projection onto v_hat,
      - ``sig_pos``       std of honest token projection onto v_hat,
      - ``delta_mu``      mean token projection gap (honest mean - syco mean),
      - ``best_layer``    the hidden_states layer index used,
      - ``steering_vector`` = v_hat * delta_mu.
    """
    X = np.concatenate(
        [X_honest[:, layer].astype(np.float32), X_syco[:, layer].astype(np.float32)],
        axis=0,
    )
    y = np.concatenate([np.ones(len(X_honest)), np.zeros(len(X_syco))]).astype(int)
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(X, y)

    w = clf.coef_[0]
    b = float(clf.intercept_[0])
    norm = float(np.linalg.norm(w))
    v_hat = w / norm  # points toward honest (class 1)
    m = -b / norm

    proj_honest = X_honest[:, layer].astype(np.float32) @ v_hat
    proj_syco = X_syco[:, layer].astype(np.float32) @ v_hat
    mu_pos = float(proj_honest.mean())
    sig_pos = float(proj_honest.std())
    delta_mu = float(proj_honest.mean() - proj_syco.mean())

    return {
        "v_hat": v_hat.astype(np.float32),
        "m": float(m),
        "mu_pos": mu_pos,
        "sig_pos": sig_pos,
        "delta_mu": delta_mu,
        "best_layer": int(layer),
        "steering_vector": (v_hat * delta_mu).astype(np.float32),
    }
