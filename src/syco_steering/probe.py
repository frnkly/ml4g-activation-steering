"""Layer sweep and steering-direction / boundary extraction via logistic regression.

torch-free: operates purely on numpy activation arrays so it can be unit-tested on
synthetic data without a model.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def _stack_xy(H_honest: np.ndarray, H_syco: np.ndarray, layer: int):
    """Build (X, y) at a layer. honest = class 1, syco = class 0."""
    X = np.concatenate([H_honest[:, layer], H_syco[:, layer]], axis=0)
    y = np.concatenate(
        [np.ones(len(H_honest)), np.zeros(len(H_syco))]
    ).astype(int)
    return X, y


def layer_sweep(
    H_honest: np.ndarray, H_syco: np.ndarray, seed: int
) -> tuple[list[float], int]:
    """Fit a probe per layer on a 75/25 stratified split; return per-layer
    held-out accuracies and the index of the best layer (honest=1, syco=0)."""
    n_layers = H_honest.shape[1]
    accs: list[float] = []
    for layer in range(n_layers):
        X, y = _stack_xy(H_honest, H_syco, layer)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=seed
        )
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(X_tr, y_tr)
        accs.append(float(clf.score(X_te, y_te)))
    best_index = int(np.argmax(accs))
    return accs, best_index


def extract_direction(
    H_honest: np.ndarray, H_syco: np.ndarray, layer: int
) -> dict:
    """Refit the probe on ALL data at ``layer`` and extract the steering geometry.

    Returns a dict with:
      - ``v_hat``         unit direction toward honest (class 1),
      - ``m``             decision boundary along v_hat = -b / ||w||,
      - ``mu_pos``        mean honest projection onto v_hat,
      - ``sig_pos``       std of honest projection onto v_hat,
      - ``delta_mu``      mean projection gap (honest mean - syco mean),
      - ``best_layer``    the hidden_states layer index used,
      - ``steering_vector`` = v_hat * delta_mu.
    """
    X, y = _stack_xy(H_honest, H_syco, layer)
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(X, y)

    w = clf.coef_[0]
    b = float(clf.intercept_[0])
    norm = float(np.linalg.norm(w))
    v_hat = w / norm  # points toward honest (class 1)
    m = -b / norm

    proj_honest = H_honest[:, layer] @ v_hat
    proj_syco = H_syco[:, layer] @ v_hat
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
