"""Layer sweep and steering-direction / boundary extraction via logistic regression.

torch-free: operates purely on numpy TOKEN-level activation arrays of shape
``(T, num_layers + 1, hidden)`` so it can be unit-tested on synthetic data.

Following the paper (arXiv 2604.08169, Eq. 1 / Alg. A.1), the probe is fit on
RESPONSE-AVERAGED embeddings: each response contributes one training example,
the mean hidden state over its response tokens. The inference-time StTP/StMP
gate still compares *individual token* projections to the boundary ``m`` — this
works because averaging shrinks the within-class variance but leaves the class
means unchanged, so the pooled boundary sits between the same means that the
(wider) per-token distributions straddle. The per-token ``layer_sweep`` below is
kept as a cheap layer-shortlist heuristic and for gate diagnostics.
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
    """Fit a TOKEN-level probe per layer on a 75/25 GROUP-aware split (groups =
    response id, so tokens of one response never straddle the split); return
    per-layer held-out accuracies and the index of the best layer (honest=1,
    syco=0).

    This is a cheap layer-shortlist heuristic, not the paper's selection rule:
    the paper picks its operating point (layer, coefficient, position) from a
    downstream steering sweep judged on trait + coherence. The final direction
    is calibrated separately on response-averaged embeddings
    (``extract_direction``).

    ``max_tokens`` caps the number of tokens per fit (random subsample) to keep
    the sweep fast.
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


def pool_by_response(X: np.ndarray, groups: np.ndarray, layer: int) -> np.ndarray:
    """One response-averaged embedding per generation (paper Eq. 1):
    the mean hidden state over each response's tokens at ``layer``.
    Rows are ordered by ``np.unique(groups)``."""
    ids = np.unique(groups)
    return np.stack(
        [X[groups == i, layer].astype(np.float32).mean(axis=0) for i in ids]
    )


def extract_direction(
    X_honest: np.ndarray,
    X_syco: np.ndarray,
    g_honest: np.ndarray,
    g_syco: np.ndarray,
    layer: int,
) -> dict:
    """Fit the probe on RESPONSE-AVERAGED embeddings at ``layer`` (paper Eq. 1 /
    Alg. A.1) and extract the steering geometry.

    All statistics are in pooled-projection units — the paper's units, in which
    e.g. StTP's target ``s = mu_pos + alpha * sig_pos`` needs the paper's large
    alpha grid ({0, 6, ..., 36}) to move far into the positive distribution.
    The inference-time gate still compares individual token projections to
    ``m``; see the module docstring for why that calibration transfers.

    Returns a dict with:
      - ``v_hat``         unit direction toward honest (class 1),
      - ``m``             decision boundary along v_hat = -b / ||w||
                          (Alg. A.1's delta-mu rescaling reduces to this),
      - ``mu_pos``        mean honest RESPONSE-AVERAGED projection onto v_hat,
      - ``sig_pos``       std of honest response-averaged projections,
      - ``delta_mu``      pooled projection gap (honest mean - syco mean),
      - ``best_layer``    the hidden_states layer index used,
      - ``steering_vector`` = v_hat * delta_mu   (paper: ||v|| = delta_mu, so
                          SwFC's alpha = 1 shifts by one unit of class separation).
    """
    E_pos = pool_by_response(X_honest, g_honest, layer)
    E_neg = pool_by_response(X_syco, g_syco, layer)
    X = np.concatenate([E_pos, E_neg], axis=0)
    y = np.concatenate([np.ones(len(E_pos)), np.zeros(len(E_neg))]).astype(int)
    clf = LogisticRegression(C=1.0, max_iter=2000)
    clf.fit(X, y)

    w = clf.coef_[0]
    b = float(clf.intercept_[0])
    norm = float(np.linalg.norm(w))
    v_hat = w / norm  # points toward honest (class 1)
    m = -b / norm

    proj_pos = E_pos @ v_hat
    proj_neg = E_neg @ v_hat
    mu_pos = float(proj_pos.mean())
    sig_pos = float(proj_pos.std())
    delta_mu = float(proj_pos.mean() - proj_neg.mean())

    return {
        "v_hat": v_hat.astype(np.float32),
        "m": float(m),
        "mu_pos": mu_pos,
        "sig_pos": sig_pos,
        "delta_mu": delta_mu,
        "best_layer": int(layer),
        "steering_vector": (v_hat * delta_mu).astype(np.float32),
    }
