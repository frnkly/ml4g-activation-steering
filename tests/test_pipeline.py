"""CPU smoke test: run probe + validate on synthetic per-token Gaussian activations.

No model download required. Builds two separable clusters of fake token
"activations" shaped like (T, num_layers + 1, hidden) with group ids (which
response each token came from) and checks the probe/validate stack runs
end-to-end and separates the classes.
"""

import numpy as np

from syco_steering.probe import (
    extract_direction,
    layer_sweep,
    pool_by_response,
    stack_tokens,
)
from syco_steering.validate import validate


def _synthetic_token_acts(
    n_groups=20, tokens_per_group=10, n_layers=6, hidden=16, sep=4.0, seed=0
):
    """Two separable Gaussian token clusters with group structure. A middle layer
    is made most separable so the sweep should not pick the embedding layer."""
    rng = np.random.default_rng(seed)
    n = n_groups * tokens_per_group
    X_honest = rng.standard_normal((n, n_layers, hidden)).astype(np.float16)
    X_syco = rng.standard_normal((n, n_layers, hidden)).astype(np.float16)
    # Inject a class offset along dim 0 that grows toward the middle layer.
    for layer in range(n_layers):
        scale = sep * (1.0 - abs(layer - n_layers // 2) / n_layers)
        X_honest[:, layer, 0] += scale
        X_syco[:, layer, 0] -= scale
    groups = np.repeat(np.arange(n_groups), tokens_per_group)
    return X_honest, X_syco, groups.copy(), groups.copy()


def test_stack_tokens_keeps_groups_disjoint():
    X_honest, X_syco, g_honest, g_syco = _synthetic_token_acts()
    X, y, groups = stack_tokens(X_honest, X_syco, g_honest, g_syco, layer=0)
    assert len(X) == len(y) == len(groups) == len(X_honest) + len(X_syco)
    honest_groups = set(groups[y == 1])
    syco_groups = set(groups[y == 0])
    assert honest_groups.isdisjoint(syco_groups)


def test_layer_sweep_and_direction():
    X_honest, X_syco, g_honest, g_syco = _synthetic_token_acts()
    accs, best_layer = layer_sweep(X_honest, X_syco, g_honest, g_syco, seed=0)

    assert len(accs) == X_honest.shape[1]
    assert all(0.0 <= a <= 1.0 for a in accs)
    # The most separable layer is the middle, not the embedding (index 0).
    assert best_layer != 0
    assert accs[best_layer] >= 0.9

    direction = extract_direction(X_honest, X_syco, g_honest, g_syco, best_layer)
    for key in [
        "v_hat",
        "m",
        "mu_pos",
        "sig_pos",
        "delta_mu",
        "best_layer",
        "steering_vector",
    ]:
        assert key in direction
    assert direction["v_hat"].shape == (X_honest.shape[2],)
    assert np.isclose(np.linalg.norm(direction["v_hat"]), 1.0, atol=1e-5)
    # honest responses project higher than syco -> positive pooled gap.
    assert direction["delta_mu"] > 0
    np.testing.assert_allclose(
        direction["steering_vector"], direction["v_hat"] * direction["delta_mu"], rtol=1e-5
    )
    # The POOLED boundary still gates individual TOKENS: pooling changes the
    # class variances, not the class means, so the (wider) per-token
    # distributions straddle the same boundary.
    proj_syco = X_syco[:, best_layer].astype(np.float32) @ direction["v_hat"]
    proj_honest = X_honest[:, best_layer].astype(np.float32) @ direction["v_hat"]
    assert (proj_syco < direction["m"]).mean() > 0.9
    assert (proj_honest < direction["m"]).mean() < 0.1


def test_pool_by_response():
    X_honest, _, g_honest, _ = _synthetic_token_acts(n_groups=5, tokens_per_group=7)
    pooled = pool_by_response(X_honest, g_honest, layer=2)
    assert pooled.shape == (5, X_honest.shape[2])
    # Row i is the mean of group i's token activations at the layer.
    np.testing.assert_allclose(
        pooled[0],
        X_honest[g_honest == 0, 2].astype(np.float32).mean(axis=0),
        rtol=1e-5,
    )


def test_layer_sweep_subsample_cap():
    X_honest, X_syco, g_honest, g_syco = _synthetic_token_acts()
    accs, best_layer = layer_sweep(
        X_honest, X_syco, g_honest, g_syco, seed=0, max_tokens=100
    )
    assert len(accs) == X_honest.shape[1]
    assert accs[best_layer] >= 0.8


def test_validate_metrics_and_plots(tmp_path):
    X_honest, X_syco, g_honest, g_syco = _synthetic_token_acts()
    accs, best_layer = layer_sweep(X_honest, X_syco, g_honest, g_syco, seed=0)
    direction = extract_direction(X_honest, X_syco, g_honest, g_syco, best_layer)

    metrics = validate(
        X_honest,
        X_syco,
        g_honest,
        g_syco,
        best_layer,
        direction,
        str(tmp_path),
        accs=accs,
        sweep_seed=0,
    )

    assert 0.0 <= metrics["test_acc"] <= 1.0
    assert 0.0 <= metrics["auroc"] <= 1.0
    assert metrics["test_acc"] >= 0.85
    assert metrics["auroc"] >= 0.9
    # CAA cosine should be high for cleanly separable clusters.
    assert metrics["caa_cosine"] >= 0.8
    assert (tmp_path / "projection_hist.png").exists()
    assert (tmp_path / "layer_acc.png").exists()
