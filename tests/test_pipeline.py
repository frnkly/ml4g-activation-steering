"""CPU smoke test: run probe + validate on synthetic Gaussian activations.

No model download required. Builds two separable clusters of fake "activations"
shaped like (N, num_layers + 1, hidden) and checks the probe/validate stack runs
end-to-end and separates the classes.
"""

import numpy as np

from syco_steering.probe import extract_direction, layer_sweep
from syco_steering.validate import validate


def _synthetic_acts(n=80, n_layers=6, hidden=16, sep=4.0, seed=0):
    """Two separable Gaussian clusters. A middle layer is made most separable so
    the sweep should not pick the embedding layer."""
    rng = np.random.default_rng(seed)
    H_honest = rng.standard_normal((n, n_layers, hidden)).astype(np.float32)
    H_syco = rng.standard_normal((n, n_layers, hidden)).astype(np.float32)
    # Inject a class offset along dim 0 that grows toward the middle layer.
    for layer in range(n_layers):
        scale = sep * (1.0 - abs(layer - n_layers // 2) / n_layers)
        H_honest[:, layer, 0] += scale
        H_syco[:, layer, 0] -= scale
    return H_honest, H_syco


def test_layer_sweep_and_direction(tmp_path):
    H_honest, H_syco = _synthetic_acts()
    accs, best_layer = layer_sweep(H_honest, H_syco, seed=0)

    assert len(accs) == H_honest.shape[1]
    assert all(0.0 <= a <= 1.0 for a in accs)
    # The most separable layer is the middle, not the embedding (index 0).
    assert best_layer != 0
    assert accs[best_layer] >= 0.9

    direction = extract_direction(H_honest, H_syco, best_layer)
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
    assert direction["v_hat"].shape == (H_honest.shape[2],)
    assert np.isclose(np.linalg.norm(direction["v_hat"]), 1.0, atol=1e-5)
    # honest projects higher than syco -> positive gap.
    assert direction["delta_mu"] > 0
    np.testing.assert_allclose(
        direction["steering_vector"], direction["v_hat"] * direction["delta_mu"], rtol=1e-5
    )


def test_validate_metrics_and_plots(tmp_path):
    H_honest, H_syco = _synthetic_acts()
    accs, best_layer = layer_sweep(H_honest, H_syco, seed=0)
    direction = extract_direction(H_honest, H_syco, best_layer)

    metrics = validate(
        H_honest, H_syco, best_layer, direction, str(tmp_path), accs=accs, sweep_seed=0
    )

    assert 0.0 <= metrics["test_acc"] <= 1.0
    assert 0.0 <= metrics["auroc"] <= 1.0
    assert metrics["test_acc"] >= 0.85
    assert metrics["auroc"] >= 0.9
    # CAA cosine should be high for cleanly separable clusters.
    assert metrics["caa_cosine"] >= 0.8
    assert (tmp_path / "projection_hist.png").exists()
    assert (tmp_path / "layer_acc.png").exists()
