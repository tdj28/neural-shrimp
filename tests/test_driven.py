import numpy as np

from neural_shrimp.capacity import simulate
from neural_shrimp.driven import simulate_soa


def test_soa_driven_engine_matches_reference_engine():
    rng = np.random.default_rng(3)
    weights = rng.normal(size=(5, 3, 3)) * 2.0
    x0 = rng.normal(size=(5, 3))
    tangent = rng.normal(size=(5, 3))
    inputs = rng.uniform(-1, 1, 40)
    for feature in ("activation", "state"):
        ref = simulate(weights, x0, inputs, 0.3, 0.01, 20, 4, feature=feature, tangent=tangent)
        new = simulate_soa(weights, x0, inputs, 0.3, 0.01, 20, 4, feature=feature, tangent=tangent)
        np.testing.assert_allclose(new[0], ref[0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(new[1], ref[1], rtol=0, atol=1e-12)
        np.testing.assert_allclose(new[2], ref[2], rtol=0, atol=1e-10)
    no_tangent = simulate_soa(weights, x0, inputs, np.linspace(0.1, 0.5, 5), 0.01, 20, 2)
    reference = simulate(weights, x0, inputs, np.linspace(0.1, 0.5, 5), 0.01, 20, 2)
    np.testing.assert_allclose(no_tangent[0], reference[0], rtol=0, atol=1e-12)
    assert no_tangent[2] is None
