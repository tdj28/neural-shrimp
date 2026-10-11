import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import explore_momentum_maps as emm  # noqa: E402


def test_beta_zero_is_plain_gradient_descent():
    mm = emm.MomentumMap("S2", 28011, 0.0)
    rates = mm.rates(1.8, -0.08)
    theta = mm.theta0[None, :]
    new_theta, _ = mm.step(theta, np.zeros_like(theta), rates)
    plain = theta - rates * mm.net.gradient(theta, mm.x, mm.y)
    np.testing.assert_allclose(new_theta, plain, rtol=0, atol=1e-15)


def test_momentum_jacobian_matches_central_differences():
    mm = emm.MomentumMap("S2", 28011, 0.7)
    rates = mm.rates(1.7, -0.1)
    rng = np.random.default_rng(0)
    theta, m = mm.theta0.copy(), 0.1 * rng.normal(size=mm.n)
    jac = mm.jacobian(theta, m, rates)
    z = np.concatenate([theta, m])
    h = 1e-6
    for j in range(2 * mm.n):
        e = np.zeros(2 * mm.n)
        e[j] = h
        plus = np.concatenate(mm.step((z + e)[None, :mm.n], (z + e)[None, mm.n:], rates), axis=1)[0]
        minus = np.concatenate(mm.step((z - e)[None, :mm.n], (z - e)[None, mm.n:], rates), axis=1)[0]
        np.testing.assert_allclose(jac[:, j], (plus - minus) / (2 * h), rtol=1e-6, atol=1e-8)
