import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import explore_inference_memory as eim  # noqa: E402


def test_token_step_without_momentum_is_a_ridge_gradient_step():
    mm = eim.MemoryMap(context=4)
    theta, lam = 0.3, 0.02
    w = mm.w0[None, :] + 0.1
    new_w, _ = mm.token(w, np.zeros_like(w), 2, np.array([theta]), np.array([theta * lam]), 0.0)
    ridge_net = dataclasses.replace(mm.net, ridge=lam)
    expected = w - theta * ridge_net.gradient(w, mm.keys[2:3], mm.values[2:3])
    np.testing.assert_allclose(new_w, expected, rtol=0, atol=1e-14)


def test_period_jacobian_matches_central_differences():
    mm = eim.MemoryMap(context=3)
    theta, alpha, eta = 0.4, 0.01, 0.5
    rng = np.random.default_rng(1)
    z = np.concatenate([mm.w0 + 0.2 * rng.normal(size=mm.n), 0.05 * rng.normal(size=mm.n)])
    value, jac = mm.period_jacobian(z, theta, alpha, eta)

    def period(y):
        w, s = mm.period(y[None, :mm.n], y[None, mm.n:], np.array([theta]), np.array([alpha]), eta)
        return np.concatenate([w[0], s[0]])

    np.testing.assert_allclose(value, period(z), rtol=0, atol=1e-14)
    h = 1e-6
    numeric = np.empty_like(jac)
    for j in range(2 * mm.n):
        e = np.zeros(2 * mm.n)
        e[j] = h
        numeric[:, j] = (period(z + e) - period(z - e)) / (2 * h)
    np.testing.assert_allclose(jac, numeric, rtol=0, atol=1e-7)


def test_strong_forgetting_gives_memory_that_repeats_with_the_context():
    task = (8, 3, 2, 28011, 128011, 0.2, np.array([0.05]), np.array([0.01]), 0.0, 400, 64, 8, 1e-8)
    out = eim.classify_block(task)
    assert out["finite"][0]
    assert out["period"][0] == 1


def test_recall_error_per_cell():
    mm = eim.MemoryMap(context=5)
    w = np.stack([mm.w0, mm.w0 + 0.3])
    err = mm.recall_error(w)
    assert err.shape == (2,)
    for c in range(2):
        pred = mm.net.predict(w[c:c + 1], mm.keys)[:, 0]
        np.testing.assert_allclose(err[c], 0.5 * np.mean((pred - mm.values) ** 2), rtol=1e-14)
