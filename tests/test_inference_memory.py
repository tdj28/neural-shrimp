import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import explore_inference_memory as eim  # noqa: E402
from neural_shrimp.trainability import make_s2  # noqa: E402


def test_token_step_without_momentum_is_a_ridge_gradient_step():
    mm = eim.MemoryMap(context=4)
    theta, lam = 0.3, 0.02
    w = mm.w0[None, :] + 0.1
    rates = mm.rates(np.array([theta]), np.array([theta]))
    new_w, _ = mm.update(w, np.zeros_like(w), 2, rates, np.array([theta * lam]), 0.0)
    ridge_net = dataclasses.replace(mm.net, ridge=lam)
    expected = w - theta * ridge_net.gradient(w, mm.keys[2:3], mm.values[2:3])
    np.testing.assert_allclose(new_w, expected, rtol=0, atol=1e-14)


def test_whole_context_chunk_is_the_s2_training_map():
    net, x, y, theta0 = make_s2(28011)
    mm = eim.MemoryMap(context=24, chunk=24, coupled_decay=1e-3)
    np.testing.assert_array_equal(mm.keys, x)
    np.testing.assert_array_equal(mm.values, y)
    np.testing.assert_array_equal(mm.w0, theta0)
    eta_h, eta_o = 10.0 ** 1.853, 10.0 ** -0.0857
    rates = mm.rates(np.array([eta_h]), np.array([eta_o]))
    w, _ = mm.period(theta0[None, :], np.zeros((1, mm.n)), rates, np.array([0.0]), 0.0)
    expected = theta0[None, :] - net.rate_vector(eta_h, eta_o) * net.gradient(theta0[None, :], x, y)
    np.testing.assert_allclose(w, expected, rtol=0, atol=1e-13)


def test_period_jacobian_matches_central_differences():
    mm = eim.MemoryMap(context=6, chunk=3, coupled_decay=1e-3)
    alpha, eta = 0.01, 0.5
    rates = mm.rates(np.array([0.4]), np.array([0.05]))[0]
    rng = np.random.default_rng(1)
    z = np.concatenate([mm.w0 + 0.2 * rng.normal(size=mm.n), 0.05 * rng.normal(size=mm.n)])
    value, jac = mm.period_jacobian(z, rates, alpha, eta)

    def period(v):
        w, s = mm.period(v[None, :mm.n], v[None, mm.n:], rates[None, :], np.array([alpha]), eta)
        return np.concatenate([w[0], s[0]])

    np.testing.assert_allclose(value, period(z), rtol=0, atol=1e-14)
    h = 1e-6
    numeric = np.empty_like(jac)
    for j in range(2 * mm.n):
        e = np.zeros(2 * mm.n)
        e[j] = h
        numeric[:, j] = (period(z + e) - period(z - e)) / (2 * h)
    np.testing.assert_allclose(jac, numeric, rtol=0, atol=1e-7)


def test_scaled_chunks_apply_the_same_total_step_per_context():
    online = eim.MemoryMap(context=24, chunk=1, scale_by_chunk=True)
    whole = eim.MemoryMap(context=24, chunk=24, scale_by_chunk=True)
    np.testing.assert_allclose(online.rates(np.array([2.4]), np.array([0.24]))[0] * 24,
                               whole.rates(np.array([2.4]), np.array([0.24]))[0], rtol=1e-15)


def test_strong_forgetting_gives_memory_that_repeats_with_the_context():
    memory = {"context": 8, "width": 3, "d": 2, "teacher_seed": 28011, "init_seed": 128011, "init_scale": 0.2,
              "chunk": 1, "coupled_decay": 0.0, "scale_by_chunk": False}
    task = {"memory": memory, "theta": np.array([0.05]), "theta_o": np.array([0.05]), "alpha": np.array([0.01]),
            "eta": 0.0, "transient": 400, "samples": 64, "max_period": 8, "tol": 1e-8}
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
