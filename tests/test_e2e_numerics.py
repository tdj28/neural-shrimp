"""tools/e2e_numerics.py on maps with known answers."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import e2e_numerics as en  # noqa: E402


def nonnormal(n, lead, rest_scale=1e-3, seed=0):
    """M = S diag(lead..., small) S^-1 with a badly conditioned S: non-normal, dominant multipliers `lead`."""
    rng = np.random.default_rng(seed)
    lam = np.concatenate([np.asarray(lead, dtype=float), rest_scale * rng.uniform(-1, 1, size=n - len(lead))])
    S = np.eye(n) + 0.8 * rng.normal(size=(n, n)) / np.sqrt(n)
    return S @ np.diag(lam) @ np.linalg.inv(S), lam


def test_subspace_iteration_finds_dominant_multipliers():
    M, lam = nonnormal(60, [0.97, -0.4])
    Q, H, vals, ok, _ = en.subspace_iteration(lambda V: M @ V, 60, 4, iters=40)
    assert ok
    assert np.allclose(sorted(vals[:2].real), sorted([0.97, -0.4]), atol=1e-9)
    assert np.allclose(H, Q.T @ M @ Q)


@pytest.mark.parametrize("lead", [[0.995], [-1.7], [0.6, -0.3]])
def test_newton_picard_converges_near_fold_and_unstable(lead):
    n = 50
    M, _ = nonnormal(n, lead, seed=1)
    xs = np.linspace(-1, 1, n)

    def Fp(x):                                         # fixed point xs, linear part M, quadratic terms
        e = x - xs
        return xs + M @ e + 0.05 * e ** 2

    Q, H, _, _, _ = en.subspace_iteration(lambda V: M @ V, n, 4, iters=30)
    x0 = xs + 1e-3 * np.random.default_rng(2).normal(size=n)
    x, hist, ok = en.newton_picard(Fp, lambda V: M @ V, x0, Q, H, tol=1e-13)
    assert ok, hist
    assert np.linalg.norm(x - xs) < 1e-11


def test_classify_refuses_to_mix_types():
    assert en.classify([0.99, 0.01])["type"] == "real +1"
    c = en.classify([-1.02, 0.3])
    assert c["type"] == "real -1" and abs(c["next_modulus"] - 0.3) < 1e-15
    c = en.classify([0.5 + 0.9j, 0.5 - 0.9j, 0.1])
    assert c["type"] == "complex pair" and abs(c["next_modulus"] - 0.1) < 1e-15


def test_fold_extrapolate_on_normal_form():
    a, b, eps_c = -4.8, -25.4, 0.0
    lams = np.array([-4e-4, -3e-4, -2e-4, -1e-4])     # eps = lambda - lambda_c < 0: two fixed points
    mus = 1 - 2 * np.sqrt(a * b * np.abs(lams - eps_c))   # the stable fixed point: mu = 1 - 2 sqrt(ab |eps|)
    lam_c, slope, res = en.fold_extrapolate(lams, mus)
    assert abs(lam_c - eps_c) < 1e-15 and res < 1e-12
    with pytest.raises(ValueError):
        en.fold_extrapolate(lams, -mus)


def test_flip_root():
    assert abs(en.flip_root(0.0, -0.9, 1.0, -1.1) - 0.5) < 1e-15
    with pytest.raises(ValueError):
        en.flip_root(0.0, -0.9, 1.0, -0.95)


def test_covariance_spectrum_matches_dense_formula():
    rng = np.random.default_rng(4)
    E = rng.normal(size=(300, 12))
    z = rng.normal(size=300) * 3
    p = np.exp(z - z.max())
    p /= p.sum()
    m = p @ E
    C = E.T @ (E * p[:, None]) - np.outer(m, m)
    w = np.sort(np.linalg.eigvalsh(C))[::-1]
    vals, U, tr, ns = en.covariance_spectrum(E, p, tol=1e-15, k=5)
    assert np.allclose(vals, w[:5], rtol=1e-10, atol=1e-13)
    assert abs(tr - np.trace(C)) < 1e-10


def test_covariance_two_tokens_is_rank_one_along_difference():
    E = np.eye(5)
    p = np.array([0.7, 0.3, 0, 0, 0.0])
    vals, U, _, _ = en.covariance_spectrum(E, p, k=2)
    assert abs(vals[0] - 0.21 * 2) < 1e-12 and abs(vals[1]) < 1e-12
    assert abs(en.pair_alignment(U[:, 0], E[0], E[1]) - 1) < 1e-12


def test_gridmap_logistic_orbits_and_lyapunov():
    s, f = en.logistic_grid(3.2)
    g = en.GridMap(s, f)
    orbs = g.periodic_orbits(2)
    assert len(orbs) == 1
    assert abs(orbs[0]["multiplier"] - (4 + 2 * 3.2 - 3.2 ** 2)) < 1e-6
    g4 = en.GridMap(*en.logistic_grid(4.0, n=20001))
    assert abs(g4.lyapunov(0.1234, transient=100, steps=200000) - np.log(2)) < 0.02
    ex = g.extrema()
    assert len(ex) == 1 and ex[0]["kind"] == "max" and abs(ex[0]["at"] - 0.5) < 1e-9


def test_dominant_block_speeds_up_newton_picard_with_a_rough_subspace():
    n = 60
    M, _ = nonnormal(n, [0.6, -0.35, 0.2], rest_scale=0.02, seed=5)
    xs = np.linspace(-1, 1, n)

    def Fp(x):
        e = x - xs
        return xs + M @ e + 0.05 * e ** 2

    Q, H, vals, _, _ = en.subspace_iteration(lambda V: M @ V, n, 6, iters=3)     # deliberately few iterations
    V, Hm = en.dominant_block(Q, H, 3)
    assert V.shape[1] == 3 and np.allclose(V.T @ V, np.eye(3))
    assert np.allclose(sorted(np.linalg.eigvals(Hm).real), sorted(vals[:3].real))
    x0 = xs + 1e-2 * np.random.default_rng(6).normal(size=n)
    _, hist_full, ok1 = en.newton_picard(Fp, lambda W: M @ W, x0, Q, H, tol=1e-13)
    _, hist_dom, ok2 = en.newton_picard(Fp, lambda W: M @ W, x0, V, Hm, tol=1e-13)
    assert ok2 and len(hist_dom) <= len(hist_full)


def test_dominant_block_keeps_complex_pairs_whole():
    H = np.array([[0.3, -0.5, 0.0], [0.5, 0.3, 0.0], [0.0, 0.0, 0.1]])
    V, Hm = en.dominant_block(np.eye(3), H, 1)
    assert Hm.shape == (2, 2)


def test_gridmap_ignores_clamped_end_points():
    s = np.linspace(-5, 5, 401)
    g = en.GridMap(s, 2.0 * s)                  # leaves the interval at both ends; only 0 is a fixed point
    orbs = g.periodic_orbits(1)
    assert len(orbs) == 1 and abs(orbs[0]["points"][0]) < 1e-9 and abs(orbs[0]["multiplier"] - 2) < 1e-9
