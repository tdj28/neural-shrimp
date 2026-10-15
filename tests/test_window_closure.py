import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import window_closure as wcl  # noqa: E402


def _henon_family(third=None, n=10, seed=0):
    """A map on R^n and its attracting fixed point: the Henon map (a = b = 0.3; fixed-point multipliers -0.92 and
    0.32) in two coordinates; with third = (multiplier, coupling) a third coordinate w with that multiplier drives u;
    the remaining coordinates contract by 0.01 and are slaved to u. Rotated by a fixed random orthogonal matrix."""
    a, b = 0.3, 0.3
    us = (-(1 - b) + np.sqrt((1 - b) ** 2 + 4 * a)) / (2 * a)
    R = np.linalg.qr(np.random.default_rng(seed).normal(size=(n, n)))[0]

    def raw(z):
        u, v = z[0], z[1]
        out = np.empty_like(z)
        w = z[2] if third else 0.0
        out[0] = 1 - a * u * u + v + (third[1] * w if third else 0.0)
        out[1] = b * u
        start = 2
        if third:
            out[2] = third[0] * w + 0.5 * (u - us) ** 2
            start = 3
        out[start:] = 0.01 * z[start:] + 0.2 * (u - us) ** 2
        return out

    # fixed point by iteration of the raw map (it attracts)
    z = np.zeros(n)
    z[0], z[1] = us, b * us
    for _ in range(20000):
        z = raw(z)
    f = lambda x: R @ raw(R.T @ x)                       # noqa: E731
    xs = R @ z
    h = 1e-7
    J = np.array([(f(xs + h * e) - f(xs - h * e)) / (2 * h) for e in np.eye(n)]).T
    return f, xs, J


def _trajectories(f, xs, rng, seeds, A):
    n = xs.size
    D = np.zeros((seeds, wcl.RETURNS + 1, n))
    for s in range(seeds):
        d = rng.normal(size=n)
        x = xs + A * np.linalg.norm(xs) * d / np.linalg.norm(d)
        D[s, 0] = x - xs
        for r in range(1, wcl.RETURNS + 1):
            x = f(x)
            D[s, r] = x - xs
    return D


def _score(f, xs, J):
    rng = np.random.default_rng(1)
    vals, V = np.linalg.eig(J)
    order = np.argsort(-np.abs(vals))
    vals, V = vals[order], V[:, order]
    B = wcl.modal_coordinates(vals, V, 4)
    Dtr = np.concatenate([_trajectories(f, xs, rng, 32, A) for A in wcl.AMPLITUDES])
    Dte = _trajectories(f, xs, rng, 32, wcl.AMPLITUDES[1])
    return wcl.gates({k: v for k, v in wcl.fit_and_score(Dtr, Dte, B, J).items() if k != "linear"})


def test_embedded_henon_passes_two_dimensional_closure():
    g = _score(*_henon_family())
    assert g["closure_2d"], g
    assert not g["closure_1d"], g


def test_slow_third_coordinate_fails_two_dimensional_closure():
    g = _score(*_henon_family(third=(0.5, 0.6)))
    assert not g["closure_2d"], g
    assert g["median"][4] < 0.5 * g["median"][2], g


def test_degenerate_fit_sets_no_closure_flag():
    # run CL2, complex-1 at amplitude 1e-2: E4 diverged and alone satisfied "E4 >= 0.8 E2"
    g = wcl.gates({1: [0.344], 2: [0.00457], 4: [1.04e12]})
    assert g["degenerate_k"] == [4] and not g["closure_2d"] and g["dimension_at_0.05"] is None, g
    g = wcl.gates({1: [0.344], 2: [0.00457], 4: [0.004]})
    assert g["closure_2d"] and not g["degenerate_k"], g


def test_quad_features_and_modal_coordinates():
    Y = np.array([[1.0, 2.0]])
    assert np.allclose(wcl.quad_features(Y), [[1, 1, 2, 1, 2, 4]])
    rng = np.random.default_rng(0)
    V = rng.normal(size=(5, 5))                          # a non-orthogonal eigenbasis
    vals = np.array([0.9, 0.5, 0.2, 0.1, 0.05])
    P = wcl.modal_coordinates(vals, V.astype(complex), 2)
    d = 3.0 * V[:, 0] + 7.0 * V[:, 2]                    # modes 1 and 3 only
    assert np.allclose(P @ d, [3.0, 0.0])                # the spectral projection does not mix mode 3 into mode 2
