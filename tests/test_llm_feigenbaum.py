"""Tests for tools/llm_feigenbaum.py on a toy map with a known period-doubling cascade: the logistic map in the first
coordinate, embedded in R^n with strongly contracting, driven remaining coordinates (as in the soft-token map, one
multiplier of order one and the rest small)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import llm_feigenbaum as lf  # noqa: E402

torch = pytest.importorskip("torch")

# flip parameters of the logistic map r x (1 - x): r_1 = 3, r_2 = 1 + sqrt(6), r_3, r_4 (Feigenbaum cascade)
LOGISTIC_FLIPS = [3.0, 1.0 + np.sqrt(6.0), 3.5440903595519228536, 3.5644072660954325977]


class ToyCell:
    """x0' = r x0 (1 - x0); x_i' = c_i x_i + 0.1 x0^2 + 0.05 x_{i-1} (driven, contracting, no feedback to x0)."""

    def __init__(self, r, n=6):
        self.torch = torch
        self.r = float(r)
        self.c = 0.05 * torch.linspace(1.0, 0.5, n - 1, dtype=torch.float64)
        self.start = torch.full((1, 1, n), 0.3, dtype=torch.float64)

    def _g(self, X):
        x0, rest = X[..., :1], X[..., 1:]
        prev = torch.cat([x0, rest[..., :-1]], dim=-1)
        return torch.cat([self.r * x0 * (1 - x0), self.c * rest + 0.1 * x0 ** 2 + 0.05 * prev], dim=-1)

    def f(self, X):
        return self._g(X)

    def jvp_rows(self, X, V):
        return torch.func.jvp(self._g, (X.repeat(V.shape[0], 1, 1),), (V,))[1]

    def iterate(self, X, steps):
        for _ in range(steps):
            X = self.f(X)
        return X


def test_newton_and_ritz_find_the_two_cycle_and_its_multiplier():
    r = 3.2
    cell = ToyCell(r)
    X = cell.iterate(cell.start, 500)
    Q = np.linalg.qr(np.random.default_rng(0).normal(size=(6, 3)))[0]
    X, Q, hist = lf.newton(cell, X + 1e-4, 2, Q)
    assert hist[-1] < 1e-12
    vals, vecs, residuals, _, _ = lf.ritz(cell, X, 2, Q)
    # 2-cycle multiplier of the logistic map: 4 + 2r - r^2
    assert abs(vals[0].real - (4 + 2 * r - r * r)) < 1e-10
    assert residuals[0] < 1e-10
    assert abs(vecs[0, 0]) > 0.1                            # the critical direction includes the driven response


def test_cascade_reproduces_the_logistic_flips_and_ratio():
    Q0 = np.linalg.qr(np.random.default_rng(1).normal(size=(6, 4)))[0]
    cell = ToyCell(2.8)
    X0 = cell.iterate(cell.start, 400)
    flips = lf.cascade(ToyCell, X0, Q0, 1, 2.8, 3.2, 4, +1, steps_per_period=40, k=4, log=lambda s: None)
    lams = [f["flip"]["lambda"] for f in flips]
    assert [f["period"] for f in flips] == [1, 2, 4, 8]
    assert np.allclose(lams, LOGISTIC_FLIPS, rtol=0, atol=1e-9)
    for f in flips:
        mu = f["flip"]["critical_multiplier"]
        assert abs(mu[0] + 1) < 1e-8 and abs(mu[1]) < 1e-12   # real -1 at every located flip
    ratios = lf.interval_ratios(lams)
    assert abs(ratios[0] - 4.7514) < 1e-3 and abs(ratios[1] - 4.6562) < 1e-3


class Rotation(ToyCell):
    """A contracting rotation by `angle` with modulus 0.95: its multipliers are a complex pair."""

    def __init__(self, angle):
        super().__init__(0.0)
        self.angle = angle

    def _g(self, X):
        a, b, rest = X[..., :1], X[..., 1:2], X[..., 2:]
        c, s = 0.95 * np.cos(self.angle), 0.95 * np.sin(self.angle)
        return torch.cat([c * a - s * b + 0.01 * a * a, s * a + c * b, 0.05 * rest], dim=-1)


def test_refuses_a_complex_critical_pair_near_minus_one():
    X = torch.zeros((1, 1, 6), dtype=torch.float64)
    Q = np.linalg.qr(np.random.default_rng(2).normal(size=(6, 3)))[0]
    with pytest.raises(lf.SearchError, match="complex pair near -1"):
        lf.evaluate(lambda lam: Rotation(2.8), 0.0, X, 1, Q)          # Re = 0.95 cos 2.8 = -0.89
    ev = lf.evaluate(lambda lam: Rotation(1.0), 0.0, X, 1, Q)         # a complex pair far from -1 is accepted
    assert ev.g > 0.5 and np.isclose(np.abs(ev.vals), 0.95, rtol=0, atol=1e-8).sum() == 2


def test_interval_ratios():
    assert lf.interval_ratios([0.0, 1.0, 1.25, 1.3125]) == [4.0, 4.0]


def test_tolerance_grows_with_the_period():
    assert lf.tol_for(3, 1e-12) == 1e-12
    assert abs(lf.tol_for(96, 1e-12) - 9.6e-12) < 1e-24
    assert lf.tol_for(384, 1e-12) > 3e-11


def test_edges_of_the_period_three_window():
    """The logistic period-3 window: born in a fold at r = 1 + sqrt(8), loses stability in a flip at r = 3.8415."""
    cell = ToyCell(3.835)
    X = cell.iterate(cell.start, 3000)
    Q0 = np.linalg.qr(np.random.default_rng(3).normal(size=(6, 4)))[0]
    recs = lf.edges(ToyCell, X, Q0, 3, 3.835, 1e-3, min_step=1e-9, log=lambda s: None)
    up = [r for r in recs if r["direction"] > 0][0]
    down = [r for r in recs if r["direction"] < 0][0]
    assert up["kind"] == "flip" and abs(up["lambda"] - 3.8414990075) < 1e-6
    assert down["kind"] == "fold" and abs(down["lambda"] - (1 + np.sqrt(8))) < 1e-6
