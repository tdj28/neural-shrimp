"""Turning-point counting of tools/turning_scan.py on maps with known numbers of turning points."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import turning_scan as ts  # noqa: E402


def orbit(f, x0, n, burn=200):
    x = x0
    for _ in range(burn):
        x = f(x)
    out = np.empty(n)
    for i in range(n):
        x = f(x)
        out[i] = x
    return out


def test_logistic_has_one_turning_point():
    s = orbit(lambda x: 3.99 * x * (1 - x), 0.31, 6000)
    _, cy = ts.return_curve(s, 40)
    assert ts.count_turning(cy, 0.05) == 1


def test_chebyshev_cubic_has_two():
    s = orbit(lambda x: 2.98 * x - x ** 3, 0.3, 6000)
    _, cy = ts.return_curve(s, 40)
    assert ts.count_turning(cy, 0.05) == 2


def test_noisy_monotone_has_none():
    rng = np.random.default_rng(0)
    s = np.empty(6000)
    s[0] = 0.0
    for i in range(1, len(s)):
        s[i] = 0.6 * s[i - 1] + rng.normal()
    _, cy = ts.return_curve(s, 40)
    assert ts.count_turning(cy, 0.05) == 0


def test_symbolic_period():
    assert ts.symbolic_period([1, 2, 3] * 20, 24) == 3
    assert ts.symbolic_period([1, 2, 3, 4] * 10 + [9], 24) == 1
