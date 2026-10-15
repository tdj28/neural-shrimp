import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import dss_window as dw  # noqa: E402


def test_checkerboard_marks_a_crossing_and_not_a_single_curve():
    x = np.linspace(-1, 1, 8)                       # even count: the crossing falls between cells
    X, Y = np.meshgrid(x, x)
    valid = np.ones_like(X, dtype=bool)
    cross = dw.checkerboards(np.sign(X * Y), valid)
    assert cross == [(3, 3)]
    assert dw.checkerboards(np.sign(X + 0.3 * Y), valid) == []
    assert dw.sign_changes(np.sign(X + 0.3 * Y), valid) > 0
    valid[4, 4] = False                              # a block with an invalid cell is not read
    assert dw.checkerboards(np.sign(X * Y), valid) == []


def test_stats_reads_the_modulus_not_the_real_part():
    n = 4
    mult = np.zeros((n, n, 3, 2))
    mult[..., 0, 0] = 0.5                            # real leading multiplier 0.5 everywhere
    mult[..., 1, 0] = 0.01
    mult[1, 2, 0] = [-0.0005, 0.03]                  # a complex pair with tiny real part
    mult[1, 2, 1] = [-0.0005, -0.03]
    mult[3, 3, 0, 0] = -1.5                          # one repelling cell
    z = {"ok": np.ones((n, n), dtype=bool), "clean": np.ones((n, n), dtype=bool),
         "same_tokens": np.ones((n, n), dtype=bool), "multipliers": mult,
         "stretchings": np.full((n, n, 5), 0.8)}
    s = dw.stats(z, 5)
    assert abs(s["min_leading_modulus"] - np.hypot(0.0005, 0.03)) < 1e-12
    assert s["min_leading_at_row_col"] == [1, 2]
    assert s["attracting"] == n * n - 1 and s["repelling"] == 1
    assert s["attracting_complex_leading"] == 1
    assert s["pair_product_negative_attracting"] == 0
