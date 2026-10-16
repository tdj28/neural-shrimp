import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import pi_f64_twins as pt  # noqa: E402


def test_dmax_scales_with_large_offsets():
    assert pt.dmax_for(1e-10) == 1e-3
    assert pt.dmax_for(1e-6) == 1e-3
    assert math.isclose(pt.dmax_for(1e-3), 0.1)


def test_growth_rate_recovers_an_exponential_and_its_crossing():
    eps0, lam = 1e-10, 0.5
    d = eps0 * np.exp(lam * np.arange(64))
    g, first = pt.growth_rate(d, eps0)
    assert abs(g - lam) < 1e-9
    assert first == int(np.argmax(d > 1e-3)) + 1


def test_contracting_twin_has_no_crossing_and_negative_rate():
    eps0 = 1e-10
    d = eps0 * np.exp(-0.3 * np.arange(64))
    g, first = pt.growth_rate(d, eps0)
    assert first is None and g < 0


def test_log10_marks_exact_merges():
    assert pt.log10_or_none(0.0) is None
    assert pt.log10_or_none(1e-3) == -3.0
