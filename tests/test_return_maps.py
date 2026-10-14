import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import return_maps as rm


def test_fit_recovers_logistic_map_exponent():
    r, x = 3.9, 0.3
    xs = []
    for _ in range(3000):
        x = r * x * (1 - x)
        xs.append(x)
    f = rm.fit(np.array(xs[500:]), 8)
    exact = np.mean(np.log(np.abs(r * (1 - 2 * np.array(xs[500:-1])))))
    assert f["residual_rel"] < 1e-10
    assert abs(f["exponent_per_repetition"] - exact) < 1e-6
    assert f["critical_points_inside"] == 1


def test_fit_flags_two_valued_data():
    rng = np.random.default_rng(0)
    f = rm.fit(rng.random(500), 8)
    assert f["residual_rel"] > 0.1


def test_cv_fit_recovers_logistic_map_exponent():
    r, x = 3.9, 0.3
    xs = []
    for _ in range(3000):
        x = r * x * (1 - x)
        xs.append(x)
    f = rm.cv_fit(np.array(xs[500:]))
    exact = np.mean(np.log(np.abs(r * (1 - 2 * np.array(xs[500:-1])))))
    assert f["cv_residual_rel"] < 1e-10
    assert abs(f["exponent_per_repetition"] - exact) < 1e-6
    assert f["critical_points_inside"] == 1
