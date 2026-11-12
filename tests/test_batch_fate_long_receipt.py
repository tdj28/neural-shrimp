"""Tests of the pure functions of tools/batch_fate_long_receipt.py (no model, no run data)."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import batch_fate_long_receipt as rc  # noqa: E402


def test_cyclic_distance_finds_the_shift_and_refuses_other_periods():
    rng = np.random.default_rng(0)
    C = rng.normal(size=(4, 6))
    assert rc.cyclic_distance(np.roll(C, 3, axis=0), C) == 0.0
    assert rc.cyclic_distance(C + 1e-9, C) < 1e-8
    assert rc.cyclic_distance(C[[1, 0, 2, 3]], C) > 0.1          # a reordering that is not a shift
    assert rc.cyclic_distance(np.concatenate([C, C]), C) == math.inf


def test_primitive_gap_is_zero_for_a_doubled_cycle():
    rng = np.random.default_rng(1)
    C = rng.normal(size=(4, 6))
    assert rc.primitive_gap(np.concatenate([C, C])) == 0.0
    assert rc.primitive_gap(C) > 0.1


def test_exposure_and_rate():
    cap = np.array([-1, 10, 60, 100])
    assert rc.exposure(cap, 100) == 100 + 10 + 60 + 100
    assert rc.exposure(cap, 50) == 50 + 10 + 50 + 50
    assert rc.exposure(cap, 100, 50) == 50 + 0 + 10 + 50
    r = rc.rate_record(3, 270)
    assert r["per_step"] == pytest.approx(3 / 270)
    assert r["interval95_per_step"][0] < 3 / 270 < r["interval95_per_step"][1]


def test_poisson_interval_matches_known_values():
    assert rc.poisson_interval(0) == pytest.approx((0.0, 3.6889), abs=1e-4)
    assert rc.poisson_interval(14) == pytest.approx((7.654, 23.490), abs=2e-3)


def test_reading_follows_the_run_card():
    H = 100
    assert rc.reading(np.array([10, 20, 60, -1]), H) == "long transient"                 # half or more, some late
    assert rc.reading(np.array([10] + [-1] * 7), H) == "attractor at this horizon"       # few, none late
    assert rc.reading(np.array([10, 20, 30, -1]), H) == "unresolved"                     # many, but none late
    assert rc.reading(np.array([60] + [-1] * 7), H) == "unresolved"                      # few, but one late


def test_constant_rate_tests_on_exponential_and_on_early_captures():
    rng = np.random.default_rng(2)
    H = 100000
    t = rng.exponential(60000, size=400).astype(np.int64) + 1
    flat = rc.constant_rate_tests(np.where(t <= H, t, -1), H)
    assert flat["halves"]["binomial_p"] > 0.01 and flat["uniform"]["ks_p"] > 0.01
    early = np.where(rng.random(400) < 0.3, rng.integers(1, 5000, size=400), -1)         # captures stop early
    stop = rc.constant_rate_tests(early, H)
    assert stop["halves"]["captures"][1] == 0 and stop["halves"]["binomial_p"] < 1e-6
    assert stop["uniform"]["ks_p"] < 1e-6


def test_nearest_pair():
    X = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1e-6], [3.0, 3.0]])
    i, j, d = rc.nearest_pair(X)
    assert (i, j) == (0, 2) and d == pytest.approx(1e-6, rel=1e-3)


def test_densest_window_and_cluster_test():
    late = np.array([100, 5000, 9000, 60000, 60010, 60020, 60030, 60040, 60050] + [-1] * 55)   # six at once, late
    tail, count, expected, first = rc.densest_window(late, 100000, 5000)
    assert (count, first) == (6, 60000) and expected < 1.0 and tail < 1e-3
    out = rc.cluster_test(late, 100000, width=5000, draws=200)
    assert out["captures"] == 6 and out["simulated_share_with_a_window_as_improbable"] < 0.05
    rng = np.random.default_rng(3)
    t = np.ceil(rng.exponential(60000, size=64)).astype(np.int64)
    flat = rc.cluster_test(np.where(t <= 100000, t, -1), 100000, width=5000, draws=200)
    assert flat["simulated_share_with_a_window_as_improbable"] > 0.05
