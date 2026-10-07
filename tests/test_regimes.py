import numpy as np
import pytest

from neural_shrimp.regimes import (
    block_bootstrap, blocked_cv_r2, classify, cycle_rotation, dominant_ar_root, partial_correlation,
    peak_frequencies, rotation_number, spectrum_features,
)


def _embed(planar, dim=6, seed=0):
    rng = np.random.default_rng(seed)
    basis = np.linalg.qr(rng.normal(size=(dim, 2)))[0]
    return planar @ basis.T + rng.normal(size=dim)


def test_rotation_number_recovers_rigid_rotation():
    k = np.arange(2000)
    for alpha in (0.2194, 0.1322):
        circle = np.column_stack((np.cos(2 * np.pi * alpha * k), 2.0 * np.sin(2 * np.pi * alpha * k)))
        rho, first, second, share = rotation_number(_embed(circle))
        assert abs(rho - alpha) < 2e-3
        assert abs(first - second) < 2e-3
        assert share > 0.999


def test_rotation_number_folds_to_half_interval():
    k = np.arange(1000)
    circle = np.column_stack((np.cos(2 * np.pi * 0.7 * k), np.sin(2 * np.pi * 0.7 * k)))
    assert abs(rotation_number(_embed(circle))[0] - 0.3) < 1e-9


def test_cycle_rotation_of_two_lobe_period_eighteen_orbit():
    states = []
    for t in range(18):
        j = t // 2
        angle = 2 * np.pi * (2 / 9) * j
        lobe = np.array([5.0, 0.0]) if t % 2 == 0 else np.array([-5.0, 0.0])
        states.append(lobe + np.array([np.cos(angle), np.sin(angle)]))
    m, q, rho = cycle_rotation(_embed(np.array(states)))
    assert (m, q) == (2, 9) and abs(rho - 2 / 9) < 1e-12


def test_cycle_rotation_of_period_two_has_no_turns():
    assert cycle_rotation(np.array([[0.0, 1.0], [1.0, 0.0]])) == (0, 1, 0.0)


def test_spectrum_features_separate_lines_from_noise():
    t = np.arange(4096)
    lines, entropy_lines = spectrum_features(np.sin(2 * np.pi * 0.11 * t) + 0.5 * np.sin(2 * np.pi * 0.31 * t))
    noise, entropy_noise = spectrum_features(np.random.default_rng(1).normal(size=4096))
    assert lines > 0.95 and noise < 0.2
    assert entropy_lines < 0.3 < 0.8 < entropy_noise
    assert spectrum_features(np.ones(64)) == (1.0, 0.0)


def test_peak_frequencies_locate_a_line():
    t = np.arange(4096)
    peaks = peak_frequencies(np.sin(2 * np.pi * 0.1234 * t) + 0.1 * np.sin(2 * np.pi * 0.3 * t))
    assert abs(peaks[0] - 0.1234) < 5e-4 and abs(peaks[1] - 0.3) < 5e-4


@pytest.mark.parametrize("args, label", [
    ((0, 0, (np.nan, np.nan), (0, 0), None, np.nan, 0), "divergent"),
    ((1, 1, (-0.1, -0.1), (0, 0), None, 0.0, 0), "fixed"),
    ((2, 2, (-0.1, -0.1), (0, 0), 0.0, 0.0, 0), "period2"),
    ((2, 18, (-0.01, -0.01), (0.2222, 0.2222), 0.2222, 0.0, 2), "locked"),
    ((2, 5, (-0.01, -0.01), (0, 0), 0.0, 0.0, 1), "other_periodic"),
    ((2, 4, (-0.01, -0.01), (0, 0), 0.0, 0.0, 0), "other_periodic"),
    ((4, 0, (2e-4, -1e-4), (0.2194, 0.2195), 0.21945, 3e-9, 0), "quasiperiodic"),
    ((3, 0, (0.02, 0.021), (0.23, 0.25), 0.24, 0.3, 0), "chaotic"),
    ((3, 0, (0.003, 0.004), (0.23, 0.25), 0.24, 0.3, 0), "irregular"),
    ((4, 0, (2e-4, -1e-4), (0.2194, 0.2195), 0.21945, 0.3, 0), "irregular"),
])
def test_classify_rules(args, label):
    assert classify(*args) == label


def test_nested_models_and_partial_correlation():
    rng = np.random.default_rng(3)
    x1, x2, junk = rng.normal(size=(3, 400))
    y = x1 + 0.8 * x2 + 0.1 * rng.normal(size=400)
    blocks = np.arange(400) // 40
    base = blocked_cv_r2([x1], y, blocks)
    full = blocked_cv_r2([x1, x2], y, blocks)
    assert full > base + 0.3 and full > 0.95
    assert blocked_cv_r2([], y, blocks) < 0.01
    assert partial_correlation(x2, y, [x1]) > 0.95
    assert abs(partial_correlation(junk, y, [x1, x2])) < 0.15


def test_block_bootstrap_interval_of_constant_statistic():
    low, high = block_bootstrap(lambda index: 2.0, 100, 10, draws=50)
    assert low == high == 2.0
    values = np.arange(100.0)
    low, high = block_bootstrap(lambda index: values[index].mean(), 100, 10, draws=200)
    assert low < 49.5 < high


def test_dominant_ar_root_recovers_damped_oscillation():
    rng = np.random.default_rng(5)
    n, r, f = 8192, 0.99, 0.2125
    a1, a2 = 2 * r * np.cos(2 * np.pi * f), -r * r
    osc = np.zeros(n)
    slow = np.zeros(n)
    for k in range(2, n):
        osc[k] = a1 * osc[k - 1] + a2 * osc[k - 2] + rng.normal()
        slow[k] = 0.5 * slow[k - 1] + rng.normal()
    modulus, frequency = dominant_ar_root(osc + 0.5 * slow + 0.1 * rng.normal(size=n))
    assert abs(modulus - r) < 0.01 and abs(frequency - f) < 2e-3
    assert np.isnan(dominant_ar_root(rng.normal(size=n), order=1)[0])
