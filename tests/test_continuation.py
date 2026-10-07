import numpy as np
import pytest
from scipy.optimize import brentq

from neural_shrimp.continuation import classify_multiplier, continue_ray, ray_directions


def henon_family(params):
    a, b = params

    def map_fn(state):
        x, y = state
        return np.array([1.0 - a * x * x + y, b * x])

    def jacobian(state):
        return np.array([[-2.0 * a * state[0], 1.0], [b, 0.0]])

    return map_fn, jacobian


def _three_cycle_multiplier(c):
    """Multiplier of the 3-cycle of y -> c - y^2 that passes near y = 0 (Newton on f^3)."""
    y = 0.0
    for _ in range(50):
        a = c - y * y
        b = c - a * a
        g = c - b * b - y
        dg = (-2 * b) * (-2 * a) * (-2 * y) - 1.0
        y -= g / dg
    a = c - y * y
    b = c - a * a
    return -8.0 * y * a * b


def test_period_three_window_fold_and_flip_are_located_and_typed():
    # b = 0 reduces Henon to x -> 1 - a x^2, conjugate to y -> a - y^2: the period-3
    # window opens by a saddle-node at a = 7/4 and period-doubles where the
    # cycle multiplier equals -1.
    superstable = brentq(lambda c: c ** 3 - 2 * c ** 2 + c - 1, 1.7, 1.8)
    flip = brentq(lambda c: _three_cycle_multiplier(c) + 1.0, superstable + 1e-6, 1.770)
    seed = np.array([0.0, 0.0])
    map_fn, _ = henon_family((superstable, 0.0))
    for _ in range(3000):
        seed = map_fn(seed)
    left = continue_ray(henon_family, seed, 3, (superstable, 0.0), (-1.0, 0.0), 0.05, 1e-4,
                        bisection_tolerance=1e-10)
    right = continue_ray(henon_family, seed, 3, (superstable, 0.0), (1.0, 0.0), 0.05, 1e-4,
                         bisection_tolerance=1e-10)
    assert left["type"] == "fold"
    assert superstable - left["s_boundary"] == pytest.approx(1.75, abs=1e-6)
    assert right["type"] == "flip"
    assert superstable + right["s_boundary"] == pytest.approx(flip, abs=1e-6)


def test_ray_that_stays_inside_reports_none_within_range():
    superstable = brentq(lambda c: c ** 3 - 2 * c ** 2 + c - 1, 1.7, 1.8)
    seed = np.array([0.0, 0.0])
    map_fn, _ = henon_family((superstable, 0.0))
    for _ in range(3000):
        seed = map_fn(seed)
    result = continue_ray(henon_family, seed, 3, (superstable, 0.0), (1.0, 0.0), 0.001, 1e-4)
    assert result["type"] == "none_within_range"


def test_classify_multiplier_and_directions():
    assert classify_multiplier(0.995 + 0j) == "fold"
    assert classify_multiplier(-0.999 + 0j) == "flip"
    assert classify_multiplier(complex(0.6, 0.79)) == "neimark_sacker"
    assert classify_multiplier(0.5 + 0j) == "unresolved"
    d = ray_directions(8)
    np.testing.assert_allclose(np.linalg.norm(d, axis=1), 1.0)
    np.testing.assert_allclose(d[2], [0.0, 1.0], atol=1e-15)


def test_gradient_descent_family_flip_in_log_rate_coordinates():
    from neural_shrimp.gdmaps import GridMap, TwoLayerNet

    net = TwoLayerNet(d=1, width=1, activation="identity", bias=False, loss_scale=1.0, ridge=0.1)
    x, y = np.array([[1.0]]), np.array([1.0])

    def family(params):
        grid = GridMap(net, x, y, [10.0 ** params[0]], [10.0 ** params[1]])
        return grid.cell_map(0)

    s = np.sqrt(0.95)
    center = np.log10([0.4, 0.4])
    direction = np.array([1.0, 1.0]) / np.sqrt(2.0)
    result = continue_ray(family, np.array([s, s]), 1, center, direction, 0.5, 0.01,
                          bisection_tolerance=1e-9)
    assert result["type"] == "flip"
    expected = np.sqrt(2.0) * (np.log10(2.0 / 3.8) - np.log10(0.4))
    assert result["s_boundary"] == pytest.approx(expected, abs=1e-7)
