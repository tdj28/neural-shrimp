import numpy as np
import pytest

from neural_shrimp.autonomous import autonomous_map, floquet_multipliers, screen_events
from neural_shrimp.capacity import hopfield_weights

# Published-reproduction coordinates (EXP-004/005): focus-grid gamma index 53 is the core.
CORE_GAMMA = 10.14 + 53 * (0.41 / 180)


def test_screen_events_finds_smallest_full_state_period():
    cells, count = 3, 60
    pattern = np.random.default_rng(1).normal(size=(4, 3))
    periodic = pattern[np.arange(count) % 4]
    doubled = periodic.copy()
    times = np.tile(np.arange(count, dtype=float), (cells, 1))
    aperiodic = np.random.default_rng(2).normal(size=(count, 3))
    events = np.stack([periodic, aperiodic, doubled])
    events[2, 30:] = np.nan  # fewer valid events still suffice for period 4
    period, cycle = screen_events(events, times, max_period=8, repeats=4)
    np.testing.assert_array_equal(period, [4, 0, 4])
    assert cycle[0] == pytest.approx(4.0)


def test_floquet_multipliers_of_linear_decay():
    weights = np.zeros((2, 3, 3))
    eigenvalues, _ = floquet_multipliers(weights, np.zeros((2, 3)), np.array([1.0, 2.5]), 0.01)
    np.testing.assert_allclose(np.abs(eigenvalues[0]), np.exp(-1.0), rtol=1e-9)
    np.testing.assert_allclose(np.abs(eigenvalues[1]), np.exp(-2.5), rtol=1e-9)


def test_reproduces_published_core_cycle_and_irregular_endpoints():
    weights = hopfield_weights([1.064, 1.064, 1.064], [CORE_GAMMA, 10.14, 10.55])
    x0 = np.full((3, 3), 0.1)
    result = autonomous_map(weights, x0, 0.01, settle=2000.0, measure=1200.0)
    assert result["period"][0] == 9
    assert result["cycle_time"][0] == pytest.approx(37.63637538, abs=2e-3)
    assert abs(result["ftle"][0]) < 0.005
    assert list(result["period"][1:]) == [0, 0]
    assert np.all(result["ftle"][1:] > 0.1)
    valid = result["event_counts"][0]
    start = result["event_states"][0, min(valid, 160) - 1]
    eigenvalues, end = floquet_multipliers(weights[:1], start[None], result["cycle_time"][:1], 0.01)
    moduli = np.abs(eigenvalues[0])
    assert moduli[0] == pytest.approx(1.0, abs=2e-3)
    assert moduli[1] == pytest.approx(0.06506, abs=2e-3)
    np.testing.assert_allclose(end[0], start, atol=1e-2)
