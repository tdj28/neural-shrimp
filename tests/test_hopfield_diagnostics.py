"""Analytic-fixture audits; no target Hopfield regime outcomes are generated."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "tools" / "hopfield_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("hopfield_diagnostics", PATH)
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


@pytest.mark.parametrize("state", [[0.0, 0.0, 0.0], [0.4, -0.8, 1.1], [8.0, -10.0, 2.0]])
def test_analytic_neural_jacobian_matches_finite_differences(state):
    state = np.asarray(state)
    alpha, gamma = -1.37, 3.21
    increment = 1e-5
    numerical = np.column_stack([
        (diagnostics.hopfield_rhs(0, state + increment * axis, alpha, gamma) -
         diagnostics.hopfield_rhs(0, state - increment * axis, alpha, gamma)) /
        (2 * increment) for axis in np.eye(3)
    ])
    np.testing.assert_allclose(diagnostics.hopfield_jacobian(0, state, alpha, gamma),
                               numerical, atol=1e-8, rtol=1e-8)


def analytic_rhs(t, state):
    x, y, z = state
    radial = 1 - x * x - y * y
    return np.array([radial * x - y, x + radial * y, -0.7 * z])


def analytic_jacobian(t, state):
    x, y, _ = state
    radial = 1 - x * x - y * y
    return np.array([[radial - 2 * x * x, -1 - 2 * x * y, 0],
                     [1 - 2 * x * y, radial - 2 * y * y, 0],
                     [0, 0, -0.7]])


def test_known_continuous_cycle_has_neutral_flow_and_two_contractions():
    initial = np.array([1.0, 0.0, 0.0])
    period = 2 * np.pi
    solution = diagnostics._integrate_variational(analytic_rhs, analytic_jacobian,
                                                 initial, period)
    np.testing.assert_allclose(solution.y[:3, -1], initial, atol=1e-9)
    monodromy = solution.y[3:, -1].reshape(3, 3)
    result = diagnostics._floquet_diagnostics(monodromy, analytic_rhs(0, initial))
    actual = sorted(value["modulus"] for value in result["multipliers"])
    np.testing.assert_allclose(actual, sorted([1.0, np.exp(-4 * np.pi),
                                              np.exp(-1.4 * np.pi)]), atol=1e-9)
    assert result["neutral_verified"]
    assert result["transversely_attracting"]
    assert result["neutral_flow_alignment"] > 0.999999


def test_neutral_mode_is_not_counted_as_transverse_contraction():
    result = diagnostics._floquet_diagnostics(np.diag([1.0, 0.5, 0.2]),
                                             np.array([1.0, 0.0, 0.0]))
    assert result["neutral_verified"]
    assert result["transverse_spectral_radius"] == 0.5
    assert len(result["multipliers"]) == 3
    assert len(result["transverse_multipliers"]) == 2


def test_wrong_neutral_direction_and_unstable_transverse_are_rejected():
    wrong = diagnostics._floquet_diagnostics(np.diag([1.0, 0.5, 0.2]),
                                            np.array([0.0, 1.0, 0.0]))
    assert not wrong["neutral_verified"]
    unstable = diagnostics._floquet_diagnostics(np.diag([1.0, 1.01, 0.2]),
                                               np.array([1.0, 0.0, 0.0]))
    assert unstable["neutral_verified"]
    assert not unstable["transversely_attracting"]


def test_equilibrium_cannot_supply_a_neutral_flow_certificate():
    result = diagnostics._floquet_diagnostics(np.eye(3), np.zeros(3))
    assert not result["neutral_verified"]
    assert result["neutral_flow_residual"] is None
    json.dumps(result, allow_nan=False)


def test_invalid_inputs_fail_before_target_integration():
    with pytest.raises(ValueError, match="positive integer"):
        diagnostics.certify(0.0, 0.0, [0, 0, 0], 1.0, 0)
    with pytest.raises(ValueError, match="positive"):
        diagnostics.certify(0.0, 0.0, [0, 0, 0], -1.0, 1)
    with pytest.raises(ValueError, match="five"):
        diagnostics.behavior_probe(0.0, 0.0, [0, 0, 0], 1.0, 0.5)
