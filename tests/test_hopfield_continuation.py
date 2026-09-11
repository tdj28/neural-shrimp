"""Analytic fixtures only: no neural target orbit is generated here."""
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location("hopfield_continuation", TOOLS / "hopfield_continuation.py")
continuation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = continuation
SPEC.loader.exec_module(continuation)


def fold_rhs(t, state, parameter):
    x, y, z = state
    radius_squared = x * x + y * y
    radial = parameter - (radius_squared - 1) ** 2
    return np.array([radial * x - y, radial * y + x, -0.7 * z])


def fold_jacobian(t, state, parameter):
    x, y, _ = state
    radius_squared = x * x + y * y
    radial = parameter - (radius_squared - 1) ** 2
    gx, gy = -4 * x * (radius_squared - 1), -4 * y * (radius_squared - 1)
    return np.array([[radial + x * gx, x * gy - 1, 0],
                     [y * gx + 1, radial + y * gy, 0], [0, 0, -0.7]])


def fold_parameter_derivative(t, state, parameter):
    return np.array([state[0], state[1], 0.0])


def initial_cycle(parameter=0.04, angle=0.0):
    radius = np.sqrt(1 + np.sqrt(parameter))
    return np.array([radius * np.cos(angle), radius * np.sin(angle), 0.0])


def test_scaled_shooting_jacobian_matches_central_differences():
    config = continuation.ContinuationConfig()
    state = initial_cycle(angle=0.3)
    scales = np.array([0.7, 1.3, 3.0])
    parameter_scale = 0.8
    normal = fold_rhs(0, state, 0.04)
    normal /= np.linalg.norm(normal)
    z = np.r_[state / scales, np.log(2 * np.pi), 0.04 / parameter_scale]
    def system(value):
        return continuation.shooting_system(value, state, normal, fold_rhs,
                 fold_jacobian, fold_parameter_derivative, scales,
                 parameter_scale, config)
    _, analytic, _ = system(z)
    epsilon = 1e-6
    numerical = np.column_stack([(system(z + epsilon * axis)[0] -
                                  system(z - epsilon * axis)[0]) / (2 * epsilon)
                                 for axis in np.eye(5)])
    np.testing.assert_allclose(analytic, numerical, rtol=2e-6, atol=2e-7)


def test_fold_continuation_reverses_parameter_and_retains_unstable_branch():
    result = continuation.continue_ray(
        fold_rhs, fold_jacobian, fold_parameter_derivative, initial_cycle(), 2 * np.pi,
        initial_parameter=0.04, direction=-1, state_scales=(1, 1, 1),
        config={"initial_step": 0.01, "maximum_step": 0.02,
                "maximum_accepted": 18, "maximum_attempts": 30,
                "parameter_minimum": -0.1, "parameter_maximum": 0.3})
    assert result["accepted_count"] == 18
    assert result["stop_reason"] == "maximum_accepted"
    assert any(node["parameter_reversal"] for node in result["nodes"])
    assert any(node["unstable"] for node in result["nodes"])
    assert any("plus_one_bracket" in node["bracket_flags"] for node in result["nodes"])
    for node in result["nodes"]:
        assert node["verified"]
        radius_squared = np.dot(node["state"][:2], node["state"][:2])
        np.testing.assert_allclose(node["parameter"], (radius_squared - 1) ** 2, atol=2e-8)
        np.testing.assert_allclose(node["period"], 2 * np.pi, atol=2e-8)
        expected = sorted([np.exp(-8 * np.pi * radius_squared * (radius_squared - 1)),
                           np.exp(-1.4 * np.pi)])
        actual = sorted(value["modulus"] for value in node["transverse_multipliers"])
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-8)
    json.dumps(result, allow_nan=False)


def test_flow_projection_has_no_neutral_eigenvalue_ambiguity_at_fold():
    state = np.array([1.0, 0.0, 0.0])
    config = continuation.ContinuationConfig()
    _, monodromy, _ = continuation.variational_shoot(
        fold_rhs, fold_jacobian, fold_parameter_derivative, state, 2 * np.pi, 0.0, config)
    flow = fold_rhs(0, state, 0.0)
    result = continuation.poincare_diagnostics(monodromy, flow, flow / np.linalg.norm(flow))
    np.testing.assert_allclose(sorted(v["modulus"] for v in result["transverse_multipliers"]),
                               [np.exp(-1.4 * np.pi), 1.0], atol=2e-8)
    assert result["flow_residual"] < 1e-8


def test_rotated_phase_continues_the_same_analytic_branch():
    results = [continuation.continue_ray(
        fold_rhs, fold_jacobian, fold_parameter_derivative, initial_cycle(angle=angle), 2 * np.pi,
        initial_parameter=0.04, direction=-1, state_scales=(1, 1, 1),
        config={"maximum_accepted": 3, "maximum_attempts": 6}) for angle in (0.0, 0.71)]
    for a, b in zip(results[0]["nodes"], results[1]["nodes"]):
        np.testing.assert_allclose(a["parameter"], b["parameter"], atol=1e-8)
        np.testing.assert_allclose(a["transverse_spectral_radius"], b["transverse_spectral_radius"], atol=1e-8)


def test_failed_correctors_are_retained_and_not_called_branch_loss():
    emitted_attempts, emitted_nodes = [], []
    result = continuation.continue_ray(
        fold_rhs, fold_jacobian, fold_parameter_derivative, initial_cycle(), 2 * np.pi,
        initial_parameter=0.04, state_scales=(1, 1, 1),
        config={"initial_step": 0.004, "minimum_step": 0.001,
                "maximum_step": 0.004, "maximum_attempts": 10},
        node_validator=lambda node: {"gates": {"forced_failure_fixture": node["index"] == 0}},
        on_attempt=emitted_attempts.append, on_node=emitted_nodes.append)
    assert len(result["nodes"]) == len(emitted_nodes) == 1
    assert len(result["attempts"]) == len(emitted_attempts) == 3
    assert all(not attempt["accepted"] for attempt in result["attempts"])
    assert all("forced_failure_fixture" in attempt["reason"] for attempt in result["attempts"])
    assert all("corrected_node" in attempt for attempt in result["attempts"])
    assert result["stop_reason"] == "minimum_step_after_failed_corrector"
    json.dumps(result, allow_nan=False)


def test_synthetic_flip_crossing_is_only_labeled_a_bracket():
    def node(values):
        return {"transverse_multipliers": [
            {"real": value, "imag": 0, "modulus": abs(value)} for value in values]}
    assert continuation.bracket_flags(node([-0.9, -0.02]), node([-1.1, -0.021])) == [
        "minus_one_bracket", "unit_circle_bracket"]
    assert continuation.bracket_flags(node([0.9, 0.02]), node([1.1, 0.021])) == [
        "plus_one_bracket", "unit_circle_bracket"]
    assert continuation.bracket_flags(node([0.9, 0.02]), node([0.8, 0.021])) == []


def test_out_of_bounds_predictor_failure_is_logged():
    result = continuation.continue_ray(
        fold_rhs, fold_jacobian, fold_parameter_derivative, initial_cycle(), 2 * np.pi,
        initial_parameter=0.04, direction=1, state_scales=(1, 1, 1),
        config={"initial_step": 0.005, "minimum_step": 0.002, "maximum_step": 0.005,
                "parameter_minimum": 0.0, "parameter_maximum": 0.04000001})
    assert len(result["attempts"]) == 2
    assert all(not row["accepted"] for row in result["attempts"])
    assert all("bounds" in row["reason"] for row in result["attempts"])
