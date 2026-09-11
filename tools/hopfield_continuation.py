"""Bounded pseudo-arclength continuation of autonomous periodic orbits.

All corrector attempts are returned, including failures. Loss of attraction is
not a stopping condition. Numerical brackets are not bifurcation theorems.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

import hopfield_diagnostics as diagnostics


@dataclass(frozen=True)
class ContinuationConfig:
    initial_step: float = 0.005
    minimum_step: float = 1e-5
    maximum_step: float = 0.01
    maximum_accepted: int = 48
    maximum_attempts: int = 96
    maximum_newton_iterations: int = 10
    maximum_line_halvings: int = 6
    residual_tolerance: float = 1e-9
    closure_tolerance: float = 1e-8
    flow_residual_tolerance: float = 1e-6
    phase_tolerance: float = 1e-8
    rtol: float = 1e-10
    atol: float = 1e-12
    maximum_ivp_step: float = 0.2
    parameter_minimum: float = -np.inf
    parameter_maximum: float = np.inf


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(item) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def variational_shoot(rhs, jacobian, parameter_derivative, state, period,
                      parameter, config):
    """Integrate x, monodromy M and fixed-initial-state sensitivity S."""
    state = np.asarray(state, float)
    size = state.size

    def combined(t, packed):
        x = packed[:size]
        tangent = packed[size:size + size * size].reshape(size, size)
        sensitivity = packed[-size:]
        jac = jacobian(t, x, parameter)
        return np.r_[rhs(t, x, parameter), (jac @ tangent).ravel(),
                     jac @ sensitivity + parameter_derivative(t, x, parameter)]

    packed = np.r_[state, np.eye(size).ravel(), np.zeros(size)]
    solved = solve_ivp(combined, (0.0, float(period)), packed, method="DOP853",
                       rtol=config.rtol, atol=config.atol,
                       max_step=min(config.maximum_ivp_step, period / 100.0))
    if not solved.success or not np.isfinite(solved.y).all():
        raise RuntimeError(f"variational integration failed: {solved.message}")
    return (solved.y[:size, -1],
            solved.y[size:size + size * size, -1].reshape(size, size),
            solved.y[-size:, -1])


def poincare_diagnostics(monodromy, flow, normal):
    """Remove the neutral direction geometrically, including near a +1 fold."""
    flow, normal = np.asarray(flow, float), np.asarray(normal, float)
    norm_flow = float(np.linalg.norm(flow))
    denominator = float(normal @ flow)
    if norm_flow <= 1e-7 or abs(denominator) <= 1e-8 * norm_flow:
        raise ValueError("phase section is tangent to the flow or stationary")
    basis = np.linalg.svd(normal[None, :], full_matrices=True)[2][1:].T
    projection = np.eye(flow.size) - np.outer(flow, normal) / denominator
    reduced = basis.T @ projection @ monodromy @ basis
    values = np.linalg.eigvals(reduced)
    values = values[np.lexsort((values.imag, values.real))]
    return {
        "poincare_matrix": reduced.tolist(),
        "transverse_multipliers": diagnostics._complex_values(values),
        "transverse_spectral_radius": float(np.max(np.abs(values))),
        "attracting": bool(np.max(np.abs(values)) < 1 - 1e-5),
        "unstable": bool(np.max(np.abs(values)) > 1 + 1e-5),
        "flow_residual": float(np.linalg.norm(monodromy @ flow - flow) / norm_flow),
        "section_transversality": float(abs(denominator) / norm_flow),
    }


def shooting_system(z, reference_state, normal, rhs, jacobian,
                    parameter_derivative, state_scales, parameter_scale, config):
    """Return scaled closure/phase residual and its exact variational Jacobian."""
    scales = np.asarray(state_scales, float)
    size = scales.size
    state, period, parameter = z[:size] * scales, np.exp(z[size]), z[-1] * parameter_scale
    endpoint, monodromy, sensitivity = variational_shoot(
        rhs, jacobian, parameter_derivative, state, period, parameter, config)
    phase_scale = float(np.linalg.norm(normal * scales))
    residual = np.r_[(endpoint - state) / scales,
                     normal @ (state - reference_state) / phase_scale]
    matrix = np.zeros((size + 1, size + 2))
    matrix[:size, :size] = (monodromy - np.eye(size)) * scales[None, :] / scales[:, None]
    matrix[:size, size] = period * rhs(period, endpoint, parameter) / scales
    matrix[:size, -1] = sensitivity * parameter_scale / scales
    matrix[-1, :size] = normal * scales / phase_scale
    details = {"state": state, "period": period, "parameter": parameter,
               "endpoint": endpoint, "monodromy": monodromy,
               "sensitivity": sensitivity}
    return residual, matrix, details


def bracket_flags(previous, current):
    """Endpoint sign brackets, never exact bifurcation classifications."""
    a = np.array([complex(v["real"], v["imag"]) for v in previous["transverse_multipliers"]])
    b = np.array([complex(v["real"], v["imag"]) for v in current["transverse_multipliers"]])
    # There are exactly two transverse modes for the three-neuron target.
    if a.size == 2 and abs(a[0] - b[1]) + abs(a[1] - b[0]) < abs(a[0] - b[0]) + abs(a[1] - b[1]):
        b = b[::-1]
    flags = []
    for target, label in [(1.0, "plus_one_bracket"), (-1.0, "minus_one_bracket")]:
        if any(abs(x.imag) < 1e-7 and abs(y.imag) < 1e-7 and
               (x.real - target) * (y.real - target) < 0 for x, y in zip(a, b)):
            flags.append(label)
    if (max(abs(a)) - 1) * (max(abs(b)) - 1) < 0:
        flags.append("unit_circle_bracket")
    return flags


def _config(config):
    config = ContinuationConfig(**config) if isinstance(config, dict) else config or ContinuationConfig()
    if not (0 < config.minimum_step <= config.initial_step <= config.maximum_step):
        raise ValueError("invalid continuation step bounds")
    if min(config.maximum_accepted, config.maximum_attempts, config.maximum_newton_iterations) < 1:
        raise ValueError("continuation caps must be positive")
    if config.maximum_line_halvings < 0 or config.parameter_minimum >= config.parameter_maximum:
        raise ValueError("invalid line-search or parameter bounds")
    return config


def continue_ray(rhs: Callable, jacobian: Callable, parameter_derivative: Callable,
                 initial_state, period, *, initial_parameter=0.0, direction=1,
                 state_scales=(1.0, 1.0, 20.0), parameter_scale=1.0,
                 config=None, node_validator=None, on_attempt=None, on_node=None):
    """Continue one oriented branch; callbacks receive durable JSON-safe records.

    `rhs`, `jacobian`, and `parameter_derivative` take (t, x, scalar_parameter).
    The parameter derivative is partial f/partial parameter, holding x fixed.
    `node_validator(node)` may add a `gates` mapping and compact diagnostics.
    Failure of any supplied gate rejects the corrector and is retained.
    """
    config = _config(config)
    initial = np.asarray(initial_state, float)
    scales = np.asarray(state_scales, float)
    if (initial.ndim != 1 or scales.shape != initial.shape or initial.size < 2 or
        not np.isfinite(initial).all() or not np.isfinite(scales).all() or
        np.any(scales <= 0) or not np.isfinite([period, initial_parameter, parameter_scale]).all() or
        period <= 0 or parameter_scale <= 0 or direction not in (-1, 1)):
        raise ValueError("invalid seed, scaling, period, or direction")
    if not config.parameter_minimum <= initial_parameter <= config.parameter_maximum:
        raise ValueError("initial parameter outside declared bounds")
    size = initial.size
    z = np.r_[initial / scales, np.log(period), initial_parameter / parameter_scale]
    period_bounds = (float(period / 4), float(period * 4))
    nodes, attempts = [], []
    step, tangent = config.initial_step, None
    accepted_count = 0
    stop_reason = "maximum_attempts"

    def permitted(candidate):
        return (np.isfinite(candidate).all() and
                np.log(period_bounds[0]) <= candidate[size] <= np.log(period_bounds[1]) and
                config.parameter_minimum <= candidate[-1] * parameter_scale <= config.parameter_maximum)

    def evaluate(candidate, reference_state, normal):
        if not permitted(candidate):
            raise ValueError("candidate outside declared period or parameter bounds")
        return shooting_system(candidate, reference_state, normal, rhs, jacobian,
                               parameter_derivative, scales, parameter_scale, config)

    def make_node(candidate, details, normal, reference_state, index):
        state, endpoint = details["state"], details["endpoint"]
        flow = rhs(0.0, state, details["parameter"])
        diagnostics_ = poincare_diagnostics(details["monodromy"], flow, normal)
        closure = float(np.linalg.norm(endpoint - state) / max(1.0, np.linalg.norm(state)))
        phase = float(abs(normal @ (state - reference_state)))
        node = {"index": index, "state": state.tolist(), "period": float(details["period"]),
                "parameter": float(details["parameter"]), "scaled_coordinates": candidate.tolist(),
                "monodromy": details["monodromy"].tolist(), "relative_closure": closure,
                "phase_residual": phase, "phase_normal": normal.tolist(),
                "reference_state": reference_state.tolist(), **diagnostics_,
                "gates": {"closure": closure <= config.closure_tolerance,
                          "phase": phase <= config.phase_tolerance,
                          "neutral_flow": diagnostics_["flow_residual"] <= config.flow_residual_tolerance}}
        if node_validator:
            extra = node_validator(node)
            node["gates"].update(extra.pop("gates", {}))
            node.update(extra)
        node["verified"] = bool(all(node["gates"].values()))
        return node

    initial_flow = rhs(0.0, initial, initial_parameter)
    if np.linalg.norm(initial_flow) <= 1e-7:
        raise ValueError("initial orbit is stationary")
    normal = initial_flow / np.linalg.norm(initial_flow)
    residual, matrix, details = evaluate(z, initial, normal)
    seed = make_node(z, details, normal, initial, 0)
    seed["bracket_flags"] = []
    seed["parameter_reversal"] = False
    nodes.append(json_safe(seed))
    if on_node:
        on_node(nodes[-1])
    if not seed["verified"]:
        return {"config": json_safe(asdict(config)), "nodes": nodes, "attempts": [],
                "stop_reason": "seed_validation_failed", "period_bounds": period_bounds}

    for attempt_index in range(config.maximum_attempts):
        attempt = {"attempt": attempt_index, "from_node": accepted_count,
                   "step": float(step), "iterations": [], "accepted": False}
        corrector_ok = False
        try:
            reference = z[:size] * scales
            flow = rhs(0.0, reference, z[-1] * parameter_scale)
            normal = flow / np.linalg.norm(flow)
            residual, matrix, details = evaluate(z, reference, normal)
            proposed_tangent = np.linalg.svd(matrix, full_matrices=True)[2][-1]
            if tangent is None:
                proposed_tangent *= direction if proposed_tangent[-1] >= 0 else -direction
            elif proposed_tangent @ tangent < 0:
                proposed_tangent *= -1
            tangent = proposed_tangent / np.linalg.norm(proposed_tangent)
            predictor = z + step * tangent
            candidate = predictor.copy()
            attempt.update(tangent=tangent.tolist(), predictor=predictor.tolist())
            for iteration in range(config.maximum_newton_iterations + 1):
                residual, matrix, details = evaluate(candidate, reference, normal)
                bordered_residual = np.r_[residual, tangent @ (candidate - predictor)]
                norm = float(np.linalg.norm(bordered_residual))
                record = {"iteration": iteration, "coordinates": candidate.tolist(),
                          "residual_norm": norm, "line_search": []}
                attempt["iterations"].append(record)
                if norm <= config.residual_tolerance:
                    corrector_ok = True
                    break
                if iteration == config.maximum_newton_iterations:
                    break
                correction = np.linalg.solve(np.vstack((matrix, tangent)), -bordered_residual)
                moved = False
                for half in range(config.maximum_line_halvings + 1):
                    factor = 2.0 ** -half
                    trial = candidate + factor * correction
                    line = {"factor": factor, "coordinates": trial.tolist()}
                    try:
                        trial_residual, _, _ = evaluate(trial, reference, normal)
                        trial_norm = float(np.linalg.norm(np.r_[trial_residual, tangent @ (trial - predictor)]))
                        line["residual_norm"] = trial_norm
                        if trial_norm <= (1 - 1e-4 * factor) * norm:
                            line["accepted"] = True
                            record["line_search"].append(line)
                            candidate = trial
                            moved = True
                            break
                        line["accepted"] = False
                    except (ValueError, RuntimeError, FloatingPointError) as error:
                        line.update(accepted=False, reason=str(error))
                    record["line_search"].append(line)
                if not moved:
                    raise RuntimeError("bounded line search failed")
            if not corrector_ok:
                raise RuntimeError("maximum Newton iterations reached")
            node = make_node(candidate, details, normal, reference, accepted_count + 1)
            attempt["corrected_node"] = node
            if not node["verified"]:
                raise RuntimeError("node validation failed: " + ", ".join(k for k, v in node["gates"].items() if not v))
            node["bracket_flags"] = bracket_flags(nodes[-1], node)
            delta = node["parameter"] - nodes[-1]["parameter"]
            previous_delta = (nodes[-1]["parameter"] - nodes[-2]["parameter"]) if len(nodes) > 1 else direction
            node["parameter_reversal"] = bool(delta * previous_delta < 0)
            attempt["accepted"] = True
            attempt["reason"] = "passed"
            accepted_count += 1
            z = candidate
            nodes.append(json_safe(node))
            if on_node:
                on_node(nodes[-1])
            if iteration <= 3:
                step = min(config.maximum_step, step * 1.25)
            elif iteration > 6:
                step = max(config.minimum_step, step / 2)
        except (ValueError, RuntimeError, FloatingPointError, np.linalg.LinAlgError) as error:
            attempt["reason"] = str(error)
            step /= 2
        attempts.append(json_safe(attempt))
        if on_attempt:
            on_attempt(attempts[-1])
        if accepted_count >= config.maximum_accepted:
            stop_reason = "maximum_accepted"
            break
        if step < config.minimum_step:
            stop_reason = "minimum_step_after_failed_corrector"
            break
    return {"config": json_safe(asdict(config)), "nodes": nodes, "attempts": attempts,
            "stop_reason": stop_reason, "period_bounds": period_bounds,
            "accepted_count": accepted_count,
            "scope": "Numerical branch continuation; failed correction does not establish branch or basin loss."}


def _hopfield_primitive(node, alpha, gamma):
    """Count maxima and reject a multiple cover from repeated maximum states."""
    state, period = np.asarray(node["state"]), node["period"]
    rhs = lambda t, x: diagnostics.hopfield_rhs(t, x, alpha, gamma)
    def maxima(t, x):
        return rhs(t, x)[2]
    maxima.direction = -1
    maxima.terminal = False
    solved = solve_ivp(rhs, (0.0, period), state, method="DOP853", rtol=1e-10,
                       atol=1e-12, events=maxima, dense_output=True,
                       max_step=min(0.05, period / 900.0))
    if not solved.success:
        return {"gates": {"primitive_nine_peaks": False}, "peak_check_error": solved.message}
    times = solved.t_events[0]
    tolerance = max(1e-7, 1e-6 * period)
    # A boundary event counts once, whichever side of t=0 numerical phase lands.
    if len(times) > 1 and times[0] < tolerance and period - times[-1] < tolerance:
        times = times[:-1]
    states = solved.sol(times).T
    minimum_distance = np.inf
    if len(states) > 1:
        for index in range(len(states) - 1):
            distances = np.linalg.norm(states[index + 1:] - states[index], axis=1)
            minimum_distance = min(minimum_distance, float(min(distances)) / max(1.0, np.linalg.norm(states[index])))
    primitive = len(times) == 9 and minimum_distance > 1e-7
    return {"gates": {"primitive_nine_peaks": bool(primitive)}, "peak_count": len(times),
            "minimum_peak_return_distance": float(minimum_distance),
            "first_maximum_state": states[0].tolist() if len(states) else None}


def continue_hopfield_branch(alpha, gamma, section_state, period, direction, *,
                             config=None, on_attempt=None, on_node=None):
    """Frozen EXP-006 gamma branch: gamma(s)=gamma_seed+0.41*s; alpha fixed."""
    config = _config(config)
    config = replace(config, parameter_minimum=(9.9 - gamma) / 0.41,
                     parameter_maximum=(10.7 - gamma) / 0.41)
    rhs = lambda t, x, s: diagnostics.hopfield_rhs(t, x, alpha, gamma + 0.41 * s)
    jacobian = lambda t, x, s: diagnostics.hopfield_jacobian(t, x, alpha, gamma + 0.41 * s)
    derivative = lambda t, x, s: np.array([0.0, 0.0, 0.41 * np.tanh(x[0])])

    def validator(node):
        return _hopfield_primitive(node, alpha, gamma + 0.41 * node["parameter"])

    checkpoint_certificates = []
    def callback(node):
        node["alpha"] = float(alpha)
        node["gamma"] = float(gamma + 0.41 * node["parameter"])
        if node["index"] % 8 == 0 and node["first_maximum_state"] is not None:
            node["checkpoint_certificate"] = _checkpoint(node)
            checkpoint_certificates.append(node["checkpoint_certificate"])
        if on_node:
            on_node(node)

    def _checkpoint(node):
        certificate = diagnostics.certify(alpha, node["gamma"], node["first_maximum_state"], node["period"], 9)
        # Dense traces are not needed in a branch receipt; retain all diagnostic gates.
        compact = {key: value for key, value in certificate.items()
                   if key not in ("orbit_times", "orbit_states", "orbit_activations")}
        compact["continuation_node"] = node["index"]
        compact["period_agreement"] = bool(certificate.get("period") is not None and
                                             abs(certificate["period"] / node["period"] - 1) <= 1e-6)
        compact["state_agreement"] = bool(certificate.get("section_state") is not None and
            np.linalg.norm(np.asarray(certificate["section_state"]) - node["first_maximum_state"]) /
            max(1.0, np.linalg.norm(node["first_maximum_state"])) <= 1e-6)
        compact["continued_orbit_agreement"] = bool(certificate.get("verified", False) and
            compact["period_agreement"] and compact["state_agreement"])
        return compact

    result = continue_ray(rhs, jacobian, derivative, section_state, period,
                          direction=direction, state_scales=(1.0, 1.0, 20.0),
                          config=config, node_validator=validator,
                          on_attempt=on_attempt, on_node=callback)
    if result["nodes"][-1]["index"] % 8 != 0 and result["nodes"][-1].get("first_maximum_state") is not None:
        checkpoint_certificates.append(_checkpoint(result["nodes"][-1]))
    result.update(alpha_seed=float(alpha), gamma_seed=float(gamma), direction=int(direction),
                  parameterization="alpha=alpha_seed; gamma=gamma_seed+0.41*s",
                  state_scales=[1.0, 1.0, 20.0], parameter_scale=1.0,
                  checkpoint_certificates=checkpoint_certificates)
    return json_safe(result)
