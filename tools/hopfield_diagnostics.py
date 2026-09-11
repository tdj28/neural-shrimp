"""Independent continuous-time orbit certificates and neuron-level probes.

The scan's discretization is never used for certification. Shooting and the
variational equation use adaptive DOP853 integration. An autonomous limit cycle
has one *neutral* Floquet multiplier; only the other two must contract.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.spatial import cKDTree

RTOL = 1e-10
ATOL = 1e-12
STABILITY_MARGIN = 1e-5
CLOSURE_TOL = 1e-8
NEUTRAL_TOL = 1e-6
FLOW_ALIGNMENT_MIN = 0.99


def weight_matrix(alpha: float, gamma: float) -> np.ndarray:
    return np.asarray([[1.5, 2.9, alpha], [-1.5, 1.18, 0.0],
                       [gamma, -22.0, 0.47]], dtype=float)


def hopfield_rhs(t: float, state: np.ndarray, alpha: float,
                 gamma: float) -> np.ndarray:
    del t
    return -np.asarray(state) + weight_matrix(alpha, gamma) @ np.tanh(state)


def hopfield_jacobian(t: float, state: np.ndarray, alpha: float,
                      gamma: float) -> np.ndarray:
    del t
    activation = np.tanh(state)
    return -np.eye(3) + weight_matrix(alpha, gamma) * (1.0 - activation**2)[None, :]


def _integrate_variational(rhs: Callable, jacobian: Callable,
                           state: np.ndarray, period: float,
                           *, rtol: float = RTOL, atol: float = ATOL):
    """Return solve_ivp result for state followed by row-major tangent matrix."""
    state = np.asarray(state, dtype=float)
    size = state.size
    if not np.isfinite(period) or period <= 0:
        raise ValueError("period must be positive and finite")

    def combined(t, packed):
        x = packed[:size]
        tangent = packed[size:].reshape(size, size)
        return np.concatenate((rhs(t, x), (jacobian(t, x) @ tangent).ravel()))

    initial = np.concatenate((state, np.eye(size).ravel()))
    solution = solve_ivp(combined, (0.0, period), initial, method="DOP853",
                         rtol=rtol, atol=atol, dense_output=True,
                         max_step=min(0.2, period / 100.0))
    if not solution.success or not np.isfinite(solution.y).all():
        raise RuntimeError(f"variational integration failed: {solution.message}")
    return solution


def _complex_values(values):
    return [{"real": float(value.real), "imag": float(value.imag),
             "modulus": float(abs(value))} for value in values]


def _floquet_diagnostics(monodromy: np.ndarray, flow: np.ndarray) -> dict:
    values, vectors = np.linalg.eig(monodromy)
    neutral_index = int(np.argmin(np.abs(values - 1.0)))
    transverse = np.delete(values, neutral_index)
    norm_flow = float(np.linalg.norm(flow))
    if norm_flow <= 1e-12:
        alignment = 0.0
        flow_residual = None
    else:
        vector = vectors[:, neutral_index]
        alignment = float(abs(np.vdot(vector, flow)) /
                          (np.linalg.norm(vector) * norm_flow))
        flow_residual = float(np.linalg.norm(monodromy @ flow - flow) / norm_flow)
    neutral_error = float(abs(values[neutral_index] - 1.0))
    radius = float(np.max(np.abs(transverse)))
    # Sorting is not needed: retain the eigensolver order and identify neutral.
    return {
        "multipliers": _complex_values(values),
        "neutral_index": neutral_index,
        "neutral_multiplier_error": neutral_error,
        "neutral_flow_alignment": alignment,
        "neutral_flow_residual": flow_residual,
        "transverse_multipliers": _complex_values(transverse),
        "transverse_spectral_radius": radius,
        "neutral_verified": bool(flow_residual is not None and
                                  neutral_error <= NEUTRAL_TOL and
                                  flow_residual <= NEUTRAL_TOL and
                                  alignment >= FLOW_ALIGNMENT_MIN),
        "transversely_attracting": bool(radius < 1.0 - STABILITY_MARGIN),
    }


def _failure(reason: str, **fields) -> dict:
    return {"verified": False, "attracting": False, "reason": reason, **fields}


def certify(alpha: float, gamma: float, section_state,
            period_time: float, expected_peak_count: int) -> dict:
    """Shoot a primitive cycle starting at an x3 maximum and certify stability.

    ``section_state`` and ``period_time`` are scan estimates. All failure gates
    are retained in the result. A root that reaches an equilibrium, a multiple
    cover, the wrong maximum count, or a nonneutral tangent mode is rejected.
    The period is parameterized by log(T), preserving positivity during root
    solving. T remains within [T_seed/4, 4*T_seed] at every root evaluation.
    """
    initial = np.asarray(section_state, dtype=float)
    if initial.shape != (3,) or not np.isfinite(initial).all():
        raise ValueError("section_state must contain three finite values")
    if not np.isfinite([alpha, gamma, period_time]).all() or period_time <= 0:
        raise ValueError("parameters must be finite and period_time positive")
    if not isinstance(expected_peak_count, (int, np.integer)) or expected_peak_count < 1:
        raise ValueError("expected_peak_count must be a positive integer")
    rhs = lambda t, x: hopfield_rhs(t, x, alpha, gamma)
    jacobian = lambda t, x: hopfield_jacobian(t, x, alpha, gamma)
    lower_log = np.log(period_time / 4.0)
    upper_log = np.log(period_time * 4.0)
    evaluations = 0

    def residual(candidate):
        nonlocal evaluations
        evaluations += 1
        if not np.isfinite(candidate).all() or not lower_log <= candidate[3] <= upper_log:
            raise ValueError("shooting period left the declared seed neighborhood")
        cycle_period = float(np.exp(candidate[3]))
        solution = solve_ivp(rhs, (0.0, cycle_period), candidate[:3],
                             method="DOP853", rtol=RTOL, atol=ATOL,
                             max_step=min(0.2, cycle_period / 100.0))
        if not solution.success or not np.isfinite(solution.y).all():
            raise RuntimeError("shooting integration failed")
        return np.concatenate((solution.y[:, -1] - candidate[:3],
                               [rhs(0.0, candidate[:3])[2]]))

    solver_fields = {
        "alpha": float(alpha), "gamma": float(gamma),
        "seed_section_state": initial.tolist(), "seed_period": float(period_time),
        "expected_peak_count": int(expected_peak_count),
        "integration": {"method": "DOP853", "rtol": RTOL, "atol": ATOL,
                        "max_step": "min(0.2, T/100)"},
        "thresholds": {"relative_closure": CLOSURE_TOL,
                       "neutral_error": NEUTRAL_TOL,
                       "neutral_flow_residual": NEUTRAL_TOL,
                       "neutral_flow_alignment_min": FLOW_ALIGNMENT_MIN,
                       "transverse_stability_margin": STABILITY_MARGIN},
    }
    try:
        solved = root(residual, np.r_[initial, np.log(period_time)],
                      method="hybr", options={"xtol": 1e-10, "maxfev": 120})
        state = np.asarray(solved.x[:3])
        period = float(np.exp(solved.x[3]))
        shooting_residual = residual(solved.x)
        solution = _integrate_variational(rhs, jacobian, state, period)
    except (ValueError, RuntimeError, FloatingPointError) as error:
        return _failure(str(error), **solver_fields, root_evaluations=evaluations)

    endpoint = solution.y[:3, -1]
    monodromy = solution.y[3:, -1].reshape(3, 3)
    flow = rhs(0.0, state)
    relative_closure = float(np.linalg.norm(endpoint - state) /
                             max(1.0, np.linalg.norm(state)))
    section_residual = float(abs(flow[2]))
    section_curvature = float((jacobian(0.0, state) @ flow)[2])
    flow_norm = float(np.linalg.norm(flow))
    floquet = _floquet_diagnostics(monodromy, flow)

    def maxima(t, x):
        return rhs(t, x)[2]
    maxima.direction = -1
    maxima.terminal = False
    events = solve_ivp(rhs, (0.0, period), state, method="DOP853",
                       rtol=RTOL, atol=ATOL, events=maxima, dense_output=True,
                       max_step=min(0.05, period / (100.0 * expected_peak_count)))
    if not events.success:
        return _failure("maximum-event integration failed", **solver_fields,
                        root_evaluations=evaluations)
    # The chosen starting maximum counts exactly once. Events close to either
    # endpoint represent that same maximum and are excluded from the interior.
    endpoint_tolerance = max(1e-7, 1e-6 * period)
    interior_times = events.t_events[0]
    interior_times = interior_times[(interior_times > endpoint_tolerance) &
                                    (interior_times < period - endpoint_tolerance)]
    interior_states = events.sol(interior_times).T
    distances = (np.linalg.norm(interior_states - state, axis=1) /
                 max(1.0, np.linalg.norm(state)))
    primitive = bool(np.all(distances > 10 * CLOSURE_TOL))
    peak_count = 1 + len(interior_times)
    times = np.linspace(0.0, period, max(1001, 300 * expected_peak_count + 1))
    states = solution.sol(times)[:3].T
    amplitude = float(np.max(np.ptp(states, axis=0)))
    gates = {
        "closure": relative_closure <= CLOSURE_TOL,
        "phase_condition": section_residual <= CLOSURE_TOL,
        "section_is_maximum": section_curvature < -1e-8,
        "nonstationary": flow_norm > 1e-7 and amplitude > 1e-6,
        "expected_peak_count": peak_count == expected_peak_count,
        "primitive": primitive,
        "neutral_mode": floquet["neutral_verified"],
        "transverse_stability": floquet["transversely_attracting"],
    }
    orbit_verified = all(value for name, value in gates.items()
                         if name != "transverse_stability")
    return {
        **solver_fields,
        "verified": bool(orbit_verified),
        "attracting": bool(all(gates.values())),
        "reason": "passed" if all(gates.values()) else ", ".join(
            name for name, passed in gates.items() if not passed),
        "gates": {name: bool(value) for name, value in gates.items()},
        "root_success": bool(solved.success), "root_message": str(solved.message),
        "root_evaluations": evaluations,
        "shooting_residual_norm": float(np.linalg.norm(shooting_residual)),
        "seed_state_correction": float(np.linalg.norm(state - initial)),
        "section_state": state.tolist(), "period": period,
        "relative_closure": relative_closure,
        "section_residual": section_residual,
        "section_curvature": section_curvature,
        "flow_norm": flow_norm, "state_amplitude": amplitude,
        "peak_count": int(peak_count), "primitive": primitive,
        "interior_maximum_times": interior_times.tolist(),
        "interior_maximum_return_distances": distances.tolist(),
        "monodromy": monodromy.tolist(), **floquet,
        "orbit_times": times.tolist(), "orbit_states": states.tolist(),
        "orbit_activations": np.tanh(states).tolist(),
        "scope": "Numerical autonomous-ODE cycle certificate, not a rigorous interval proof.",
    }


def behavior_probe(alpha: float, gamma: float, initial_state, duration: float = 300.0,
                   sample_dt: float = 0.05, certificate: dict | None = None) -> dict:
    """Record all neuron states, bounded activations, and paired perturbation.

    Summary moments use the final half of the fixed duration, without choosing
    an outcome-dependent burn-in. Separation is measured at equal times; when
    a certificate is supplied, distance to its sampled orbit is also reported
    because autonomous oscillators can retain a neutral phase displacement.
    """
    initial = np.asarray(initial_state, dtype=float)
    if initial.shape != (3,) or not np.isfinite(initial).all():
        raise ValueError("initial_state must contain three finite values")
    if not np.isfinite([alpha, gamma, duration, sample_dt]).all() or min(duration, sample_dt) <= 0:
        raise ValueError("parameters must be finite and durations positive")
    if sample_dt > duration / 4:
        raise ValueError("sample_dt must provide at least five time samples")
    epsilon = 1e-6
    direction = np.ones(3) / np.sqrt(3.0)
    times = np.arange(int(np.floor(duration / sample_dt)) + 1) * sample_dt
    if times[-1] < duration - 1e-12:
        times = np.r_[times, duration]
    else:
        times[-1] = duration
    rhs = lambda t, x: hopfield_rhs(t, x, alpha, gamma)

    def integrate(state):
        solution = solve_ivp(rhs, (0.0, duration), state, method="DOP853",
                             rtol=RTOL, atol=ATOL, t_eval=times, dense_output=True,
                             max_step=0.2)
        if not solution.success or not np.isfinite(solution.y).all():
            raise RuntimeError(f"behavior integration failed: {solution.message}")
        return solution

    baseline = integrate(initial)
    perturbed = integrate(initial + epsilon * direction)
    states = baseline.y.T
    perturbed_states = perturbed.y.T
    activations = np.tanh(states)
    perturbed_activations = np.tanh(perturbed_states)
    separation = np.linalg.norm(states - perturbed_states, axis=1)
    activation_separation = np.linalg.norm(activations - perturbed_activations, axis=1)
    stationary = times >= duration / 2

    def moments(values):
        tail = values[stationary]
        return {"mean": np.mean(tail, axis=0).tolist(),
                "std": np.std(tail, axis=0, ddof=0).tolist(),
                "minimum": np.min(tail, axis=0).tolist(),
                "maximum": np.max(tail, axis=0).tolist()}

    metrics = {
        "alpha": float(alpha), "gamma": float(gamma),
        "initial_state": initial.tolist(), "duration": duration,
        "sample_dt": sample_dt, "summary_window": [duration / 2, duration],
        "state_moments": moments(states), "activation_moments": moments(activations),
        "perturbation": {"epsilon": epsilon, "direction": direction.tolist(),
                         "initial_separation": float(separation[0]),
                         "final_separation": float(separation[-1]),
                         "max_separation": float(np.max(separation)),
                         "rms_tail_separation": float(np.sqrt(np.mean(separation[stationary]**2))),
                         "final_activation_separation": float(activation_separation[-1]),
                         "max_activation_separation": float(np.max(activation_separation)),
                         "finite_horizon_log_amplification_per_time":
                             float(np.log(max(separation[-1], np.finfo(float).tiny) /
                                          separation[0]) / duration)},
        "integration": {"method": "DOP853", "rtol": RTOL, "atol": ATOL,
                        "max_step": 0.2},
        "scope": "Neuron state and activation observables; no task-accuracy interpretation.",
    }
    arrays = {
        "times": times, "states": states, "activations": activations,
        "perturbed_states": perturbed_states,
        "perturbed_activations": perturbed_activations,
        "separation": separation, "activation_separation": activation_separation,
    }
    if certificate and certificate.get("verified"):
        period = float(certificate["period"])
        paired_times = times[(times >= duration / 2) & (times + period <= duration)]
        if paired_times.size:
            residual = (baseline.sol(paired_times + period) - baseline.sol(paired_times)).T
            metrics["cycle_repetition"] = {
                "period": period, "sample_count": int(paired_times.size),
                "per_neuron_state_rms": np.sqrt(np.mean(residual**2, axis=0)).tolist(),
                "state_rms": float(np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
            }
        orbit = np.asarray(certificate["orbit_states"])
        tree = cKDTree(orbit)
        baseline_distance, _ = tree.query(states)
        perturbed_distance, _ = tree.query(perturbed_states)
        arrays.update(sampled_orbit_distance=baseline_distance,
                      perturbed_sampled_orbit_distance=perturbed_distance)
        metrics["sampled_orbit_distance"] = {
            "orbit_sample_count": len(orbit),
            "baseline_tail_rms": float(np.sqrt(np.mean(baseline_distance[stationary]**2))),
            "perturbed_tail_rms": float(np.sqrt(np.mean(perturbed_distance[stationary]**2))),
            "interpretation": "Nearest sampled-orbit point; includes discretization error, not exact phase projection.",
        }
    return {**arrays, "metrics": metrics}
