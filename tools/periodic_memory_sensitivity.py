"""Total return-trace sensitivity to memory, including periodic-orbit motion.

At beta=0, F_beta(x)=(1-beta)F(x)+beta*x has forcing x-F(x).
Callbacks supply the original Jacobians and Hessian-direction products. These
utilities do not load a model or differentiate an incompletely resolved orbit.
"""

import numpy as np

from trace_pair_sensitivity import isolated_leading_pair


def _real_array(value, label, shape=None):
    value = np.asarray(value)
    if (not np.isfinite(value).all() or
            (np.iscomplexobj(value) and np.any(value.imag != 0)) or
            (shape is not None and value.shape != shape)):
        raise ValueError(f"{label} must have finite real entries and the required shape")
    return np.asarray(value.real, dtype=np.float64)


def _positive_gates(**gates):
    if any(not np.isfinite(v) or v <= 0 for v in gates.values()):
        raise ValueError("all sensitivity tolerances must be finite and positive")


def periodic_memory_tangent(matrix, states, step_callbacks, *,
                            orbit_closure_tolerance=1e-11,
                            solve_tolerance=1e-10,
                            tangent_closure_tolerance=1e-7,
                            resolvent_condition_limit=1e10):
    """Solve and independently replay the periodic orbit's beta tangent.

``states`` has shape (p+1,n), including the measured final return. There is one
``apply_J(V)`` callback per step, accepting n-by-k column batches. The tangent
obeys eta[i+1]=J[i]eta[i]+states[i]-states[i+1]. Its initial value solves
(I-M)eta[0]=b, where b is the zero-start forced tangent after one period.

Closure here qualifies a supplied orbit; this utility does not establish its
primitive period, which remains the caller's responsibility.
"""
    matrix = _real_array(matrix, "monodromy")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or not len(matrix):
        raise ValueError("a nonempty square monodromy is required")
    n = len(matrix)
    states = _real_array(states, "orbit states")
    callbacks = list(step_callbacks)
    if states.ndim != 2 or states.shape[1] != n or len(states) < 2 or len(callbacks) != len(states)-1:
        raise ValueError("p+1 orbit states and exactly p Jacobian callbacks are required")
    _positive_gates(orbit_closure_tolerance=orbit_closure_tolerance,
                    solve_tolerance=solve_tolerance,
                    tangent_closure_tolerance=tangent_closure_tolerance,
                    resolvent_condition_limit=resolvent_condition_limit)
    state_scale = float(np.linalg.norm(states[0]))
    closure = float(np.linalg.norm(states[-1]-states[0]) / max(state_scale, 1e-300))
    if closure >= orbit_closure_tolerance:
        raise ValueError("the supplied periodic orbit failed the closure gate")
    forcing = states[:-1] - states[1:]

    def applied(callback, vector):
        return _real_array(callback(vector[:, None]), "Jacobian action", (n, 1))[:, 0]

    b = np.zeros(n)
    for callback, force in zip(callbacks, forcing):
        b = applied(callback, b) + force
    resolvent = np.eye(n) - matrix
    condition = float(np.linalg.cond(resolvent))
    if not np.isfinite(condition) or condition > resolvent_condition_limit:
        raise ValueError("the periodic tangent resolvent is ill-conditioned")
    initial = np.linalg.solve(resolvent, b)
    solve_error = float(np.linalg.norm(resolvent @ initial - b) /
                        max(np.linalg.norm(resolvent @ initial), np.linalg.norm(b), 1e-300))
    if not np.isfinite(solve_error) or solve_error >= solve_tolerance:
        raise ValueError("the periodic tangent linear solve failed its residual gate")
    tangents = [initial]
    for callback, force in zip(callbacks, forcing):
        tangents.append(applied(callback, tangents[-1]) + force)
    tangents = np.stack(tangents)
    tangent_scale = max(float(np.linalg.norm(tangents, axis=1).max()), 1e-300)
    tangent_absolute_closure = float(np.linalg.norm(tangents[-1]-tangents[0]))
    tangent_closure = tangent_absolute_closure / tangent_scale
    if not np.isfinite(tangent_closure) or tangent_closure >= tangent_closure_tolerance:
        raise ValueError("independently chained periodic tangent failed closure")
    return {"passed": True, "tangents": tangents, "forcing": forcing,
            "zero_start_return": b, "orbit_relative_closure": closure,
            "solve_relative_residual": solve_error,
            "tangent_relative_closure": tangent_closure,
            "tangent_absolute_closure": tangent_absolute_closure,
            "maximum_phase_tangent_norm": tangent_scale,
            "resolvent_condition": condition,
            "gates": {"orbit_closure_tolerance": orbit_closure_tolerance,
                      "solve_tolerance": solve_tolerance,
                      "tangent_closure_tolerance": tangent_closure_tolerance,
                      "resolvent_condition_limit": resolvent_condition_limit}}


def memory_product_derivatives(step_callbacks, hessian_callbacks, tangents, directions):
    """Propagate return tangents and both pieces of the total beta derivative.

``apply_H(eta,V)`` returns columns D^2F(x_i)[eta,V] at the saved point.
``tangents`` contains p+1 rows from ``periodic_memory_tangent``. Each Jacobian
callback is applied once per step to concatenated [w,explicit,orbit] columns.
The explicit and orbit-mediated contributions are propagated separately,
avoiding their extraction by subtraction of two nearly equal trace values.
"""
    callbacks, hessians = list(step_callbacks), list(hessian_callbacks)
    w = _real_array(directions, "directions")
    if w.ndim != 2 or not all(w.shape) or not callbacks or len(callbacks) != len(hessians):
        raise ValueError("nonempty tangent columns and matched Jacobian/Hessian callbacks are required")
    tangents = _real_array(tangents, "periodic tangents", (len(callbacks)+1, w.shape[0]))
    w = w.copy()
    explicit = np.zeros_like(w)
    mediated = np.zeros_like(w)
    n, k = w.shape
    for index, (apply_j, apply_h) in enumerate(zip(callbacks, hessians)):
        applied = _real_array(apply_j(np.concatenate([w, explicit, mediated], axis=1)),
                              "Jacobian action", (n, 3*k))
        jw, je, jo = np.split(applied, 3, axis=1)
        h = _real_array(apply_h(tangents[index].copy(), w.copy()), "Hessian-direction action", (n, k))
        explicit, mediated, w = je + w - jw, jo + h, jw
    total = explicit + mediated
    if not np.isfinite(total).all():
        raise ValueError("nonfinite total return derivative")
    return {"propagated": w, "explicit": explicit,
            "orbit_mediated": mediated, "total": total}


def periodic_memory_trace_sensitivity(matrix, states, step_callbacks, hessian_callbacks, *,
                                      tangent_gates=None, pair_gates=None,
                                      operator_tolerance=1e-7):
    """Return gated, basis-invariant pair-trace derivative contributions.

The total derivative includes motion of the periodic orbit at fixed original
alpha and temperature. It equals the derivative of a moving saddle's height
only when those coordinates are stationary; stationarity is not checked here.
Hessian callbacks must already have their own arithmetic/convergence checks.
"""
    _positive_gates(operator_tolerance=operator_tolerance)
    callbacks, hessians = list(step_callbacks), list(hessian_callbacks)
    matrix = _real_array(matrix, "monodromy")
    orbit = periodic_memory_tangent(matrix, states, callbacks, **(tangent_gates or {}))
    pair = isolated_leading_pair(matrix, **(pair_gates or {}))
    columns = memory_product_derivatives(callbacks, hessians, orbit["tangents"], pair["right"])
    dense = matrix @ pair["right"]
    errors = np.linalg.norm(columns["propagated"] - dense, axis=0) / np.maximum(
        np.maximum(np.linalg.norm(columns["propagated"], axis=0), np.linalg.norm(dense, axis=0)), 1e-300)
    if not np.isfinite(errors).all() or errors.max() >= operator_tolerance:
        raise ValueError("composed Jacobian does not match the supplied monodromy on the pair")
    derivatives = {name: np.einsum("ij,ij->j", pair["left"], columns[name])
                   for name in ("explicit", "orbit_mediated", "total")}
    if not all(np.isfinite(v).all() for v in derivatives.values()):
        raise ValueError("nonfinite multiplier derivative")
    return {"passed": True, "orbit_tangent": orbit, "pair": pair,
            "operator_relative_errors": errors, "operator_tolerance": operator_tolerance,
            "derivative_columns": columns, "multiplier_derivatives": derivatives,
            "trace_derivatives": {name: float(v.sum()) for name, v in derivatives.items()},
            "scope": "Total fixed-alpha, fixed-temperature memory response with periodic-orbit motion; not a saddle-height or native coupling certificate."}
