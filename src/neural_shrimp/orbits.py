"""Hardened numerical certification of periodic orbits of smooth maps.

``certify_orbit`` refines ``F^p(x) = x`` by damped Newton iteration with the
exact Jacobian ``M(x) - I``, where ``M`` is the ordered product of map
Jacobians along the orbit. Acceptance requires all of: Newton convergence
with a small final correction, normalized closure, proximity to the observed
candidate, and a primitive period separated from every proper divisor by a
margin. Multipliers come from ``M`` at the solution, with conditioning
diagnostics. The historical ``diagnostics.certify_cycle`` is unchanged.

This is floating-point verification, not interval arithmetic.
"""

import numpy as np


def _advance(map_fn, x, count):
    state = np.asarray(x, dtype=np.float64).copy()
    orbit = [state.copy()]
    for _ in range(count):
        state = np.asarray(map_fn(state), dtype=np.float64)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError("nonfinite orbit")
        orbit.append(state.copy())
    return orbit


def _monodromy(jacobian, orbit_points):
    n = len(orbit_points[0])
    product = np.eye(n)
    for point in orbit_points:
        matrix = np.asarray(jacobian(point), dtype=np.float64)
        if matrix.shape != (n, n) or not np.all(np.isfinite(matrix)):
            raise FloatingPointError("invalid jacobian")
        product = matrix @ product
    return product


def _normalized(a, b, atol, rtol):
    return float(np.max(np.abs(a - b) / (atol + rtol * np.maximum(np.abs(a), np.abs(b)))))


def certify_orbit(map_fn, jacobian, x0, period, atol=1e-9, rtol=1e-8, newton_tolerance=1e-10,
                  stability_margin=1e-7, candidate_tolerance=1e-3, divisor_separation=1e-6,
                  max_iterations=60, max_halvings=30):
    """Verify a primitive period-``period`` orbit near ``x0`` and its stability.

    Returns a dict with ``verified``, ``attracting``, ``stability``
    (attracting / unstable / neutral_or_unresolved), ``primitive_period``,
    ``spectral_radius``, all multipliers, the orbit, Newton diagnostics
    (iterations, final relative correction, convergence), the normalized
    closure, the candidate distance, the minimum proper-divisor separation,
    ``monodromy_norm``, the condition number of ``M - I`` and the eigenvalue
    condition number of the dominant multiplier, and a ``reason``.
    """
    if period < 1 or int(period) != period:
        raise ValueError("period must be a positive integer")
    period = int(period)
    seed = np.asarray(x0, dtype=np.float64).copy()
    scale_seed = max(1.0, float(np.max(np.abs(seed))))
    result = {"verified": False, "attracting": False, "stability": "unresolved",
              "requested_period": period, "primitive_period": None, "spectral_radius": None,
              "multipliers": [], "orbit": [], "newton_iterations": 0,
              "final_relative_correction": None, "newton_converged": False,
              "normalized_closure": None, "candidate_distance": None,
              "min_divisor_separation": None, "monodromy_norm": None,
              "condition_m_minus_i": None, "dominant_eigenvalue_condition": None,
              "method": "damped Newton with exact chain-rule Jacobian"}
    x = seed.copy()
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            for iteration in range(1, max_iterations + 1):
                orbit = _advance(map_fn, x, period)
                g = orbit[-1] - x
                m = _monodromy(jacobian, orbit[:-1])
                step = np.linalg.solve(m - np.eye(len(x)), -g)
                # Convergence is judged on the undamped Newton correction, so a
                # stalled line search can never masquerade as convergence.
                correction = float(np.max(np.abs(step))) / max(1.0, float(np.max(np.abs(x))))
                result["newton_iterations"] = iteration
                result["final_relative_correction"] = correction
                if correction <= newton_tolerance:
                    x = x + step
                    result["newton_converged"] = True
                    break
                base = float(np.max(np.abs(g)))
                t = 1.0
                accepted = False
                for _ in range(max_halvings):
                    trial = x + t * step
                    trial_orbit = _advance(map_fn, trial, period)
                    if float(np.max(np.abs(trial_orbit[-1] - trial))) < base:
                        accepted = True
                        break
                    t *= 0.5
                if not accepted:
                    return {**result, "reason": "line_search_stalled"}
                x = trial
            orbit = _advance(map_fn, x, period)
            closure = _normalized(orbit[-1], orbit[0], atol, rtol)
            result["normalized_closure"] = closure
            result["orbit"] = np.asarray(orbit[:-1]).tolist()
            distance = min(float(np.max(np.abs(p - seed))) for p in orbit[:-1]) / scale_seed
            result["candidate_distance"] = distance
            m = _monodromy(jacobian, orbit[:-1])
            eigenvalues, right = np.linalg.eig(m)
            order = np.argsort(-np.abs(eigenvalues))
            eigenvalues, right = eigenvalues[order], right[:, order]
            radius = float(np.abs(eigenvalues[0]))
            left_values, left = np.linalg.eig(m.T)
            match = int(np.argmin(np.abs(left_values - eigenvalues[0])))
            y, v = left[:, match], right[:, 0]
            overlap = abs(np.vdot(y, v)) / (np.linalg.norm(y) * np.linalg.norm(v))
            result["multipliers"] = [{"real": float(z.real), "imag": float(z.imag),
                                      "modulus": float(abs(z))} for z in eigenvalues]
            result["spectral_radius"] = radius
            result["monodromy_norm"] = float(np.linalg.norm(m, 2))
            result["condition_m_minus_i"] = float(np.linalg.cond(m - np.eye(len(x))))
            result["dominant_eigenvalue_condition"] = float(1.0 / overlap) if overlap > 0 else None
            scale = max(1.0, float(np.max(np.abs(orbit[0]))))
            separations = [float(np.max(np.abs(orbit[d] - orbit[0]))) / scale
                           for d in range(1, period) if period % d == 0]
            result["min_divisor_separation"] = min(separations) if separations else None
            if not result["newton_converged"]:
                return {**result, "reason": "newton_not_converged"}
            if closure > 1.0:
                return {**result, "reason": "closure_failed"}
            if distance > candidate_tolerance:
                return {**result, "reason": "root_remote_from_candidate"}
            if separations and min(separations) < divisor_separation:
                return {**result, "reason": "nonprimitive_or_unseparated_period"}
            stability = ("attracting" if radius < 1.0 - stability_margin else
                         "unstable" if radius > 1.0 + stability_margin else "neutral_or_unresolved")
            return {**result, "verified": True, "primitive_period": period,
                    "attracting": stability == "attracting", "stability": stability,
                    "reason": "verified_primitive_orbit"}
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
        return {**result, "reason": f"numerical_failure:{type(error).__name__}:{error}"}
