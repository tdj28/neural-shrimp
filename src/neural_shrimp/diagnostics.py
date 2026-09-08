"""Conservative finite-time screening and independent orbit verification.

No function in this module labels an island a shrimp. Finite recurrence is a
candidate, positive finite-time expansion is a chaos candidate, and neither
is an asymptotic theorem. Shrimp geometry requires parameter continuation.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.optimize import root


def _state(value):
    result = np.atleast_1d(np.asarray(value, dtype=np.float64))
    if result.ndim != 1 or len(result) < 1:
        raise ValueError("State must be a nonempty vector")
    return result


def _relative_residual(a, b, atol, rtol):
    scale = atol + rtol * np.maximum(np.abs(a), np.abs(b))
    return float(np.max(np.abs(a - b) / scale))


def finite_difference_jacobian(map_fn, state, relative_step=None):
    """Central differences, with coordinate-scaled float64 steps.

    Used as a fallback or independent derivative audit, not as an exact
    stability certificate. Cancellation and ill-conditioning still matter.
    """
    x = _state(state)
    step = np.cbrt(np.finfo(np.float64).eps) if relative_step is None else relative_step
    if not np.isfinite(step) or step <= 0:
        raise ValueError("relative_step must be finite and positive")
    result = np.empty((len(x), len(x)), dtype=np.float64)
    for i in range(len(x)):
        delta = step * max(1.0, abs(x[i]))
        plus, minus = x.copy(), x.copy()
        plus[i] += delta
        minus[i] -= delta
        result[:, i] = (_state(map_fn(plus)) - _state(map_fn(minus))) / (2.0 * delta)
    return result


@dataclass(frozen=True)
class TrajectoryResult:
    states: np.ndarray
    status: str
    steps_completed: int
    failure_reason: str | None = None


def iterate_map(map_fn, x0, steps, tail=None, divergence_threshold=1e8):
    """Iterate a map, optionally retaining only its final ``tail`` states.

    Initial state is included when retention permits. A nonfinite iterate,
    arithmetic overflow, or a state norm beyond the prespecified threshold
    produces ``divergent``. Threshold crossing is an operational stopping
    label, not a proof of mathematical escape to infinity. Bad programmer
    shapes raise an exception rather than being silently called divergence.
    """
    if steps < 0 or int(steps) != steps or (tail is not None and tail < 1):
        raise ValueError("steps must be nonnegative and tail positive")
    if not np.isfinite(divergence_threshold) or divergence_threshold <= 0:
        raise ValueError("divergence_threshold must be finite and positive")
    state = _state(x0).copy()
    history = deque([state.copy()], maxlen=tail)
    if not np.all(np.isfinite(state)) or np.max(np.abs(state)) > divergence_threshold:
        return TrajectoryResult(np.asarray(history), "divergent", 0, "invalid_initial_state")
    for step in range(1, steps + 1):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                next_state = _state(map_fn(state))
        except (FloatingPointError, OverflowError) as error:
            return TrajectoryResult(np.asarray(history), "divergent", step - 1,
                                    type(error).__name__)
        if next_state.shape != state.shape:
            raise ValueError("Map changed the state dimension")
        state = next_state.copy()
        history.append(state)
        if not np.all(np.isfinite(state)):
            return TrajectoryResult(np.asarray(history), "divergent", step, "nonfinite_state")
        if np.max(np.abs(state)) > divergence_threshold:
            return TrajectoryResult(np.asarray(history), "divergent", step, "state_threshold")
    return TrajectoryResult(np.asarray(history), "completed", int(steps))


def classify_tail(states, max_period=16, atol=1e-8, rtol=1e-6, min_repeats=6,
                  lyapunov=None, chaos_threshold=1e-4, divergence_threshold=1e8):
    """Screen full-state recurrence across every retained cycle block.

    The smallest repeated period is reported. Every candidate phase must
    agree across ``min_repeats`` blocks; comparing only the last pair could
    mistake transient crossings for recurrence. Even this test cannot rule
    out exceptionally slow drift or long transients: independent root,
    primitive-period and multiplier checks remain necessary.

    ``fixed_point`` is a finite-tail label, not an attraction certificate.
    Nonrecurrent bounded tails are ``unresolved`` unless a supplied positive
    FTLE supports ``chaotic_candidate``. FTLE may be a scalar or the dict
    returned by ``largest_lyapunov``. No claim of rigorous chaos is made.
    """
    states = np.asarray(states, dtype=np.float64)
    if states.ndim == 1:
        states = states[:, None]
    if states.ndim != 2 or len(states) < 1:
        raise ValueError("states must be a nonempty (time, dimension) array")
    if max_period < 1 or min_repeats < 2 or atol <= 0 or rtol < 0:
        raise ValueError("Invalid recurrence settings")
    base = {"classification": "unresolved", "period": None, "residual": None,
            "normalized_residual": None, "evidence": "finite_tail",
            "tail_length": len(states), "min_repeats": int(min_repeats)}
    if not np.all(np.isfinite(states)) or np.max(np.abs(states)) > divergence_threshold:
        return {**base, "classification": "divergent"}
    for period in range(1, int(max_period) + 1):
        count = period * int(min_repeats)
        if len(states) < count:
            break
        blocks = states[-count:].reshape(min_repeats, period, states.shape[1])
        reference = blocks[-1:]
        normalized = _relative_residual(blocks, reference, atol, rtol)
        if normalized <= 1.0:
            return {**base,
                    "classification": "fixed_point" if period == 1 else "periodic_candidate",
                    "period": period, "residual": float(np.max(np.abs(blocks - reference))),
                    "normalized_residual": normalized}
    exponent = lyapunov.get("exponent") if isinstance(lyapunov, dict) else lyapunov
    valid_ftle = not isinstance(lyapunov, dict) or lyapunov.get("status") == "ok"
    if exponent is not None and valid_ftle and np.isfinite(exponent) and exponent > chaos_threshold:
        return {**base, "classification": "chaotic_candidate", "ftle": float(exponent)}
    return base


def certify_cycle(map_fn, x0, period, jacobian=None, atol=1e-9, rtol=1e-8,
                  root_tolerance=1e-10, stability_margin=1e-7,
                  candidate_tolerance=1e-3, max_evaluations=2000):
    """Numerically verify a nearby prime periodic root and assess stability.

    Solve F^p(x)-x=0, independently re-evaluate the orbit, reject recurrence
    at proper divisors, and form the ordered Jacobian product. ``verified``
    certifies a numerical prime periodic root; ``attracting`` additionally
    requires all Floquet multiplier moduli < 1 - stability_margin. Neutral
    or ill-conditioned directions are reported without declaring attraction.
    ``candidate_tolerance`` bounds distance to the observed seed relative to
    max(1, seed infinity norm), preventing convergence to a remote orbit from
    being silently attributed to the observed tail. This is floating-point
    verification, not interval arithmetic or a computer-assisted proof.
    """
    if period < 1 or int(period) != period:
        raise ValueError("period must be a positive integer")
    if min(atol, root_tolerance, candidate_tolerance) <= 0 or rtol < 0:
        raise ValueError("Invalid numerical tolerances")
    seed = _state(x0).copy()
    period = int(period)
    result = {"verified": False, "attracting": False, "requested_period": period,
              "primitive_period": None, "classification": "unresolved",
              "closure_residual": None, "normalized_closure_residual": None,
              "spectral_radius": None, "multipliers": [], "orbit": [],
              "stability": "unresolved", "solver_success": False,
              "seed_distance": None, "jacobian_method": "analytic" if jacobian else "central_difference"}

    def advance(x, count):
        state = _state(x).copy()
        for _ in range(count):
            state = _state(map_fn(state))
            if not np.all(np.isfinite(state)):
                raise FloatingPointError("nonfinite_orbit")
        return state

    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            solution = root(lambda x: advance(x, period) - x, seed, method="hybr",
                            options={"xtol": root_tolerance, "maxfev": max_evaluations})
            result["solver_success"] = bool(solution.success)
            result["solver_message"] = str(solution.message)
            center = _state(solution.x)
            orbit = [center.copy()]
            for _ in range(period):
                orbit.append(advance(orbit[-1], 1))
            closure = _relative_residual(orbit[-1], orbit[0], atol, rtol)
            result["closure_residual"] = float(np.max(np.abs(orbit[-1] - orbit[0])))
            result["normalized_closure_residual"] = closure
            result["orbit"] = np.asarray(orbit[:-1]).tolist()
            seed_distance = min(float(np.max(np.abs(point - seed))) for point in orbit[:-1])
            result["seed_distance"] = seed_distance
            if closure > 1.0:
                return {**result, "reason": "closure_failed"}
            if seed_distance > candidate_tolerance * max(1.0, float(np.max(np.abs(seed)))):
                return {**result, "reason": "root_remote_from_candidate"}
            for divisor in range(1, period):
                if period % divisor == 0 and _relative_residual(orbit[divisor], orbit[0], atol, rtol) <= 1.0:
                    return {**result, "reason": "nonprimitive_period", "primitive_period": divisor}
            result["primitive_period"] = period
            derivative = jacobian if jacobian is not None else lambda x: finite_difference_jacobian(map_fn, x)
            monodromy = np.eye(len(seed), dtype=np.float64)
            for point in orbit[:-1]:
                matrix = np.asarray(derivative(point), dtype=np.float64)
                if matrix.shape != monodromy.shape or not np.all(np.isfinite(matrix)):
                    return {**result, "reason": "invalid_jacobian"}
                monodromy = matrix @ monodromy
            eigenvalues = np.linalg.eigvals(monodromy)
            radius = float(np.max(np.abs(eigenvalues)))
            result["multipliers"] = [{"real": float(v.real), "imag": float(v.imag),
                                      "modulus": float(abs(v))} for v in eigenvalues]
            result["spectral_radius"] = radius
            attracting = radius < 1.0 - stability_margin
            stability = "attracting" if attracting else ("unstable" if radius > 1.0 + stability_margin
                                                         else "neutral_or_unresolved")
            return {**result, "verified": True, "attracting": bool(attracting),
                    "stability": stability, "classification": "verified_periodic",
                    "reason": "numerical_prime_cycle_verified"}
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
        return {**result, "reason": f"numerical_failure:{type(error).__name__}"}


def largest_lyapunov(map_fn, jacobian, x0, steps, burn_in=0, seed=0,
                     divergence_threshold=1e8):
    """Estimate the largest finite-time Lyapunov exponent using full QR.

    Tangents evolve during burn-in; logarithms accumulate only afterward.
    Full QR avoids mistaking annihilation of one chosen tangent for collapse
    of every direction. Exact zero QR diagonals represent negative-infinite
    exponents and are serialized as None with explicit index metadata. A
    positive finite-time estimate alone does not establish sustained chaos.
    """
    if steps < 1 or burn_in < 0 or int(steps) != steps or int(burn_in) != burn_in:
        raise ValueError("steps must be positive and burn_in nonnegative integers")
    state = _state(x0).copy()
    d = len(state)
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    totals = np.zeros(d)
    singular = np.zeros(d, dtype=bool)
    singular_steps = 0
    completed = 0
    for iteration in range(burn_in + steps):
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                if not np.all(np.isfinite(state)) or np.max(np.abs(state)) > divergence_threshold:
                    raise FloatingPointError("state_threshold")
                matrix = np.asarray(jacobian(state), dtype=np.float64)
                if matrix.shape != (d, d) or not np.all(np.isfinite(matrix)):
                    raise FloatingPointError("invalid_jacobian")
                q, r = np.linalg.qr(matrix @ q)
                diagonal = np.abs(np.diag(r))
                state = _state(map_fn(state))
                if not np.all(np.isfinite(state)) or np.max(np.abs(state)) > divergence_threshold:
                    raise FloatingPointError("state_threshold")
                if iteration >= burn_in:
                    zeros = diagonal == 0.0
                    singular |= zeros
                    singular_steps += int(np.any(zeros))
                    totals[~zeros] += np.log(diagonal[~zeros])
                    completed += 1
        except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as error:
            return {"status": "failed", "exponent": None, "steps_completed": completed,
                    "reason": type(error).__name__, "finite_time": True}
    values = totals / steps
    finite_values = values[~singular]
    exponent = float(np.max(finite_values)) if len(finite_values) else None
    return {"status": "ok", "exponent": exponent, "finite_time": True,
            "steps_completed": completed, "burn_in": int(burn_in),
            "spectrum": [None if singular[i] else float(values[i]) for i in range(d)],
            "negative_infinity_indices": np.flatnonzero(singular).tolist(),
            "largest_is_negative_infinity": bool(np.all(singular)),
            "singular_steps": singular_steps, "method": "full_qr"}
