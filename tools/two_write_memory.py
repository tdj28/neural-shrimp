"""Pure support for a prospective two-write, fixed-linear-observer study.

There is no data access, neural integration, model fitting or experiment gate
here. Callers supply every array and choose qualification rules prospectively.
Array inputs must already be finite float64 ndarrays; no dtype conversion or
projector repair is implicit. Balanced accuracies assume a zero-threshold
observer with independent Gaussian noise on each observation coordinate.

The covariance observer is not a binary-mixture Bayes oracle or necessarily
the accuracy-optimal linear classifier. An exact mixture oracle is deferred:
chance performance of this observer does not imply that information is erased.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np


FIRST_SIGNS = (-1, 1)
SECOND_SIGNS = (-1, 0, 1)
INTERVENTIONS = ("identity", "timing", "transverse")


def _array(value, name, ndim=None, shape=None):
    if type(value) is not np.ndarray or value.dtype != np.float64:
        raise ValueError(f"{name} must be a float64 ndarray")
    if ndim is not None and value.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if shape is not None and value.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if value.size == 0 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return value


def _scalar(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} is not representable") from exc
    if not math.isfinite(result) or (positive and result <= 0) or (
        nonnegative and result < 0
    ):
        raise ValueError(f"{name} has an invalid value")
    return result


def _finite(value, name):
    if not np.isfinite(value).all():
        raise ValueError(f"{name} is not representable as finite float64")
    return value


def _norm(value):
    return _scalar(math.hypot(*value.ravel()), "Euclidean norm", nonnegative=True)


def _maxabs(value):
    return float(np.max(np.abs(value)))


def _observer(h, sigma):
    h = _array(h, "h", ndim=1)
    sigma = _scalar(sigma, "sigma", positive=True)
    norm = _norm(h)
    if norm == 0:
        raise ValueError("A zero observer has no defined Gaussian sign accuracy")
    noise = _scalar(sigma * norm, "projected noise SD", positive=True)
    return h, sigma, noise


def linear_observer(A, B, sigma):
    """Return unit h proportional to (sigma² I + B Bᵀ)^-1 A.

    B is the response to a unit-sign, equiprobable second bit, not a fitted
    nuisance vector. Orientation satisfies h·A > 0. Zero A is undefined and
    rejected rather than assigned an arbitrary direction. Matrix conditioning
    and equation residual are diagnostics, not scientific acceptance gates.
    """
    A = _array(A, "A", ndim=1)
    B = _array(B, "B", shape=A.shape)
    sigma = _scalar(sigma, "sigma", positive=True)
    a_norm, b_norm = _norm(A), _norm(B)
    if a_norm == 0:
        raise ValueError("A is zero; the first-signal observer is undefined")
    scale = max(sigma, b_norm)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        b = B / scale
        noise_variance = (sigma / scale) ** 2
        covariance = noise_variance * np.eye(A.size) + np.outer(b, b)
    if noise_variance == 0:
        raise ValueError("Scaled noise variance is not representable")
    _finite(covariance, "scaled covariance")
    try:
        factor = np.linalg.cholesky(covariance)
        rhs = A / a_norm
        solution = np.linalg.solve(factor.T, np.linalg.solve(factor, rhs))
    except np.linalg.LinAlgError as exc:
        raise ValueError("Noise covariance is not numerically positive definite") from exc
    _finite(solution, "covariance solution")
    solution_norm = _norm(solution)
    if solution_norm == 0:
        raise ValueError("Covariance solution is zero")
    h = solution / solution_norm
    alignment = float(h @ A)
    if alignment < 0:
        h = -h
        alignment = -alignment
    if alignment <= 0 or not math.isfinite(alignment):
        raise ValueError("Observer orientation is undefined")
    residual = covariance @ solution - rhs
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        condition_value = np.float64(1.0) + np.dot(b, b) / np.float64(noise_variance)
    condition = _scalar(condition_value, "analytic covariance condition", positive=True)
    return {
        "h": h,
        "first_projection": alignment,
        "second_projection": _scalar(float(h @ B), "second projection"),
        "noise_sd": _observer(h, sigma)[2],
        "covariance_scale": scale,
        "covariance_condition": condition,
        "scaled_equation_residual": _finite(residual, "equation residual"),
        "scaled_equation_residual_norm": _norm(residual),
        "scope": "covariance/SNR linear observer; not a binary-mixture Bayes oracle",
    }


def linear_means(A, B, offset=None):
    """All six sign/absence means, indexed FIRST_SIGNS × SECOND_SIGNS.

    offset, if supplied, is fixed across both unknown bits. A caller must not
    subtract a baseline conditioned on the unknown second-bit sign.
    """
    A = _array(A, "A", ndim=1)
    B = _array(B, "B", shape=A.shape)
    offset = np.zeros_like(A) if offset is None else _array(offset, "offset", shape=A.shape)
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.asarray(
            [[offset + first * A + second * B for second in SECOND_SIGNS]
             for first in FIRST_SIGNS], dtype=np.float64
        )
    return _finite(result, "linear means")


def balanced_accuracy(h, means, sigma):
    """Analytic Gaussian probabilities for a fixed zero-threshold observer.

    means has shape (2, 3, n): first signs (-1,+1), second signs (-1,0,+1).
    Each mixture is balanced exactly; there are no sampled noise trials.
    """
    h, sigma, noise = _observer(h, sigma)
    means = _array(means, "means", shape=(2, 3, h.size))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        projected = means @ h
        signed = projected * np.asarray(FIRST_SIGNS, dtype=np.float64)[:, None] / noise
    _finite(projected, "projected means")
    _finite(signed, "signed standardized means")
    probabilities = np.asarray(
        [[0.5 * math.erfc(-float(z) / math.sqrt(2.0)) for z in row] for row in signed],
        dtype=np.float64,
    )
    by_second = np.asarray(
        [math.fsum(probabilities[:, j]) / 2.0 for j in range(3)], dtype=np.float64
    )
    present = math.fsum(probabilities[:, [0, 2]].ravel()) / 4.0
    absent = float(by_second[1])
    return {
        "h": h.copy(), "sigma": sigma, "noise_sd": noise,
        "first_signs": FIRST_SIGNS, "second_signs": SECOND_SIGNS,
        "projected_means": projected, "signed_standardized_means": signed,
        "probability_correct": probabilities,
        "balanced_accuracy_by_second": by_second,
        "with_second": present, "without_second": absent,
        "interference_loss": absent - present,
    }


def no_first_write_control(h, means, sigma):
    """Assign balanced pseudo-labels to identical no-first-write means.

    means has shape (3,n), one row per second sign/absence. Chance follows
    analytically even with a nonzero nuisance mean; it is not a fitted test.
    """
    h, _, _ = _observer(h, sigma)
    means = _array(means, "no-first-write means", shape=(3, h.size))
    result = balanced_accuracy(h, np.stack((means, means)), sigma)
    result["maximum_chance_deviation"] = _maxabs(result["balanced_accuracy_by_second"] - 0.5)
    return result


def fixed_observer_contrasts(h, identity, interventions, sigma):
    """Compare every supplied arm with identity using exactly the same h.

    Positive contrasts mean identity has higher accuracy. These describe a
    fixed observer, not all information recoverable after state replacement.
    """
    if not isinstance(interventions, Mapping) or not interventions:
        raise ValueError("interventions must be a nonempty mapping")
    if any(not isinstance(k, str) or not k or k == "identity" for k in interventions):
        raise ValueError("Intervention names must be nonempty and distinct from identity")
    baseline = balanced_accuracy(h, identity, sigma)
    arms = {}
    for name in sorted(interventions):
        score = balanced_accuracy(h, interventions[name], sigma)
        arms[name] = {
            "score": score,
            "identity_minus_arm_with_second": baseline["with_second"] - score["with_second"],
            "identity_minus_arm_without_second": baseline["without_second"] - score["without_second"],
            "identity_minus_arm_probability": baseline["probability_correct"] - score["probability_correct"],
        }
    return {"identity": baseline, "interventions": arms}


def oblique_components(flow, gradient, displacement):
    """Unrepaired Q=f Zᵀ, P=I-Q and local component diagnostics.

    Supplied flow/gradient must belong to the intervention-time reference.
    This function cannot certify their provenance or infer a nonlinear phase.
    Zero displacement has undefined amplification (None), not zero or one.
    """
    flow = _array(flow, "flow", ndim=1)
    gradient = _array(gradient, "gradient", shape=flow.shape)
    d = _array(displacement, "displacement", shape=flow.shape)
    with np.errstate(over="ignore", invalid="ignore"):
        Q = np.outer(flow, gradient)
        P = np.eye(flow.size, dtype=np.float64) - Q
        phase, transverse = Q @ d, P @ d
        raw = {
            "Q": Q, "P": P, "phase": phase, "transverse": transverse,
            "partition_residual": phase + transverse - d,
            "Q_idempotence_residual": Q @ Q - Q,
            "P_idempotence_residual": P @ P - P,
            "gradient_P": gradient @ P,
            "Q_transverse": Q @ transverse, "P_phase": P @ phase,
            "timing_replacement": phase - d, "transverse_replacement": transverse - d,
        }
    for name, value in raw.items():
        _finite(value, name)
    d_norm, phase_norm, transverse_norm = _norm(d), _norm(phase), _norm(transverse)
    normalization = _scalar(float(gradient @ flow), "Z dot f")
    raw.update({
        "normalization": normalization, "normalization_residual": normalization - 1.0,
        "initial_linear_time_shift": _scalar(float(gradient @ d), "initial shift"),
        "retained_phase_linear_time_shift": _scalar(float(gradient @ phase), "phase shift"),
        "transverse_linear_time_leakage": _scalar(float(gradient @ transverse), "transverse leakage"),
        "displacement_norm": d_norm, "phase_norm": phase_norm, "transverse_norm": transverse_norm,
        "phase_amplification": None if d_norm == 0 else _scalar(phase_norm / d_norm, "phase amplification"),
        "transverse_amplification": None if d_norm == 0 else _scalar(transverse_norm / d_norm, "transverse amplification"),
        "Q_operator_norm": _scalar(_norm(flow) * _norm(gradient), "Q norm"),
        "P_operator_norm": _scalar(float(np.linalg.norm(P, 2)), "P norm"),
    })
    return raw


def intervene(reference, state, flow, gradient, mode):
    """Apply a local replacement and retain requested versus realized deltas.

    Identity retains state bit-for-bit while recording the same displacement diagnostics.
    No exact nonlinear isochron or physically norm-matched intervention is claimed.
    """
    reference = _array(reference, "reference", ndim=1)
    state = _array(state, "state", shape=reference.shape)
    if not isinstance(mode, str) or mode not in INTERVENTIONS:
        raise ValueError("Unknown intervention")
    d = _finite(state - reference, "initial displacement")
    before = oblique_components(flow, gradient, d)
    requested = {"identity": d, "timing": before["phase"],
                 "transverse": before["transverse"]}[mode].copy()
    with np.errstate(over="ignore", invalid="ignore"):
        replaced = state.copy() if mode == "identity" else _finite(reference + requested, "replaced state")
        realized = _finite(replaced - reference, "realized displacement")
        replacement = _finite(replaced - state, "realized replacement")
        realization_error = _finite(realized - requested, "realization error")
    return {
        "mode": mode, "state": replaced, "before": before,
        "requested_displacement": requested, "realized_displacement": realized,
        "realization_error": realization_error,
        "realized_replacement": replacement, "replacement_norm": _norm(replacement),
        "after": oblique_components(flow, gradient, realized),
    }


def compose_two_write(phi_before, phi_after, H, first, second):
    """Propagate first write, apply H at T-minus, then add second at T-plus.

    phi_before is n×n; phi_after is n×n or m×n×n for fixed observations after
    the event. All arrays are supplied; this function neither advances a clock
    nor assumes periodic closure. The second write is never multiplied by H.
    """
    first = _array(first, "first", ndim=1)
    second = _array(second, "second", shape=first.shape)
    n = first.size
    phi_before = _array(phi_before, "phi_before", shape=(n, n))
    H = _array(H, "H", shape=(n, n))
    phi_after = _array(phi_after, "phi_after")
    if phi_after.ndim not in (2, 3) or phi_after.shape[-2:] != (n, n):
        raise ValueError("phi_after must have shape (n,n) or (m,n,n)")
    with np.errstate(over="ignore", invalid="ignore"):
        before = phi_before @ first
        retained = H @ before
        after = retained + second
        first_operator = (phi_after @ H) @ phi_before
        first_response = first_operator @ first
        second_response = phi_after @ second
        total = phi_after @ after
        result = {
            "before_intervention": before, "after_intervention": retained,
            "after_second_write": after, "first_operator": first_operator,
            "second_operator": phi_after.copy(), "first_response": first_response,
            "second_response": second_response, "total": total,
            "composition_residual": first_response + second_response - total,
        }
    for name, value in result.items():
        _finite(value, name)
    return result


def one_mode_response(elapsed_after, spacing, decay_rate, first_gain, second_gain, retention=1.0):
    """Declared scalar exponential-memory baseline, sampled after write two.

    State obeys dx/dt=-decay_rate*x; writes add first_gain*s1 and
    second_gain*s2; retention multiplies the state just before write two.
    No parameter is fitted. Observation coordinates share this one latent mode.
    """
    elapsed_after = _array(elapsed_after, "elapsed_after", ndim=1)
    if np.any(elapsed_after < 0):
        raise ValueError("Observations must follow the second write")
    spacing = _scalar(spacing, "spacing", positive=True)
    decay_rate = _scalar(decay_rate, "decay_rate", nonnegative=True)
    first_gain = _scalar(first_gain, "first_gain")
    second_gain = _scalar(second_gain, "second_gain")
    retention = _scalar(retention, "retention")
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        exponent = _finite(-decay_rate * elapsed_after, "decay exponent")
        before_exponent = _scalar(-decay_rate * spacing, "pre-event exponent")
        shape = np.exp(exponent)
        A = _finite(shape * (retention * math.exp(before_exponent) * first_gain), "A")
        B = _finite(shape * second_gain, "B")
    return {
        "A": A, "B": B, "means": linear_means(A, B),
        "spacing": spacing, "decay_rate": decay_rate,
        "first_gain": first_gain, "second_gain": second_gain, "retention": retention,
    }


def periodic_single_mode(B, rho, retention=1.0):
    """Window-matched periodic one-mode approximation A=retention*rho*B.

    B is the supplied actual-reference second-write observation vector; rho
    is a declared signed, real transverse multiplier for one write spacing.
    This matches B at this window, not the full neural ODE or its other modes.
    No multiplier is fitted and no stability/qualification gate is imposed.
    """
    B = _array(B, "B", ndim=1)
    rho = _scalar(rho, "rho")
    retention = _scalar(retention, "retention")
    with np.errstate(over="ignore", invalid="ignore"):
        A = _finite((retention * rho) * B, "periodic one-mode A")
    return {"A": A, "B": B.copy(), "rho": rho, "retention": retention,
            "means": linear_means(A, B),
            "scope": "periodic single-mode approximation matched to this window's B"}
