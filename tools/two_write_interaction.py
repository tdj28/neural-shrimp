"""Synthetic-testable parity mathematics for the proposed EXP-024.

No file access, integration, fitting of target data, or qualification gates.
Rows always mean (++,+-,-+,--); signs are balanced and noise is independent
Gaussian with SD sigma on each coordinate. Reference observer construction
must precede access to finite test means. A caller can compute an affine oracle
from test contrasts, but must label that separately from a reference prediction.
"""

from __future__ import annotations

import math

import numpy as np


SIGN_PAIRS = ((1, 1), (1, -1), (-1, 1), (-1, -1))
_S1 = np.array([1., 1., -1., -1.])
_S2 = np.array([1., -1., 1., -1.])
_PARITY = _S1 * _S2


def _array(value, name, *, ndim=None, shape=None):
    if type(value) is not np.ndarray or value.dtype != np.float64:
        raise ValueError(f"{name} must be a float64 ndarray")
    if (ndim is not None and value.ndim != ndim) or (
        shape is not None and value.shape != shape
    ):
        raise ValueError(f"{name} has incorrect dimensions")
    if not value.size or not np.isfinite(value).all():
        raise ValueError(f"{name} must be nonempty and finite")
    return value


def _sigma(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise ValueError("sigma must be positive and finite")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError("sigma must be representable") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError("sigma must be positive and finite")
    return result


def _finite(value, name):
    if not np.isfinite(value).all():
        raise ValueError(f"{name} is not representable as finite float64")
    return value


def reference_means(m, A, B, C):
    """Construct m + s1*A + s2*B + s1*s2*C in explicit SIGN_PAIRS order."""
    m = _array(m, "m", ndim=1)
    A, B, C = (_array(v, n, shape=m.shape) for v, n in ((A, "A"), (B, "B"), (C, "C")))
    with np.errstate(over="ignore", invalid="ignore"):
        means = m + _S1[:, None] * A + _S2[:, None] * B + _PARITY[:, None] * C
    return _finite(means, "reference means")


def factorial_components(means):
    """Return the exact four-row factorial decomposition, not fitted dynamics.

    C includes all finite-amplitude cross effects, not just the Hessian term.
    Applying this to test outcomes is a descriptive diagnostic/oracle operation.
    """
    means = _array(means, "means", ndim=2)
    if means.shape[0] != 4:
        raise ValueError("means must have four rows in SIGN_PAIRS order")
    # Scale before summation to avoid overflow of otherwise finite averages.
    quarter = means * 0.25
    return {"m": quarter.sum(axis=0), "A": _S1 @ quarter,
            "B": _S2 @ quarter, "C": _PARITY @ quarter}


def reference_observer(A, B, C, sigma):
    """Linear MMSE parity weights from reference response coefficients.

    Solve (AA^T + BB^T + CC^T + sigma^2 I) w = C. Unlike classification
    weights these have a meaningful magnitude and must not be normalized.
    Zero C returns zero weights: a valid constant parity predictor.
    SVD of the scaled d-by-3 template matrix avoids a dense d-by-d inverse.
    """
    A = _array(A, "A", ndim=1)
    B, C = _array(B, "B", shape=A.shape), _array(C, "C", shape=A.shape)
    sigma = _sigma(sigma)
    templates = np.column_stack((A, B, C))
    scale = max(sigma, float(np.max(np.abs(templates))))
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        scaled = templates / scale
        noise_variance = (sigma / scale) ** 2
    if noise_variance == 0:
        raise ValueError("scaled noise variance is not representable")
    try:
        u, s, vt = np.linalg.svd(scaled, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError("reference SVD failed") from exc
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        solution = u @ ((s / (s * s + noise_variance)) * vt[:, 2])
        w = solution / scale
        residual = scaled @ (scaled.T @ solution) + noise_variance * solution - scaled[:, 2]
    return {"w": _finite(w, "observer weights"),
            "scaled_equation_residual": _finite(residual, "equation residual"),
            "scale": scale, "scaled_noise_variance": noise_variance}


def parity_risk(w, m, means, sigma):
    """Exact balanced-sign squared risk, analytically averaged over noise.

    The prediction is w^T(Y-m); m is the sealed reference center, never
    silently recentered on test outcomes. R-squared is 1 minus risk because
    the balanced parity target has variance one. Negative scores are retained.
    """
    w = _array(w, "w", ndim=1)
    m = _array(m, "m", shape=w.shape)
    means = _array(means, "means", shape=(4, w.size))
    sigma = _sigma(sigma)
    with np.errstate(over="ignore", invalid="ignore"):
        predictions = (means - m) @ w
        sigma_w = sigma * w
        noise_risk = float(sigma_w @ sigma_w)
        per_sign_risk = (_PARITY - predictions) ** 2 + noise_risk
        risk = float(np.mean(per_sign_risk))
    _finite(predictions, "predictions")
    _finite(per_sign_risk, "per-sign risk")
    _finite(risk, "risk")
    return {"sign_pairs": SIGN_PAIRS, "predictions": predictions,
            "noise_risk": noise_risk, "per_sign_risk": per_sign_risk,
            "risk": risk, "r_squared": 1.0 - risk}
