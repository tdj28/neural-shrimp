"""Invariant sensitivity of an isolated real leading pair of return multipliers.

For the memory map F_beta(x) = (1-beta) F(x) + beta x, the derivative
implemented here holds the saved orbit points fixed. It measures the explicit
identity feedback path, not the derivative of a continued periodic orbit and
not a causal decomposition of the original network's internal modes.
"""

import numpy as np
from scipy.linalg import eig


def isolated_leading_pair(matrix, *, relative_residual_tolerance=1e-6,
                          separation_factor=100.0, condition_limit=1e8,
                          real_tolerance=1e-10, pair_gap_tolerance=1e-10):
    """Gate a distinct real leading pair and return biorthogonal vectors.

Both left and right residuals are relative to |mu| times the vector norm,
not to the norm of a possibly nonnormal matrix. ``condition_limit`` bounds
the individual eigenvalue condition estimates ||left|| ||right|| / |left^T right|.
The selected pair is ordered by modulus, but its trace sensitivity is invariant
under exchanging that order or making a constant similarity transformation.
"""
    matrix = np.asarray(matrix)
    if (matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or
            len(matrix) < 2 or not np.isfinite(matrix).all() or
            (np.iscomplexobj(matrix) and np.any(matrix.imag != 0))):
        raise ValueError("a finite real square matrix with at least two rows is required")
    matrix = np.asarray(matrix.real, dtype=np.float64)
    limits = [relative_residual_tolerance, separation_factor, condition_limit,
              real_tolerance, pair_gap_tolerance]
    if not np.isfinite(limits).all() or min(limits) <= 0 or separation_factor <= 1:
        raise ValueError("finite positive gates and separation_factor > 1 are required")
    values, left, right = eig(matrix, left=True, right=True)
    order = np.argsort(-np.abs(values), kind="stable")
    values, left, right = values[order], left[:, order], right[:, order]
    pair = values[:2]
    if (not np.isfinite(values).all() or np.any(abs(pair) == 0) or
            np.any(abs(pair.imag) > real_tolerance * abs(pair))):
        raise ValueError("the leading pair must contain two finite nonzero real multipliers")
    if abs(pair[0] - pair[1]) <= pair_gap_tolerance * max(abs(pair)):
        raise ValueError("individual leading multipliers are not sufficiently separated")
    third = float(abs(values[2])) if len(values) > 2 else 0.0
    ratio = float(abs(pair[1]) / third) if third > 0 else None
    if third > 0 and abs(pair[1]) <= separation_factor * third:
        raise ValueError("the leading pair is not isolated from the third multiplier")

    pair = pair.real
    left, right = left[:, :2].real.copy(), right[:, :2].real.copy()
    overlaps = np.einsum("ij,ij->j", left, right)
    condition = (np.linalg.norm(left, axis=0) * np.linalg.norm(right, axis=0) /
                 np.maximum(abs(overlaps), np.finfo(float).tiny))
    if not np.isfinite(condition).all() or condition.max() > condition_limit or np.any(overlaps == 0):
        raise ValueError("the leading eigenvectors are too ill-conditioned")
    left /= overlaps
    right_residual = (np.linalg.norm(matrix @ right - right * pair, axis=0) /
                      (abs(pair) * np.linalg.norm(right, axis=0)))
    left_residual = (np.linalg.norm(matrix.T @ left - left * pair, axis=0) /
                     (abs(pair) * np.linalg.norm(left, axis=0)))
    if (not np.isfinite([right_residual, left_residual]).all() or
            max(right_residual.max(), left_residual.max()) >= relative_residual_tolerance):
        raise ValueError("a leading left or right eigenvector failed the relative residual gate")
    gram = left.T @ right
    if not np.allclose(gram, np.eye(2), rtol=0,
                       atol=max(1e-10, 10 * relative_residual_tolerance)):
        raise ValueError("the leading eigenvectors are not biorthogonal")
    return {"passed": True, "multipliers": pair, "right": right, "left": left,
            "trace": float(pair.sum()), "determinant": float(pair.prod()),
            "right_relative_residuals": right_residual,
            "left_relative_residuals": left_residual,
            "eigenvalue_condition_estimates": condition,
            "biorthogonality_error": float(np.linalg.norm(gram - np.eye(2), ord=np.inf)),
            "third_modulus": third, "pair_to_third_ratio": ratio,
            "gates": {"relative_residual_tolerance": relative_residual_tolerance,
                      "separation_factor": separation_factor, "condition_limit": condition_limit,
                      "real_tolerance": real_tolerance, "pair_gap_tolerance": pair_gap_tolerance}}


def trace_pair_sensitivity(matrix, apply_prime, **pair_gates):
    """Compute d(mu1+mu2)/ds from a derivative-operator callback.

``apply_prime(V)`` takes an n-by-2 matrix of right eigenvectors and must return
M'(s) V, including exactly the parameter dependence intended by the caller.
This is tr(P M'), with P the isolated pair's spectral projector. Merely changing
the arbitrary eigenvector basis cannot change the result.
"""
    pair = isolated_leading_pair(matrix, **pair_gates)
    applied = np.asarray(apply_prime(pair["right"].copy()))
    if (applied.shape != pair["right"].shape or not np.isfinite(applied).all() or
            (np.iscomplexobj(applied) and np.any(applied.imag != 0))):
        raise ValueError("the derivative callback must return finite real n-by-2 columns")
    derivatives = np.einsum("ij,ij->j", pair["left"], applied.real)
    if not np.isfinite(derivatives).all():
        raise ValueError("nonfinite eigenvalue sensitivity")
    return {**pair, "applied_derivative": applied.real,
            "multiplier_derivatives": derivatives,
            "trace_derivative": float(derivatives.sum())}


def memory_tangent_derivative(step_callbacks, directions):
    """Apply M and its explicit memory derivative at beta=0 to columns V.

For each saved orbit point, a callback applies the original one-step Jacobian
J_k to arbitrary column batches. The orbit is held fixed. Propagate
    w_next = J_k w;  z_next = J_k z + (I-J_k) w,
starting with w=V, z=0. One callback batch contains both w and z. The result is
``(M V, M'_explicit V)``; the callback must not include beta itself.
"""
    callbacks = list(step_callbacks)
    w = np.asarray(directions)
    if (not callbacks or w.ndim != 2 or not all(w.shape) or
            not np.isfinite(w).all() or
            (np.iscomplexobj(w) and np.any(w.imag != 0))):
        raise ValueError("finite real tangent columns and at least one step callback are required")
    w = np.asarray(w.real, dtype=np.float64).copy()
    z = np.zeros_like(w)
    for apply_step in callbacks:
        applied = np.asarray(apply_step(np.concatenate([w, z], axis=1)))
        if (applied.shape != (w.shape[0], 2 * w.shape[1]) or
                not np.isfinite(applied).all() or
                (np.iscomplexobj(applied) and np.any(applied.imag != 0))):
            raise ValueError("step callback returned invalid tangent columns")
        jw, jz = np.split(applied.real, 2, axis=1)
        z, w = jz + w - jw, jw
    return w, z
