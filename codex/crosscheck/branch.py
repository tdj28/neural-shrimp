"""Offline stable-branch extrapolation of a saved fold coefficient product.

No model evaluations, research-tool imports, or file I/O occur here. The caller
owns artifact hashes and a committed analysis/run card. All convergence labels
are assertions in the supplied records, not independently verified eigenpairs.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np


def _finite_scalar(value, name):
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _multipliers(values):
    result = []
    for value in values:
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) < 2:
                raise ValueError("encoded multiplier needs real and imaginary parts")
            result.append(complex(float(value[0]), float(value[1])))
        else:
            result.append(complex(value))
    result = np.asarray(result, dtype=np.complex128)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError("finite nonempty saved multiplier list required")
    return result


def _polynomial_fit(x, y, degree):
    # Rescale only the design columns, without changing the intercept or fit.
    scale = float(np.max(np.abs(x)))
    design = np.vander(x / scale, N=degree + 1, increasing=True)
    scaled, _, rank, singular = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ scaled
    residuals = y - fitted
    coefficients = scaled / (scale ** np.arange(degree + 1))
    return dict(degree=degree, intercept=float(coefficients[0]),
                coefficients_ascending=coefficients.tolist(),
                coefficient_basis="1, sqrt(delta), delta" if degree == 2 else "1, sqrt(delta)",
                design_scale=scale, scaled_coefficients_ascending=scaled.tolist(),
                design_rank=int(rank), full_rank=bool(rank == degree + 1),
                scaled_design_singular_values=singular.tolist(),
                scaled_design_condition=float(singular[0] / singular[-1]) if singular[-1] else None,
                fitted_values=fitted.tolist(), residuals_y_minus_fit=residuals.tolist(),
                residual_sum_squares=float(residuals @ residuals),
                residual_root_mean_square=float(np.sqrt(np.mean(residuals ** 2))),
                max_absolute_residual=float(np.max(np.abs(residuals))))


def branch_extrapolation(branch_points, alpha_c, *, alpha_key="lambda",
                         delta_range=(5e-5, 4e-4), expected_count=18,
                         closure_tol=1e-11, imag_tol=1e-9,
                         spectral_metadata_tol=1e-10, reference_ab=None):
    """Fit y=(1-mu)^2/(4*delta) versus sqrt(delta), delta=alpha_c-alpha.

    ``branch_points`` is an iterable of saved point dictionaries. ``alpha_key``
    must explicitly denote PHYSICAL alpha; only ``lambda`` (the producer's
    unscaled serialized branch coordinate) and ``alpha`` are supported. Scaled
    continuation theta[1]=10*alpha is never inferred or converted here.

    The inclusive fixed delta range selects all rows before outcome gating.
    Each selected record must assert multipliers_converged=True, have relative
    closure residual<closure_tol, and a positive, real dominant multiplier<1.
    Its saved spectral_radius must agree with the largest supplied modulus.
    A selected row that fails is retained and makes the whole analysis fail;
    it is never silently excluded. expected_count=None disables only the count
    gate and is recorded as a design change by the caller. No fit is computed
    when a selection or quality gate fails.

    Output ``passed`` means saved-record quality and full-rank fitting gates
    passed. It does NOT mean an interval proof, independent forward replication,
    agreement with a target coefficient, or validity of asymptotic extrapolation
    over the supplied finite distance range. Fit residuals are deterministic
    diagnostics, not estimates of measurement uncertainty.
    """
    alpha_c = _finite_scalar(alpha_c, "alpha_c")
    if alpha_key not in ("lambda", "alpha"):
        raise ValueError("alpha_key must designate physical 'lambda' or 'alpha'")
    low, high = map(float, delta_range)
    if not (np.isfinite(low) and np.isfinite(high) and 0 < low < high):
        raise ValueError("finite ordered positive delta_range required")
    for name, value in (("closure_tol", closure_tol), ("imag_tol", imag_tol),
                        ("spectral_metadata_tol", spectral_metadata_tol)):
        if not _finite_scalar(value, name) > 0:
            raise ValueError(f"{name} must be positive")
    if expected_count is not None and (type(expected_count) is not int or expected_count < 4):
        raise ValueError("expected_count must be an integer>=4 or None")
    if reference_ab is not None:
        reference_ab = _finite_scalar(reference_ab, "reference_ab")
    points = list(branch_points)
    selection, rows = [], []
    for ordinal, record in enumerate(points):
        if not isinstance(record, dict) or alpha_key not in record:
            raise ValueError(f"point {ordinal} is missing its explicit physical alpha")
        alpha = _finite_scalar(record[alpha_key], f"point {ordinal} alpha")
        delta = alpha_c - alpha
        selected = bool(low <= delta <= high)
        selection.append(dict(input_ordinal=ordinal, saved_index=deepcopy(record.get("index")),
                              physical_alpha=alpha, delta=delta, selected=selected))
        if not selected:
            continue
        row = dict(input_ordinal=ordinal, saved_index=deepcopy(record.get("index")),
                   input_record=deepcopy(record), physical_alpha=alpha, delta=delta)
        gates = {"saved_multipliers_converged": record.get("multipliers_converged") is True}
        try:
            residual = _finite_scalar(record["residual"], "saved closure residual")
            row["saved_relative_closure"] = residual
            gates["saved_closure"] = bool(0 <= residual < closure_tol)
        except (KeyError, TypeError, ValueError):
            gates["saved_closure"] = False
        try:
            values = _multipliers(record["multipliers"])
            k = int(np.argmax(np.abs(values)))
            dominant = values[k]
            radius = float(abs(dominant))
            row.update(dominant_saved_position=k,
                       dominant_multiplier=[float(dominant.real), float(dominant.imag)],
                       largest_saved_modulus=radius)
            gates["saved_dominant_real"] = bool(abs(dominant.imag) < imag_tol)
            gates["saved_dominant_positive_attracting"] = bool(0 < dominant.real < 1 and radius < 1)
            recorded_radius = _finite_scalar(record["spectral_radius"], "saved spectral radius")
            row["saved_spectral_radius"] = recorded_radius
            gates["saved_spectral_metadata_consistent"] = bool(
                recorded_radius >= 0 and abs(recorded_radius-radius) <= spectral_metadata_tol*max(1., radius))
            if gates["saved_dominant_real"] and gates["saved_dominant_positive_attracting"]:
                row["sqrt_delta"] = float(np.sqrt(delta))
                row["y"] = float((1-dominant.real)**2 / (4*delta))
        except (KeyError, TypeError, ValueError, OverflowError):
            gates["saved_spectrum_valid"] = False
        row.update(gates=gates, passed=bool(all(gates.values())))
        rows.append(row)
    prerequisites = dict(selected_count=bool(expected_count is None or len(rows) == expected_count),
                         enough_distinct_points=bool(len({r["physical_alpha"] for r in rows}) >= 4),
                         every_selected_row_qualified=bool(rows and all(r["passed"] for r in rows)))
    result = dict(analysis_kind="saved-data stable-branch extrapolation",
                  independent_forward_reevaluation=False,
                  evidence_scope="Saved convergence metadata and saved multipliers only; no model or eigenpair reevaluation.",
                  formula="delta=alpha_c-alpha; y=(1-mu)^2/(4*delta)",
                  alpha_coordinate="physical alpha", alpha_key=alpha_key, alpha_c=alpha_c,
                  delta_range_inclusive=[low, high], expected_count=expected_count,
                  input_count=len(points), selected_count=len(rows), selection=selection,
                  selected_rows=rows, prerequisite_gates=prerequisites,
                  tolerances=dict(closure=closure_tol, imaginary=imag_tol,
                                  spectral_metadata=spectral_metadata_tol),
                  passed=False, stage="saved-record-gates")
    if not all(prerequisites.values()):
        return result
    x, y = np.array([r["sqrt_delta"] for r in rows]), np.array([r["y"] for r in rows])
    quadratic, linear = _polynomial_fit(x, y, 2), _polynomial_fit(x, y, 1)
    result.update(quadratic=quadratic, linear=linear,
                  quadratic_minus_linear_intercept=quadratic["intercept"]-linear["intercept"],
                  passed=bool(quadratic["full_rank"] and linear["full_rank"]), stage="fit")
    if reference_ab is not None:
        result["reference_comparison"] = dict(
            reference_ab=reference_ab,
            quadratic_difference=quadratic["intercept"]-reference_ab,
            linear_difference=linear["intercept"]-reference_ab,
            quadratic_relative_difference=abs(quadratic["intercept"]-reference_ab)/abs(reference_ab)
                if reference_ab else None)
    return result
