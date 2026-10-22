"""Pure local geometry for a real, opposite-sign multiplier pair.

All derivatives and eigenvectors use one caller-declared *scaled Euclidean*
coordinate system. They transform covariantly under orthogonal changes of
that frame; an anisotropic rescaling changes the valley convention. No model
calls, coordinate rescaling, curve continuation, empirical admission tolerance,
or claim that this pair dominates the full spectrum is supplied here.

A scalar jet is a mapping with ``value``, a length-two ``gradient``, and a
symmetric 2-by-2 ``hessian``. Smooth-field jets may be estimated from stencils
that straddle trace zero; the Hessian of the nonsmooth radius itself must not
be estimated across that crease. Floating-point sign/gap diagnostics below
are algebraic conditions, not uncertainty-qualified experimental decisions.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real
import math

import numpy as np


METRIC = "caller-declared scaled Euclidean coordinates (orthogonal covariance only)"
PAIR_SCOPE = (
    "Local algebra for an isolated real opposite-sign multiplier pair; "
    "full-spectrum dominance and numerical uncertainty require external checks."
)


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite real scalar")
    try:
        result = float(value)
    except (ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be a finite real scalar") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _array(value, shape, name):
    try:
        raw = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite real array of shape {shape}") from error
    # Do not silently cast complex, Boolean, string or object data to float.
    if raw.shape != shape or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real array of shape {shape}")
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(raw, dtype=float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def _jet(jet, name):
    if not isinstance(jet, Mapping) or not {"value", "gradient", "hessian"} <= jet.keys():
        raise ValueError(f"{name} must contain value, gradient and hessian")
    value = _scalar(jet["value"], f"{name} value")
    gradient = _array(jet["gradient"], (2,), f"{name} gradient")
    hessian = _array(jet["hessian"], (2, 2), f"{name} hessian")
    # The stencil builder produces an exactly symmetric array. Do not hide a
    # caller's asymmetric derivative estimate by averaging its two triangles.
    if not np.array_equal(hessian, hessian.T):
        raise ValueError(f"{name} hessian must be symmetric; symmetrize explicitly upstream")
    return value, gradient, hessian


def _finite_result(*values):
    if any(not np.all(np.isfinite(value)) for value in values):
        raise ValueError("local geometry is not finite in the working precision")


def _unit(vector, name):
    vector = _array(vector, (2,), name)
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError(f"{name} must have a finite nonzero norm")
    return vector / norm, norm


def quadratic_jet(jet, displacement):
    """Evaluate the declared quadratic Taylor model, not a new measurement.

    Returns its value and first/second derivatives at ``displacement`` from
    the input jet's origin, in the same scaled Euclidean frame. The Hessian
    is held fixed. No bound on the omitted higher-order terms is inferred.
    """
    value, gradient, hessian = _jet(jet, "field")
    q = _array(displacement, (2,), "displacement")
    try:
        with np.errstate(over="raise", invalid="raise"):
            prediction = value + gradient @ q + .5 * q @ hessian @ q
            predicted_gradient = gradient + hessian @ q
    except FloatingPointError as error:
        raise ValueError("quadratic prediction is not finite in the working precision") from error
    _finite_result(prediction, predicted_gradient)
    return {
        "value": float(prediction), "gradient": predicted_gradient.tolist(),
        "hessian": hessian.tolist(), "displacement": q.tolist(), "metric": METRIC,
        "scope": "Quadratic Taylor prediction only; omitted higher-order terms unbounded.",
    }


def smooth_radius_jet(tau, delta, *, sign):
    """Differentiate rho=(sqrt(tau**2-4*delta)+abs(tau))/2 away from tau=0.

    ``sign`` must be integer +1 or -1 and match the nonzero trace. Delta must
    be strictly negative, so the two multipliers are real with opposite signs.
    The formulas are exact chain rules on this declared active smooth branch,
    subject to the accuracy of the input jets and floating-point arithmetic.
    They are not a smooth extension through the absolute-value crease.
    """
    t, gt, ht = _jet(tau, "tau")
    d, gd, hd = _jet(delta, "delta")
    if isinstance(sign, (bool, np.bool_)) or not isinstance(sign, Integral) or sign not in (-1, 1):
        raise ValueError("sign must be integer +1 or -1")
    if t == 0 or sign * t <= 0:
        raise ValueError("a nonzero tau matching the declared sign is required")
    if d >= 0:
        raise ValueError("delta must be strictly negative")
    # hypot avoids unnecessary overflow in tau*tau - 4*delta.
    separation = math.hypot(t, 2 * math.sqrt(-d))
    radius = .5 * separation + .5 * abs(t)
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            b_over_s = (t / separation) * gt - (2 / separation) * gd
            gradient = .5 * (b_over_s + sign * gt)
            hessian = .5 * (
                np.outer(gt, gt) / separation
                + (t / separation) * ht - (2 / separation) * hd
                - np.outer(b_over_s, b_over_s) / separation + sign * ht
            )
    except (FloatingPointError, OverflowError, ZeroDivisionError) as error:
        raise ValueError("radius derivatives are not finite in the working precision") from error
    _finite_result(radius, gradient, hessian, separation)
    return {
        "value": float(radius), "gradient": gradient.tolist(), "hessian": hessian.tolist(),
        "tau": t, "delta": d, "branch_sign": int(sign),
        "pair_eigenvalue_separation": separation, "metric": METRIC, "scope": PAIR_SCOPE,
    }


def hessian_valley_feature(radius_jet, *, reference):
    """Return f=e_high dot grad rho with a transported eigenvector sign.

    The largest algebraic Hessian eigenvalue must be positive and simple.
    Its sign is chosen by positive overlap with the supplied nonzero reference
    direction (normally the preceding continuation normal). No component-sign
    convention is imposed. Exactly zero overlap leaves orientation undefined.
    Tiny positive curvature/gap/overlap is reported, never promoted by an
    invented numerical threshold. The caller must apply measured uncertainties.

    Undefined features return ``feature=None`` with diagnostics and a reason;
    they never fall through to a default eigenvector/zero. ``normal`` denotes
    the Hessian direction, which need not be normal to the eventual f=0 locus.
    No feature-gradient or curve tangent is inferred from a second-order jet.
    """
    value, gradient, hessian = _jet(radius_jet, "radius")
    if value <= 0:
        raise ValueError("radius value must be positive")
    ref, _ = _unit(reference, "reference")
    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    gap = float(eigenvalues[1] - eigenvalues[0])
    _finite_result(eigenvalues, eigenvectors, gap)
    result = {
        "defined": False, "feature": None, "normal": None,
        "eigenvalues": eigenvalues.tolist(), "largest_eigenvalue": float(eigenvalues[1]),
        "eigenvalue_gap": gap, "positive_largest": bool(eigenvalues[1] > 0),
        "simple_largest": bool(gap > 0), "reference": ref.tolist(),
        "reference_overlap": None, "metric": METRIC,
        "scope": "Hessian-valley feature only; no empirical gate or locus-normal inference.",
    }
    if gap == 0:
        return {**result, "status": "degenerate_largest_eigenvalue"}
    if eigenvalues[1] <= 0:
        return {**result, "status": "nonpositive_largest_eigenvalue"}
    normal = eigenvectors[:, 1]
    overlap = float(normal @ ref)
    result["reference_overlap"] = abs(overlap)
    if overlap == 0:
        return {**result, "status": "orthogonal_reference"}
    if overlap < 0:
        normal = -normal
    feature = float(normal @ gradient)
    _finite_result(feature)
    return {
        **result, "defined": True, "status": "defined", "feature": feature,
        "normal": normal.tolist(),
    }


def cusp_geometry(tau, delta):
    """First-order normal-minimum diagnostics at an exact regular tau=0 crease.

    Uses its geometric unit normal n=grad(tau)/||grad(tau)||, directed toward
    positive tau. The returned left/right derivatives are both with respect
    to the signed coordinate s in x+s*n (not two outward directions).
    A strict first-order minimum requires left<0<right. Zero slope is
    unresolved by this first-order test and may need higher-order analysis.
    Nonzero input tau is refused instead of silently snapped onto the crease.
    No radius Hessian across the cusp is returned.
    """
    t, gt, _ = _jet(tau, "tau")
    d, gd, _ = _jet(delta, "delta")
    if t != 0:
        raise ValueError("cusp geometry requires exact tau=0; project/refine explicitly upstream")
    if d >= 0:
        raise ValueError("delta must be strictly negative")
    normal, a = _unit(gt, "tau gradient")
    radius = math.sqrt(-d)
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            normal_delta = float(normal @ gd)
            smooth_slope = -normal_delta / (2 * radius)
            left = smooth_slope - .5 * a
            right = smooth_slope + .5 * a
    except (FloatingPointError, OverflowError, ZeroDivisionError) as error:
        raise ValueError("cusp slopes are not finite in the working precision") from error
    _finite_result(normal_delta, left, right)
    strict = left < 0 < right
    status = ("strict_first_order_minimum" if strict else
              "not_a_normal_minimum" if left > 0 or right < 0 else
              "first_order_degenerate")
    return {
        "value": radius, "normal": normal.tolist(), "tau_gradient_norm": a,
        "normal_delta_gradient": normal_delta, "left_slope": left, "right_slope": right,
        "slope_margin": min(-left, right), "strict_first_order_minimum": bool(strict),
        "status": status, "metric": METRIC, "scope": PAIR_SCOPE,
    }


def cusp_branch_limit_jets(tau, delta):
    """Two separate smooth-branch derivative limits at a regular tau=0 cusp.

    ``positive_tau_side`` is the limit of (S+tau)/2 from tau>0;
    ``negative_tau_side`` is the limit of (S-tau)/2 from tau<0, where
    S=sqrt(tau**2-4*delta). Each analytic extension has its own gradient and
    Hessian at the crease. They are NOT one common gradient/Hessian of rho.
    No epsilon displacement or trace snapping is performed. Exact tau=0,
    nonzero grad(tau), and delta<0 are required. The geometric cusp diagnostics
    and branch identity are retained alongside the two jets.

    The two limit jets can be passed separately to hessian_valley_feature;
    simultaneous feature zeros describe candidate boundary limits only.
    Their active-side continuation and any mixed-junction interpretation still
    require separate empirical checks.
    """
    # Also applies the exact regular-crease and finite one-sided slope gates.
    cusp = cusp_geometry(tau, delta)
    _, gt, ht = _jet(tau, "tau")
    _, gd, hd = _jet(delta, "delta")
    radius = cusp["value"]
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            half_gt = .5 * gt
            determinant_slope = gd / (2 * radius)
            common_hessian = (
                np.outer(half_gt, half_gt) / radius - hd / (2 * radius)
                - np.outer(determinant_slope, determinant_slope) / radius
            )
            branches = {}
            for sign, key, side in ((1, "positive_tau_side", "tau>0"),
                                    (-1, "negative_tau_side", "tau<0")):
                gradient = -determinant_slope + sign * half_gt
                hessian = common_hessian + sign * .5 * ht
                _finite_result(gradient, hessian)
                branches[key] = {
                    "value": radius, "gradient": gradient.tolist(), "hessian": hessian.tolist(),
                    "branch_sign": sign, "limit_from": side, "at_exact_trace_zero": True,
                    "metric": METRIC,
                    "scope": f"One-sided {side} smooth-extension derivative limit; not a Hessian of rho across its cusp.",
                }
    except (FloatingPointError, OverflowError, ZeroDivisionError) as error:
        raise ValueError("cusp branch limits are not finite in the working precision") from error
    return {
        **branches, "cusp": cusp, "metric": METRIC,
        "scope": "Two distinct one-sided smooth-extension jets; no common cusp Hessian or junction claim.",
    }


def stationary_trace_zero_gradient(tau, delta):
    """Conditional gradient identity at tau=0 and grad(tau)=0, delta<0.

    At this trace-stationary zero the |tau| term is second order, so rho is
    differentiable and grad(rho)=-grad(delta)/(2*sqrt(-delta)). A radius Hessian
    need not exist. This identity neither establishes a local minimum nor
    resolves valley topology. Exact algebraic preconditions are required;
    uncertainty-qualified near-zero inputs must be handled explicitly upstream.
    """
    t, gt, _ = _jet(tau, "tau")
    d, gd, _ = _jet(delta, "delta")
    if t != 0 or np.any(gt != 0):
        raise ValueError("stationary trace identity requires exact tau=0 and gradient=0")
    if d >= 0:
        raise ValueError("delta must be strictly negative")
    radius = math.sqrt(-d)
    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            gradient = -gd / (2 * radius)
    except FloatingPointError as error:
        raise ValueError("stationary gradient is not finite in the working precision") from error
    _finite_result(gradient)
    return {
        "value": radius, "gradient": gradient.tolist(), "metric": METRIC,
        "scope": "Conditional differentiability identity; no Hessian or minimum/topology claim.",
    }
