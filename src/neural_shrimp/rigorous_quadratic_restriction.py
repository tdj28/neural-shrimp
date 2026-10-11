"""Exact restriction of an already proved quadratic implicit root family.

The caller authenticates the original root correction radii and residual
Taylor coefficients. New Jacobians must be acquired on the returned fresh
domain; no old whole-Jacobian domain coverage is assumed after recentering.
"""
from flint import arb, arb_mat
from .rigorous_connection import require
from .rigorous_parameter_localization import finite


def restrict_quadratic(center, slope, curvature, weights, original_halfwidth,
                       radius, shift, halfwidth, center_coefficients,
                       whole_coefficients):
    n = len(center)
    require(1 <= n <= 705 and len(slope) == len(curvature) == len(weights) == n,
            'full original quadratic predictor dimensions')
    require(all(v.is_exact() and v.is_finite() for v in
                center + slope + curvature + weights +
                [original_halfwidth, radius, shift, halfwidth]),
            'exact finite original predictor and restriction')
    require(original_halfwidth > 0 and radius > 0 and halfwidth > 0 and
            all(w > 0 for w in weights), 'positive original and local scales')
    require(abs(shift) + halfwidth <= original_halfwidth,
            'closed restricted parameter interval within proved original')
    c, w = center_coefficients, whole_coefficients
    require(c.nrows() == w.nrows() == n and c.ncols() == w.ncols() == 4
            and finite(c) and finite(w), 'authenticated full cubic Taylor data')
    local_center = [center[i] + slope[i]*shift + curvature[i]*shift*shift/2
                    for i in range(n)]
    local_slope = [slope[i] + curvature[i]*shift for i in range(n)]
    require(all(v.is_exact() for v in local_center + local_slope),
            'exact recentering without rounded polynomial coefficients')
    d = arb(0, halfwidth)
    delta = shift + d
    # Arb may round this arithmetic multiplier slightly beyond the exact
    # closed subset. It only multiplies already bounded Taylor coefficients;
    # no old derivative is evaluated or asserted valid on that extra sliver.
    # The exact shift+halfwidth test above supplies the actual parameter set.
    # R(delta) and R'(delta) are bounded using their respective Taylor
    # theorems at the ORIGINAL center. We never differentiate a remainder.
    residual = arb_mat([[c[i,0] + c[i,1]*delta + c[i,2]*delta*delta
                         + w[i,3]*delta*delta*delta] for i in range(n)])
    derivative = arb_mat([[c[i,1] + 2*c[i,2]*delta
                           + 3*w[i,3]*delta*delta] for i in range(n)])
    radii = [radius*v for v in weights]
    require(all(v.is_exact() for v in radii), 'exact inherited correction radii')
    fresh_domain = [local_center[i] + local_slope[i]*d + curvature[i]*d*d/2
                    + arb(0, radii[i]) for i in range(n)]
    require(all(v.is_finite() for v in fresh_domain) and
            finite(residual) and finite(derivative), 'finite restricted domains')
    return dict(center=local_center, slope=local_slope, curvature=list(curvature),
                weights=list(weights), halfwidth=halfwidth, radius=radius,
                fresh_domain=fresh_domain, path_residual=residual,
                residual_derivative=derivative, original_delta=delta)
