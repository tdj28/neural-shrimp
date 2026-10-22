"""Independent analytic and finite-difference checks; no model calls."""
import copy
import math
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import contraction_valley_geometry as geometry


def jet(value=1., gradient=(.2, -.3), hessian=((.4, .1), (.1, -.2))):
    return dict(value=value, gradient=list(gradient), hessian=np.asarray(hessian).tolist())


def value_at(j, q):
    q = np.asarray(q)
    return j["value"] + np.asarray(j["gradient"]) @ q + .5*q @ np.asarray(j["hessian"]) @ q


def pair_radius(tau, delta, q):
    # Independent 2x2 characteristic matrix, rather than reusing the helper's
    # radius expression or derivatives.
    t, d = value_at(tau, q), value_at(delta, q)
    return max(abs(np.linalg.eigvals([[t, -d], [1., 0.]])))


def finite_differences(function, h):
    zero, e = np.zeros(2), np.eye(2)*h
    gradient = np.array([(function(e[i])-function(-e[i]))/(2*h) for i in range(2)])
    hessian = np.zeros((2, 2))
    for i in range(2):
        hessian[i, i] = (function(e[i])-2*function(zero)+function(-e[i]))/h**2
    hessian[0, 1] = hessian[1, 0] = (
        function(e[0]+e[1])-function(e[0]-e[1])
        -function(-e[0]+e[1])+function(-e[0]-e[1])
    )/(4*h*h)
    return gradient, hessian


def transform(j, Q):
    h = Q @ np.array(j["hessian"]) @ Q.T
    # Explicit roundoff symmetrization belongs to the caller; the helper does
    # not silently change asymmetric input matrices.
    return jet(j["value"], Q @ j["gradient"], .5*(h+h.T))


@pytest.mark.parametrize("sign", [-1, 1])
def test_smooth_chain_rule_recovers_known_multiplier_jets(sign):
    positive = jet(2. if sign == 1 else 1., [.13, -.27], [[.7, .3], [.3, -.4]])
    negative = jet(-1. if sign == 1 else -2., [.31, .19], [[-.6, -.2], [-.2, .8]])
    p, n = positive["value"], negative["value"]
    gp, gn = np.array(positive["gradient"]), np.array(negative["gradient"])
    hp, hn = np.array(positive["hessian"]), np.array(negative["hessian"])
    tau = jet(p+n, gp+gn, hp+hn)
    cross = np.outer(gp, gn)
    delta = jet(p*n, n*gp+p*gn, n*hp+p*hn+(cross+cross.T))
    result = geometry.smooth_radius_jet(tau, delta, sign=sign)
    expected = positive if sign == 1 else negative
    assert result["value"] == pytest.approx(sign*expected["value"], abs=1e-15)
    np.testing.assert_allclose(result["gradient"], sign*np.array(expected["gradient"]), atol=1e-15)
    np.testing.assert_allclose(result["hessian"], sign*np.array(expected["hessian"]), atol=1e-15)
    assert result["pair_eigenvalue_separation"] == pytest.approx(p-n)
    assert "full-spectrum dominance" in result["scope"]


@pytest.mark.parametrize("sign", [-1, 1])
def test_radius_gradient_and_hessian_agree_with_composed_matrix_differences(sign):
    tau = jet(sign*.41, [.7, -.4], [[.8, -.3], [-.3, .6]])
    delta = jet(-.9, [-.3, .27], [[-.4, .2], [.2, .5]])
    result = geometry.smooth_radius_jet(tau, delta, sign=sign)
    function = lambda q: pair_radius(tau, delta, q)
    assert result["value"] == pytest.approx(function([0., 0.]))
    errors = []
    for h in (.004, .002):
        gradient, hessian = finite_differences(function, h)
        errors.append(np.linalg.norm(gradient-result["gradient"])
                      + np.linalg.norm(hessian-result["hessian"]))
    assert errors[0]/errors[1] == pytest.approx(4., rel=.002)
    assert errors[-1] < 3e-6


@pytest.mark.parametrize("scale", [1e-100, 1e-12, 1e-4, 1e100])
def test_common_multiplier_scaling_preserves_geometry_across_magnitudes(scale):
    tau, delta = jet(.4), jet(-1.3)
    def scaled(j, factor):
        return jet(factor*j["value"], factor*np.array(j["gradient"]),
                   factor*np.array(j["hessian"]))
    original = geometry.smooth_radius_jet(tau, delta, sign=1)
    result = geometry.smooth_radius_jet(scaled(tau, scale), scaled(delta, scale**2), sign=1)
    assert result["value"]/scale == pytest.approx(original["value"], rel=3e-15)
    np.testing.assert_allclose(np.array(result["gradient"])/scale, original["gradient"],
                               rtol=3e-15, atol=1e-16)
    np.testing.assert_allclose(np.array(result["hessian"])/scale, original["hessian"],
                               rtol=3e-15, atol=1e-16)


@pytest.mark.parametrize("angle", [.31, 1.43, 2.7])
@pytest.mark.parametrize("reflection", [False, True])
def test_smooth_jet_and_transported_feature_have_orthogonal_covariance(angle, reflection):
    Q = np.array([[math.cos(angle), -math.sin(angle)],
                  [math.sin(angle), math.cos(angle)]])
    if reflection:
        Q[0] *= -1
    tau = jet(.7, [.3, -.2], [[.9, .1], [.1, -.4]])
    delta = jet(-1.3, [.2, .6], [[-.2, .07], [.07, .3]])
    original = geometry.smooth_radius_jet(tau, delta, sign=1)
    rotated = geometry.smooth_radius_jet(transform(tau, Q), transform(delta, Q), sign=1)
    assert rotated["value"] == pytest.approx(original["value"])
    np.testing.assert_allclose(rotated["gradient"], Q @ original["gradient"], atol=3e-16)
    np.testing.assert_allclose(rotated["hessian"], Q @ np.array(original["hessian"]) @ Q.T, atol=3e-16)
    ref = np.array([.8, -.2])
    f = geometry.hessian_valley_feature(original, reference=ref)
    fq = geometry.hessian_valley_feature(rotated, reference=Q@ref)
    assert f["defined"] and fq["defined"]
    assert fq["feature"] == pytest.approx(f["feature"], abs=2e-16)
    np.testing.assert_allclose(fq["normal"], Q @ f["normal"], atol=4e-16)
    np.testing.assert_allclose(fq["eigenvalues"], f["eigenvalues"], atol=4e-16)
    assert fq["reference_overlap"] == pytest.approx(f["reference_overlap"])


def test_quadratic_prediction_translates_a_full_mixed_jet_without_mutation():
    original = jet(1.3, [.4, -.8], [[.7, .25], [.25, -.6]])
    saved = copy.deepcopy(original)
    q, r = np.array([.2, -.3]), np.array([-.1, .7])
    first = geometry.quadratic_jet(original, q)
    twice = geometry.quadratic_jet(first, r)
    once = geometry.quadratic_jet(original, q+r)
    assert first["value"] == pytest.approx(value_at(original, q))
    assert twice["value"] == pytest.approx(once["value"])
    np.testing.assert_allclose(twice["gradient"], once["gradient"], atol=2e-16)
    assert first["hessian"] == original["hessian"]
    assert original == saved
    first["hessian"][0][0] = 99.
    assert original == saved
    assert "prediction only" in first["scope"]


def test_feature_transports_sign_instead_of_restarting_a_component_gauge():
    field = jet(1., [2., 3.], [[1., 0.], [0., 4.]])
    plus = geometry.hessian_valley_feature(field, reference=[.01, 2.])
    minus = geometry.hessian_valley_feature(field, reference=[.01, -2.])
    assert plus["feature"] == 3.
    assert minus["feature"] == -3.
    np.testing.assert_array_equal(plus["normal"], -np.array(minus["normal"]))
    assert plus["eigenvalue_gap"] == 3.
    assert "locus-normal" in plus["scope"]
    assert "curve_normal" not in plus and "feature_gradient" not in plus


@pytest.mark.parametrize("hessian,status", [
    ([[2., 0.], [0., 2.]], "degenerate_largest_eigenvalue"),
    ([[-2., 0.], [0., -1.]], "nonpositive_largest_eigenvalue"),
    ([[-2., 0.], [0., 0.]], "nonpositive_largest_eigenvalue"),
])
def test_nonadmissible_hessian_retains_diagnostics_without_default_feature(hessian, status):
    result = geometry.hessian_valley_feature(jet(hessian=hessian), reference=[1., 1.])
    assert result["status"] == status
    assert not result["defined"]
    assert result["feature"] is None and result["normal"] is None
    assert len(result["eigenvalues"]) == 2


def test_orthogonal_reference_leaves_orientation_undefined():
    result = geometry.hessian_valley_feature(jet(hessian=[[1., 0.], [0., 2.]]), reference=[1., 0.])
    assert result["status"] == "orthogonal_reference"
    assert result["positive_largest"] and result["simple_largest"]
    assert result["reference_overlap"] == 0.
    assert result["feature"] is None and result["normal"] is None


def test_tiny_positive_diagnostics_are_reported_without_invented_thresholds():
    result = geometry.hessian_valley_feature(
        jet(hessian=[[1e-20, 0.], [0., 2e-20]]), reference=[1., 1e-15])
    assert result["defined"]
    assert result["largest_eigenvalue"] == 2e-20
    assert result["eigenvalue_gap"] == 1e-20
    assert result["reference_overlap"] == pytest.approx(1e-15)


@pytest.mark.parametrize("delta_gradient,status,strict", [
    ([.4, -.1], "strict_first_order_minimum", True),
    ([5., 0.], "not_a_normal_minimum", False),
    ([-5., 0.], "not_a_normal_minimum", False),
    ([4., 0.], "first_order_degenerate", False),
    ([-4., 0.], "first_order_degenerate", False),
])
def test_cusp_one_sided_signed_derivatives_and_strict_minimum_gate(delta_gradient, status, strict):
    tau = jet(0., [2., 0.], [[.4, -.3], [-.3, .2]])
    delta = jet(-4., delta_gradient, [[.1, .2], [.2, -.4]])
    result = geometry.cusp_geometry(tau, delta)
    assert result["status"] == status
    assert result["strict_first_order_minimum"] is strict
    assert "hessian" not in result
    assert result["slope_margin"] == min(-result["left_slope"], result["right_slope"])
    n, h = np.array(result["normal"]), 1e-7
    left = (pair_radius(tau, delta, -h*n)-result["value"])/(-h)
    right = (pair_radius(tau, delta, h*n)-result["value"])/h
    assert result["left_slope"] == pytest.approx(left, abs=1e-7)
    assert result["right_slope"] == pytest.approx(right, abs=1e-7)


def test_cusp_orientation_and_slopes_rotate_with_trace_normal():
    Q = np.array([[.6, -.8], [.8, .6]])
    tau, delta = jet(0., [1.2, -.4]), jet(-2., [.4, .2])
    initial = geometry.cusp_geometry(tau, delta)
    rotated = geometry.cusp_geometry(transform(tau, Q), transform(delta, Q))
    np.testing.assert_allclose(rotated["normal"], Q @ initial["normal"], atol=2e-16)
    for key in ("left_slope", "right_slope", "slope_margin", "normal_delta_gradient"):
        assert rotated[key] == pytest.approx(initial[key], abs=1e-16)
    reversed_tau = jet(0., -np.array(tau["gradient"]), -np.array(tau["hessian"]))
    reverse = geometry.cusp_geometry(reversed_tau, delta)
    np.testing.assert_array_equal(reverse["normal"], -np.array(initial["normal"]))
    assert reverse["left_slope"] == pytest.approx(-initial["right_slope"])
    assert reverse["right_slope"] == pytest.approx(-initial["left_slope"])


def test_cusp_branch_limits_recover_two_known_signed_multiplier_jets():
    positive = jet(2., [.13, -.27], [[.7, .3], [.3, -.4]])
    negative = jet(-2., [.31, .19], [[-.6, -.2], [-.2, .8]])
    gp, gn = np.array(positive["gradient"]), np.array(negative["gradient"])
    hp, hn = np.array(positive["hessian"]), np.array(negative["hessian"])
    cross = np.outer(gp, gn)
    tau = jet(0., gp+gn, hp+hn)
    delta = jet(-4., -2*gp+2*gn, -2*hp+2*hn+(cross+cross.T))
    result = geometry.cusp_branch_limit_jets(tau, delta)
    for key, multiplier, sign in (("positive_tau_side", positive, 1),
                                   ("negative_tau_side", negative, -1)):
        limit = result[key]
        assert limit["value"] == 2.
        np.testing.assert_allclose(limit["gradient"], sign*np.array(multiplier["gradient"]), atol=1e-16)
        np.testing.assert_allclose(limit["hessian"], sign*np.array(multiplier["hessian"]), atol=3e-16)
        assert limit["branch_sign"] == sign
        assert limit["at_exact_trace_zero"] is True
        assert "not a Hessian of rho across its cusp" in limit["scope"]
    assert "hessian" not in result and "gradient" not in result
    np.testing.assert_allclose(
        np.array(result["positive_tau_side"]["gradient"])-result["negative_tau_side"]["gradient"],
        tau["gradient"], atol=2e-16)
    np.testing.assert_allclose(
        np.array(result["positive_tau_side"]["hessian"])-result["negative_tau_side"]["hessian"],
        tau["hessian"], atol=2e-16)


@pytest.mark.parametrize("sign,key", [(1, "positive_tau_side"), (-1, "negative_tau_side")])
def test_each_active_side_smooth_jet_approaches_its_own_cusp_limit(sign, key):
    tau = jet(0., [.7, -.4], [[.8, -.3], [-.3, .6]])
    delta = jet(-.9, [-.3, .27], [[-.4, .2], [.2, .5]])
    limits = geometry.cusp_branch_limit_jets(tau, delta)
    normal = np.array(limits["cusp"]["normal"])
    errors = []
    for distance in (.001, .0005):
        q = sign*distance*normal
        shifted_tau = geometry.quadratic_jet(tau, q)
        assert sign*shifted_tau["value"] > 0
        smooth = geometry.smooth_radius_jet(shifted_tau, geometry.quadratic_jet(delta, q), sign=sign)
        error = np.linalg.norm(np.array(smooth["gradient"])-limits[key]["gradient"])
        error += np.linalg.norm(np.array(smooth["hessian"])-limits[key]["hessian"])
        errors.append(error)
    assert errors[0]/errors[1] == pytest.approx(2., rel=.002)
    assert errors[1] < .001
    # Normal projections of the two limiting gradients must equal the signed
    # left/right physical-radius derivatives, not outward-slope magnitudes.
    slope_key = "right_slope" if sign == 1 else "left_slope"
    assert normal @ limits[key]["gradient"] == pytest.approx(limits["cusp"][slope_key], abs=3e-16)


@pytest.mark.parametrize("sign,key", [(1, "positive_tau_side"), (-1, "negative_tau_side")])
def test_cusp_limits_match_independent_smooth_signed_eigenvalue_differences(sign, key):
    tau = jet(0., [.7, -.4], [[.8, -.3], [-.3, .6]])
    delta = jet(-.9, [-.3, .27], [[-.4, .2], [.2, .5]])
    limit = geometry.cusp_branch_limit_jets(tau, delta)[key]
    def smooth_extension(q):
        t, d = value_at(tau, q), value_at(delta, q)
        pair = np.linalg.eigvals([[t, -d], [1., 0.]])
        assert np.all(pair.imag == 0)
        pair = np.sort(pair.real)
        # This selected signed eigenvalue is an analytic extension on BOTH
        # sides. It is deliberately not the nonsmooth physical pair maximum.
        return pair[-1] if sign == 1 else -pair[0]
    gradient, hessian = finite_differences(smooth_extension, .001)
    np.testing.assert_allclose(gradient, limit["gradient"], atol=1e-7)
    np.testing.assert_allclose(hessian, limit["hessian"], atol=2e-7)


def test_cusp_limit_jets_are_orthogonally_covariant():
    tau, delta = jet(0., [.6, -.9]), jet(-1.3, [.2, .4])
    Q = np.array([[.6, -.8], [.8, .6]])
    before = geometry.cusp_branch_limit_jets(tau, delta)
    after = geometry.cusp_branch_limit_jets(transform(tau, Q), transform(delta, Q))
    for key in ("positive_tau_side", "negative_tau_side"):
        np.testing.assert_allclose(after[key]["gradient"], Q @ before[key]["gradient"], atol=2e-16)
        np.testing.assert_allclose(after[key]["hessian"], Q @ np.array(before[key]["hessian"]) @ Q.T, atol=2e-16)


@pytest.mark.parametrize("tau,delta", [
    (jet(1e-30), jet(-1.)), (jet(-1e-30), jet(-1.)),
    (jet(0., [0., 0.]), jet(-1.)), (jet(0.), jet(0.)), (jet(0.), jet(.1)),
])
def test_cusp_limits_require_an_exact_regular_negative_determinant_crease(tau, delta):
    with pytest.raises(ValueError):
        geometry.cusp_branch_limit_jets(tau, delta)


def test_stationary_trace_identity_allows_nonzero_radius_gradient_and_no_hessian():
    tau = jet(0., [0., 0.], [[-.4, .1], [.1, .2]])
    delta = jet(-4., [.6, -.8], [[.1, -.2], [-.2, .3]])
    result = geometry.stationary_trace_zero_gradient(tau, delta)
    np.testing.assert_allclose(result["gradient"], [-.15, .2], atol=1e-16)
    for direction in (np.array([1., 0.]), np.array([0., 1.]), np.array([.6, .8])):
        h = 1e-6
        empirical = (pair_radius(tau, delta, h*direction)-pair_radius(tau, delta, -h*direction))/(2*h)
        assert empirical == pytest.approx(np.array(result["gradient"]) @ direction, abs=4e-10)
    assert "hessian" not in result
    assert "no Hessian or minimum/topology claim" in result["scope"]


@pytest.mark.parametrize("sign", [0, 2, -2, True, 1., "1", None])
def test_invalid_branch_declarations_refused(sign):
    with pytest.raises(ValueError, match="sign"):
        geometry.smooth_radius_jet(jet(.2), jet(-1.), sign=sign)


@pytest.mark.parametrize("tau,sign", [(0., 1), (-0., -1), (.2, -1), (-.2, 1)])
def test_cusp_or_mismatched_trace_refused_in_smooth_formula(tau, sign):
    with pytest.raises(ValueError, match="nonzero tau"):
        geometry.smooth_radius_jet(jet(tau), jet(-1.), sign=sign)


@pytest.mark.parametrize("delta", [0., .2])
@pytest.mark.parametrize("function", ["smooth", "cusp", "stationary"])
def test_pair_regime_must_have_strictly_negative_determinant(delta, function):
    with pytest.raises(ValueError, match="strictly negative"):
        if function == "smooth":
            geometry.smooth_radius_jet(jet(.2), jet(delta), sign=1)
        elif function == "cusp":
            geometry.cusp_geometry(jet(0.), jet(delta))
        else:
            geometry.stationary_trace_zero_gradient(jet(0., [0., 0.]), jet(delta))


@pytest.mark.parametrize("value", [1e-30, -.01])
def test_cusp_does_not_silently_project_near_zero_trace(value):
    with pytest.raises(ValueError, match="exact tau=0"):
        geometry.cusp_geometry(jet(value), jet(-1.))


def test_singular_trace_zero_uses_separate_conditional_identity():
    with pytest.raises(ValueError, match="nonzero norm"):
        geometry.cusp_geometry(jet(0., [0., 0.]), jet(-1.))
    with pytest.raises(ValueError, match="exact tau=0 and gradient=0"):
        geometry.stationary_trace_zero_gradient(jet(0., [1e-30, 0.]), jet(-1.))
    with pytest.raises(ValueError, match="exact tau=0 and gradient=0"):
        geometry.stationary_trace_zero_gradient(jet(1e-30, [0., 0.]), jet(-1.))


@pytest.mark.parametrize("key,bad", [
    ("value", True), ("value", "1"), ("value", 1+0j),
    ("value", np.nan), ("value", np.inf), ("value", [1.]),
    ("gradient", [1.]), ("gradient", [[1., 2.]]),
    ("gradient", [np.nan, 0.]), ("gradient", [np.inf, 0.]),
    ("gradient", [True, False]), ("gradient", [1+0j, 0j]),
    ("gradient", ["1", "2"]), ("gradient", np.array([1., 2.], dtype=object)),
    ("hessian", np.eye(3)), ("hessian", [[1., 1e-30], [0., 1.]]),
    ("hessian", [[1., np.nan], [np.nan, 1.]]), ("hessian", [[1., 0.], [0., np.inf]]),
])
def test_malformed_jets_are_rejected_by_all_entry_points(key, bad):
    broken = jet(0., [0., 0.])
    broken[key] = bad
    for call in (
        lambda: geometry.quadratic_jet(broken, [0., 0.]),
        lambda: geometry.smooth_radius_jet(jet(.2), broken, sign=1),
        lambda: geometry.hessian_valley_feature(broken, reference=[1., 1.]),
        lambda: geometry.cusp_geometry(broken, jet(-1.)),
        lambda: geometry.cusp_branch_limit_jets(broken, jet(-1.)),
        lambda: geometry.stationary_trace_zero_gradient(broken, jet(-1.)),
    ):
        with pytest.raises(ValueError):
            call()


@pytest.mark.parametrize("broken", [None, [], {}, {"value": 1., "gradient": [0., 0.]}])
def test_missing_jet_fields_fail_loudly(broken):
    with pytest.raises(ValueError, match="must contain"):
        geometry.quadratic_jet(broken, [0., 0.])


@pytest.mark.parametrize("reference", [[0., 0.], [np.nan, 1.], [1.], [1+0j, 1.]])
def test_bad_orientation_references_fail(reference):
    with pytest.raises(ValueError):
        geometry.hessian_valley_feature(jet(), reference=reference)


@pytest.mark.parametrize("q", [[0.], [np.nan, 0.], [np.inf, 0.], ["1", "2"]])
def test_bad_displacements_fail(q):
    with pytest.raises(ValueError):
        geometry.quadratic_jet(jet(), q)


def test_nonfinite_computed_geometry_fails_instead_of_emitting_nan_or_inf():
    with pytest.raises(ValueError, match="not finite"):
        geometry.quadratic_jet(jet(), [1e308, 1e308])
    with pytest.raises(ValueError, match="not finite"):
        geometry.smooth_radius_jet(jet(.1, [1e308, 1e308]), jet(-1.), sign=1)


def test_all_results_declare_the_coordinate_metric():
    tau, delta = jet(.2), jet(-1.)
    smooth = geometry.smooth_radius_jet(tau, delta, sign=1)
    results = [smooth, geometry.quadratic_jet(tau, [.1, .2]),
               geometry.hessian_valley_feature(smooth, reference=[1., 1.]),
               geometry.cusp_geometry(jet(0.), delta),
               geometry.cusp_branch_limit_jets(jet(0.), delta),
               geometry.stationary_trace_zero_gradient(jet(0., [0., 0.]), delta)]
    assert all(result["metric"] == geometry.METRIC for result in results)
    assert "scaled Euclidean" in geometry.METRIC and "orthogonal" in geometry.METRIC
