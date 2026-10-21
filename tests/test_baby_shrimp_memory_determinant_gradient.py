"""Analytic stencil and conditional cusp controls; no model or provider calls."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import baby_shrimp_memory_determinant_gradient as diagnostic


def stencil(center=(0., 0.), step=.01, cubic=0.):
    center = np.asarray(center, float)
    mapping, indices = {}, np.arange(9).reshape((3, 3))
    tau = np.empty((3, 3))
    for iy, dy in enumerate((-1, 0, 1)):
        for ix, dx in enumerate((-1, 0, 1)):
            p = center + step * np.array([dx, dy])
            x, y = p
            value = .02-x*x+y*y+cubic*x**3
            tau[iy, ix] = value
            index = int(indices[iy, ix])
            mapping[index] = {"index": index+100, "coordinate": p.tolist(), "beta": .003,
                              "qualified": True, "tau": value,
                              "delta": -.01+.002*x+.003*y+.004*x*x-.001*x*y}
    field = diagnostic.stencil_field(tau, step)
    H, g = np.asarray(field["hessian"]), np.asarray(field["gradient"])
    row = {"center": center.tolist(), "parameter": .003, "stencil_step": step,
           "point_indices": indices.tolist(), "tau": field["value"],
           "gradient": g.tolist(), "hessian": H.tolist(), "correction": (-np.linalg.solve(H, g)).tolist()}
    return row, mapping


def test_nine_point_gradient_and_coordinate_scaling_are_recomputed():
    row, mapping = stencil(center=(.01, -.02))
    result = diagnostic.stencil_diagnostics(row, mapping, np.array([.5, 2.]))
    np.testing.assert_allclose(result["delta"]["gradient"], [.0021, .00299], atol=1e-15)
    np.testing.assert_allclose(result["delta"]["physical_coordinate_gradient"], [.0042, .001495], atol=1e-15)
    np.testing.assert_allclose(result["tau"]["gradient"], [-.02, -.04], atol=1e-14)
    np.testing.assert_allclose(result["determinant_gradient_at_quadratically_corrected_center"], [.002, .003], atol=1e-14)
    assert result["centering_gradient_change_norm"] > 0
    assert "not automatically" in result["conditional_zero_trace_stationary_tilt"]["scope"]


@pytest.mark.parametrize("change,match", [("coordinate", "misplaced"), ("delta", "opposite-sign"), ("trace", "trace stencil")])
def test_rejects_corrupted_or_misplaced_stencil_values(change, match):
    row, mapping = stencil()
    if change == "coordinate":
        mapping[0]["coordinate"][0] += .001
    elif change == "delta":
        mapping[0]["delta"] = .001
    else:
        mapping[0]["tau"] += .0001
    with pytest.raises(ValueError, match=match):
        diagnostic.stencil_diagnostics(row, mapping, np.ones(2))


def test_matched_center_richardson_removes_cubic_trace_gradient_bias():
    rows = [diagnostic.stencil_diagnostics(*stencil(step=h, cubic=.7), np.ones(2)) for h in (.1, .03)]
    result = diagnostic.compare_scales(*rows, np.ones(2))
    assert result["centers_identical"] and result["richardson"]["applied"]
    np.testing.assert_allclose(result["richardson"]["tau_gradient"], [0., 0.], atol=1e-15)
    np.testing.assert_allclose(result["richardson"]["delta_gradient"], [.002, .003], atol=1e-15)
    assert "do not establish convergence order" in result["richardson"]["scope"]


def test_different_centers_use_quadratic_transport_and_no_richardson():
    coarse = diagnostic.stencil_diagnostics(*stencil(center=(.01, -.02), step=.1), np.ones(2))
    fine = diagnostic.stencil_diagnostics(*stencil(center=(.003, .004), step=.03), np.ones(2))
    result = diagnostic.compare_scales(coarse, fine, np.ones(2))
    assert not result["richardson"]["applied"]
    for field in result["fields"].values():
        assert field["gradient_transport_residual_norm"] < 1e-14
        assert abs(field["value_transport_residual"]) < 1e-16


def test_conditional_stationary_tilt_and_physical_units():
    result = diagnostic.conditional_tilt(-4., [2., -1.], [.5, 2.])
    np.testing.assert_allclose(result["scaled_coordinate_gradient"], [-.5, .25])
    np.testing.assert_allclose(result["physical_coordinate_gradient"], [-1., .125])
    with pytest.raises(ValueError, match="negative finite"):
        diagnostic.conditional_tilt(0., [2., -1.], [.5, 2.])


def test_cusp_strict_sufficiency_reverse_exclusion_and_inconclusive_equality():
    assert "sufficient" in diagnostic.transverse_cusp_condition(1, .5, -1)["status"]
    assert "rules out" in diagnostic.transverse_cusp_condition(1, 2, -1)["status"]
    assert "equality: inconclusive" in diagnostic.transverse_cusp_condition(1, 1, -1)["status"]
    # tau=t, Delta=-1+t-t² has equality at0, but higher orders give a strict
    # minimum: the right derivative is0 and the right quadratic term positive.
    for t in (-.01, -.001, .001, .01):
        delta = -1+t-t*t
        radius = (np.sqrt(t*t-4*delta)+abs(t))/2
        assert radius > 1
    with pytest.raises(ValueError, match="regular trace-zero"):
        diagnostic.transverse_cusp_condition(0, 1, -1)
