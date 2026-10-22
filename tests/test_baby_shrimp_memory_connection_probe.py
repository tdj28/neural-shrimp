"""Analytic saddle controls only; no model, GPU or provider calls."""
import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import baby_shrimp_memory_connection_probe as probe


class HyperbolicEvaluator:
    """Known trace contours, with a spatially varying negative determinant."""

    def __init__(self, height=1e-4, center=(0., 0.), fail_after=None,
                 curvatures=(-.02, .01), cubic=0.):
        self.height = height
        self.center = np.asarray(center, float)
        self.fail_after = fail_after
        self.curvatures = curvatures
        self.cubic = cubic
        self.rows = []

    def __call__(self, coordinate, beta, cyclic=False):
        if self.fail_after is not None and len(self.rows) >= self.fail_after:
            raise TimeoutError("synthetic evaluation budget expired")
        coordinate = np.asarray(coordinate, float)
        x, y = coordinate-self.center
        tau = float(self.height+self.curvatures[0]*x*x+self.curvatures[1]*y*y
                    + self.cubic*y**3)
        delta = float(-1e-8+1e-9*x)
        radius = float((np.sqrt(tau*tau-4*delta)+abs(tau))/2)
        row = dict(index=len(self.rows), coordinate=coordinate.tolist(), beta=beta,
                   qualified=True, tau=tau, delta=delta, radius=radius,
                   phases=[dict(phase=phase, tau=tau, delta=delta, radius=radius)
                           for phase in ((0, 6, 12) if cyclic else (6,))])
        self.rows.append(row)
        return row


def config():
    return dict(before_beta=.006, after_beta=.007,
                axis_offsets=[-.1, -.05, .05, .1], stencil_steps=[.003, .001],
                slice_offsets=[0., -.08, .08], root_bracket=.3,
                maximum_correction_norm=3e-5, root_coordinate_tolerance=1e-7,
                root_trace_tolerance=1e-10, root_max_iterations=20,
                normal_offsets=[-.003, .003],
                candidate_tilt_offsets=[-.0003, -.0001, .0001, .0003])


def plan():
    return dict(id="BSC4-direct-connection", period=18, maximum_evaluations=120,
                connection_probe=config())


def test_validate_probe_accepts_the_bounded_p18_design():
    p = plan()
    assert probe.validate_probe(p) == p["connection_probe"]


@pytest.mark.parametrize("key,value", [("id", "different"), ("period", 6),
                                      ("maximum_evaluations", 121),
                                      ("maximum_evaluations", 0)])
def test_validate_probe_rejects_wrong_family_or_budget(key, value):
    p = plan()
    p[key] = value
    with pytest.raises(ValueError):
        probe.validate_probe(p)


@pytest.mark.parametrize("key,value", [
    ("before_beta", .005), ("after_beta", .006),
    ("axis_offsets", [-.1, .1]), ("stencil_steps", [.003, .003]),
    ("stencil_steps", [.01, .003]), ("slice_offsets", [-.08, 0., .08]),
    ("root_bracket", .31), ("maximum_correction_norm", 3.1e-5),
    ("root_coordinate_tolerance", 1.1e-7), ("root_trace_tolerance", 1.1e-10),
    ("root_max_iterations", 21), ("normal_offsets", [-.003, .002]),
    ("candidate_tilt_offsets", [-.001, -.0001, .0001, .001]),
])
def test_validate_probe_rejects_changed_geometry_or_loosened_gates(key, value):
    p = plan()
    p["connection_probe"][key] = value
    with pytest.raises(ValueError):
        probe.validate_probe(p)


def test_matched_stencils_keep_the_exact_center_and_recover_the_saddle():
    center = np.array([.23, -.43])
    evaluate = HyperbolicEvaluator(center=center)
    rows = probe.matched_stencils(evaluate, center, .006, [.01, .003], 1e-5)
    assert len(rows) == 2
    assert all(row["center"] == center.tolist() for row in rows)
    for row in rows:
        np.testing.assert_allclose(row["gradient"], [0., 0.], atol=1e-14)
        np.testing.assert_allclose(row["hessian"], np.diag([-.04, .02]), atol=1e-12)
        np.testing.assert_allclose(row["correction"], [0., 0.], atol=1e-12)
        indices = np.asarray(row["point_indices"])
        assert indices.shape == (3, 3)
        for iy, dy in enumerate((-1, 0, 1)):
            for ix, dx in enumerate((-1, 0, 1)):
                point = evaluate.rows[int(indices[iy, ix])]
                expected = center+row["step"]*np.array([dx, dy])
                np.testing.assert_array_equal(point["coordinate"], expected)


def test_matched_stencils_refuse_a_definite_hessian():
    with pytest.raises((ValueError, RuntimeError), match="saddle|Hessian"):
        probe.matched_stencils(HyperbolicEvaluator(curvatures=(.02, .01)),
                               [0., 0.], .006, [.01, .003], 1e-5)


def test_matched_stencils_refuse_an_uncorrected_center_without_moving_it():
    evaluate = HyperbolicEvaluator()
    center = np.array([.01, -.02])
    before = center.copy()
    with pytest.raises((ValueError, RuntimeError), match="correction|center"):
        probe.matched_stencils(evaluate, center, .006, [.01, .003], 1e-5)
    np.testing.assert_array_equal(center, before)


@pytest.mark.parametrize("height,bracket_axis", [(4e-5, 0), (-4e-5, 1)])
def test_direct_axis_samples_reverse_the_sign_bracket_axis(height, bracket_axis):
    center = np.array([.23, -.43])
    evaluate = HyperbolicEvaluator(height=height, center=center)
    result = probe.axis_diagnostics(evaluate, center, .006,
                                    np.diag([-.04, .02]), config()["axis_offsets"])
    assert result["center"] == center.tolist()
    for col, axis in enumerate(result["axes"]):
        assert [p["offset"] for p in axis["samples"]] == [-.1, -.05, 0., .05, .1]
        assert len(axis["strict_sign_brackets"]) == (2 if col == bracket_axis else 0)
        for bracket in axis["strict_sign_brackets"]:
            assert np.prod(bracket["heights"]) < 0
            assert bracket["heights"] == [evaluate.rows[i]["tau"] for i in bracket["point_indices"]]
        for sample in axis["samples"]:
            np.testing.assert_array_equal(sample["point"]["coordinate"],
                                          center+sample["offset"]*np.array(axis["direction"]))


@pytest.mark.parametrize("hessian", [np.eye(2), np.diag([-1., 0.]),
                                    [[-1., 1.], [0., 1.]]])
def test_axis_diagnostics_refuses_invalid_saddle_geometry(hessian):
    evaluate = HyperbolicEvaluator()
    with pytest.raises(ValueError, match="Hessian"):
        probe.axis_diagnostics(evaluate, [0., 0.], .006, hessian, config()["axis_offsets"])
    assert not evaluate.rows


@pytest.mark.parametrize("key,value", [("qualified", False), ("beta", .005),
                                      ("coordinate", [0., .001]),
                                      ("tau", float("nan")), ("delta", 0.),
                                      ("radius", 1.)])
def test_each_evaluation_must_pass_location_and_spectral_gates(key, value):
    original = HyperbolicEvaluator()
    def evaluate(coordinate, beta, cyclic=False):
        row = original(coordinate, beta, cyclic=cyclic)
        row[key] = value
        return row
    with pytest.raises(ValueError, match="unqualified|misplaced|invalid attracting"):
        probe.axis_diagnostics(evaluate, [0., 0.], .006,
                                np.diag([-.04, .02]), config()["axis_offsets"])


def test_radius_tilt_retains_finite_offsets_and_recovers_analytic_slopes():
    center = np.array([.2, -.3])
    evaluate = HyperbolicEvaluator(height=0., center=center)
    saved = []
    result = probe.radius_tilt(evaluate, center, .00645, [1., 0.],
                               config()["candidate_tilt_offsets"],
                               lambda row: saved.append(copy.deepcopy(row)))
    assert result["complete"] and saved[-1] == result
    assert [row["offset"] for row in result["samples"]] == config()["candidate_tilt_offsets"]
    assert [row["step"] for row in result["symmetric_slopes"]] == [.0001, .0003]
    for sample in result["samples"]:
        np.testing.assert_array_equal(sample["point"]["coordinate"], center+[sample["offset"], 0.])
    for slope in result["symmetric_slopes"]:
        assert slope["tau"] == pytest.approx(0., abs=1e-15)
        assert slope["delta"] == pytest.approx(1e-9, rel=1e-10)
        # At the stationary trace zero, d rho/dx=-Delta_x/(2 sqrt(-Delta)).
        assert slope["radius"] == pytest.approx(-5e-6, rel=1e-7)


def test_radius_tilt_preserves_completed_samples_on_callback_failure():
    evaluate = HyperbolicEvaluator(height=0., fail_after=3)
    saved = []
    with pytest.raises(TimeoutError, match="synthetic evaluation budget"):
        probe.radius_tilt(evaluate, [0., 0.], .00645, [1., 0.],
                           config()["candidate_tilt_offsets"],
                           lambda row: saved.append(copy.deepcopy(row)))
    assert not saved[-1]["complete"]
    assert len(saved[-1]["samples"]) == 2
    assert saved[-1]["symmetric_slopes"] == []
    assert [s["point"]["index"] for s in saved[-1]["samples"]] == [1, 2]


def test_radius_tilt_refuses_a_nonunit_direction_before_evaluation():
    evaluate = HyperbolicEvaluator()
    with pytest.raises(ValueError, match="unit tilt direction"):
        probe.radius_tilt(evaluate, [0., 0.], .00645, [2., 0.],
                           config()["candidate_tilt_offsets"], lambda row: None)
    assert not evaluate.rows


def analytic_stencil():
    return dict(hessian=np.diag([-.04, .02]).tolist(), gradient=[0., 0.])


def test_direct_slices_find_both_actual_branches_with_cyclic_and_radius_evidence():
    center = np.array([.23, -.43])
    evaluate = HyperbolicEvaluator(height=-5e-5, center=center)
    saved = []
    roots = probe.direct_slices(evaluate, center, .007, analytic_stencil(), config(),
                               lambda rows: saved.append(copy.deepcopy(rows)))
    assert len(roots) == 6
    assert [(r["transverse"], r["side"]) for r in roots] == [
        (offset, side) for offset in config()["slice_offsets"] for side in ("negative", "positive")]
    for root in roots:
        assert root["complete"]
        assert np.prod(root["bracket_values"]) < 0
        expected_y = np.sqrt((5e-5+.02*root["transverse"]**2)/.01)
        expected_y *= -1 if root["side"] == "negative" else 1
        np.testing.assert_allclose(np.array(root["point"]["coordinate"])-center,
                                   [root["transverse"], expected_y], rtol=0, atol=1e-7)
        assert abs(root["point"]["tau"]) < config()["root_trace_tolerance"]
        cyclic = root["cyclic_point"]
        assert [p["phase"] for p in cyclic["phases"]] == [0, 6, 12]
        np.testing.assert_array_equal(cyclic["coordinate"], root["point"]["coordinate"])
        normal = np.asarray(root["normal_direction"])
        assert np.linalg.norm(normal) == pytest.approx(1.)
        assert len(root["normal_offsets"]) == 2
        for sample in root["normal_offsets"]:
            point = sample["point"]
            np.testing.assert_array_equal(point["coordinate"],
                                          np.array(cyclic["coordinate"])+sample["offset"]*normal)
            assert sample["radius_difference_from_root"] == point["radius"]-cyclic["radius"]
    assert saved[-1] == roots


def test_direct_roots_use_measured_trace_instead_of_the_quadratic_prediction():
    evaluate = HyperbolicEvaluator(height=-5e-5, cubic=.02)
    roots = probe.direct_slices(evaluate, [0., 0.], .007, analytic_stencil(), config(), lambda rows: None)
    for root in roots:
        x, y = root["point"]["coordinate"]
        assert abs(-5e-5-.02*x*x+.01*y*y+.02*y**3) < 1e-10
        quadratic_y = np.sqrt((5e-5+.02*x*x)/.01)
        assert abs(abs(y)-quadratic_y) > 1e-4


def test_missing_direct_root_bracket_raises_without_a_default_root():
    evaluate = HyperbolicEvaluator(height=-.01)
    saved = []
    with pytest.raises(ValueError, match="no trace-zero bracket"):
        probe.direct_slices(evaluate, [0., 0.], .007, analytic_stencil(), config(), saved.append)
    assert saved == []


def test_direct_slices_require_negative_after_height():
    with pytest.raises(ValueError, match="not negative"):
        probe.direct_slices(HyperbolicEvaluator(height=1e-4), [0., 0.], .007,
                            analytic_stencil(), config(), lambda rows: None)


@pytest.mark.parametrize("fail_stage", ["cyclic", "second offset"])
def test_failed_followup_preserves_the_root_and_every_completed_offset(fail_stage):
    evaluate = HyperbolicEvaluator(height=-5e-5)
    saved = []
    def progress(rows):
        saved.append(copy.deepcopy(rows))
        root = rows[-1]
        if (fail_stage == "cyclic" and "cyclic_point" not in root
                or fail_stage == "second offset" and len(root["normal_offsets"]) == 1):
            evaluate.fail_after = len(evaluate.rows)
    with pytest.raises(TimeoutError, match="synthetic evaluation budget"):
        probe.direct_slices(evaluate, [0., 0.], .007, analytic_stencil(), config(), progress)
    assert len(saved[-1]) == 1
    retained = saved[-1][0]
    assert not retained["complete"] and abs(retained["point"]["tau"]) < 1e-10
    assert len(retained["normal_offsets"]) == (0 if fail_stage == "cyclic" else 1)
    assert ("cyclic_point" in retained) == (fail_stage == "second offset")


@pytest.mark.parametrize("change,match", [("phases", "three cyclic phases"),
                                          ("tau", "trace residual")])
def test_cyclic_root_must_requalify_the_saved_location(change, match):
    original = HyperbolicEvaluator(height=-5e-5)
    def evaluate(coordinate, beta, cyclic=False):
        row = original(coordinate, beta, cyclic=cyclic)
        if cyclic:
            if change == "phases":
                row["phases"] = row["phases"][:1]
            else:
                row["tau"] = 2e-10
        return row
    saved = []
    with pytest.raises(ValueError, match=match):
        probe.direct_slices(evaluate, [0., 0.], .007, analytic_stencil(), config(),
                            lambda rows: saved.append(copy.deepcopy(rows)))
    assert len(saved[-1]) == 1 and not saved[-1][0]["complete"]
