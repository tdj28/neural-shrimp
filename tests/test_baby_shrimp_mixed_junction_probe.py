"""Analytic scalar-jet fixtures only; no model, target run or GPU access."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import baby_shrimp_mixed_junction_probe as probe


def configuration():
    return {"allow_scientifically_partial_bsc5": False,
        "target": {"beta": .006, "coordinate": [0., 0.], "moving_center_beta": [0., 0.],
                   "references": {"positive_tau_side": [.2, 1.], "negative_tau_side": [1., 0.]}},
        "beta_offsets": [0., -3e-9, 3e-9], "stencil_steps": [.0003, .0001], "primary_step": .0003,
        "maximum_corrections": 2, "maximum_step": .0005, "maximum_total_displacement": .0005,
        "coordinate_tolerance": 2e-6, "trace_tolerance": 1e-12, "f_plus_tolerance": 1e-8, "f_minus_tolerance": 1e-8,
        "maximum_projection_displacement": 2e-6, "maximum_predictor_projection_displacement": .0005,
        "minimum_trace_gradient": 1e-7, "jacobian_step": 1e-6, "maximum_jacobian_condition": 1e8,
        "minimum_normalized_jacobian_singular_value": .1, "minimum_reference_overlap": .2,
        "noise_multiplier": 1., "signal_multiple": 10., "maximum_feature_uncertainty": 1e-5,
        "maximum_f_minus_total_uncertainty": 1e-5, "maximum_intersection_location_uncertainty": 5e-6,
        "maximum_jacobian_perturbation": .25, "orbit_alignment_tolerance": 1e-4,
        "field_noise_floor": {"tau": 1e-20, "delta": 1e-20},
        "jet_difference_limits": {name: {"value": 1e-8, "gradient": 1e-5, "hessian": 1e-3} for name in ("tau", "delta")}}


def synthetic_fields(q=(0., 0.), beta_shift=0.):
    """An actual analytic crease with two distinct smooth branch normals."""
    q = np.asarray(q)
    H = np.array([[.8, .1], [.1, .9]])
    T = np.array([[-.7, .2], [.2, 1.1]])
    gt0 = np.array([.01, .015])
    normals = [np.linalg.eigh(H+sign*T/2)[1][:, -1] for sign in (1, -1)]
    G0 = np.linalg.solve(normals, [-np.dot(normals[0], gt0)/2, np.dot(normals[1], gt0)/2])
    G0 = G0+beta_shift*np.array([1e4, -2e3])
    t = float(gt0 @ q+.5*q @ T @ q)
    gt = gt0+T @ q
    R = float(.01+G0 @ q+.5*q @ H @ q)
    G = G0+H @ q
    delta = t*t/4-R*R
    gd = .5*t*gt-2*R*G
    hd = .5*np.outer(gt, gt)+.5*t*T-2*np.outer(G, G)-2*R*H
    return {"tau": {"value": t, "gradient": gt.tolist(), "hessian": T.tolist()},
            "delta": {"value": delta, "gradient": gd.tolist(), "hessian": hd.tolist()}}, R+abs(t)/2


class AnalyticEvaluator:
    def __init__(self, *, fail_after=None, max_points=120):
        self.rows, self.cache, self.fail_after, self.max_points = [], {}, fail_after, max_points

    def __call__(self, coordinate, beta, cyclic=False):
        key = (*coordinate, beta)
        if key in self.cache and (not cyclic or len(self.cache[key]["phases"]) == 3):
            return self.cache[key]
        if self.fail_after is not None and len(self.rows) >= self.fail_after:
            raise RuntimeError("synthetic incomplete model attempt")
        if len(self.rows) >= self.max_points:
            raise TimeoutError("synthetic point budget")
        fields, radius = synthetic_fields(coordinate, beta-.006)
        values = {"tau": fields["tau"]["value"], "delta": fields["delta"]["value"], "radius": radius}
        p = {"index": len(self.rows), "qualified": True, "coordinate": np.asarray(coordinate).tolist(), "beta": beta,
             **values, "phases": [{"phase": s, **values} for s in ((0, 6, 12) if cyclic else (0,))]}
        self.rows.append(p); self.cache[key] = p
        return p


def test_explicit_projection_keeps_observed_trace_and_fields_unchanged():
    c = configuration(); fields, _ = synthetic_fields([2e-5, -1e-5]); before = copy.deepcopy(fields)
    result = probe.conditional_geometry(fields, c["target"]["references"], c)
    assert fields == before
    assert result["input_tau"] != 0
    assert result["projection"]["input_tau"] == fields["tau"]["value"]
    assert result["projection"]["enforced_tau"] == 0
    assert not result["projection"]["spatial_displacement_applied"]
    assert result["projection"]["predicted_displacement_norm"] > 0
    assert result["branches"][probe.SIDES[0]]["jet"]["hessian"] != result["branches"][probe.SIDES[1]]["jet"]["hessian"]


def test_true_analytic_junction_has_zero_features_and_strict_geometric_cusp():
    c = configuration(); fields, _ = synthetic_fields()
    geometry = probe.conditional_geometry(fields, c["target"]["references"], c)
    for branch in geometry["branches"].values():
        assert abs(branch["valley"]["feature"]) < 1e-16
    assert geometry["cusp"]["left_slope"] < 0 < geometry["cusp"]["right_slope"]
    correction = probe.newton_correction(fields, c["target"]["references"], c)
    assert correction["correction_norm"] < 1e-14
    assert correction["normalized_jacobian_singular_values"][-1] > .1


def test_joint_corrector_changes_coordinates_instead_of_fixed_target_slice():
    c = configuration(); fields, _ = synthetic_fields([3e-5, -2e-5])
    initial = probe.newton_correction(fields, c["target"]["references"], c)
    correction = np.asarray(initial["correction"])
    assert np.linalg.norm(correction) > 1e-5
    predicted = probe.local_equations(fields, correction, c["target"]["references"], c)
    assert np.linalg.norm(predicted[:2]) < .01*np.linalg.norm(initial["equations"])


def test_measured_stencil_uses_fresh_field_values_and_retains_nonzero_tau():
    evaluate = AnalyticEvaluator(); center = np.array([3e-5, -2e-5])
    row = probe.measured_jet(evaluate, center, .006, .0003)
    assert len(evaluate.rows) == 9
    assert row["fields"]["tau"]["value"] != 0
    fields, _ = synthetic_fields(center)
    np.testing.assert_allclose(row["fields"]["tau"]["gradient"], fields["tau"]["gradient"], atol=1e-14)
    np.testing.assert_allclose(row["fields"]["tau"]["hessian"], fields["tau"]["hessian"], atol=1e-12)
    assert len({i for ids in row["point_indices"] for i in ids}) == 9


@pytest.mark.parametrize("offset", [0., -3e-9, 3e-9])
def test_bounded_fresh_corrector_and_primary_scale_are_explicit(offset):
    c = configuration(); evaluate = AnalyticEvaluator(); snapshots = []
    result = probe.probe_case(evaluate, [2e-5, -1e-5], .006+offset, offset,
        c["target"]["references"], {"tau": 1e-20, "delta": 1e-20}, c,
        lambda row: snapshots.append(copy.deepcopy(row)))
    assert result["complete"] and result["status"] == "admitted"
    assert len(result["iterations"]) <= 3
    assert not np.array_equal(result["final_center"], result["predicted_center"])
    assert len(evaluate.rows) <= 40
    final = result["iterations"][-1]
    assert final["uncertainty"]["primary_step"] == .0003
    assert final["uncertainty"]["check_step"] == .0001
    assert final["admission"]["f_minus"]["value"] == final["coarse_correction"]["f_minus"]
    assert [p["phase"] for p in evaluate.rows[result["final_point_index"]]["phases"]] == [0, 6, 12]
    assert all(v for v in final["admission"]["checks"].values())


def valid_uncertainty_input():
    c = configuration(); fields, _ = synthetic_fields()
    primary = {"fields": fields, "step": .0003}
    check = {"fields": copy.deepcopy(fields), "step": .0001}
    geometry = probe.conditional_geometry(fields, c["target"]["references"], c)
    corrector = probe.newton_correction(fields, c["target"]["references"], c)
    return c, primary, check, geometry, corrector


def test_empirical_noise_is_amplified_and_location_transfer_is_not_omitted():
    c, primary, check, geom, corr = valid_uncertainty_input()
    uncertainty = probe.empirical_uncertainty(primary, check, geom, geom, corr, corr,
        {"tau": 1e-16, "delta": 1e-18}, {"tau": 0., "delta": 0.}, c)
    assert uncertainty["field_error_balls"]["tau"]["gradient"] == pytest.approx(np.sqrt(2)*1e-16/.0003)
    assert uncertainty["field_error_balls"]["tau"]["hessian"] == pytest.approx(5e-16/.0003**2)
    assert uncertainty["intersection"]["location_error_norm"] > 0
    assert uncertainty["intersection"]["f_minus_location_transfer"] > 0
    assert uncertainty["intersection"]["f_minus_total_error"] > uncertainty["branch_envelopes"][probe.SIDES[1]]["feature_error"]


def test_differing_center_values_increase_field_error_ball_not_only_discrepancy_gate():
    c, primary, check, geom, corr = valid_uncertainty_input()
    check["fields"]["tau"]["value"] = 1e-15
    spans = {"tau": 1e-16, "delta": 1e-18}
    result = probe.empirical_uncertainty(primary, check, geom, geom, corr, corr, spans, spans, c)
    assert result["field_error_balls"]["tau"]["value"] == pytest.approx(1.1e-15)


def test_actual_residuals_increase_intersection_location_error():
    c, primary, check, geom, corr = valid_uncertainty_input()
    spans = {"tau": 1e-18, "delta": 1e-20}
    baseline = probe.empirical_uncertainty(primary, check, geom, geom, corr, corr, spans, spans, c)
    shifted = copy.deepcopy(corr); shifted["equations"] = [1e-12, 1e-8]
    result = probe.empirical_uncertainty(primary, check, geom, geom, shifted, shifted, spans, spans, c)
    assert result["intersection"]["location_error_norm"] > baseline["intersection"]["location_error_norm"]
    assert result["intersection"]["f_minus_total_error"] > baseline["intersection"]["f_minus_total_error"]


def test_rotated_inverse_cannot_hide_transfer_when_nominal_direction_is_zero():
    c = configuration()
    primary = {"equation_scales": [1., 1.], "jacobian": [[1., 0.], [0., 1.]],
               "jacobian_half_step": [[1., 0.], [0., 1.]], "f_minus_gradient_half_step": [0., 1.],
               "equations": [0., 0.], "f_minus_gradient": [0., 1.]}
    check = {**primary, "jacobian": [[1., 0.], [.1, 1.]], "jacobian_half_step": [[1., 0.], [.1, 1.]]}
    result = probe.intersection_transfer(primary, check, [1., 0.], c)
    exact_transfer = abs((np.linalg.inv(check["jacobian"]) @ [1., 0.])[1])
    assert exact_transfer == pytest.approx(.1)
    assert result["f_minus_nominal_transfer"] == 0
    assert result["inverse_rotation_transfer"] > exact_transfer
    assert result["f_minus_location_transfer"] > exact_transfer
    assert result["nominal_location_box_components"] == [1., 0.]
    assert result["location_error_norm"] > 1


@pytest.mark.parametrize("offset", [-3e-9, 3e-9])
def test_outward_predictor_keeps_fresh_central_correction_and_parent_binding(offset):
    target = configuration()["target"]
    target["moving_center_beta"] = [1e4, -2e4]
    central = {"complete": True, "status": "admitted", "beta_offset": 0., "beta": target["beta"],
               "final_center": [.0001, -.0002], "final_point_index": 42}
    predicted, parent = probe.case_predictor(target, offset, central)
    np.testing.assert_array_equal(predicted, np.asarray(central["final_center"])+offset*np.asarray(target["moving_center_beta"]))
    assert not np.array_equal(predicted, np.asarray(target["coordinate"])+offset*np.asarray(target["moving_center_beta"]))
    assert parent["point_index"] == 42 and parent["coordinate"] == central["final_center"]
    central["complete"] = False
    with pytest.raises(ValueError, match="admitted central"):
        probe.case_predictor(target, offset, central)


@pytest.mark.parametrize("change", ["reversed primary", "negative determinant ball", "regular gradient", "normal gap", "jacobian disagreement"])
def test_uncertainty_refuses_unresolved_assumptions(change):
    c, primary, check, geom, corr = valid_uncertainty_input()
    spans = {"tau": 1e-20, "delta": 1e-20}
    other = copy.deepcopy(corr)
    if change == "reversed primary":
        primary, check = check, primary
    elif change == "negative determinant ball":
        spans["delta"] = .001
    elif change == "regular gradient":
        spans["tau"] = .01
    elif change == "normal gap":
        spans["tau"] = 1e-7
    else:
        other["jacobian"][0][0] *= 2
    with pytest.raises(ValueError):
        probe.empirical_uncertainty(primary, check, geom, geom, corr, other, spans, spans, c)


@pytest.mark.parametrize("change", ["stationary crease", "huge projection", "positive delta", "orthogonal reference"])
def test_conditional_limits_fail_loudly_without_regular_admission(change):
    c = configuration(); fields, _ = synthetic_fields(); refs = c["target"]["references"]
    if change == "stationary crease":
        fields["tau"]["gradient"] = [0., 0.]
    elif change == "huge projection":
        fields["tau"]["value"] = .1
    elif change == "positive delta":
        fields["delta"]["value"] = .01
    else:
        original = probe.conditional_geometry(fields, refs, c)["branches"][probe.SIDES[0]]["valley"]["normal"]
        refs[probe.SIDES[0]] = [-original[1], original[0]]
    with pytest.raises(ValueError):
        probe.conditional_geometry(fields, refs, c)


@pytest.mark.parametrize("failure", ["incomplete", "point budget", "trust", "precision", "correction budget"])
def test_failure_keeps_partial_case_and_never_invents_a_target(failure):
    c = configuration(); snapshots = []
    evaluate = AnalyticEvaluator(fail_after=4 if failure == "incomplete" else None,
                                 max_points=12 if failure == "point budget" else 120)
    if failure == "trust":
        c["maximum_step"] = 1e-10
    elif failure == "precision":
        c["maximum_f_minus_total_uncertainty"] = 1e-30
    elif failure == "correction budget":
        c["maximum_corrections"] = 0
    with pytest.raises((ValueError, TimeoutError, RuntimeError)):
        probe.probe_case(evaluate, [2e-5, -1e-5], .006, 0., c["target"]["references"],
                         {"tau": 1e-20, "delta": 1e-20}, c, lambda r: snapshots.append(copy.deepcopy(r)))
    assert snapshots and not snapshots[-1]["complete"]
    assert "final_center" not in snapshots[-1]
    assert evaluate.rows


def test_f_minus_sign_needs_total_envelope_and_capped_uncertainty():
    c, primary, check, geom, corr = valid_uncertainty_input()
    spans = {"tau": 1e-20, "delta": 1e-20}
    unc = probe.empirical_uncertainty(primary, check, geom, geom, corr, corr, spans, spans, c)
    cyclic = {"phases": [{"tau": 0.}]*3}
    corr["f_minus"] = 2e-6
    unc["intersection"]["f_minus_total_error"] = 3e-7
    assert probe.admission(geom, corr, unc, cyclic, c)["f_minus"]["status"] == "unresolved_sign"
    unc["intersection"]["f_minus_total_error"] = 1e-7
    assert probe.admission(geom, corr, unc, cyclic, c)["f_minus"]["status"] == "resolved_positive"
    c["maximum_f_minus_total_uncertainty"] = 5e-8
    rejected = probe.admission(geom, corr, unc, cyclic, c)
    assert not rejected["passed"]
    assert rejected["f_minus"]["status"] == "not_admitted"
    assert rejected["f_minus"]["ungated_sign_diagnostic"] == "resolved_positive"


@pytest.mark.parametrize("key,value", [("primary_step", .0001), ("maximum_corrections", 3),
    ("maximum_step", .001), ("trace_tolerance", 1e-10), ("coordinate_tolerance", 1e-4),
    ("allow_scientifically_partial_bsc5", 1), ("signal_multiple", 0), ("beta_offsets", [0., -1e-9, 3e-9])])
def test_declared_plan_refuses_changed_design_or_loosened_caps(key, value):
    c = configuration()
    c.update(minimum_normalized_jacobian_singular_value=.2, maximum_f_minus_total_uncertainty=5e-8)
    plan = {"id": "BSC6-mixed-junction-probe", "period": 18, "maximum_evaluations": 120,
            "maximum_abs_beta": .007, "mixed_junction": c}
    assert probe.validate_plan(plan) is c
    c[key] = value
    with pytest.raises(ValueError):
        probe.validate_plan(plan)


def prior_fixture(tmp_path, monkeypatch, *, partial=False):
    import baby_shrimp_contraction_minima_finish as audit
    import baby_shrimp_memory_orbit_alignment as binding
    root, data = tmp_path/"repo", tmp_path/"data"
    root.mkdir(); data.mkdir()
    monkeypatch.setattr(probe, "ROOT", root)
    monkeypatch.setattr(probe, "DATA", data)
    sources = {}
    for source in audit.SOURCES:
        p = root/source; p.parent.mkdir(parents=True, exist_ok=True); p.write_text("synthetic source: "+source)
        sources[source] = hashlib.sha256(p.read_bytes()).hexdigest()
    monkeypatch.setattr(binding, "git_bytes", lambda commit, source: (root/source).read_bytes())
    run = data/"explore/bsc5"; directory = run/"point-0001"; directory.mkdir(parents=True)
    point = {"index": 1, "qualified": True, "coordinate": [0., 0.], "beta": .006,
             "phases": [{"phase": phase} for phase in (0, 6, 12)]}
    (directory/"point.json").write_text(json.dumps(point))
    np.save(directory/"points.npy", np.ones((18, 2)))
    for phase in (0, 6, 12):
        np.savez(directory/f"phase-{phase}.npz", q=np.eye(2))
    metadata = {"id": "BSC5-contraction-minima", "model_revision": "test-revision", "period": 18,
                "start": [.073, -.15], "coordinate_scales": [2.5e-6, 7.5e-6],
                "anchor_directory": "synthetic-anchor", "anchor_hashes": {}, "intervention": "synthetic"}
    sections = [{"arm_id": "test-arm", "longitudinal": .002,
                 "search": {"status": "budget_exhausted" if partial else "converged", "complete": not partial}}]
    receipt = {"id": "BSC5-contraction-minima-offline-audit", "audit_finished": True, "complete": not partial,
        "center_qualified": True, "scientific_metadata": metadata, "analysis_commit": "a"*40,
        "analysis_source_sha256": sources, "lifecycle": {"python_finalization": True, "controller": None},
        "raw_directory": str(run), "baseline": {"passed": True}, "seed_replay": {"passed": True},
        "expanded_spectrum": {"complete": True, "point_index": 1}, "center_point_index": 1,
        "local_jets": [{"test": "coarse"}, {"test": "fine"}], "sections": sections,
        "untested_or_incomplete_stages": ["later sections unresolved"] if partial else [],
        "generated_utc": "synthetic-time", "threads": 1}
    path = data/"bsc5-receipt.json"
    def refresh():
        path.write_text(json.dumps(receipt))
        return {"path": path.name, "sha256": probe.digest(path), "bytes": path.stat().st_size}
    plan = {**metadata, "id": "BSC6-mixed-junction-probe", "mixed_junction": configuration(), "bsc5_audit": refresh()}
    plan["mixed_junction"]["allow_scientifically_partial_bsc5"] = partial
    plan["mixed_junction"]["partial_bsc5_review"] = {
        "receipt_sha256": plan["bsc5_audit"]["sha256"], "retained_unresolved_stages": receipt["untested_or_incomplete_stages"],
        "section_statuses": [{"arm_id": s["arm_id"], "longitudinal": s["longitudinal"],
                              "status": s["search"]["status"], "complete": s["search"]["complete"]} for s in sections]}
    calls = []
    def replay(path, **kwargs):
        calls.append((path, kwargs))
        return {k: copy.deepcopy(v) for k, v in receipt.items() if k not in ("analysis_commit", "analysis_source_sha256", "threads")}
    monkeypatch.setattr(audit, "run_audit", replay)
    return plan, receipt, refresh, calls


@pytest.mark.parametrize("partial", [False, True])
def test_bound_prior_replays_exact_auditor_and_preserves_partial_science(tmp_path, monkeypatch, partial):
    plan, receipt, _, calls = prior_fixture(tmp_path, monkeypatch, partial=partial)
    bound = probe.bind_prior(plan)
    assert len(calls) == 1 and calls[0][1]["allow_partial"] is partial
    assert bound["provenance"]["scientific_complete"] is not partial
    assert bound["provenance"]["retained_unresolved_stages"] == receipt["untested_or_incomplete_stages"]
    assert bound["seed"]["point"]["index"] == 1
    assert set(bound["seed"]["bases"]) == {0, 6, 12}


@pytest.mark.parametrize("change", ["hash", "unfinished audit", "unqualified center", "changed source", "baseline", "seed", "expanded", "jets", "silent partial", "missing review", "omitted status"])
def test_prior_admission_refuses_missing_support_or_silent_partial_promotion(tmp_path, monkeypatch, change):
    plan, receipt, refresh, calls = prior_fixture(tmp_path, monkeypatch, partial=change in ("silent partial", "missing review", "omitted status"))
    if change == "hash":
        plan["bsc5_audit"]["sha256"] = "0"*64
    elif change == "changed source":
        source = next(iter(receipt["analysis_source_sha256"]))
        (probe.ROOT/source).write_text("changed")
    elif change == "silent partial":
        plan["mixed_junction"]["allow_scientifically_partial_bsc5"] = False
    elif change == "missing review":
        plan["mixed_junction"]["partial_bsc5_review"]["receipt_sha256"] = "0"*64
    elif change == "omitted status":
        plan["mixed_junction"]["partial_bsc5_review"]["section_statuses"] = []
    else:
        if change == "unfinished audit":
            receipt["audit_finished"] = False
        elif change == "unqualified center":
            receipt["center_qualified"] = False
        elif change in ("baseline", "seed"):
            receipt["baseline" if change == "baseline" else "seed_replay"]["passed"] = False
        elif change == "expanded":
            receipt["expanded_spectrum"]["complete"] = False
        else:
            receipt["local_jets"] = receipt["local_jets"][:1]
        plan["bsc5_audit"] = refresh()
    with pytest.raises(ValueError):
        probe.bind_prior(plan)


def test_prior_replay_mismatch_is_not_replaced_with_saved_result(tmp_path, monkeypatch):
    import baby_shrimp_contraction_minima_finish as audit
    plan, _, _, _ = prior_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(audit, "run_audit", lambda *a, **kw: {"different": "scientific record"})
    with pytest.raises(ValueError, match="replay differs"):
        probe.bind_prior(plan)


@pytest.mark.parametrize("shift,closed", [(0, True), (1, True), (0, False)])
def test_every_actual_point_requires_saved_array_closure_and_full_phase_alignment(tmp_path, monkeypatch, shift, closed):
    import baby_shrimp_memory_finish as audit
    plan = {"mixed_junction": configuration()}
    reference = np.column_stack([2+np.arange(18)/18, np.sin(np.arange(18))])
    for index, orbit in ((0, reference), (1, np.roll(reference, shift, axis=0))):
        p = tmp_path/f"point-{index:04d}"; p.mkdir(); np.save(p/"points.npy", orbit)
    point = {"index": 1, "qualified": True, "coordinate": [0., 0.], "beta": .006,
             "tau": 1e-13, "delta": -1e-6, "radius": .001, "phases": [{"phase": 0}]}
    class Evaluator:
        out = tmp_path
        def __call__(self, *args, **kwargs):
            return point
    monkeypatch.setattr(audit, "audit_point", lambda *args: (point, {"closure_recomputed_from_saved_endpoint": closed}))
    records = []
    evaluate = probe.AdmittedEvaluator(Evaluator(), {"index": 0}, plan, records, lambda: None)
    if shift == 0 and closed:
        assert evaluate([0., 0.], .006) is point
        assert evaluate([0., 0.], .006) is point
        assert len(records) == 1 and records[0]["passed"]
    else:
        with pytest.raises(ValueError, match="alignment|closure"):
            evaluate([0., 0.], .006)
        assert records[0]["status"] == "failed"
        assert "all_alignment_maximum_relative_changes" in records[0]
