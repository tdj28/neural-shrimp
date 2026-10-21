"""Independent saved-evidence audits, with synthetic orbits/operators only."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import baby_shrimp_memory_finish as finish
from trace_saddle_continuation import ContinuationError, TraceSaddleContinuation


def write_json(path, row):
    path.write_text(json.dumps(row, allow_nan=False))


def synthetic_point(tmp_path, dimension=8):
    directory = tmp_path / "point-0000"
    directory.mkdir()
    period = 18
    angle = np.arange(period) * 2 * np.pi / period
    points = np.zeros((period, dimension))
    points[:, 0], points[:, 1] = np.cos(angle), np.sin(angle)
    np.save(directory / "points.npy", points)
    np.save(directory / "return-state.npy", points[0])
    lower = np.linalg.norm(points[1:] - points[0], axis=1)
    values = np.array([1e-3, -1e-4, 1e-12, -1e-14, 1e-16, -1e-18], dtype=np.complex128)
    q = np.eye(dimension, 6)
    w = q * values.real
    vectors = q[:, :2]
    direct = vectors * values[:2].real
    phase = dict(phase=0, multipliers=np.column_stack([values.real, values.imag]).tolist(),
                 relative_residuals=[0., 0.], tau=float(sum(values[:2]).real),
                 delta=float(np.prod(values[:2]).real), radius=float(max(abs(values))),
                 history=[dict(iteration=i, multipliers=np.column_stack([values.real, values.imag]).tolist(),
                               relative_residuals=[0.] * 6) for i in (0, 1)])
    row = dict(index=0, qualified=True, coordinate=[0., 0.], beta=0., alpha=.1, log10_temperature=-.2,
               closure_history=[.1, 0.], tokens=[271, 785, 3405, 271, 785, 13885] * 3,
               primitive=dict(closed=True, no_lower_period=True, all_returns=[*lower, 0.],
                              minimum_lower_return=float(min(lower)), primitive_separation_tolerance=1e-8),
               phases=[phase], tau=phase["tau"], delta=phase["delta"], radius=phase["radius"])
    write_json(directory / "point.json", row)
    np.savez(directory / "phase-0.npz", values=values, q=q, w=w, vectors=vectors,
             direct=direct, relative_residuals=np.zeros(2))
    plan = dict(start=[.1, -.2], coordinate_scales=[1e-5, 2e-5], maximum_abs_beta=.01,
                maximum_scaled_coordinate=1000, period=period)
    return directory, plan


def test_recomputes_orbit_and_spectral_gates_and_marks_missing_endpoint(tmp_path):
    directory, plan = synthetic_point(tmp_path)
    _, summary = finish.audit_point(directory, plan, dimension=8)
    assert summary["qualified"] and summary["closure_recomputed_from_saved_endpoint"]
    assert summary["maximum_relative_residual"] == 0
    (directory / "return-state.npy").unlink()
    _, summary = finish.audit_point(directory, plan, dimension=8)
    assert not summary["closure_recomputed_from_saved_endpoint"]


@pytest.mark.parametrize("array,change,match", [
    ("q", lambda a: a * 2, "orthonormal"),
    ("w", lambda a: a * 1.1, "Ritz values"),
    ("direct", lambda a: a + 1e-7, "residual gate"),
])
def test_rejects_corrupted_spectral_evidence(tmp_path, array, change, match):
    directory, plan = synthetic_point(tmp_path)
    with np.load(directory / "phase-0.npz") as archive:
        values = {key: archive[key] for key in archive.files}
    values[array] = change(values[array])
    np.savez(directory / "phase-0.npz", **values)
    with pytest.raises(ValueError, match=match):
        finish.audit_point(directory, plan, dimension=8)


def test_rejects_changed_saved_orbit_and_return_state(tmp_path):
    directory, plan = synthetic_point(tmp_path)
    points = np.load(directory / "points.npy")
    points[3] = points[0]
    np.save(directory / "points.npy", points)
    with pytest.raises(ValueError, match="lower-lag returns"):
        finish.audit_point(directory, plan, dimension=8)
    other = tmp_path / "other"
    other.mkdir()
    directory, plan = synthetic_point(other)
    np.save(directory / "return-state.npy", np.ones(8))
    with pytest.raises(ValueError, match="return-state closure"):
        finish.audit_point(directory, plan, dimension=8)


def saddle_evidence():
    points, events = {}, []
    def evaluate(center, beta):
        x, y = center - [0.4 * beta, -0.2 * beta]
        row = {"index": 100 + len(points), "coordinate": center.tolist(), "beta": beta,
               "qualified": True, "tau": float(beta - .3 - 2*x*x + .5*x*y + 3*y*y),
               "delta": -1e-8, "radius": .5, "phases": [{"phase": 0}]}
        points[row["index"]] = row
        return row
    engine = TraceSaddleContinuation(evaluate, record=events.append, correction_tolerance=1e-10)
    saddle = engine.correct(.2, [0, 0])
    mapping = {e["index"]: points[e["callback"]["index"]]
               for e in events if e["stage"] == "evaluation_complete"}
    cyclic = copy.deepcopy(mapping[saddle["point_index"]])
    cyclic.update(index=1000, phases=[{"phase": shift, "tau": cyclic["tau"]} for shift in (0, 6, 12)])
    points[1000] = cyclic
    saddle["cyclic_endpoint"] = cyclic
    return saddle, mapping, points


def test_recomputes_nine_point_saddle_and_cyclic_center_link():
    saddle, mapping, points = saddle_evidence()
    summary = finish.audit_saddle(saddle, mapping, points, [.1, .03], 1e-10)
    assert summary["kind"] == "continued saddle"
    np.testing.assert_allclose(summary["hessian"], [[-4, .5], [.5, 6]], atol=1e-10)
    assert summary["tau"] == pytest.approx(-.1)
    assert summary["height_sensitivity"]["height_sign_consistent_across_estimates"]
    assert len(summary["stencil_diagnostics"]) == 2
    assert "not an uncertainty bound" in summary["height_sensitivity"]["scope"]
    altered = copy.deepcopy(saddle)
    altered["history"][-1]["hessian"][0][0] += 1e-3
    with pytest.raises(ValueError, match="stencil differs: hessian"):
        finish.audit_saddle(altered, mapping, points, [.1, .03], 1e-10)
    altered = copy.deepcopy(saddle)
    altered["cyclic_endpoint"]["coordinate"][0] += .1
    with pytest.raises(ValueError, match="bound cyclic endpoint"):
        finish.audit_saddle(altered, mapping, points, [.1, .03], 1e-10)


def test_stencil_cannot_use_a_wrong_parameter_point():
    saddle, mapping, points = saddle_evidence()
    wrong = copy.deepcopy(mapping)
    index = saddle["history"][0]["point_indices"][0][0]
    wrong[index]["beta"] += .01
    with pytest.raises(ValueError, match="wrong parameter"):
        finish.audit_saddle(saddle, wrong, points, [.1, .03], 1e-10)


def corrector_attempt_with_rejected_move(failed_beta=.3):
    """One valid saddle, followed by a valid stencil proposing an unsafe move."""
    points, events, artifacts = {}, [], []
    def evaluate(center, beta):
        x, y = center - [.4*beta, -.2*beta]
        row = dict(index=len(points), coordinate=center.tolist(), beta=beta, qualified=True,
                   tau=float(beta-.3-2*x*x+.5*x*y+3*y*y), delta=-1e-8, radius=.5,
                   phases=[dict(phase=0)])
        points[row["index"]] = row
        return row
    def record(row):
        if row["stage"] == "saddle_complete":
            cyclic = copy.deepcopy(evaluate(np.asarray(row["center"]), row["parameter"]))
            cyclic["phases"] = [dict(phase=shift, tau=cyclic["tau"]) for shift in (0, 6, 12)]
            points[cyclic["index"]] = cyclic
            row["cyclic_endpoint"] = cyclic
            artifacts.append(row)
        events.append(row)
    engine = TraceSaddleContinuation(evaluate, correction_tolerance=1e-10, record=record)
    accepted = engine.correct(.2, [0., 0.])
    with pytest.raises(ContinuationError, match="trust radius"):
        engine.correct(failed_beta, [-3., 0.])
    mapping = {event["index"]: points[event["callback"]["index"]]
               for event in events if event["stage"] == "evaluation_complete"}
    rejected = [event for event in events if event["stage"] == "saddle_iteration"][-1]
    return accepted, artifacts[0], rejected, points, mapping, events


def test_rejected_trust_radius_stencil_is_evidence_but_never_an_accepted_saddle():
    accepted, artifact, rejected, points, mapping, _ = corrector_attempt_with_rejected_move(failed_beta=.2)
    checked = finish.audit_stencil(rejected, mapping, 1e-10)
    assert not checked["trust_radius_passed"] and not checked["converged"]
    assert checked["correction_norm"] > 3
    valid = finish.audit_saddle(artifact, mapping, points, [.1, .03], 1e-10)
    assert valid["trust_radius_passed"] and valid["tau"] == accepted["tau"]
    # Even a later, fully converged two-scale ending cannot legitimize a
    # rejected step earlier in the claimed accepted history.
    forged = copy.deepcopy(artifact)
    forged["history"].insert(0, rejected)
    with pytest.raises(ValueError, match="accepted saddle contains a rejected trust-radius step"):
        finish.audit_saddle(forged, mapping, points, [.1, .03], 1e-10)
    altered = copy.deepcopy(rejected)
    altered["correction_norm"] = 1.9
    with pytest.raises(ValueError, match="recomputed stencil differs: correction_norm"):
        finish.audit_stencil(altered, mapping, 1e-10)


@pytest.mark.parametrize("claimed_complete", [False, True])
def test_partial_audit_retains_rejected_corrector_without_accepting_it(tmp_path, monkeypatch, claimed_complete):
    accepted, artifact, rejected, points, _, events = corrector_attempt_with_rejected_move()
    run, out = tmp_path / "run", tmp_path / "audit"
    run.mkdir()
    out.mkdir()
    for index, point in points.items():
        directory = run / f"point-{index:04d}"
        directory.mkdir()
        write_json(directory / "point.json", point)
    write_json(run / "saddle-000.json", artifact)
    (run / "progress.jsonl").write_text("\n".join(json.dumps(dict(stage="continuation", event=e)) for e in events)+"\n")
    protocol = dict(commit="a"*40, plan={}, plan_sha256="b"*64,
                    args=dict(mode="continue", stencils=[.1, .03], correction_tolerance=1e-10,
                              betas=[.2] if claimed_complete else [.2, .3]))
    write_json(run / "protocol.json", protocol)
    write_json(run / "runtime.json", dict(norm_sensitivity=1e-12, rotary_float64=True, threads=8))
    write_json(run / "result.json", dict(complete=claimed_complete, points=[accepted]))
    # Orbit/operator audits are separately tested above. This fixture isolates
    # the real callback/stencil replay and receipt semantics; no failure.json
    # is supplied, so a rejected move alone must veto a complete label.
    monkeypatch.setattr(finish, "bind_protocol", lambda protocol, root, data: tmp_path)
    monkeypatch.setattr(finish, "audit_point", lambda directory, plan: (
        finish.read(directory / "point.json"), {"index": finish.read(directory / "point.json")["index"], "qualified": True}))
    monkeypatch.setattr(finish, "audit_baseline", lambda run, raw, anchor: {"passed": True})
    monkeypatch.setattr(finish, "plot_centers", lambda centers, out: [])
    receipt = finish.audit_run(run, out)
    assert not receipt["complete"] and not receipt["unqualified_or_incomplete_points"]
    assert [row["parameter"] for row in receipt["accepted_centers"]] == [.2]
    assert len(receipt["rejected_corrector_steps"]) == 1
    row = receipt["rejected_corrector_steps"][0]
    assert row["parameter"] == .3 and row["center"] == [-3., 0.]
    assert row["correction_norm"] == pytest.approx(rejected["correction_norm"])
    assert row["correction_norm"] > row["trust_radius"] == 2.
    assert row["accepted"] is False and "trust radius" in row["reason"]
    assert "progress.jsonl" in {entry["path"] for entry in receipt["manifest"]}
    assert isinstance(finish.accepted_centers(run, protocol, points, dict(points=[accepted])), tuple)


def test_execution_sources_and_plan_are_bound_to_git_bytes(tmp_path, monkeypatch):
    plan = {"period": 18, "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            "anchor_directory": "anchor", "anchor_hashes": {}}
    (tmp_path / "anchor").mkdir()
    blobs = {p: p.encode() for p in finish.BASE_SOURCES}
    blobs["plan.json"] = json.dumps(plan).encode()
    protocol = dict(commit="a" * 40, plan=plan, args=dict(mode="pilot", plan="plan.json"),
                    source_sha256={p: hashlib.sha256(blobs[p]).hexdigest() for p in finish.BASE_SOURCES},
                    plan_sha256=hashlib.sha256(blobs["plan.json"]).hexdigest())
    monkeypatch.setattr(finish, "git_bytes", lambda commit, path, root: blobs[path])
    assert finish.bind_protocol(protocol, data=tmp_path) == tmp_path / "anchor"
    changed = copy.deepcopy(protocol)
    changed["source_sha256"][finish.BASE_SOURCES[0]] = "0" * 64
    with pytest.raises(ValueError, match="source hash"):
        finish.bind_protocol(changed, data=tmp_path)
    changed = copy.deepcopy(protocol)
    changed["plan"]["period"] = 36
    with pytest.raises(ValueError, match="plan hash or contents"):
        finish.bind_protocol(changed, data=tmp_path)


def test_incomplete_point_remains_manifested_and_prevents_complete_label(tmp_path, monkeypatch):
    run, out, anchor = (tmp_path / name for name in ("run", "audit", "anchor"))
    for directory in (run, out, anchor):
        directory.mkdir()
    directory, plan = synthetic_point(run, dimension=896)
    row = finish.read(directory / "point.json")
    with np.load(directory / "phase-0.npz") as archive:
        arrays = {key: archive[key] for key in archive.files}
    row["phases"] = [{**copy.deepcopy(row["phases"][0]), "phase": shift} for shift in (0, 6, 12)]
    row["cyclic_pair_absolute_discrepancy"] = 0.
    for shift in (6, 12):
        np.savez(directory / f"phase-{shift}.npz", **arrays)
    write_json(directory / "point.json", row)
    matrix = np.zeros((896, 896))
    np.fill_diagonal(matrix, np.r_[arrays["values"].real, np.zeros(890)])
    for shift in (0, 6, 12):
        np.savez(anchor / f"phase-{shift}.npz", matrix=matrix, multipliers=arrays["values"])
    write_json(anchor / "point.json", dict(tau=row["tau"], spectral_radius=row["radius"]))
    directions = np.eye(896, 3)
    action = matrix @ directions
    np.savez(run / "baseline-operator-check.npz", directions=directions, direct=action, dense=action, errors=np.zeros(3))
    baseline = dict(point=row, passed=True, composed_jvp_relative_errors=[0.] * 3,
                    tau_error_relative_to_radius=0., cyclic_dense_comparisons=[
                        dict(phase=shift, pair_relative_error=0., determinant_relative_error=0.) for shift in (0, 6, 12)])
    write_json(run / "baseline.json", baseline)
    write_json(run / "protocol.json", dict(commit="a" * 40, plan=plan, plan_sha256="b" * 64,
                                            args=dict(mode="pilot", betas=[0.])))
    write_json(run / "runtime.json", dict(norm_sensitivity=1e-12, rotary_float64=True, threads=8))
    write_json(run / "result.json", dict(complete=True, rows=[row]))
    monkeypatch.setattr(finish, "bind_protocol", lambda protocol, root, data: anchor)
    complete = finish.audit_run(run, out)
    assert complete["complete"] and complete["figures"] == []
    incomplete = run / "point-0001"
    incomplete.mkdir()
    write_json(incomplete / "point.json", dict(index=1, qualified=False))
    partial = finish.audit_run(run, out)
    assert not partial["complete"]
    assert len(partial["qualified_points"]) == 1
    assert partial["unqualified_or_incomplete_points"][0]["index"] == 1
    assert "point-0001/point.json" in {row["path"] for row in partial["manifest"]}


def diagnostic_stencil(center, step, tau, gradient, hessian, parameter=0.):
    return dict(center=list(center), stencil_step=step, tau=tau, parameter=parameter,
                **finish.stationary_height_diagnostics(tau, gradient, hessian))


def test_saddle_signed_height_correction_cannot_hide_curvature_cancellation():
    row = finish.stationary_height_diagnostics(1e-12, [2e-3, 2e-3], [[-2, 0], [0, 2]])
    assert row["quadratic_height_shift"] == pytest.approx(0, abs=1e-20)
    assert row["absolute_curvature_height_scale"] == pytest.approx(2e-6)
    sensitivity = finish.height_sensitivity([
        dict(row, tau=1e-12, center=[0., 0.], parameter=0., stencil_step=.01)], 0.)
    assert sensitivity["height_sign_consistent_across_estimates"]
    assert sensitivity["height_to_sensitivity_ratio"] == pytest.approx(5e-7)
    assert "sign not certified" in sensitivity["status"]


def test_different_centers_are_quadratically_transported_not_richardson_extrapolated():
    hessian = np.array([[-2., .4], [.4, 3.]])
    height = .017
    def row(center, step):
        center = np.array(center)
        return diagnostic_stencil(center, step, height + .5 * center @ hessian @ center,
                                  hessian @ center, hessian)
    coarse, fine = row([.1, .2], .1), row([.003, -.001], .03)
    comparison = finish.compare_stencil_scales(coarse, fine)
    assert comparison["raw_height_difference"] != pytest.approx(0)
    assert comparison["trace_transport_residual"] == pytest.approx(0, abs=1e-16)
    np.testing.assert_allclose(comparison["gradient_transport_residual"], 0., atol=1e-16)
    assert comparison["quadratic_corrected_height_difference"] == pytest.approx(0, abs=1e-16)
    assert not comparison["richardson"]["applied"]
    assert "centers differ" in comparison["richardson"]["reason"]


def test_matched_center_richardson_removes_cubic_gradient_bias_but_is_not_a_bound():
    # tau=.02-x²+y²+.7x³: at the origin, centered differences have g_x=.7h².
    coarse = diagnostic_stencil([0., 0.], .1, .02, [.7 * .1**2, 0.], [[-2, 0], [0, 2]])
    fine = diagnostic_stencil([0., 0.], .03, .02, [.7 * .03**2, 0.], [[-2, 0], [0, 2]])
    comparison = finish.compare_stencil_scales(coarse, fine)
    richardson = comparison["richardson"]
    assert richardson["applied"] and richardson["saddle_hessian_retained"]
    np.testing.assert_allclose(richardson["gradient"], [0., 0.], atol=1e-16)
    assert richardson["quadratic_corrected_height"] == pytest.approx(.02)
    assert "two scales do not establish" in richardson["scope"]


def test_height_changes_use_corrected_heights_and_retain_unavailable_legacy_sensitivity():
    centers = []
    # The saddle translates but has constant height; residual locations differ.
    for beta, offset in [(0., [.1, .2]), (1e-7, [.05, -.03])]:
        H, e = np.diag([-2., 3.]), np.asarray(offset)
        tau = .02 + .5 * e @ H @ e
        stencil = diagnostic_stencil(e, .01, tau, H @ e, H, beta)
        centers.append(dict(parameter=beta, center=(e + beta * np.array([1e4, 2e4])).tolist(),
                            tau=tau, kind="continued saddle", gradient=stencil["gradient"], correction=stencil["correction"],
                            height_sensitivity=finish.height_sensitivity([stencil], 0.)))
    centers.append(dict(parameter=-1e-7, center=[0., 0.], tau=.02, kind="old saddle estimate"))
    result = finish.assess_height_changes(centers)
    assert not result["centers"][0]["sensitivity_available"]
    assert len(result["adjacent_changes"]) == 1
    change = result["adjacent_changes"][0]
    assert change["quadratic_height_change"] == pytest.approx(0., abs=5e-17)
    assert change["secant_sensitivity_scale"] > 0
    np.testing.assert_allclose(change["quadratic_center_secant_velocity"], [1e4, 2e4], rtol=1e-12)
    assert len(change["residual_gradient_dot_center_secant_velocity"]) == 2
    assert "No root, derivative sign, noncrossing interval" in result["scope"]


def test_small_newton_step_does_not_promote_an_unresolved_height_sign():
    # A location correction below 3e-5 can still overwhelm a 1e-12 height.
    e, H = np.array([1e-5, 0.]), np.diag([-2., 3.])
    stencil = diagnostic_stencil(e, .003, 1e-12 + .5 * e @ H @ e, H @ e, H)
    assert stencil["correction_norm"] < 3e-5
    sensitivity = finish.height_sensitivity([stencil], 0.)
    assert not sensitivity["height_sign_consistent_across_estimates"]
    assert sensitivity["height_to_sensitivity_ratio"] < 1
