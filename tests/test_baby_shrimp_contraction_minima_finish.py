"""Corruption and incomplete-evidence controls for the BSC5 offline audit."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import baby_shrimp_contraction_minima_finish as finish
from contraction_section_minima import search_section_minima


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def runtime(run):
    save(run/"runtime.json", {"norm_sensitivity": 3.5e-15, "rotary_float64": True, "threads": 8})


def test_snapshot_checks_exact_files_and_hashes(tmp_path):
    save(tmp_path/"evidence.json", {"complete": False})
    entries = [finish.artifact(tmp_path/"evidence.json", tmp_path)]
    finish.verify_snapshot(tmp_path, entries)
    save(tmp_path/"unexpected.json", {})
    with pytest.raises(ValueError, match="unmanifested"):
        finish.verify_snapshot(tmp_path, entries)


@pytest.mark.parametrize("change", ["hash", "size", "duplicate", "escape", "link"])
def test_raw_manifest_corruption_is_rejected(tmp_path, change):
    path = tmp_path/"evidence.json"
    save(path, {})
    entries = [finish.artifact(path, tmp_path)]
    if change == "hash":
        entries[0]["sha256"] = "0"*64
    elif change == "size":
        entries[0]["bytes"] += 1
    elif change == "duplicate":
        entries.append(entries[0])
    elif change == "escape":
        entries[0]["path"] = "../evidence.json"
    else:
        (tmp_path/"link.json").symlink_to(path)
    with pytest.raises(ValueError):
        finish.verify_snapshot(tmp_path, entries)


def test_unfinished_points_are_preserved_without_becoming_qualified(tmp_path, monkeypatch):
    runtime(tmp_path)
    (tmp_path/"point-0000").mkdir()
    save(tmp_path/"point-0001/point.json", {"index": 1, "qualified": False, "beta": .006, "coordinate": [1, 2]})
    (tmp_path/"point-0002").mkdir()
    (tmp_path/"point-0002/point.json").write_text('{"index":')
    monkeypatch.setattr(finish.point_audit, "audit_point", lambda *args: pytest.fail("must not qualify unfinished points"))
    checked = finish.audit_saved_points(tmp_path, {"maximum_evaluations": 3})
    assert checked["raw"] == {} and checked["qualified_points"] == []
    assert len(checked["unqualified_or_incomplete_points"]) == 3
    assert checked["attempted_points"] == 3


@pytest.mark.parametrize("change", ["missing runtime", "precision", "threads", "budget", "gap", "index", "closure"])
def test_point_preconditions_cannot_be_silently_relaxed(tmp_path, monkeypatch, change):
    runtime(tmp_path)
    point = {"index": 0, "qualified": True}
    save(tmp_path/"point-0000/point.json", point)
    monkeypatch.setattr(finish.point_audit, "audit_point", lambda *args: (point, {"closure_recomputed_from_saved_endpoint": change != "closure"}))
    if change == "missing runtime":
        (tmp_path/"runtime.json").unlink()
    elif change in ("precision", "threads"):
        data = finish.read(tmp_path/"runtime.json")
        data["rotary_float64" if change == "precision" else "threads"] = False if change == "precision" else 32
        save(tmp_path/"runtime.json", data)
    elif change == "gap":
        (tmp_path/"point-0000").rename(tmp_path/"point-0001")
    elif change == "index":
        point["index"] = True
        save(tmp_path/"point-0000/point.json", point)
    with pytest.raises(ValueError):
        finish.audit_saved_points(tmp_path, {"maximum_evaluations": 0 if change == "budget" else 1})


@pytest.mark.parametrize("change", [None, "commit", "source", "missing source", "plan"])
def test_source_and_plan_are_bound_to_exact_git_blobs(tmp_path, monkeypatch, change):
    plan = {"id": "BSC5-test", "period": 18, "anchor_directory": "anchor", "anchor_hashes": {}}
    blobs = {"driver.py": b"frozen source", "plan.json": json.dumps(plan).encode()}
    protocol = {"commit": "a"*40, "source_sha256": {"driver.py": hashlib.sha256(blobs["driver.py"]).hexdigest()},
                "args": {"plan": "plan.json"}, "plan": plan,
                "plan_sha256": hashlib.sha256(blobs["plan.json"]).hexdigest()}
    monkeypatch.setattr(finish, "DATA", tmp_path)
    monkeypatch.setattr(finish.point_audit, "git_bytes", lambda commit, name: blobs[name])
    if change == "commit":
        protocol["commit"] = "main"
    elif change == "source":
        protocol["source_sha256"]["driver.py"] = "0"*64
    elif change == "missing source":
        protocol["source_sha256"] = {}
    elif change == "plan":
        protocol["plan_sha256"] = "0"*64
    if change is None:
        assert finish.bind_execution(protocol, expected_id="BSC5-test", required_sources={"driver.py"}) == tmp_path/"anchor"
    else:
        with pytest.raises(ValueError):
            finish.bind_execution(protocol, expected_id="BSC5-test", required_sources={"driver.py"})


def jet_evidence():
    center, beta, h = np.array([1., 2.]), .0064, .001
    raw = {}
    for iy, y in enumerate((-1, 0, 1)):
        for ix, x in enumerate((-1, 0, 1)):
            q = h*np.array([x, y]); index = 3*iy+ix
            tau = 1e-10 + np.array([1e-9, 2e-9])@q + .5*q@np.diag([-.04, .02])@q
            delta = -1e-8+np.array([1e-9, -2e-9])@q
            radius = .5*(np.sqrt(tau*tau-4*delta)+abs(tau))
            raw[index] = dict(index=index, qualified=True, beta=beta, coordinate=(center+q).tolist(),
                              tau=float(tau), delta=float(delta), radius=float(radius))
    H = np.diag([-.04, .02]); correction = -np.linalg.solve(H, [1e-9, 2e-9])
    jet = dict(center=center.tolist(), beta=beta, step=h, point_indices=np.arange(9).reshape(3, 3).tolist(),
               fields=dict(tau=dict(value=1e-10, gradient=[1e-9, 2e-9], hessian=H.tolist()),
                           delta=dict(value=-1e-8, gradient=[1e-9, -2e-9], hessian=[[0., 0.], [0., 0.]])),
               correction=correction.tolist(), correction_norm=float(np.linalg.norm(correction)))
    return jet, raw


def test_trace_and_determinant_jets_replay_saved_points():
    jet, raw = jet_evidence()
    result = finish.recompute_jet(jet, raw)
    np.testing.assert_allclose(result["fields"]["tau"]["hessian"], np.diag([-.04, .02]), atol=1e-13)
    np.testing.assert_allclose(result["fields"]["delta"]["gradient"], [1e-9, -2e-9], atol=1e-18)


@pytest.mark.parametrize("change", ["coordinate", "unqualified", "missing", "indices", "value", "gradient", "hessian", "correction"])
def test_jet_corruption_is_rejected(change):
    jet, raw = jet_evidence()
    if change == "coordinate":
        raw[0]["coordinate"][0] += .001
    elif change == "unqualified":
        raw[0]["qualified"] = False
    elif change == "missing":
        del raw[0]
    elif change == "indices":
        jet["point_indices"][0][0] = .5
    elif change == "value":
        jet["fields"]["tau"]["value"] += 1e-6
    elif change == "gradient":
        jet["fields"]["delta"]["gradient"][0] += 1e-8
    elif change == "hessian":
        jet["fields"]["tau"]["hessian"][0][0] += .01
    else:
        jet["correction_norm"] += .01
    with pytest.raises(ValueError):
        finish.recompute_jet(jet, raw)


def section_evidence(budget=100, fail_at=None):
    raw = {}
    def objective(s):
        if fail_at is not None and len(raw) == fail_at:
            raise ValueError("synthetic model gate failed")
        radius = 1e-4+1e-6*(s*s-1)**2
        index = len(raw)
        raw[index] = {"index": index, "coordinate": [s, 0.], "beta": .0064, "qualified": True,
                      "radius": radius, "tau": 0., "delta": -radius**2}
        return {"index": index, "radius": radius}
    result = search_section_minima(objective, [-2, -1, 0, 1, 2], coordinate_tolerance=1e-4,
                                   max_evaluations=budget, radius_floor=1e-14, signal_multiple=10,
                                   validation_offsets=[.03, .01], max_refinement_steps=80)
    return result, raw


@pytest.mark.parametrize("budget,fail_at", [(100, None), (3, None), (12, None), (100, 11)])
def test_all_competing_minima_and_partial_searches_are_replayed(budget, fail_at):
    result, raw = section_evidence(budget, fail_at)
    checked = finish.audit_section_search(result, raw, [0, 0], [1, 0], .0064)
    assert checked["status"] == result["status"]
    if budget == 100 and fail_at is None:
        assert checked["complete"] and len(checked["minima"]) == 2
    else:
        assert not checked["complete"]


@pytest.mark.parametrize("change", ["objective", "coordinate", "mesh omission", "minimum omission", "width", "steps", "contrast", "lower neighbor", "complete", "settings"])
def test_section_search_tampering_is_rejected(change):
    result, raw = section_evidence()
    if change == "objective":
        result["evaluations"][0]["radius"] += 1e-6
    elif change == "coordinate":
        raw[0]["coordinate"][0] += 1e-4
    elif change == "mesh omission":
        result["mesh_brackets"].pop()
    elif change == "minimum omission":
        result["minima"].pop()
    elif change == "width":
        result["candidates"][0]["refinement"]["width"] *= .5
    elif change == "steps":
        result["candidates"][0]["refinement"]["steps"] -= 1
    elif change == "contrast":
        result["candidates"][0]["validation"][0]["contrasts"][0] *= 2
    elif change == "lower neighbor":
        result["candidates"][0]["validation"][0]["lower_neighbor_resolved"] = True
    elif change == "complete":
        result["complete"] = False
    else:
        result["settings"]["signal_threshold"] *= 2
    with pytest.raises(ValueError):
        finish.audit_section_search(result, raw, [0, 0], [1, 0], .0064)


def terminal_packet(tmp_path, monkeypatch, scientific_complete=True, python_manifest=True):
    monkeypatch.setattr(finish, "DATA", tmp_path)
    run = tmp_path/"explore/run"; run.mkdir(parents=True)
    protocol = {"commit": "a"*40, "args": {"plan": "plan.json", "device": "cuda", "seconds": 10800}}
    save(run/"result.json", {"complete": scientific_complete, "execution_finished": True})
    if python_manifest:
        save(run/"manifest.json", {"complete": scientific_complete, "files": [finish.artifact(run/"result.json", run)]})
    manifest = tmp_path/"controller/manifest.json"
    save(manifest, [finish.artifact(p, run) for p in sorted(run.rglob("*")) if p.is_file()])
    status = tmp_path/"controller/status.json"
    command = ["python", "tools/baby_shrimp_contraction_minima.py", "--source-commit", "a"*40,
               "--out", str(run), "--plan", "plan.json", "--device", "cuda", "--seconds", "10800"]
    save(status, dict(worker_reaped=True, source_commit="a"*40, output=str(run), exit_code=0,
                     timed_out=False, complete=True, scientific_complete=scientific_complete, ended_utc="2026-10-22T00:00:00Z", hard_seconds=10800,
                     termination_grace_seconds=10, command=command, worker_pid=10, controller_pid=9, elapsed_seconds=100))
    return run, protocol, manifest, status


def terminal_call(run, protocol, manifest, status):
    return finish.terminal_evidence(run, protocol, manifest, finish.digest(manifest), status, finish.digest(status))


@pytest.mark.parametrize("scientific_complete,python_manifest", [(True, True), (False, True), (False, False)])
def test_terminal_execution_is_distinct_from_scientific_resolution(tmp_path, monkeypatch, scientific_complete, python_manifest):
    run, protocol, manifest, status = terminal_packet(tmp_path, monkeypatch, scientific_complete, python_manifest)
    evidence = terminal_call(run, protocol, manifest, status)
    assert evidence["controller"]["execution_complete"]
    assert evidence["python_complete"] == (scientific_complete and python_manifest)


@pytest.mark.parametrize("change", ["worker", "commit", "out", "seconds", "device", "plan", "timed out", "finished", "hash", "extra raw"])
def test_terminal_binding_corruptions_rejected(tmp_path, monkeypatch, change):
    run, protocol, manifest, status = terminal_packet(tmp_path, monkeypatch)
    row = finish.read(status)
    if change == "worker": row["worker_reaped"] = False
    elif change == "commit": row["source_commit"] = "b"*40
    elif change == "out": row["output"] = str(tmp_path/"other")
    elif change in ("seconds", "device", "plan"):
        row["command"][row["command"].index("--"+change)+1] = "wrong"
    elif change == "timed out": row["timed_out"] = True
    elif change == "finished": save(run/"result.json", {"complete": False, "execution_finished": False})
    elif change == "extra raw": save(run/"extra.json", {})
    save(status, row)
    with pytest.raises(ValueError):
        finish.terminal_evidence(run, protocol, manifest, "0"*64 if change == "hash" else finish.digest(manifest), status, finish.digest(status))


def expanded_evidence(tmp_path):
    directory = tmp_path/"expanded-spectrum"; directory.mkdir()
    values = np.array([1e-4, -.9e-4]+[1e-8/(i+1) for i in range(10)])
    q = np.eye(896, 12); w = q*values
    pairs = [[float(v), 0.] for v in values]
    point = {"index": 0, "phases": []}
    rows = []
    for phase in (0, 6, 12):
        row = {"phase": phase, "columns": 12, "random_seed": 5801+phase, "multipliers": pairs,
               "relative_residuals": [0., 0.], "history": [{"multipliers": pairs, "relative_residuals": [0., 0.]}]*2,
               "tau": float(sum(values[:2])), "delta": float(np.prod(values[:2])), "radius": float(max(abs(values))),
               "pair_relative_difference": 0., "passed": True}
        rows.append(row); point["phases"].append(dict(row))
        np.savez(directory/f"phase-{phase}.npz", values=values, q=q, w=w, vectors=q[:, :2],
                 direct=w[:, :2], relative_residuals=np.zeros(2))
    check = {"point_index": 0, "rows": rows, "variations": {k: 0. for k in ("tau", "delta", "radius")}, "complete": True}
    save(directory/"check.json", check)
    return check, {0: point}


def test_expanded_independent_arrays_and_cyclic_spread_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(finish, "DATA", tmp_path)
    check, raw = expanded_evidence(tmp_path)
    result = finish.audit_expanded(tmp_path, check, raw, 12)
    assert result["complete"] and len(result["rows"]) == 3 and result["variations"]["radius"] == 0.


@pytest.mark.parametrize("change", ["rank", "seed", "phase", "variation", "pair", "passed", "matrix"])
def test_expanded_corruption_rejected(tmp_path, monkeypatch, change):
    monkeypatch.setattr(finish, "DATA", tmp_path)
    check, raw = expanded_evidence(tmp_path)
    if change == "rank": check["rows"][0]["columns"] = 6
    elif change == "seed": check["rows"][0]["random_seed"] += 1
    elif change == "phase": check["rows"][0]["phase"] = 6
    elif change == "variation": check["variations"]["radius"] = 1e-12
    elif change == "pair": check["rows"][0]["pair_relative_difference"] = .01
    elif change == "passed": check["rows"][0]["passed"] = False
    elif change == "matrix":
        path = tmp_path/"expanded-spectrum/phase-0.npz"
        with np.load(path) as z: arrays = dict(z)
        arrays["w"][:, 0] *= 1.1
        np.savez(path, **arrays)
    save(tmp_path/"expanded-spectrum/check.json", check)
    with pytest.raises(ValueError): finish.audit_expanded(tmp_path, check, raw, 12)


def test_partial_expanded_prefix_is_not_promoted(tmp_path, monkeypatch):
    monkeypatch.setattr(finish, "DATA", tmp_path)
    check, raw = expanded_evidence(tmp_path)
    check.update(rows=check["rows"][:1], complete=False); check.pop("variations")
    save(tmp_path/"expanded-spectrum/check.json", check)
    result = finish.audit_expanded(tmp_path, None, raw, 12)
    assert not result["complete"] and len(result["rows"]) == 1


def test_null_arm_frame_is_orthonormal_and_has_zero_quadratic_form():
    H = np.array([[-.04, .005], [.005, .02]])
    arms = finish.null_arms(H)
    assert len(arms) == 4
    for arm in arms:
        t, n = np.asarray(arm["tangent"]), np.asarray(arm["normal"])
        assert abs(t@H@t) < 1e-17 and abs(t@n) < 1e-15
        assert abs(np.linalg.norm(t)-1) < 1e-15


def refinement_evidence(monkeypatch):
    jet = {"center": [1., 2.], "beta": .0064, "step": .001,
        "fields": {"tau": {"value": 1e-10, "gradient": [1e-9, 2e-9], "hessian": [[-.04, 0], [0, .02]]},
                   "delta": {"value": -1e-8, "gradient": [1e-9, -2e-9], "hessian": [[0,0],[0,0]]}},
        "correction_norm": 1e-7}
    fine = json.loads(json.dumps(jet)); fine["step"] = .0003
    derivatives = {"height_secant": -.1, "center_secant": [10., -20.]}
    dp = -np.linalg.solve(jet["fields"]["tau"]["hessian"], jet["fields"]["tau"]["gradient"])
    height = jet["fields"]["tau"]["value"]+.5*np.asarray(jet["fields"]["tau"]["gradient"])@dp
    db = -height/derivatives["height_secant"]
    row = {"old_center": jet["center"], "old_beta": jet["beta"], "stationary_height_predictor": height,
        "step_center": dp.tolist(), "step_beta": db, "new_center": (np.asarray(jet["center"])+dp+np.asarray(derivatives["center_secant"])*db).tolist(),
        "new_beta": jet["beta"]+db, "point_index": 0}
    raw = {0: {"index": 0, "qualified": True, "coordinate": row["new_center"], "beta": row["new_beta"],
               "tau": 1e-14, "delta": -1e-8, "radius": 1e-4}}
    cfg = {"maximum_refinements": 2, "jet_steps": [.001,.0003], "maximum_beta_correction": 1e-7,
           "refinement_step": .0003, "height_tolerance": 1e-12, "center_correction_tolerance": 1e-5}
    monkeypatch.setattr(finish, "recompute_jet", lambda j, r: j)
    return {"local_jets": [jet, fine], "refinement": [row]}, raw, {"center": jet["center"], "parameter": jet["beta"]}, derivatives, cfg


def test_partial_refinement_preserves_correction_without_inventing_verification(tmp_path, monkeypatch):
    args = refinement_evidence(monkeypatch)
    audited = finish.audit_refinements(*args)
    assert len(audited["refinement"]) == 1 and audited["center_point_index"] is None


@pytest.mark.parametrize("change", ["height", "beta", "center", "step", "point", "early admission"])
def test_refinement_predictor_corruption_rejected(monkeypatch, change):
    result, raw, seed, derivatives, cfg = refinement_evidence(monkeypatch)
    row = result["refinement"][0]
    if change == "height": row["stationary_height_predictor"] += 1e-6
    elif change == "beta": row["step_beta"] += 1e-6
    elif change == "center": row["new_center"] = [1.1,2.]
    elif change == "step": result["local_jets"][1]["step"] = .03
    elif change == "point": raw[0]["beta"] += 1e-7
    else: result["center_point"] = raw[0]
    with pytest.raises(ValueError): finish.audit_refinements(result, raw, seed, derivatives, cfg)


def full_section_evidence(tmp_path, fail_alignment=False):
    center, beta = np.array([1., 2.]), .0064
    jet = {"fields": {"tau": {"value": 0., "gradient": [0., 0.], "hessian": [[-.04,0],[0,.02]]},
                      "delta": {"value": -1e-8, "gradient": [1e-9,-2e-9], "hessian": [[0,0],[0,0]]}}}
    arms = finish.null_arms(jet["fields"]["tau"]["hessian"])
    arm = arms[0]; tangent, normal = np.asarray(arm["tangent"]), np.asarray(arm["normal"])
    origin = center+.002*tangent
    cfg = {"longitudinal_steps": [.002,.0006,.0002], "section_mesh": [-.0005,-.00025,0.,.00025,.0005],
           "coordinate_tolerance": 1e-5, "section_maximum_evaluations": 40, "signal_multiple": 10,
           "validation_offsets": [.0001,.00003]}
    point = {"index": 0, "coordinate": center.tolist(), "beta": beta, "qualified": True,
             "tau": 0., "delta": -1e-8, "radius": 1e-4}
    raw = {0: point}
    (tmp_path/"point-0000").mkdir()
    center_orbit = np.ones((18,896))
    np.save(tmp_path/"point-0000/points.npy", center_orbit)
    section = {**arm, "longitudinal": .002, "center": center.tolist(), "beta": beta, "origin": origin.tolist(),
               "mesh": cfg["section_mesh"], "search": None, "complete": False, "events": [], "predictor_checks": []}
    def objective(s):
        index = len(raw); coordinate = origin+s*normal
        radius = 1e-4+.001*s*s
        p = {"index": index, "coordinate": coordinate.tolist(), "beta": beta, "qualified": True,
             "tau": 0., "delta": -radius**2, "radius": radius}
        raw[index] = p
        directory = tmp_path/f"point-{index:04d}"; directory.mkdir()
        orbit = center_orbit*(1.001 if fail_alignment else 1.)
        np.save(directory/"points.npy", orbit)
        q = coordinate-center
        pred = {k: float(f["value"]+np.asarray(f["gradient"])@q+.5*q@np.asarray(f["hessian"])@q) for k,f in jet["fields"].items()}
        pred["radius"] = .5*(np.sqrt(pred["tau"]**2-4*pred["delta"])+abs(pred["tau"]))
        ds = [float(np.max(np.linalg.norm(np.roll(orbit,i,axis=0)-center_orbit,axis=1)/np.linalg.norm(center_orbit,axis=1))) for i in range(18)]
        section["predictor_checks"].append({"s": s, "point_index": index, "predicted": pred,
            "measured": {k:p[k] for k in ("tau","delta","radius")},
            "orbit_alignment": {"reference_point_index": 0, "best_right_roll": 0, "all_alignment_maximum_relative_changes": ds}})
        if fail_alignment: raise ValueError("full-state neighborhood failed")
        return {"index": index, "radius": radius}
    section["search"] = search_section_minima(objective, cfg["section_mesh"], coordinate_tolerance=1e-5,
        max_evaluations=40, radius_floor=1e-14, signal_multiple=10, validation_offsets=cfg["validation_offsets"],
        record=section["events"].append)
    section["complete"] = section["search"]["complete"]
    result = {"center_qualified": True, "center_point": point, "arms": arms, "sections": [section], "radius_floor": 1e-14}
    return result, raw, [jet], cfg


@pytest.mark.parametrize("fail_alignment", [False, True])
def test_whole_section_replays_geometry_objective_prediction_and_failed_alignment(tmp_path, fail_alignment):
    args = full_section_evidence(tmp_path, fail_alignment)
    checked = finish.audit_sections(tmp_path, *args)
    assert len(checked) == 1
    if fail_alignment:
        assert checked[0]["search"]["status"] == "evaluation_failed"
        assert checked[0]["predictor_checks"][0]["orbit_alignment_passed"] is False
    else:
        assert checked[0]["search"]["complete"] and len(checked[0]["search"]["minima"]) == 1
        assert checked[0]["maximum_absolute_prediction_error"]["radius"] > 0


@pytest.mark.parametrize("change", ["normal", "origin", "order", "prediction", "measured", "alignment", "alignment ref", "event", "callback", "complete"])
def test_whole_section_corruptions_rejected(tmp_path, change):
    result, raw, jets, cfg = full_section_evidence(tmp_path)
    section = result["sections"][0]
    if change == "normal": section["normal"] = [-x for x in section["normal"]]
    elif change == "origin": section["origin"][0] += 1e-6
    elif change == "order": section["longitudinal"] = .0006
    elif change == "prediction": section["predictor_checks"][0]["predicted"]["radius"] += 1e-8
    elif change == "measured": section["predictor_checks"][0]["measured"]["radius"] += 1e-8
    elif change == "alignment": section["predictor_checks"][0]["orbit_alignment"]["all_alignment_maximum_relative_changes"][0] = .1
    elif change == "alignment ref": section["predictor_checks"][0]["orbit_alignment"]["reference_point_index"] = 1
    elif change == "event": section["events"][0]["attempt"] = 2
    elif change == "callback": section["events"][1]["index"] = 999
    else: section["complete"] = False
    with pytest.raises(ValueError): finish.audit_sections(tmp_path, result, raw, jets, cfg)


def test_unreturned_section_is_preserved_without_promoting_pending_search(tmp_path):
    result, raw, jets, cfg = full_section_evidence(tmp_path)
    section = result["sections"][0]
    section.update(search=None, complete=False)
    checked = finish.audit_sections(tmp_path, result, raw, jets, cfg)
    assert checked[0]["search"]["status"] == "interrupted before search returned"
    assert not checked[0]["search"]["complete"] and checked[0]["search"]["minima"] == []


def test_failed_expanded_regime_is_replayed_but_never_admitted(tmp_path, monkeypatch):
    monkeypatch.setattr(finish, "DATA", tmp_path)
    check, raw = expanded_evidence(tmp_path)
    row = json.loads(json.dumps(check["rows"][0]))
    path = tmp_path/"expanded-spectrum/phase-0.npz"
    with np.load(path) as z: arrays = dict(z)
    arrays["values"][1] *= -1
    arrays["w"] = arrays["q"]*arrays["values"]
    arrays["direct"] = arrays["w"][:, :2].copy()
    np.savez(path, **arrays)
    values = arrays["values"]
    pairs = [[float(v),0.] for v in values]
    row.update(multipliers=pairs, tau=float(sum(values[:2])), delta=float(np.prod(values[:2])), passed=False,
               pair_relative_difference=float(finish.point_audit.pair_gap(values[:2], finish.point_audit.complex_values(raw[0]["phases"][0]["multipliers"])[:2])/raw[0]["phases"][0]["radius"]))
    for history in row["history"]: history["multipliers"] = pairs
    check.update(rows=[row], complete=False); check.pop("variations")
    save(tmp_path/"expanded-spectrum/check.json", check)
    replay = finish.audit_expanded(tmp_path, None, raw, 12)
    assert not replay["complete"] and replay["rows"][0]["passed"] is False
    assert replay["rows"][0]["delta"] > 0
    arrays["direct"][:,0] *= 1.1
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="residual"):
        finish.audit_expanded(tmp_path, None, raw, 12)


def failed_baseline(tmp_path):
    run, anchor = tmp_path/"run", tmp_path/"anchor"
    run.mkdir(); anchor.mkdir()
    values = np.array([1e-4,-9e-5])
    phases = [{"phase":p,"multipliers":[[float(v),0.] for v in values]} for p in (0,6,12)]
    point = {"index":0,"beta":0,"coordinate":[0,0],"phases":phases,"tau":float(sum(values))}
    for phase in (0,6,12): np.savez(anchor/f"phase-{phase}.npz", matrix=np.eye(2), multipliers=values)
    save(anchor/"point.json", {"tau":point["tau"],"spectral_radius":1e-4})
    directions = np.ones((2,1)); dense = directions; direct = directions*1.01
    errors = np.linalg.norm(direct-dense,axis=0)/np.linalg.norm(direct,axis=0)
    np.savez(run/"baseline-operator-check.npz",directions=directions,dense=dense,direct=direct,errors=errors)
    baseline = {"passed":False,"point":point,"composed_jvp_relative_errors":errors.tolist(),"tau_error_relative_to_radius":0.,
                "cyclic_dense_comparisons":[{"phase":p,"pair_relative_error":0.,"determinant_relative_error":0.} for p in (0,6,12)]}
    save(run/"baseline.json",baseline)
    return run,{0:point},anchor


def test_failed_dense_baseline_admission_preserves_independent_measurement(tmp_path):
    run, raw, anchor = failed_baseline(tmp_path)
    replay = finish.audit_baseline(run, raw, anchor)
    assert replay["passed"] is False and replay["maximum_operator_relative_error"] > 1e-3
    baseline = finish.read(run/"baseline.json"); baseline["composed_jvp_relative_errors"] = [0.]
    save(run/"baseline.json",baseline)
    with pytest.raises(ValueError, match="operator JSON"):
        finish.audit_baseline(run, raw, anchor)


def test_unrecorded_expanded_array_attempts_remain_explicitly_unverified(tmp_path, monkeypatch):
    monkeypatch.setattr(finish, "DATA", tmp_path)
    directory = tmp_path/"expanded-spectrum"; directory.mkdir()
    path = directory/"phase-0.npz"; path.write_bytes(b"partial compressed bytes")
    result = finish.audit_expanded(tmp_path, None, {}, 12)
    assert not result["complete"] and result["rows"] == []
    assert result["unrecorded_array_attempts"] == [finish.artifact(path)]
    assert "interrupted" in result["status"]


def partial_audit_fixture(tmp_path, monkeypatch, expanded_tau=0.):
    run, protocol, manifest, status = terminal_packet(tmp_path, monkeypatch, scientific_complete=False)
    plan = {key: {} for key in finish.binding.METADATA}
    plan.update(id="BSC5-contraction-minima", maximum_evaluations=300, prior_audit={},
        contraction_minima={"expanded_columns":12,"height_tolerance":1e-12,"radius_floor_minimum":1e-14,
                            "longitudinal_steps":[.002,.0006,.0002]})
    protocol.update(plan=plan, plan_sha256="f"*64, source_sha256={})
    protocol["args"].update(out=str(run),source_commit=protocol["commit"])
    result = {"complete":False,"execution_finished":True,"evaluations":1,"center_qualified":True,
              "radius_floor":1e-14,"baseline":{},"seed_replay":{},"expanded_spectrum":{},
              "local_jets":[],"sections":[]}
    save(run/"protocol.json",protocol); save(run/"baseline.json",{}); save(run/"result.json",result)
    save(run/"manifest.json",{"complete":False,"files":[finish.artifact(p,run) for p in sorted(run.rglob("*")) if p.is_file() and p.name!="manifest.json"]})
    save(manifest,[finish.artifact(p,run) for p in sorted(run.rglob("*")) if p.is_file()])
    monkeypatch.setattr(finish,"bind_execution",lambda *a,**k:tmp_path/"anchor")
    monkeypatch.setattr(finish,"audit_saved_points",lambda *a:{"raw":{},"qualified_points":[],"unqualified_or_incomplete_points":[],"runtime":{},"attempted_points":1})
    monkeypatch.setattr(finish,"audit_baseline",lambda *a:{"passed":True})
    monkeypatch.setattr(finish,"bind_prior",lambda *a:({}, {}, {}))
    monkeypatch.setattr(finish,"audit_seed",lambda *a:{"passed":True})
    monkeypatch.setattr(finish,"audit_refinements",lambda *a:{"local_jets":[],"center_point_index":0})
    monkeypatch.setattr(finish,"audit_expanded",lambda *a:{"complete":True,"point_index":0,"variations":{"radius":0.},"rows":[{"tau":expanded_tau}]*3})
    monkeypatch.setattr(finish,"audit_sections",lambda *a:[{"search":{"complete":False,"status":"nonconverged"}}]*12)
    kwargs=dict(controller_manifest=manifest,manifest_sha=finish.digest(manifest),controller_status=status,status_sha=finish.digest(status))
    return run,kwargs


def test_successful_execution_with_unresolved_sections_is_valid_explicit_partial_audit(tmp_path, monkeypatch):
    run, kwargs = partial_audit_fixture(tmp_path, monkeypatch)
    receipt=finish.run_audit(run,allow_partial=True,**kwargs)
    assert receipt["audit_finished"] and receipt["execution_finished"] and not receipt["complete"]
    assert receipt["lifecycle"]["controller"]["execution_complete"]
    with pytest.raises(ValueError,match="explicit --allow-partial"):
        finish.run_audit(run,allow_partial=False,**kwargs)


def test_expanded_height_admission_cannot_be_promoted_by_center_flag(tmp_path, monkeypatch):
    run, kwargs = partial_audit_fixture(tmp_path,monkeypatch,expanded_tau=1e-10)
    with pytest.raises(ValueError,match="expanded trace-height"):
        finish.run_audit(run,allow_partial=True,**kwargs)
