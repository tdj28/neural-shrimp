"""Synthetic plotting relationships and corruption tests; no model or GPU."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import plot_baby_shrimp_mixed_junction as plotter


def fixture_receipt(mode="complete"):
    """Synthetic display-only fields, not a simulated neural-map result."""
    plan = json.loads((plotter.ROOT/"docs/baby-shrimp-zoom/mixed-junction-plan.json").read_text())
    cfg = plan["mixed_junction"]; raw, points, admissions, cases = {}, [], [], []
    target = np.asarray(cfg["target"]["coordinate"])

    def new_point(center, beta, tau, cyclic=False):
        physical = np.asarray(plan["start"])+np.asarray(plan["coordinate_scales"])*center
        phases = [{"phase": p, "tau": tau+(p/6)*1e-14} for p in ([0, 6, 12] if cyclic else [0])]
        p = {"index": len(raw), "qualified": True, "coordinate": np.asarray(center).tolist(), "beta": beta,
             "alpha": float(physical[0]), "log10_temperature": float(physical[1]), "tau": tau,
             "delta": -1e-8, "radius": .0001, "phases": phases}
        raw[p["index"]] = p
        points.append({**p, "phases": [r["phase"] for r in phases], "closure_recomputed_from_saved_endpoint": True})
        admissions.append({"point_index": p["index"], "status": "passed", "passed": True})
        return p

    count = 0 if mode == "empty" else 1 if mode in ("partial", "failed", "unfinished_stencil") else 3
    for j, offset in enumerate(cfg["beta_offsets"][:count]):
        beta = float(cfg["target"]["beta"]+offset)
        center = target+offset*np.asarray(cfg["target"]["moving_center_beta"])
        iteration = {"center": center.tolist(), "beta": beta, "corrections_already_applied": 0}
        case = {"beta": beta, "beta_offset": offset, "complete": mode not in ("failed", "unfinished_stencil"),
                "status": "unresolved precision" if mode == "failed" else "unresolved" if mode == "unfinished_stencil" else "admitted",
                "predicted_center": center.tolist(), "iterations": [iteration]}
        cases.append(case)
        if mode == "unfinished_stencil":
            new_point(center+[-.0003, -.0003], beta, 2e-9)
            break
        for prefix, step in zip(("coarse", "fine"), cfg["stencil_steps"]):
            ids = []
            for dy in (-1, 0, 1):
                ids.append([new_point(center+step*np.array([dx, dy]), beta, float(2e-14+step*(dx*1e-5-dy*1e-6)))["index"] for dx in (-1, 0, 1)])
            iteration[prefix+"_jet"] = {"center": center.tolist(), "beta": beta, "step": step, "point_indices": ids,
                "fields": {"tau": {"value": 2e-14}, "delta": {"value": -1e-8}}}
        cyclic = new_point(center, beta, 2e-14, True)
        feature = (0., 5.1e-7, -5.6e-7)[j]
        iteration["coarse_geometry"] = {"input_tau": 2e-14,
            "projection": {"input_tau": 2e-14, "enforced_tau": 0., "spatial_displacement_applied": False},
            "branches": {side: {"valley": {"defined": True, "feature": 0. if k == 0 else feature}}
                         for k, side in enumerate(plotter.SIDES)}}
        iteration["coarse_correction"] = {"equations": [2e-14, 0.], "f_minus": feature}
        iteration["cyclic_point_index"] = cyclic["index"]
        iteration["uncertainty"] = {"intersection": {"f_minus_total_error": 1e-8, "f_minus_location_transfer": 6e-9},
            "branch_envelopes": {plotter.SIDES[1]: {"feature_error": 4e-9}}}
        status = ("residual_qualified_near_zero", "resolved_positive", "resolved_negative")[j]
        fm = {"value": feature, "empirical_total_error": 1e-8, "status": "not_admitted" if mode == "failed" else status}
        iteration["admission"] = {"passed": mode != "failed", "checks": {"example_gate": mode != "failed"}, "f_minus": fm}
        if case["complete"]:
            case.update(final_center=center.tolist(), final_point_index=cyclic["index"], f_minus=copy.deepcopy(fm))
    complete = mode == "complete"
    receipt = {"id": "BSC6-mixed-junction-offline-audit", "audit_finished": True, "complete": complete,
        "execution_finished": complete, "qualified_points": points, "point_admissions": admissions,
        "cases": cases, "attempted_points": len(points), "unqualified_or_incomplete_points": [],
        "resolved_f_minus_sign_bracket": True if complete else None,
        "untested_or_incomplete_stages": [] if complete else ["upper-beta intersection", "lower-beta intersection"],
        "terminal_failure": None if complete or mode == "partial" else {"type": "ValueError", "error": "synthetic retained failure"},
        "lifecycle": {"python_finalization": True, "python_complete": complete, "controller": None}}
    return receipt, plan, raw


def validate(receipt, plan, raw, **kw):
    return plotter.validate_plot_data(receipt, plan, raw, **kw)


def test_complete_keeps_three_cases_four_actual_trace_values_and_unjoined_points():
    receipt, plan, raw = fixture_receipt()
    data = validate(receipt, plan, raw)
    assert [r["feature_status"] for r in data["rows"]] == ["residual_qualified_near_zero", "resolved_positive", "resolved_negative"]
    assert all(len(r["actual_traces"]) == 4 and len(r["sample_indices"]) == 19 for r in data["rows"])
    assert data["rows"][0]["actual_traces"][-1]["tau"] != 0


def test_actual_independent_finisher_schema_from_analytic_measured_stencils():
    sys.path.insert(0, str(Path(__file__).parent))
    from test_baby_shrimp_mixed_junction_probe_finish import case_fixture, audit
    result, raw, admitted, spans, cfg = case_fixture()
    cases = audit.audit_cases(result, raw, admitted, spans, cfg)
    receipt, plan, _ = fixture_receipt()
    plan["mixed_junction"] = cfg
    points = []
    for p in raw.values():
        physical = np.asarray(plan["start"])+np.asarray(plan["coordinate_scales"])*p["coordinate"]
        p.update(alpha=float(physical[0]), log10_temperature=float(physical[1]))
        points.append({**p, "phases": [v["phase"] for v in p["phases"]], "closure_recomputed_from_saved_endpoint": True})
    receipt.update(cases=cases, qualified_points=points, attempted_points=len(points),
                   point_admissions=[{"point_index": i, "status": "passed", "passed": True} for i in raw],
                   resolved_f_minus_sign_bracket=result["resolved_f_minus_sign_bracket"])
    data = validate(receipt, plan, raw)
    assert len(data["rows"]) == 3 and all(r["admitted"] for r in data["rows"])


@pytest.mark.parametrize("mode", ["empty", "partial", "failed", "unfinished_stencil"])
def test_partial_requires_opt_in_and_preserves_every_planned_case(mode):
    receipt, plan, raw = fixture_receipt(mode)
    with pytest.raises(ValueError, match="allow-partial"):
        validate(receipt, plan, raw)
    data = validate(receipt, plan, raw, allow_partial=True)
    assert len(data["rows"]) == 3
    assert data["rows"][2]["status"] == "not attempted"
    if mode == "unfinished_stencil":
        assert data["rows"][0]["sample_indices"] == [0]
        assert data["rows"][0]["feature"] is None
    if mode == "failed":
        assert data["rows"][0]["feature_status"] == "not_admitted"
        assert not data["rows"][0]["admitted"]


@pytest.mark.parametrize("change", ["unfinished", "boolean_complete", "qualified", "closure", "boolean_index", "duplicate", "nan", "physical", "raw_tau", "phase", "phase_trace", "raw_extra", "state_admission", "inventory", "case_beta", "case_order", "unknown_status", "predictor", "step", "stencil_id", "stencil_coordinate", "jet_tau", "projection", "applied_projection", "feature", "equation_tau", "error", "transfer", "admission_boolean", "admission_gate", "wrong_sign", "final_center", "missing_cyclic", "hidden_partial", "sign_bracket", "missing_case"])
def test_corrupt_display_evidence_fails_closed(change):
    receipt, plan, raw = fixture_receipt()
    case = receipt["cases"][0]; row = case["iterations"][0]; p = receipt["qualified_points"][0]
    if change == "unfinished": receipt["audit_finished"] = False
    elif change == "boolean_complete": receipt["complete"] = 1
    elif change == "qualified": p["qualified"] = 1
    elif change == "closure": p["closure_recomputed_from_saved_endpoint"] = False
    elif change == "boolean_index": p["index"] = True
    elif change == "duplicate": receipt["qualified_points"].append(copy.deepcopy(p))
    elif change == "nan": p["radius"] = raw[0]["radius"] = float("nan")
    elif change == "physical": p["alpha"] = raw[0]["alpha"] = .9
    elif change == "raw_tau": raw[0]["tau"] += 1e-10
    elif change == "phase": p["phases"] = [True]
    elif change == "phase_trace": raw[0]["phases"][0]["tau"] += 1e-10
    elif change == "raw_extra": raw[999] = copy.deepcopy(raw[0])
    elif change == "state_admission": receipt["point_admissions"][0]["status"] = "failed"
    elif change == "inventory": receipt["attempted_points"] += 1
    elif change == "case_beta": case["beta"] += 1e-10
    elif change == "case_order": receipt["cases"].reverse()
    elif change == "unknown_status": case["status"] = "exact crossing proved"
    elif change == "predictor": case["predicted_center"][0] += 1e-7
    elif change == "step": row["coarse_jet"]["step"] *= 2
    elif change == "stencil_id": row["coarse_jet"]["point_indices"][0][0] = True
    elif change == "stencil_coordinate": row["coarse_jet"]["point_indices"][0][0] = 1
    elif change == "jet_tau": row["coarse_jet"]["fields"]["tau"]["value"] = 0.
    elif change == "projection": row["coarse_geometry"]["projection"]["input_tau"] = 0.
    elif change == "applied_projection": row["coarse_geometry"]["projection"]["spatial_displacement_applied"] = True
    elif change == "feature": row["coarse_correction"]["f_minus"] += 1e-7
    elif change == "equation_tau": row["coarse_correction"]["equations"][0] = 0.
    elif change == "error": row["uncertainty"]["intersection"]["f_minus_total_error"] = -1.
    elif change == "transfer": row["uncertainty"]["intersection"]["f_minus_location_transfer"] = 0.
    elif change == "admission_boolean": row["admission"]["passed"] = 1
    elif change == "admission_gate": row["admission"]["checks"]["example_gate"] = False
    elif change == "wrong_sign": row["admission"]["f_minus"]["status"] = "resolved_positive"
    elif change == "final_center": case["final_center"][0] += 1e-8
    elif change == "missing_cyclic": del row["cyclic_point_index"]
    elif change == "hidden_partial": receipt["untested_or_incomplete_stages"] = ["upper-beta intersection"]
    elif change == "sign_bracket": receipt["resolved_f_minus_sign_bracket"] = False
    elif change == "missing_case": receipt["cases"].pop()
    with pytest.raises((ValueError, KeyError)):
        validate(receipt, plan, raw)


def test_unqualified_and_alignment_failed_points_survive_in_inventory():
    receipt, plan, raw = fixture_receipt("unfinished_stencil")
    receipt["point_admissions"][0].update(status="failed", passed=False)
    receipt["attempted_points"] += 1
    receipt["unqualified_or_incomplete_points"] = [{"directory": "point-0001", "status": "not qualified; failed or interrupted",
        "beta": receipt["cases"][0]["beta"], "coordinate": receipt["cases"][0]["predicted_center"]}]
    data = validate(receipt, plan, raw, allow_partial=True)
    assert data["rows"][0]["unadmitted_sample_indices"] == [0]
    assert len(data["pending"]) == 1


def bound_fixture(tmp_path, monkeypatch, mode="complete"):
    receipt, plan, raw = fixture_receipt(mode)
    data = tmp_path/"data"; run = data/"explore"/"bsc6"; run.mkdir(parents=True)
    monkeypatch.setattr(plotter, "DATA", data)
    commit = "a"*40; plan_path = "docs/plan.json"
    blobs = {plotter.FINISHER: (f"SOURCES = {[plotter.FINISHER]!r}\nEXECUTION_SOURCES = {[plotter.DRIVER, 'tools/contraction_valley_geometry.py']!r}\n").encode(),
             plotter.DRIVER: b"worker", "tools/contraction_valley_geometry.py": b"geometry", plan_path: json.dumps(plan).encode()}
    def git_bytes(c, path):
        if c != commit or path not in blobs: raise ValueError("source lookup differs")
        return blobs[path]
    monkeypatch.setattr(plotter, "git_bytes", git_bytes)
    sha = lambda b: hashlib.sha256(b).hexdigest()
    receipt.update(analysis_commit=commit, execution_source_commit=commit,
        analysis_source_sha256={plotter.FINISHER: sha(blobs[plotter.FINISHER])},
        execution_source_sha256={p: sha(blobs[p]) for p in (plotter.DRIVER, "tools/contraction_valley_geometry.py")},
        plan_path=plan_path, plan_sha256=sha(blobs[plan_path]), scientific_metadata={k: plan[k] for k in plotter.METADATA}, raw_directory=str(run))
    protocol = {"commit": commit, "source_sha256": receipt["execution_source_sha256"], "plan_sha256": receipt["plan_sha256"],
                "plan": plan, "args": {"plan": plan_path, "out": str(run)}}
    (run/"protocol.json").write_text(json.dumps(protocol))
    for i, p in raw.items():
        directory = run/f"point-{i:04d}"; directory.mkdir(); (directory/"point.json").write_text(json.dumps(p))
    receipt["manifest"] = [{"path": p.relative_to(run).as_posix(), "bytes": p.stat().st_size, "sha256": plotter.digest(p)}
                           for p in sorted(run.rglob("*")) if p.is_file()]
    path = tmp_path/"receipt.json"
    def save():
        path.write_text(json.dumps(receipt)); return plotter.digest(path)
    return receipt, run, path, save, blobs


def test_complete_binding_loads_only_after_receipt_source_plan_and_raw_checks(tmp_path, monkeypatch):
    receipt, run, path, save, blobs = bound_fixture(tmp_path, monkeypatch)
    loaded, data = plotter.load_receipt(path, save())
    assert loaded["complete"] and len(data["rows"]) == 3


@pytest.mark.parametrize("change", ["receipt_sha", "audit_source", "missing_dependency", "execution_source", "analysis_commit", "plan_sha", "metadata", "raw_bytes", "raw_sha", "raw_extra", "raw_missing", "raw_duplicate", "raw_traversal", "raw_symlink", "protocol", "nonterminal", "controller_live", "controller_timeout", "python_partial"])
def test_source_manifest_lifecycle_and_plan_corruption_refused(tmp_path, monkeypatch, change):
    receipt, run, path, save, blobs = bound_fixture(tmp_path, monkeypatch)
    if change == "audit_source": receipt["analysis_source_sha256"][plotter.FINISHER] = "f"*64
    elif change == "missing_dependency": receipt["execution_source_sha256"].pop("tools/contraction_valley_geometry.py")
    elif change == "execution_source": receipt["execution_source_sha256"][plotter.DRIVER] = "f"*64
    elif change == "analysis_commit": receipt["analysis_commit"] = "b"*40
    elif change == "plan_sha": receipt["plan_sha256"] = "e"*64
    elif change == "metadata": receipt["scientific_metadata"]["period"] = 6
    elif change == "raw_bytes": receipt["manifest"][0]["bytes"] += 1
    elif change == "raw_sha": receipt["manifest"][0]["sha256"] = "e"*64
    elif change == "raw_extra": (run/"extra").write_text("new evidence")
    elif change == "raw_missing": (run/receipt["manifest"][0]["path"]).unlink()
    elif change == "raw_duplicate": receipt["manifest"].append(receipt["manifest"][0])
    elif change == "raw_traversal": receipt["manifest"][0]["path"] = "../escape"
    elif change == "raw_symlink": (run/"linked").symlink_to(path)
    elif change == "protocol":
        p = run/"protocol.json"; value = json.loads(p.read_text()); value["args"]["out"] = "/other"; p.write_text(json.dumps(value))
        for entry in receipt["manifest"]:
            if entry["path"] == "protocol.json": entry.update(bytes=p.stat().st_size, sha256=plotter.digest(p))
    elif change == "nonterminal": receipt["lifecycle"]["python_finalization"] = False
    elif change == "controller_live": receipt["lifecycle"]["controller"] = {"worker_reaped": False}
    elif change == "controller_timeout": receipt["lifecycle"]["controller"] = {"worker_reaped": True, "execution_complete": True, "timed_out": True}
    elif change == "python_partial": receipt["lifecycle"]["python_complete"] = False
    sha = save()
    with pytest.raises(ValueError):
        plotter.load_receipt(path, "0"*64 if change == "receipt_sha" else sha)


def test_terminal_controller_can_preserve_partial_without_python_finalization(tmp_path, monkeypatch):
    receipt, run, path, save, blobs = bound_fixture(tmp_path, monkeypatch, "empty")
    receipt["lifecycle"].update(python_finalization=False, controller={"worker_reaped": True, "execution_complete": False, "timed_out": True})
    loaded, data = plotter.load_receipt(path, save(), allow_partial=True)
    assert not loaded["complete"] and all(r["status"] == "not attempted" for r in data["rows"])


@pytest.mark.parametrize("mode", ["complete", "failed", "empty", "unfinished_stencil"])
def test_synthetic_png_pdf_render_preserves_status_and_uses_no_connecting_curve(tmp_path, monkeypatch, mode):
    receipt, plan, raw = fixture_receipt(mode)
    data = validate(receipt, plan, raw, allow_partial=mode != "complete")
    import matplotlib.axes
    def forbidden(*args, **kwargs):
        raise AssertionError("No inferred contour or connection may be drawn")
    monkeypatch.setattr(matplotlib.axes.Axes, "contour", forbidden)
    monkeypatch.setattr(matplotlib.axes.Axes, "contourf", forbidden)
    plotter.render(receipt, data, tmp_path, synthetic=True)
    assert {p.name for p in tmp_path.iterdir()} == {f"mixed-junction-{kind}.{ext}" for kind in ("summary", "locations") for ext in ("png", "pdf")}
    assert all(p.stat().st_size > 5000 for p in tmp_path.iterdir())
