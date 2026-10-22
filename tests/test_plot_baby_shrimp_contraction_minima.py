"""Synthetic receipt/plot validation; no model, GPU, or production-data reads."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import plot_baby_shrimp_contraction_minima as plotter
from contraction_section_minima import search_section_minima
from baby_shrimp_contraction_minima_finish import audit_section_search


def fixture_receipt(*, partial=False, no_center=False):
    """All four arms contain two wells, one cusp, flat and monotone sections."""
    plan = json.loads((plotter.ROOT/"docs/baby-shrimp-zoom/contraction-minima-plan.json").read_text())
    cfg = plan["contraction_minima"]
    center, beta = np.array([.025, -.08]), .00644853
    raw = {}

    def new_point(coordinate, radius=.001):
        coordinate = np.asarray(coordinate)
        q = coordinate-center
        tau = float(.5*(-.7*q[0]**2+1.1*q[1]**2))
        physical = np.asarray(plan["start"])+np.asarray(plan["coordinate_scales"])*coordinate
        p = {"index": len(raw), "qualified": True, "coordinate": coordinate.tolist(), "beta": beta,
             "tau": tau, "delta": float(radius*(abs(tau)-radius)), "radius": radius,
             "alpha": float(physical[0]), "log10_temperature": float(physical[1]), "phases": [0, 6, 12]}
        raw[p["index"]] = p
        return p

    step = .0003
    indices = [[new_point(center+step*np.array([dx, dy]))["index"] for dx in (-1, 0, 1)] for dy in (-1, 0, 1)]
    central = raw[indices[1][1]]
    t = np.array([[raw[i]["tau"] for i in line] for line in indices])
    hxy = (t[2, 2]-t[0, 2]-t[2, 0]+t[0, 0])/(4*step**2)
    H = np.array([[(t[1, 2]-2*t[1, 1]+t[1, 0])/step**2, hxy],
                  [hxy, (t[2, 1]-2*t[1, 1]+t[0, 1])/step**2]])
    jet = {"center": center.tolist(), "beta": beta, "step": step, "point_indices": indices,
           "fields": {"tau": {"value": 0., "gradient": [0., 0.], "hessian": H.tolist()}}}
    sections = []
    for longitudinal in cfg["longitudinal_steps"]:
        for i, arm in enumerate(plotter.null_arms(H)):
            tangent, normal = arm["tangent"], arm["normal"]
            origin = center+longitudinal*tangent
            rows = []

            def objective(s):
                difference = (6e-6*min(abs(s-.00025), abs(s+.00025)) if i == 0 else
                              6e-6*abs(s-.00007) if i == 1 else 0. if i == 2 else 6e-6*s)
                p = new_point(origin+s*normal, .001+difference)
                rows.append({"s": s, "point_index": p["index"],
                             **{k: p[k] for k in ("coordinate", "tau", "delta", "radius")},
                             "orbit_alignment_passed": True,
                             "orbit_alignment": {"reference_point_index": central["index"], "best_right_roll": 0,
                                  "all_alignment_maximum_relative_changes": [1e-5]+[.01]*17}})
                return {"index": p["index"], "radius": p["radius"]}

            search = search_section_minima(objective, cfg["section_mesh"],
                coordinate_tolerance=cfg["coordinate_tolerance"], max_evaluations=cfg["section_maximum_evaluations"],
                radius_floor=1e-14, signal_multiple=cfg["signal_multiple"], validation_offsets=cfg["validation_offsets"])
            audited = audit_section_search(search, raw, origin, normal, beta)
            sections.append({"arm_id": arm["arm_id"], "tangent": tangent.tolist(), "normal": normal.tolist(),
                "longitudinal": longitudinal, "center": center.tolist(), "beta": beta, "origin": origin.tolist(),
                "predictor_checks": rows, "callback_attempts": search["evaluations"], "search": audited})
    receipt = {"id": "BSC5-contraction-minima-offline-audit", "audit_finished": True,
        "complete": not partial and not no_center, "execution_finished": not partial and not no_center,
        "qualified_points": list(raw.values()), "center_qualified": not no_center,
        "center_point_index": central["index"], "local_jets": [jet], "radius_floor": 1e-14,
        "expanded_spectrum": {"variations": {"radius": 0.}}, "sections": [] if no_center else sections[:1] if partial else sections,
        "terminal_failure": None, "lifecycle": {"python_finalization": True, "python_complete": not partial and not no_center, "controller": None}}
    return receipt, plan


def test_all_competing_minima_and_nonminimum_outcomes_survive_validation():
    receipt, plan = fixture_receipt()
    data = plotter.validate_plot_data(receipt, plan)
    assert len(data["sections"]) == 12
    for offset in (0, 4, 8):
        rows = data["sections"][offset:offset+4]
        assert [len(r["section"]["search"]["minima"]) for r in rows] == [2, 1, 0, 0]
        assert [len(r["brackets"]) for r in rows] == [2, 1, 0, 0]
        assert [r["section"]["search"]["status"] for r in rows] == ["converged", "converged", "no_bracket", "no_bracket"]


@pytest.mark.parametrize("no_center", [False, True])
def test_partial_pilot_requires_opt_in_and_preserves_missing_stages(no_center):
    receipt, plan = fixture_receipt(partial=True, no_center=no_center)
    with pytest.raises(ValueError, match="allow-partial"):
        plotter.validate_plot_data(receipt, plan)
    data = plotter.validate_plot_data(receipt, plan, allow_partial=True)
    assert len(data["sections"]) == (0 if no_center else 1)
    assert (data["center"] is None) is no_center


@pytest.mark.parametrize("change", ["unfinished audit", "false qualified", "boolean index", "duplicate index", "nan radius", "wrong physical", "wrong beta", "unknown point"])
def test_unqualified_or_misbound_point_cannot_reach_a_plot(change):
    receipt, plan = fixture_receipt()
    row = receipt["sections"][0]["predictor_checks"][0]
    p = receipt["qualified_points"][row["point_index"]]
    if change == "unfinished audit":
        receipt["audit_finished"] = False
    elif change == "false qualified":
        p["qualified"] = 1
    elif change == "boolean index":
        p["index"] = True
    elif change == "duplicate index":
        receipt["qualified_points"].append(copy.deepcopy(p))
    elif change == "nan radius":
        p["radius"] = float("nan")
    elif change == "wrong physical":
        p["alpha"] += 1e-8
    elif change == "wrong beta":
        p["beta"] += .0001
    else:
        row["point_index"] = 99999
    with pytest.raises(ValueError):
        plotter.validate_plot_data(receipt, plan)


@pytest.mark.parametrize("change", ["section order", "tangent", "origin", "s", "radius", "alignment", "callback", "jet", "center phase", "floor"])
def test_observed_geometry_values_and_conditional_jet_are_checked(change):
    receipt, plan = fixture_receipt()
    section = receipt["sections"][0]
    row = section["predictor_checks"][0]
    if change == "section order":
        receipt["sections"].reverse()
    elif change == "tangent":
        section["tangent"][0] += .001
    elif change == "origin":
        section["origin"][0] += .001
    elif change == "s":
        row["s"] += .0001
    elif change == "radius":
        row["radius"] += 1e-12
    elif change == "alignment":
        row["orbit_alignment_passed"] = False
    elif change == "callback":
        section["callback_attempts"][0]["radius"] += 1e-12
    elif change == "jet":
        receipt["local_jets"][0]["fields"]["tau"]["hessian"][0][0] += .1
    elif change == "center phase":
        receipt["qualified_points"][receipt["center_point_index"]]["phases"] = [0]
    else:
        receipt["radius_floor"] *= 2
    with pytest.raises(ValueError):
        plotter.validate_plot_data(receipt, plan)


@pytest.mark.parametrize("change", ["omitted competitor", "extra minimum", "minimum coordinate", "width", "unknown bracket", "offset radius", "missing scale", "contrast", "below gate", "hidden incomplete", "invented no bracket"])
def test_all_competitors_widths_and_actual_validation_offsets_are_required(change):
    receipt, plan = fixture_receipt()
    section = receipt["sections"][0]
    search = section["search"]
    minimum = search["minima"][0]
    if change == "omitted competitor":
        search["minima"].pop()
    elif change == "extra minimum":
        search["minima"].append(copy.deepcopy(minimum))
        search["candidate_status_counts"]["converged"] += 1
    elif change == "minimum coordinate":
        minimum["coordinate"][0] += 1e-5
    elif change == "width":
        minimum["bracket_width"] *= 2
    elif change == "unknown bracket":
        minimum["refined_bracket"][0] -= 1e-10
    elif change == "offset radius":
        minimum["validation"][0]["samples"][0]["radius"] += 1e-12
    elif change == "missing scale":
        minimum["validation"].pop()
    elif change == "contrast":
        minimum["validation"][0]["contrasts"][0] *= 2
    elif change == "below gate":
        minimum["validation"][0]["resolved"] = False
    elif change == "hidden incomplete":
        search["complete"] = False
    else:
        search["status"] = "no_bracket"
    with pytest.raises(ValueError):
        plotter.validate_plot_data(receipt, plan)


def test_alignment_rejected_and_interrupted_attempts_are_retained_separately():
    receipt, plan = fixture_receipt(partial=True)
    section = receipt["sections"][0]
    section["predictor_checks"] = section["predictor_checks"][:1]
    row = section["predictor_checks"][0]
    row["orbit_alignment"]["all_alignment_maximum_relative_changes"][0] = .1
    row["orbit_alignment"]["best_right_roll"] = 1
    row["orbit_alignment_passed"] = False
    section["callback_attempts"] = [{"s": row["s"], "status": "failed", "radius": None, "index": None}]
    section["search"] = {"status": "interrupted before search returned", "complete": False, "minima": [], "evaluation_count": 1}
    data = plotter.validate_plot_data(receipt, plan, allow_partial=True)
    assert len(data["sections"][0]["observed"]) == 1
    assert not data["sections"][0]["admitted"]
    assert not data["sections"][0]["brackets"]


def test_aligned_measurement_with_unfinished_callback_is_not_an_admitted_objective():
    receipt, plan = fixture_receipt(partial=True)
    section = receipt["sections"][0]
    section["predictor_checks"] = section["predictor_checks"][:1]
    row = section["predictor_checks"][0]
    section["callback_attempts"] = [{"s": row["s"], "status": "started", "radius": None, "index": None}]
    section["search"] = {"status": "interrupted before search returned", "complete": False, "minima": [], "evaluation_count": 1}
    data = plotter.validate_plot_data(receipt, plan, allow_partial=True)
    assert row["orbit_alignment_passed"]
    assert len(data["sections"][0]["observed"]) == 1
    assert not data["sections"][0]["admitted"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def bound_fixture(tmp_path, monkeypatch, *, partial=False):
    receipt, plan = fixture_receipt(partial=partial)
    data = tmp_path/"data"
    run = data/"explore/synthetic-bsc5"
    run.mkdir(parents=True)
    monkeypatch.setattr(plotter, "DATA", data)
    execution, analysis = "a"*40, "b"*40
    execution_paths = {plotter.DRIVER, "tools/contraction_section_minima.py", "tools/synthetic-dependency.py"}
    analysis_paths = {plotter.FINISHER, "tools/synthetic-auditor-dependency.py"}
    blobs = {(execution, p): ("synthetic source "+p).encode() for p in execution_paths}
    blobs.update({(analysis, p): ("synthetic auditor "+p).encode() for p in analysis_paths})
    blobs[(analysis, plotter.FINISHER)] = ("SOURCES = "+repr(sorted(analysis_paths))+"\nEXECUTION_SOURCES = "+repr(execution_paths)+"\n").encode()
    plan_path = "docs/synthetic-plan.json"
    blobs[(execution, plan_path)] = json.dumps(plan).encode()
    receipt.update(execution_source_commit=execution, analysis_commit=analysis,
        execution_source_sha256={p: hashlib.sha256(blobs[(execution, p)]).hexdigest() for p in execution_paths},
        analysis_source_sha256={p: hashlib.sha256(blobs[(analysis, p)]).hexdigest() for p in analysis_paths},
        plan_path=plan_path, plan_sha256=hashlib.sha256(blobs[(execution, plan_path)]).hexdigest(),
        scientific_metadata={k: plan[k] for k in plotter.METADATA}, raw_directory=str(run))
    protocol = {"commit": execution, "source_sha256": receipt["execution_source_sha256"],
        "plan_sha256": receipt["plan_sha256"], "plan": plan, "args": {"out": str(run), "plan": plan_path}}
    write_json(run/"protocol.json", protocol)
    (run/"synthetic.bin").write_bytes(b"immutable synthetic raw evidence")
    receipt["manifest"] = [{"path": p.name, "bytes": p.stat().st_size, "sha256": plotter.digest(p)} for p in sorted(run.iterdir())]
    calls = []
    def git_bytes(commit, path):
        calls.append((commit, path))
        if (commit, path) not in blobs:
            raise ValueError("unexpected frozen Git input")
        return blobs[(commit, path)]
    monkeypatch.setattr(plotter, "git_bytes", git_bytes)
    path = tmp_path/"receipt.json"
    write_json(path, receipt)
    return path, receipt, blobs, calls


def test_loader_requires_and_verifies_frozen_source_plan_and_exact_raw_snapshot(tmp_path, monkeypatch):
    path, receipt, blobs, calls = bound_fixture(tmp_path, monkeypatch)
    loaded, data = plotter.load_receipt(path, plotter.digest(path))
    assert loaded == receipt
    assert set(calls) == set(blobs)
    assert len(data["sections"]) == 12


@pytest.mark.parametrize("change", ["hash", "unfinished", "partial", "analysis dependency", "execution dependency", "source hash", "plan hash", "metadata", "lifecycle", "unreaped", "protocol", "raw changed", "raw missing", "raw extra", "duplicate manifest", "manifest escape", "symlink"])
def test_loader_refuses_unbound_or_unfinished_evidence(tmp_path, monkeypatch, change):
    path, receipt, blobs, calls = bound_fixture(tmp_path, monkeypatch)
    expected = None
    run = Path(receipt["raw_directory"])
    if change == "hash":
        expected = "0"*64
    elif change == "unfinished":
        receipt["audit_finished"] = False
    elif change == "partial":
        receipt["complete"] = False
    elif change == "analysis dependency":
        del receipt["analysis_source_sha256"]["tools/synthetic-auditor-dependency.py"]
    elif change == "execution dependency":
        del receipt["execution_source_sha256"]["tools/synthetic-dependency.py"]
    elif change == "source hash":
        receipt["execution_source_sha256"][plotter.DRIVER] = "0"*64
    elif change == "plan hash":
        receipt["plan_sha256"] = "0"*64
    elif change == "metadata":
        receipt["scientific_metadata"]["period"] = 6
    elif change == "lifecycle":
        receipt["lifecycle"]["python_finalization"] = False
    elif change == "unreaped":
        receipt["lifecycle"]["controller"] = {"worker_reaped": False}
    elif change == "protocol":
        protocol = json.loads((run/"protocol.json").read_text())
        protocol["args"]["out"] = str(run.parent/"wrong")
        write_json(run/"protocol.json", protocol)
        entry = next(r for r in receipt["manifest"] if r["path"] == "protocol.json")
        entry.update(bytes=(run/"protocol.json").stat().st_size, sha256=plotter.digest(run/"protocol.json"))
    elif change == "raw changed":
        (run/"synthetic.bin").write_bytes(b"changed")
    elif change == "raw missing":
        (run/"synthetic.bin").unlink()
    elif change == "raw extra":
        (run/"extra.bin").write_bytes(b"unmanifested")
    elif change == "duplicate manifest":
        receipt["manifest"].append(copy.deepcopy(receipt["manifest"][0]))
    elif change == "manifest escape":
        receipt["manifest"][0]["path"] = "../escaped"
    else:
        (run/"link").symlink_to(run/"synthetic.bin")
    write_json(path, receipt)
    with pytest.raises(ValueError):
        plotter.load_receipt(path, expected or plotter.digest(path))
    if change in ("hash", "unfinished", "partial"):
        assert not calls


def test_explicit_partial_loader_still_requires_finished_audit(tmp_path, monkeypatch):
    path, receipt, _, _ = bound_fixture(tmp_path, monkeypatch, partial=True)
    _, data = plotter.load_receipt(path, plotter.digest(path), allow_partial=True)
    assert len(data["sections"]) == 1
    receipt["audit_finished"] = False
    write_json(path, receipt)
    with pytest.raises(ValueError, match="finished independent"):
        plotter.load_receipt(path, plotter.digest(path), allow_partial=True)
