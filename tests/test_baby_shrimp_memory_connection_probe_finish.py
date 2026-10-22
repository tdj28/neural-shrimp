"""Synthetic saved-evidence controls for the independent connection-probe audit."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import baby_shrimp_memory_connection_probe as driver
import baby_shrimp_memory_connection_probe_finish as finish


def plan():
    return dict(id="BSC4-direct-connection", period=18, maximum_evaluations=120,
                connection_probe=dict(
                    before_beta=.006, after_beta=.007,
                    axis_offsets=[-.1, -.05, .05, .1], stencil_steps=[.003, .001],
                    slice_offsets=[0., -.08, .08], root_bracket=.3,
                    maximum_correction_norm=3e-5, root_coordinate_tolerance=1e-7,
                    root_trace_tolerance=1e-10, root_max_iterations=20,
                    normal_offsets=[-.003, .003],
                    candidate_tilt_offsets=[-.0003, -.0001, .0001, .0003]))


class AnalyticPoints:
    """Keep every evaluated point under a stable raw index, as the real driver does."""

    center = np.array([.2, -.3])
    hessian = np.diag([-.04, .02])

    def __init__(self):
        self.raw, self.cache = {}, {}

    def __call__(self, coordinate, beta, cyclic=False):
        coordinate = np.asarray(coordinate, float)
        key = (*coordinate, beta, cyclic)
        if key in self.cache:
            return self.cache[key]
        x, y = coordinate-self.center
        tau = float(4e-5-.08*(beta-.006)-.02*x*x+.01*y*y)
        delta = float(-1e-8+1e-9*x)
        radius = float((np.sqrt(tau*tau-4*delta)+abs(tau))/2)
        row = dict(index=len(self.raw), coordinate=coordinate.tolist(), beta=beta,
                   qualified=True, tau=tau, delta=delta, radius=radius,
                   phases=[dict(phase=phase, tau=tau, delta=delta, radius=radius)
                           for phase in ((0, 6, 12) if cyclic else (6,))])
        self.raw[row["index"]] = row
        self.cache[key] = row
        return row


def evidence():
    p = plan()
    cfg = p["connection_probe"]
    evaluate = AnalyticPoints()
    protocol = dict(plan=p, candidate_tilt_direction=dict(
        direction=[-1., 0.], determinant_gradient=[1e-9, 0.]),
        verified_seeds=[dict(parameter=beta, center=evaluate.center.tolist(),
                             hessian=evaluate.hessian.tolist()) for beta in (.006, .007, .0065)])
    result = dict(complete=False, seed_replays=[], candidate_tilt=None,
                  axes=[], matched_stencils=[], roots=[])
    for beta in (.006, .007, .0065):
        point = evaluate(evaluate.center, beta, cyclic=True)
        result["seed_replays"].append(dict(fresh_point_index=point["index"], beta=beta, passed=True))
    for beta in (.006, .007):
        result["axes"].append(driver.axis_diagnostics(
            evaluate, evaluate.center, beta, evaluate.hessian, cfg["axis_offsets"]))
    result["candidate_tilt"] = driver.radius_tilt(
        evaluate, evaluate.center, .0065, [-1., 0.], cfg["candidate_tilt_offsets"], lambda row: None)
    result["matched_stencils"] = driver.matched_stencils(
        evaluate, evaluate.center, .007, cfg["stencil_steps"], cfg["maximum_correction_norm"])
    root_progress = []
    result["roots"] = driver.direct_slices(
        evaluate, evaluate.center, .007, result["matched_stencils"][-1], cfg,
        lambda rows: root_progress.append(copy.deepcopy(rows)))
    result["complete"] = True
    result["evaluations"] = len(evaluate.raw)
    return copy.deepcopy(result), copy.deepcopy(protocol), copy.deepcopy(evaluate.raw), root_progress


def test_complete_synthetic_probe_replays_from_its_raw_rows():
    result, protocol, raw, _ = evidence()
    checked = finish.recompute_measurements(result, protocol, raw)
    assert checked["all_measurement_stages_complete"]
    assert checked["untested_stages"] == []
    assert checked["candidate_tilt"]["status"] == "complete"
    for row in checked["candidate_tilt"]["symmetric_slopes"]:
        assert row["delta"] == pytest.approx(-1e-9, rel=1e-10)
        assert row["radius"] == pytest.approx(5e-6, rel=1e-7)
    assert [(row["negative_axis_brackets"], row["positive_axis_brackets"])
            for row in checked["axes"]] == [(2, 0), (0, 2)]
    for row in checked["matched_stencils"]:
        np.testing.assert_allclose(row["hessian"], AnalyticPoints.hessian, atol=1e-12)
        np.testing.assert_allclose(row["gradient"], [0., 0.], atol=1e-14)
    for row in checked["roots"]:
        assert row["status"] == "complete" and abs(row["tau"]) < 1e-10
        assert row["delta"] < 0 and row["radius"] > 0
        assert len(row["radius_offsets"]) == 2
    assert set(checked["used_qualified_point_indices"]) <= set(raw)
    assert len(result["roots"]) == 6 and result["evaluations"] <= 120


@pytest.mark.parametrize("change", ["index", "embedded value", "raw value", "unqualified"])
def test_embedded_points_must_match_the_indexed_qualified_raw_record(change):
    result, protocol, raw, _ = evidence()
    point = result["axes"][0]["axes"][0]["samples"][0]["point"]
    index = point["index"]
    if change == "index":
        point["index"] = (index+1) % len(raw)
    elif change == "embedded value":
        point["tau"] += 1e-6
    elif change == "raw value":
        raw[index]["tau"] += 1e-6
    else:
        point["qualified"] = raw[index]["qualified"] = False
    with pytest.raises(ValueError):
        finish.recompute_measurements(result, protocol, raw)


@pytest.mark.parametrize("change", ["direction", "brackets", "center", "eigenvalues"])
def test_axis_geometry_and_brackets_are_bound_to_the_prior_saddle(change):
    result, protocol, raw, _ = evidence()
    case = result["axes"][0]
    if change == "direction":
        case["axes"][0]["direction"] = [0., 1.]
    elif change == "brackets":
        case["axes"][0]["strict_sign_brackets"] = []
    elif change == "center":
        case["center"][0] += .001
    else:
        case["hessian_eigenvalues"][0] *= 2
    with pytest.raises(ValueError):
        finish.recompute_measurements(result, protocol, raw)


@pytest.mark.parametrize("change", ["hessian", "tau", "gradient", "correction", "norm", "index"])
def test_matched_stencil_derivatives_and_corrections_are_recomputed(change):
    result, protocol, raw, _ = evidence()
    row = result["matched_stencils"][-1]
    if change == "hessian":
        row["hessian"][0][0] *= 2
    elif change == "tau":
        row["tau"] += 1e-6
    elif change == "gradient":
        row["gradient"][0] += 1e-6
    elif change == "correction":
        row["correction"][0] += 1e-6
    elif change == "norm":
        row["correction_norm"] += 1e-6
    else:
        row["point_indices"][0][0], row["point_indices"][0][1] = (
            row["point_indices"][0][1], row["point_indices"][0][0])
    with pytest.raises(ValueError):
        finish.recompute_measurements(result, protocol, raw)


def test_fractional_stencil_index_cannot_be_silently_coerced_to_a_raw_index():
    result, protocol, raw, _ = evidence()
    result["matched_stencils"][0]["point_indices"][0][0] += .25
    with pytest.raises(ValueError):
        finish.recompute_measurements(result, protocol, raw)


@pytest.mark.parametrize("change", ["slope", "direction", "offset", "complete"])
def test_candidate_tilt_slopes_and_completeness_are_recomputed(change):
    result, protocol, raw, _ = evidence()
    tilt = result["candidate_tilt"]
    if change == "slope":
        tilt["symmetric_slopes"][0]["radius"] += 1e-6
    elif change == "direction":
        tilt["direction"] = [0., 1.]
    elif change == "offset":
        tilt["samples"][0]["offset"] *= 2
    else:
        tilt["samples"].pop()
    with pytest.raises(ValueError):
        finish.recompute_measurements(result, protocol, raw)


@pytest.mark.parametrize("change", ["basis", "coordinate", "normal", "radius difference",
                                    "comparison", "missing cyclic", "missing offset"])
def test_direct_root_geometry_and_radius_claims_are_checked(change):
    result, protocol, raw, _ = evidence()
    root = result["roots"][0]
    if change == "basis":
        root["slice_basis"] = np.eye(2).tolist()
    elif change == "coordinate":
        root["along"] += .001
    elif change == "normal":
        root["normal_direction"][0] += .01
    elif change == "radius difference":
        root["normal_offsets"][0]["radius_difference_from_root"] += 1e-5
    elif change == "comparison":
        root["sampled_radius_comparison"] = "invented stronger claim"
    elif change == "missing cyclic":
        del root["cyclic_point"]
    else:
        root["normal_offsets"].pop()
    with pytest.raises(ValueError):
        finish.recompute_measurements(result, protocol, raw)


def test_complete_probe_cannot_omit_an_entire_root():
    result, protocol, raw, _ = evidence()
    result["roots"].pop()
    with pytest.raises(ValueError):
        finish.recompute_measurements(result, protocol, raw)


@pytest.mark.parametrize("change", ["bracket", "bracket values", "iterations", "calls"])
def test_brent_search_is_replayed_from_actual_saved_evaluations(change):
    result, protocol, raw, _ = evidence()
    root = result["roots"][0]
    if change == "bracket":
        root["bracket"][0] *= .9
    elif change == "bracket values":
        root["bracket_values"][0] *= 1.1
    elif change == "iterations":
        root["iterations"] += 1
    else:
        root["function_calls"] += 1
    with pytest.raises(ValueError):
        finish.recompute_measurements(result, protocol, raw)


def test_missing_raw_bracket_endpoint_cannot_be_replaced_with_a_recorded_scalar():
    result, protocol, raw, _ = evidence()
    root = result["roots"][0]
    location = (np.array(root["center"])+np.array(root["slice_basis"])
                @ np.array([root["bracket"][0], root["transverse"]]))
    indices = [i for i, point in raw.items() if point["beta"] == root["beta"]
               and np.array_equal(point["coordinate"], location)]
    assert indices
    for index in indices:
        del raw[index]
    with pytest.raises(ValueError, match="lacks an evaluated coordinate"):
        finish.recompute_measurements(result, protocol, raw)


@pytest.mark.parametrize("change", ["trace", "phases"])
def test_raw_link_agreement_does_not_replace_cyclic_root_gates(change):
    result, protocol, raw, _ = evidence()
    point = result["roots"][0]["cyclic_point"]
    if change == "trace":
        point["tau"] = 2e-10
    else:
        point["phases"] = point["phases"][:1]
    raw[point["index"]] = copy.deepcopy(point)
    with pytest.raises(ValueError, match="cyclic|phases"):
        finish.recompute_measurements(result, protocol, raw)


@pytest.mark.parametrize("progress_index", [0, 1, 2, -1])
def test_real_partial_root_snapshots_remain_auditable(progress_index):
    result, protocol, raw, history = evidence()
    result["complete"] = False
    result["roots"] = history[progress_index]
    checked = finish.recompute_measurements(result, protocol, raw)
    assert checked["all_measurement_stages_complete"] == (progress_index == -1)
    assert len(checked["roots"]) == len(history[progress_index])
    if progress_index != -1:
        assert "all cyclic roots and finite radius offsets" in checked["untested_stages"]
        assert checked["roots"][-1]["status"].startswith("partial")


def test_partial_root_cannot_invent_a_transverse_direction_before_cyclic_replay():
    result, protocol, raw, history = evidence()
    result["complete"] = False
    result["roots"] = history[0]
    result["roots"][0]["normal_direction"] = [5., 5.]
    with pytest.raises(ValueError):
        finish.recompute_measurements(result, protocol, raw)


def test_partial_flag_does_not_hide_an_incorrect_present_radius_comparison():
    result, protocol, raw, history = evidence()
    result["complete"] = False
    result["roots"] = history[2]
    result["roots"][0]["sampled_radius_comparison"] = "invented stronger claim"
    with pytest.raises(ValueError):
        finish.recompute_measurements(result, protocol, raw)


@pytest.mark.parametrize("stage", ["empty", "one axis", "partial tilt"])
def test_not_started_and_partially_saved_stages_remain_explicit(stage):
    complete, protocol, raw, _ = evidence()
    result = dict(complete=False, seed_replays=[], candidate_tilt=None,
                  axes=[], matched_stencils=[], roots=[])
    if stage == "one axis":
        result["axes"] = complete["axes"][:1]
    elif stage == "partial tilt":
        result["axes"] = complete["axes"]
        result["candidate_tilt"] = complete["candidate_tilt"]
        result["candidate_tilt"].update(complete=False, symmetric_slopes=[])
        result["candidate_tilt"]["samples"] = result["candidate_tilt"]["samples"][:2]
    checked = finish.recompute_measurements(result, protocol, raw)
    assert not checked["all_measurement_stages_complete"]
    assert checked["matched_stencils"] == [] and checked["roots"] == []
    assert checked["candidate_tilt"]["status"] == (
        "partial; remaining offsets untested" if stage == "partial tilt" else "untested")


def write_json(path, value):
    path.write_text(json.dumps(value, allow_nan=False)+"\n")


def file_entry(path, run):
    return dict(path=path.relative_to(run).as_posix(), bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def terminal_fixture(tmp_path, monkeypatch, *, python_finalization=False):
    data = tmp_path/"external-data"
    run = data/"explore"/"probe"
    run.mkdir(parents=True)
    monkeypatch.setattr(finish, "DATA", data)
    protocol = dict(commit="a"*40, args=dict(seconds=5400, device="cuda", plan="docs/probe-plan.json",
                                            audit=str(data/"audits"/"prior.json"), audit_sha256="b"*64))
    write_json(run/"protocol.json", protocol)
    write_json(run/"probe-progress.json", {"complete": False})
    if python_finalization:
        write_json(run/"result.json", {"complete": True})
        entries = [file_entry(p, run) for p in sorted(run.iterdir())]
        write_json(run/"manifest.json", {"complete": True, "files": entries})
    controller = data/"controller"
    controller.mkdir()
    status = dict(worker_reaped=True, source_commit=protocol["commit"], output=str(run),
                  exit_code=0 if python_finalization else 124,
                  timed_out=not python_finalization, ended_utc="2026-10-22T01:00:00+00:00",
                  hard_seconds=5400, termination_grace_seconds=10,
                  worker_pid=1234, controller_pid=1233, elapsed_seconds=5401., complete=python_finalization,
                  command=[".venv/bin/python", "tools/baby_shrimp_memory_connection_probe.py",
                           "--source-commit", protocol["commit"], "--out", str(run),
                           "--plan", protocol["args"]["plan"],
                           "--device", protocol["args"]["device"], "--seconds", str(protocol["args"]["seconds"]),
                           "--audit", protocol["args"]["audit"],
                           "--audit-sha256", protocol["args"]["audit_sha256"]])
    manifest_path, status_path = controller/"manifest.json", controller/"status.json"
    write_json(manifest_path, [file_entry(p, run) for p in sorted(run.iterdir())])
    write_json(status_path, status)
    packet = dict(controller_manifest=manifest_path, manifest_sha=finish.digest(manifest_path),
                  controller_status=status_path, status_sha=finish.digest(status_path))
    return run, protocol, status, packet


def test_exact_python_finalized_snapshot_passes_without_a_controller(tmp_path, monkeypatch):
    run, protocol, _, _ = terminal_fixture(tmp_path, monkeypatch, python_finalization=True)
    checked = finish.terminal_evidence(run, protocol)
    assert checked["python_finalization"] and checked["python_complete"]
    assert checked["controller"] is None
    assert checked["python_manifest"]["sha256"] == finish.digest(run/"manifest.json")


@pytest.mark.parametrize("change", ["modified", "unmanifested", "missing", "symlink", "duplicate", "escape"])
def test_exact_terminal_manifest_rejects_mutation_and_invalid_paths(tmp_path, monkeypatch, change):
    run, _, _, packet = terminal_fixture(tmp_path, monkeypatch)
    entries = json.loads(packet["controller_manifest"].read_text())
    path = run/"probe-progress.json"
    if change == "modified":
        write_json(path, {"complete": True})
    elif change == "unmanifested":
        (run/"unexpected.txt").write_text("unlisted evidence")
    elif change == "missing":
        path.unlink()
    elif change == "symlink":
        outside = tmp_path/"progress-copy.json"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    elif change == "duplicate":
        entries.append(copy.deepcopy(entries[0]))
    else:
        entries[0]["path"] = "../escape.json"
    with pytest.raises(ValueError):
        finish.verify_files(run, entries)


def test_missing_python_manifest_requires_the_paired_hash_bound_controller_packet(tmp_path, monkeypatch):
    run, protocol, _, packet = terminal_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="hash-bound controller"):
        finish.terminal_evidence(run, protocol)
    for omitted in packet:
        incomplete = {key: value for key, value in packet.items() if key != omitted}
        with pytest.raises(ValueError, match="explicit paired hashes"):
            finish.terminal_evidence(run, protocol, **incomplete)
    checked = finish.terminal_evidence(run, protocol, **packet)
    assert not checked["python_finalization"] and not checked["python_complete"]
    assert checked["controller"]["worker_reaped"]
    assert checked["controller"]["exit_code"] == 124 and checked["controller"]["timed_out"]


@pytest.mark.parametrize("change", ["manifest hash", "status hash", "source", "unreaped", "output", "lifetime",
                                    "source flag", "out flag", "plan flag", "audit flag", "worker",
                                    "device flag", "seconds flag", "audit path flag"])
def test_controller_terminal_binding_refuses_stale_or_mismatched_evidence(tmp_path, monkeypatch, change):
    run, protocol, status, packet = terminal_fixture(tmp_path, monkeypatch)
    if change in ("manifest hash", "status hash"):
        packet["manifest_sha" if change == "manifest hash" else "status_sha"] = "f"*64
    else:
        if change == "source":
            status["source_commit"] = "c"*40
        elif change == "unreaped":
            status["worker_reaped"] = False
        elif change == "output":
            status["output"] = str(run.parent/"different-attempt")
        elif change == "lifetime":
            status["hard_seconds"] = 5401
        elif change == "worker":
            status["command"][1] = "tools/different_worker.py"
        else:
            flag = {"source flag": "--source-commit", "out flag": "--out",
                    "plan flag": "--plan", "audit flag": "--audit-sha256",
                    "device flag": "--device", "seconds flag": "--seconds", "audit path flag": "--audit"}[change]
            status["command"][status["command"].index(flag)+1] = "different"
        write_json(packet["controller_status"], status)
        packet["status_sha"] = finish.digest(packet["controller_status"])
    with pytest.raises(ValueError):
        finish.terminal_evidence(run, protocol, **packet)


def test_python_manifest_cannot_claim_finalization_without_a_terminal_outcome(tmp_path, monkeypatch):
    run, protocol, _, _ = terminal_fixture(tmp_path, monkeypatch, python_finalization=True)
    (run/"result.json").unlink()
    entries = [file_entry(p, run) for p in sorted(run.iterdir()) if p.name != "manifest.json"]
    write_json(run/"manifest.json", {"complete": True, "files": entries})
    with pytest.raises(ValueError, match="lacks terminal outcome"):
        finish.terminal_evidence(run, protocol)


def refresh_terminal_packet(run, status, packet):
    if (run/"manifest.json").exists():
        manifest = json.loads((run/"manifest.json").read_text())
        manifest["files"] = [file_entry(p, run) for p in sorted(run.rglob("*"))
                             if p.is_file() and p.name != "manifest.json"]
        write_json(run/"manifest.json", manifest)
    write_json(packet["controller_manifest"], [file_entry(p, run) for p in sorted(run.rglob("*")) if p.is_file()])
    write_json(packet["controller_status"], status)
    packet["manifest_sha"] = finish.digest(packet["controller_manifest"])
    packet["status_sha"] = finish.digest(packet["controller_status"])


@pytest.mark.parametrize("change", ["manifest incomplete", "result incomplete", "result missing", "failure present"])
def test_python_manifest_and_terminal_result_must_agree(tmp_path, monkeypatch, change):
    run, protocol, status, packet = terminal_fixture(tmp_path, monkeypatch, python_finalization=True)
    if change == "manifest incomplete":
        manifest = json.loads((run/"manifest.json").read_text())
        manifest["complete"] = False
        write_json(run/"manifest.json", manifest)
    elif change == "result incomplete":
        write_json(run/"result.json", {"complete": False})
    elif change == "result missing":
        (run/"result.json").unlink()
        write_json(run/"failure.json", {"type": "TimeoutError", "error": "synthetic timeout"})
    else:
        write_json(run/"failure.json", {"type": "ValueError", "error": "synthetic failure"})
    refresh_terminal_packet(run, status, packet)
    with pytest.raises(ValueError):
        finish.terminal_evidence(run, protocol)


@pytest.mark.parametrize("change", ["complete false", "failed exit", "timed out", "claimed success without result"])
def test_controller_completion_cannot_contradict_exit_timeout_or_result(tmp_path, monkeypatch, change):
    finalized = change != "claimed success without result"
    run, protocol, status, packet = terminal_fixture(tmp_path, monkeypatch, python_finalization=finalized)
    if change == "complete false":
        status["complete"] = False
    elif change == "failed exit":
        status["exit_code"] = 1
    elif change == "timed out":
        status["timed_out"] = True
    else:
        status.update(complete=True, exit_code=0, timed_out=False)
    refresh_terminal_packet(run, status, packet)
    with pytest.raises(ValueError):
        finish.terminal_evidence(run, protocol, **packet)


def test_controller_only_success_cannot_wrap_an_incomplete_result(tmp_path, monkeypatch):
    run, protocol, status, packet = terminal_fixture(tmp_path, monkeypatch)
    write_json(run/"result.json", {"complete": False})
    status.update(complete=True, exit_code=0, timed_out=False)
    refresh_terminal_packet(run, status, packet)
    with pytest.raises(ValueError):
        finish.terminal_evidence(run, protocol, **packet)


@pytest.mark.parametrize("flag", ["--device", "--seconds", "--audit"])
@pytest.mark.parametrize("change", ["missing", "duplicate", "missing value"])
def test_controller_binding_requires_one_complete_copy_of_each_runtime_flag(tmp_path, monkeypatch, flag, change):
    run, protocol, status, packet = terminal_fixture(tmp_path, monkeypatch)
    index = status["command"].index(flag)
    pair = status["command"][index:index+2]
    if change == "duplicate":
        status["command"].extend(pair)
    else:
        del status["command"][index:index+2]
        if change == "missing value":
            status["command"].append(flag)
    refresh_terminal_packet(run, status, packet)
    with pytest.raises(ValueError, match="controller command"):
        finish.terminal_evidence(run, protocol, **packet)


def execution_fixture(tmp_path, monkeypatch):
    monkeypatch.setattr(finish, "DATA", tmp_path)
    anchor = tmp_path/"anchor"
    anchor.mkdir()
    p = plan()
    p.update(anchor_directory="anchor", anchor_hashes={})
    path = "docs/probe-plan.json"
    blobs = {name: ("synthetic frozen source: "+name).encode() for name in driver.SOURCES}
    blobs[path] = json.dumps(p).encode()
    protocol = dict(commit="a"*40, args=dict(plan=path, seconds=5400), plan=p,
                    plan_sha256=hashlib.sha256(blobs[path]).hexdigest(),
                    source_sha256={name: hashlib.sha256(blobs[name]).hexdigest() for name in driver.SOURCES})
    calls = []
    def git_bytes(commit, name):
        calls.append((commit, name))
        return blobs[name]
    monkeypatch.setattr(finish.point_audit, "git_bytes", git_bytes)
    return protocol, anchor, calls


def test_execution_binding_requires_all_sixteen_frozen_driver_sources(tmp_path, monkeypatch):
    protocol, anchor, calls = execution_fixture(tmp_path, monkeypatch)
    assert len(driver.SOURCES) == 16
    assert finish.bind_execution(protocol) == anchor
    assert {path for _, path in calls} == {*driver.SOURCES, protocol["args"]["plan"]}


@pytest.mark.parametrize("missing", driver.SOURCES)
def test_no_frozen_execution_source_may_be_omitted(tmp_path, monkeypatch, missing):
    protocol, _, _ = execution_fixture(tmp_path, monkeypatch)
    del protocol["source_sha256"][missing]
    with pytest.raises(ValueError, match="source"):
        finish.bind_execution(protocol)


@pytest.mark.parametrize("commit", ["HEAD", "a"*39, "a"*41, "A"*40, "g"*40, "a"*39+"\n"])
def test_execution_source_commit_must_be_exact_lowercase_hex(tmp_path, monkeypatch, commit):
    protocol, _, calls = execution_fixture(tmp_path, monkeypatch)
    protocol["commit"] = commit
    with pytest.raises(ValueError, match="commit"):
        finish.bind_execution(protocol)
    assert calls == []


def runtime_fixture(tmp_path, monkeypatch, stage, with_runtime):
    """Isolate run_audit's runtime gate; numerical/source audits have their own tests."""
    data = tmp_path/"runtime-data"
    run = data/"explore"/"probe"
    run.mkdir(parents=True)
    monkeypatch.setattr(finish, "DATA", data)
    p = plan()
    p["prior_audit"] = {}
    p.update({key: p.get(key) for key in finish.binding.METADATA})
    protocol = dict(plan=p, commit="a"*40, plan_sha256="c"*64, args={"plan": "docs/probe-plan.json"},
                    source_sha256={path: "d"*64 for path in driver.SOURCES})
    write_json(run/"protocol.json", protocol)
    complete = stage == "complete"
    write_json(run/("result.json" if complete else "probe-progress.json"),
               {"complete": complete, "evaluations": 0})
    if stage in ("qualified", "unqualified", "empty point"):
        directory = run/"point-0000"
        directory.mkdir()
        if stage != "empty point":
            write_json(directory/"point.json", dict(index=0, qualified=stage == "qualified",
                                                    beta=.007, coordinate=[0., 0.]))
    if with_runtime:
        write_json(run/"runtime.json", dict(norm_sensitivity=1e-15, rotary_float64=True, threads=8))
    lifecycle = dict(python_finalization=True, python_complete=complete, controller=None)
    monkeypatch.setattr(finish, "bind_execution", lambda protocol: data/"anchor")
    monkeypatch.setattr(finish, "bind_seeds", lambda protocol: ({}, []))
    monkeypatch.setattr(finish, "terminal_evidence", lambda *args: lifecycle)
    monkeypatch.setattr(finish, "audit_replays", lambda *args: [])
    monkeypatch.setattr(finish, "recompute_measurements", lambda *args: {
        "all_measurement_stages_complete": complete, "untested_stages": [] if complete else ["candidate tilt"],
        "candidate_tilt": {"status": "untested"}, "axes": [], "matched_stencils": [], "roots": [],
        "plot_data": {"candidate_tilt": None, "axes": [], "roots": []}})
    monkeypatch.setattr(finish.point_audit, "audit_point", lambda directory, plan:
        (json.loads((directory/"point.json").read_text()), {"closure_recomputed_from_saved_endpoint": True}))
    return run


@pytest.mark.parametrize("stage", ["qualified", "unqualified", "empty point", "complete"])
def test_numerical_work_or_complete_output_requires_runtime_evidence(tmp_path, monkeypatch, stage):
    run = runtime_fixture(tmp_path, monkeypatch, stage, with_runtime=False)
    with pytest.raises(ValueError, match="runtime"):
        finish.run_audit(run, allow_partial=True)


def test_pre_evaluation_partial_attempt_may_lack_runtime_evidence(tmp_path, monkeypatch):
    run = runtime_fixture(tmp_path, monkeypatch, "preflight", with_runtime=False)
    receipt = finish.run_audit(run, allow_partial=True)
    assert not receipt["complete"] and receipt["runtime"] is None


@pytest.mark.parametrize("stage", ["qualified", "complete"])
def test_runtime_gate_accepts_the_saved_effective_precision_record(tmp_path, monkeypatch, stage):
    run = runtime_fixture(tmp_path, monkeypatch, stage, with_runtime=True)
    receipt = finish.run_audit(run, allow_partial=True)
    assert receipt["runtime"]["rotary_float64"] and receipt["runtime"]["threads"] == 8
    assert receipt["complete"] == (stage == "complete")
