"""Adversarial saved-data gates; all fixtures are synthetic, with no model calls."""
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import baby_shrimp_memory_scan_finish as finish


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def record(path, base, key="path"):
    return {key: str(path.relative_to(base)), "sha256": finish.digest(path), "bytes": path.stat().st_size}


@pytest.fixture
def sample():
    plan = dict(model="test", layer=12, threads=8, prompt="open", opener="Hello!", window=1,
                dtype="float64", weights="native", vectors="fixture", concept="philosophy",
                transient=300, samples=300, record=80, max_period=32, tol=1e-8, eps=1e-6,
                chaos_level=.001, resolution=[2, 3], alpha=[.1, .100003],
                log10_temperature=[-.2, -.199991], beta=.0064, maximum_chunk=32)
    execution = dict(chunk=4, device="cpu", mode="scan")
    values = np.full((2, 3), -.3)
    data = {"lyapunov": values, "entropy": np.full((2, 3), .5), "period": np.full((2, 3), 2),
            "family": np.zeros((2, 3), dtype=int), "text_period": np.full((2, 3), 2),
            "text_family": np.zeros((2, 3), dtype=int), "alpha": np.linspace(*plan["alpha"], 3),
            "log10_temperature": np.linspace(*plan["log10_temperature"], 2)}
    partial = {k: (v.ravel().copy() if k in finish.FIELDS else v.copy()) for k, v in data.items()}
    partial.update(done=np.array(6), family_keys=np.array("[[12, 13]]"), text_keys=np.array("[[12, 13]]"))
    args = {k: plan[k] for k in finish.SETTINGS}
    args.update(command="scan", device="cpu", chunk=4, out="fixture", log_temperature=plan["log10_temperature"],
                feedback_memory=plan["beta"], prefix=None, start_vector=None, init_states=None,
                save_states=False, projections=False, word_period=0, return_period=0)
    row = dict(period=2, cells=6, tokens=["a", "b"], token_ids=[12, 13])
    info = dict(args=args, cells=6, seconds=1., period_counts={"2": 6}, lyapunov_positive=0,
                lyapunov_range=[-.3, -.3], families=[dict(family=0, **row)], texts=[dict(text_family=0, **row)])
    return data, partial, info, plan, execution


def gate(sample):
    data, partial, info, plan, execution = sample
    return finish.gate_arrays(data, partial, info, plan, plan["beta"], execution, "fixture")


def test_valid_arrays_reproduce_all_summaries(sample):
    result = gate(sample)
    assert result["cells"] == 6
    assert result["period_counts"] == {"2": 6}


@pytest.mark.parametrize("mutation,match", [
    ("nonfinite", "invalid field"), ("float_period", "recurrence"), ("large_period", "recurrence"),
    ("reversed_axis", "grid"), ("stale_checkpoint", "checkpoint field"), ("unfinished", "incomplete"),
    ("wrong_memory", "memory differs"), ("range", "range summary"), ("counts", "period summary"),
    ("keys", "token keys"), ("family", "family period"), ("warm_start", "fixed-opener"),
    ("missing_family", "period/family"), ("extra_array", "unexpected scan arrays")])
def test_rejects_corruption_even_when_other_summaries_look_plausible(sample, mutation, match):
    data, part, info, plan, execution = sample
    if mutation == "nonfinite": data["entropy"][0, 0] = np.nan
    elif mutation == "float_period":
        data["period"] = data["period"].astype(float); part["period"] = data["period"].ravel()
    elif mutation == "large_period":
        data["period"][0, 0] = 33; part["period"] = data["period"].ravel().copy()
    elif mutation == "reversed_axis": data["alpha"] = data["alpha"][::-1]
    elif mutation == "stale_checkpoint": part["lyapunov"][0] -= .01
    elif mutation == "unfinished": part["done"] = np.array(5)
    elif mutation == "wrong_memory": info["args"]["feedback_memory"] = 0.
    elif mutation == "range": info["lyapunov_range"] = [-.4, -.3]
    elif mutation == "counts": info["period_counts"] = {"2": 5, "0": 1}
    elif mutation == "keys": part["family_keys"] = np.array("[[12, 99]]")
    elif mutation == "family": info["families"][0]["period"] = 3
    elif mutation == "warm_start": info["args"]["init_states"] = "warm"
    elif mutation == "missing_family":
        data["family"][0, 0] = -1; part["family"] = data["family"].ravel().copy()
    elif mutation == "extra_array": data["made_up_crossing"] = np.zeros((2, 3))
    with pytest.raises(ValueError, match=match): gate(sample)


def test_progress_requires_every_chunk_in_order(tmp_path, sample):
    _, _, _, plan, execution = sample
    rows = [dict(chunk_start=start, phase=phase, step=step, seconds=i)
            for i, (start, phase, step) in enumerate((s, p, k) for s in (0, 4)
            for p in ("transient", "samples") for k in (100, 200, 300))]
    path = tmp_path / "progress.jsonl"
    path.write_text("\n".join(map(json.dumps, rows)))
    finish.gate_progress(path, plan, execution)
    path.write_text("\n".join(map(json.dumps, rows[:5] + rows[6:])))
    with pytest.raises(ValueError, match="progress missing"): finish.gate_progress(path, plan, execution)


@pytest.mark.parametrize("mutation", ["precision", "flag", "step", "closure", "chunk"])
def test_runtime_checks_numbers_not_just_success_flag(sample, mutation):
    _, _, _, plan, execution = sample
    runtime = dict(threads=8, rotary_float64=True, norm_sensitivity=3e-15, torch="2.test", cuda=None, device="cpu")
    eq = dict(passed=True, period=18, one_step_relative_error=0., saved_root_relative_closure=2e-13)
    finish.gate_runtime(runtime, eq, execution, plan)
    if mutation == "precision": runtime["norm_sensitivity"] = 0.
    elif mutation == "flag": runtime["rotary_float64"] = False
    elif mutation == "step": eq["one_step_relative_error"] = 2e-12
    elif mutation == "closure": eq["saved_root_relative_closure"] = 2e-10
    elif mutation == "chunk": execution["chunk"] = 33
    with pytest.raises(ValueError): finish.gate_runtime(runtime, eq, execution, plan)


def make_manifest(folder):
    names = ["execution.json", "runtime.json", "map-equivalence.json", "scan.npz",
             "scan-partial.npz", "scan.json", "progress.jsonl", "scan.png"]
    for name in names: (folder / name).write_bytes(b"fixture")
    rows = [record(folder / name, folder, "name") for name in names]
    result = dict(complete=True, mode="scan", files=rows)
    write(folder / "complete.json", result)
    return result


@pytest.mark.parametrize("mutation,match", [("tamper", "hash differs"), ("omit", "omits"),
        ("duplicate", "duplicate"), ("traverse", "top-level"), ("benchmark", "complete scan"),
        ("size", "size differs")])
def test_completion_manifest_fails_closed(tmp_path, mutation, match):
    value = make_manifest(tmp_path)
    finish.gate_manifest(tmp_path)
    if mutation == "tamper": (tmp_path / "scan.npz").write_bytes(b"altered")
    elif mutation == "omit": value["files"] = value["files"][:-1]
    elif mutation == "duplicate": value["files"].append(value["files"][0])
    elif mutation == "traverse": value["files"][0]["name"] = "../execution.json"
    elif mutation == "benchmark": value["mode"] = "benchmark"
    elif mutation == "size": value["files"][0]["bytes"] += 1
    write(tmp_path / "complete.json", value)
    with pytest.raises(ValueError, match=match): finish.gate_manifest(tmp_path)


def test_git_source_binding_checks_exact_bytes(monkeypatch):
    monkeypatch.setattr(finish, "git_bytes", lambda commit, path: b"old code")
    with pytest.raises(ValueError, match="Git source differs"):
        finish.verify_git("a" * 40, "worker.py", hashlib.sha256(b"new code").hexdigest())


@pytest.mark.parametrize("start,end,pid", [
    ("2026-10-22T01:00:00+00:00", "2026-10-22T00:00:00+00:00", 1),
    ("2026-10-22T00:00:00", "2026-10-22T01:00:00", 1),
    ("2026-10-22T00:00:00+00:00", "2026-10-22T01:00:00+00:00", True)])
def test_completion_times_reject_bad_metadata(start, end, pid):
    with pytest.raises(ValueError):
        finish.gate_completion_times(dict(started_utc=start, pid=pid), dict(completed_utc=end))


def test_same_growth_maps_to_same_gray_in_both_panels(sample):
    data = sample[0]
    scale = dict(shared_gray_contraction_cap=.6, shared_positive_vmax=.4)
    other = copy.deepcopy(data)
    other["lyapunov"][0, 0] = -.6
    left = finish.image_data(data, scale, .001)
    right = finish.image_data(other, scale, .001)
    assert np.array_equal(left[0, 1], right[0, 1])
    assert right[0, 0, 0] < right[0, 1, 0]
    assert np.allclose(right[0, 0], [.1, .1, .1, 1.])


def test_even_grid_does_not_claim_candidate_pixel():
    data = {"alpha": np.linspace(-1.5e-6, 1.5e-6, 128),
            "log10_temperature": np.linspace(-4.5e-6, 4.5e-6, 128)}
    sampling = finish.center_sampling(data, [0., 0.])
    assert sampling["center_is_measured_cell"] is False
    assert not any(a["contains_exact_center"] for a in sampling["axes"])
    assert sampling["axes"][0]["nearest_node_distance"] == pytest.approx(1.5e-6 / 127)
    assert sampling["axes"][1]["nearest_node_distance"] == pytest.approx(4.5e-6 / 127)


def test_center_coverage_requires_a_node_on_both_axes():
    data = {"alpha": np.array([-1., 0., 1.]), "log10_temperature": np.array([-2., 2.])}
    assert finish.center_sampling(data, [0., 0.])["center_is_measured_cell"] is False
    data["log10_temperature"] = np.array([-2., 0., 2.])
    assert finish.center_sampling(data, [0., 0.])["center_is_measured_cell"] is True


@pytest.fixture
def root_fixture(tmp_path, monkeypatch):
    repo, data = tmp_path / "repo", tmp_path / "data"
    repo.mkdir(); data.mkdir()
    monkeypatch.setattr(finish, "ROOT", repo); monkeypatch.setattr(finish, "DATA", data)
    monkeypatch.setattr(finish, "verify_git", lambda commit, path, expected: b"bound fixture")
    folder = data / "raw" / "point-0001"
    folder.mkdir(parents=True)
    orbit = np.zeros((18, 896)); orbit[:, 0] = 1.; orbit[:, 1] = np.arange(18) * .01
    np.save(folder / "points.npy", orbit); np.save(folder / "return-state.npy", orbit[0])
    write(folder / "point.json", dict(qualified=True, beta=.0064, alpha=.12, log10_temperature=-.2))
    raw = data / "raw"
    full = dict(complete=True, execution_source_commit="a"*40, plan_sha256="plan",
                height_root=dict(result=dict(saddle=dict(parameter=.0064, tau=1e-10,
                                 converged=True, center=[0., 0.]))),
                manifest=[record(p, raw) for p in folder.iterdir()])
    full_path = data / "audit.json"; write(full_path, full)
    compact = dict(complete=True, execution_source_commit="a"*40, plan_sha256="plan",
                   audit=record(full_path, data), raw_directory="raw",
                   scientific_metadata=dict(start=[.12, -.2], coordinate_scales=[1e-6, 3e-6],
                                            period=18, model_revision="test"),
                   height_root=dict(complete=True, parameter=.0064, tau=1e-10, height_tolerance=1e-8,
                                    alpha=.12, log10_temperature=-.2, cyclic_point_index=1))
    compact_path = repo / "root.json"; write(compact_path, compact)
    plan = dict(beta=.0064, center=[.12, -.2], model_revision="test",
                root_state=record(folder / "points.npy", data), root_receipt=record(compact_path, repo))
    return plan, compact, compact_path, full, full_path, folder


def test_root_replays_saved_closure_and_checks_candidate(root_fixture):
    plan, *_ = root_fixture
    out = finish.gate_root(plan, "b"*40)
    assert out["recomputed_saved_endpoint_closure"] == 0.
    assert out["minimum_saved_lower_lag"] > .009
    wrong = copy.deepcopy(plan); wrong["beta"] = 0.
    with pytest.raises(ValueError, match="root candidate differs"): finish.gate_root(wrong, "b"*40)


def test_root_primitive_failure_cannot_hide_behind_rehashed_receipts(root_fixture):
    plan, compact, compact_path, full, full_path, folder = root_fixture
    state = np.load(folder / "points.npy"); state[6] = state[0]
    np.save(folder / "points.npy", state)
    full["manifest"] = [record(p, folder.parent) for p in folder.iterdir()]
    write(full_path, full); compact["audit"] = record(full_path, full_path.parent)
    write(compact_path, compact); plan["root_receipt"] = record(compact_path, compact_path.parent)
    plan["root_state"] = record(folder / "points.npy", full_path.parent)
    with pytest.raises(ValueError, match="primitive/closure"):
        finish.gate_root(plan, "b"*40)
