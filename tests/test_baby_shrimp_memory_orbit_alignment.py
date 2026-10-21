"""Full-state alignment and frozen-evidence gates, without any model execution."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import baby_shrimp_memory_orbit_alignment as alignment


def save(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def orbit():
    result = np.zeros((18, 896))
    angle = 2 * np.pi * np.arange(18) / 18
    result[:, 0], result[:, 1] = np.cos(angle), np.sin(angle)
    return result


def sample(points, beta, tokens=None):
    return {"orbit": points, "tokens": list(range(18)) if tokens is None else tokens,
            "summary": {"parameter": beta}}


def frozen_receipt(tmp_path, monkeypatch, *, complete=False, terminal=True,
                   intervention="memory feedback", commit="b" * 40):
    data = tmp_path / "data"
    run, audit = data / "raw", data / "audit"
    run.mkdir(parents=True)
    audit.mkdir()
    plan = {"id": "BSC3-memory", "period": 18, "model_revision": "7ae557604adf67be50417f59c2c2f167def9a775",
            "start": [.073, -.15], "coordinate_scales": [2.5e-6, 7.5e-6],
            "anchor_directory": "anchor", "anchor_hashes": {"points.npy": "e" * 64},
            "intervention": intervention}
    blobs = {("a" * 40, alignment.AUDITOR): b"auditor fixture",
             (commit, "tools/baby_shrimp_saddle_memory.py"): b"driver fixture",
             (commit, "plan.json"): json.dumps(plan).encode()}
    monkeypatch.setattr(alignment, "git_bytes", lambda commit, path, root: blobs[(commit, str(path))])
    plan_sha = hashlib.sha256(blobs[(commit, "plan.json")]).hexdigest()
    save(run / "protocol.json", {"commit": commit, "plan": plan, "plan_sha256": plan_sha,
                                  "args": {"plan": "plan.json"}, "source_sha256": {
                                      "tools/baby_shrimp_saddle_memory.py": hashlib.sha256(b"driver fixture").hexdigest()}})
    if terminal:
        save(run / ("result.json" if complete else "failure.json"),
             {"complete": complete, "error": None if complete else "preserved trust-radius rejection"})
    centers, qualified = [], []
    tokens = [271, 785, 3405, 271, 785, 13885] * 3
    for index, beta in enumerate((.001, .003)):
        path = run / f"point-{index:04d}"
        path.mkdir()
        points = orbit()
        points[:, 2] = index * .001
        norm = np.linalg.norm(points[0])
        lower = np.linalg.norm(points[1:] - points[0], axis=1) / norm
        np.save(path / "points.npy", points)
        np.save(path / "return-state.npy", points[0])
        row = {"index": index, "qualified": True, "beta": beta, "coordinate": [index, -index],
               "alpha": .07 + index * 1e-6, "log10_temperature": -.15 - index * 1e-6,
               "phases": [{"phase": p} for p in (0, 6, 12)], "tokens": tokens,
               "primitive": {"closed": True, "no_lower_period": True, "all_returns": [*lower, 0.]}}
        save(path / "point.json", row)
        centers.append({"kind": "continued saddle", "parameter": beta, "converged": True,
                        "center": row["coordinate"], "cyclic_point_index": index})
        qualified.append({"index": index, "qualified": True})
    manifest = [{"path": p.relative_to(run).as_posix(), "sha256": alignment.digest(p), "bytes": p.stat().st_size}
                for p in sorted(run.rglob("*")) if p.is_file()]
    receipt = {"id": "BSC3-memory-offline-audit", "mode": "continue", "complete": complete,
               "status": "all saved-data gates passed" if complete else "partial attempt; qualified subset audited",
               "analysis_commit": "a" * 40, "analysis_source_sha256": hashlib.sha256(b"auditor fixture").hexdigest(),
               "execution_source_commit": commit, "plan_sha256": plan_sha, "raw_directory": str(run),
               "manifest": manifest, "accepted_centers": centers, "qualified_points": qualified, "worker": None}
    path = audit / "receipt.json"
    save(path, receipt)
    return path, alignment.digest(path), data, run, blobs


def test_cyclic_alignment_uses_full_states_and_reports_roll_convention():
    points = orbit()
    left = sample(points, 0.)
    right = sample(np.roll(points, 5, axis=0), .001, np.roll(np.arange(18), 5).tolist())
    comparison = alignment.compare_orbits(left, right)
    assert comparison["best_right_roll"] == 13
    assert comparison["maximum_relative_orbit_change"] == 0
    assert comparison["direct_phase_maximum_relative_change"] > 1
    assert comparison["second_best_maximum_relative_change"] > .3
    assert comparison["tokens_match_at_best_full_state_alignment"]


def test_same_tokens_do_not_mask_distinct_full_cycles_or_phase_ambiguity():
    points, shifted = orbit(), orbit()
    shifted[:, 2] += .2
    tokens = [1, 2, 3] * 6
    comparison = alignment.compare_orbits(sample(points, 0, tokens), sample(shifted, 0, tokens))
    assert comparison["tokens_match_at_best_full_state_alignment"]
    assert comparison["maximum_relative_orbit_change"] == pytest.approx(.2)
    assert comparison["comparison_kind"] == "same-parameter repeat"
    assert comparison["relative_change_per_parameter_step"] is None
    assert len(comparison["all_alignment_maximum_relative_changes"]) == 18
    assert "passed" not in comparison


def test_frozen_partial_attempt_keeps_accepted_centers_and_completeness_flag(tmp_path, monkeypatch):
    path, sha, data, run, blobs = frozen_receipt(tmp_path, monkeypatch)
    result = alignment.analyze([(path, sha)], data=data)
    assert result["accepted_cycles"] == 2
    assert not result["inputs"][0]["complete"]
    assert result["inputs"][0]["status"] == "partial attempt; qualified subset audited"
    assert result["summary"]["maximum_neighbor_relative_orbit_change"] == pytest.approx(.001)
    assert result["summary"]["all_best_alignments_preserve_phase"]
    assert all(p["closure_recomputed_from_saved_endpoint"]
               for p in (result["comparisons"][0]["left_point"], result["comparisons"][0]["right_point"]))


@pytest.mark.parametrize("change,match", [
    ("receipt", "explicit SHA-256"),
    ("raw", "hash/size differs"),
    ("additional", "unmanifested"),
    ("source", "execution source differs"),
])
def test_rejects_changed_evidence_or_sources(tmp_path, monkeypatch, change, match):
    path, sha, data, run, blobs = frozen_receipt(tmp_path, monkeypatch)
    if change == "receipt":
        path.write_text(path.read_text() + "\n")
    elif change == "raw":
        points = np.load(run / "point-0000" / "points.npy")
        points[0, 3] = .1
        np.save(run / "point-0000" / "points.npy", points)
    elif change == "additional":
        (run / "still-running.txt").write_text("new progress")
    else:
        blobs[("b" * 40, "tools/baby_shrimp_saddle_memory.py")] = b"changed driver"
    with pytest.raises(ValueError, match=match):
        alignment.analyze([(path, sha)], data=data)


def test_refuses_nonterminal_snapshot_even_with_matching_manifest(tmp_path, monkeypatch):
    path, sha, data, run, blobs = frozen_receipt(tmp_path, monkeypatch, terminal=False)
    with pytest.raises(ValueError, match="no terminal result/failure"):
        alignment.analyze([(path, sha)], data=data)


def test_rechecks_frozen_snapshot_after_comparison(tmp_path, monkeypatch):
    path, sha, data, run, blobs = frozen_receipt(tmp_path, monkeypatch)
    original = alignment.compare_orbits
    def changing_compare(left, right):
        (run / "late-file.txt").write_text("late output")
        return original(left, right)
    monkeypatch.setattr(alignment, "compare_orbits", changing_compare)
    with pytest.raises(ValueError, match="unmanifested"):
        alignment.analyze([(path, sha)], data=data)


def test_rejects_different_scientific_interventions_across_individually_bound_receipts(tmp_path, monkeypatch):
    one = frozen_receipt(tmp_path / "one", monkeypatch)
    two = frozen_receipt(tmp_path / "two", monkeypatch, intervention="different feedback", commit="c" * 40)
    blobs = {**one[4], **two[4]}
    monkeypatch.setattr(alignment, "git_bytes", lambda commit, path, root: blobs[(commit, str(path))])
    with pytest.raises(ValueError, match="scientific metadata differs across receipts"):
        alignment.analyze([(one[0], one[1]), (two[0], two[1])], data=tmp_path)


def test_rejects_missing_convergence_label_even_if_center_is_listed_as_accepted(tmp_path, monkeypatch):
    path, sha, data, run, blobs = frozen_receipt(tmp_path, monkeypatch)
    receipt = alignment.read(path)
    receipt["accepted_centers"][0]["converged"] = False
    save(path, receipt)
    with pytest.raises(ValueError, match="lacks a true convergence label"):
        alignment.analyze([(path, alignment.digest(path))], data=data)
