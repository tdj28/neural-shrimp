import importlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from neural_shrimp.trainability import array_sha256, edge_rates, make_s2

ROOT = Path(__file__).resolve().parents[1]


def _runner():
    # Import by name so spawned workers can re-import the module from sys.path.
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    return importlib.import_module("run_exp028")


def _synthetic_plan(seed=99991, resolution=16):
    """A tiny artificial plan (seed outside EXP-028) exercising every stage."""
    net, x, y, theta0 = make_s2(seed)
    hidden, output = edge_rates(net, theta0, x, y)
    spec = {"map_index": 0, "map_id": f"S2-{seed}", "system": "S2", "seed": seed,
            "n_parameters": net.n_parameters, "n_examples": len(x),
            "network": {"d": net.d, "width": net.width, "activation": net.activation,
                        "in_scale": net.in_scale, "out_scale": net.out_scale, "bias": net.bias,
                        "loss_scale": net.loss_scale, "ridge": net.ridge},
            "data_sha256": {"x": array_sha256(x), "y": array_sha256(y),
                            "theta0": array_sha256(theta0)},
            "log10_hidden": {"center": math.log10(hidden), "half_width": 1.5,
                             "resolution": resolution},
            "log10_output": {"center": math.log10(output), "half_width": 1.5,
                             "resolution": resolution}}
    return {
        "experiment": "SYNTHETIC", "maps": [spec],
        "iteration": {"steps": 800, "tail_steps": 48, "max_period": 8, "min_repeats": 6,
                      "atol": 1e-8, "rtol": 1e-6, "ftle_steps": 200, "ftle_align": 50,
                      "tangent_seed": 0, "chaos_threshold": 5e-3, "divergence_threshold": 1e8,
                      "sd_horizons": [20, 100]},
        "zoom": {"window": 8, "stride": 4, "resolution": 16, "map_index_offset": 1},
        "certification": {"min_component_size": 2, "max_components": 3, "max_fixed_components": 2,
                          "atol": 1e-9, "rtol": 1e-8, "root_tolerance": 1e-10,
                          "stability_margin": 1e-7, "candidate_tolerance": 1e-3,
                          "max_evaluations": 2000},
        "chaos_diagnostics": {"count": 2, "seed_base": 2810, "steps": 800, "ftle_steps": 200,
                              "ftle_align": 50, "tangent_seed": 1, "tail_steps": 48,
                              "chaos_threshold": 5e-3},
        "resources": {"workers": 2, "chunk_cells": 64, "wall_ceiling_seconds": 600},
        "gate": {"min_period": 3, "requires_attracting": True, "requires_borders_chaos": True},
        "label": "synthetic fixture", "output": {"raw_root": "unused", "attempt": "unused"},
        "audit": {"resimulated_cells_per_coarse_map": 10, "audit_seed": 2806},
    }


def test_runner_executes_all_stages_on_an_artificial_plan(tmp_path):
    runner = _runner()
    plan = _synthetic_plan()
    out = tmp_path / "attempt-001"
    status = runner.execute(plan, "synthetic-plan", out, workers=2)
    assert status == "complete"
    end = json.loads((out / "runtime-end.json").read_text())
    assert end["status"] == "complete" and end["cleanup"]["alive_after_cleanup"] == []
    saved = np.load(out / "stage1" / "S2-99991.npz")
    assert saved["status"].shape == (16, 16)
    assert saved["final_state"].shape == (16, 16, 13)
    assert len(saved["log10_hidden"]) == 16
    zoom = json.loads((out / "stage1" / "S2-99991-zoom.json").read_text())
    if zoom["window"] is not None:
        assert (out / "stage2" / "S2-99991-zoom.npz").exists()
    certificates = json.loads((out / "stage3" / "certificates.json").read_text())
    for record in certificates:
        assert record["kind"] in {"periodic", "fixed"}
        assert "verified" in record["certificate"]
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["status"] == "complete"
    paths = {item["path"] for item in receipt["files"]}
    assert "stage1/S2-99991.npz" in paths and "runtime-end.json" in paths
    for item in receipt["files"]:
        assert runner.sha256_file(out / item["path"]) == item["sha256"]
    # Chunks reassemble the same cells as a single direct run.
    chunk = np.load(out / "stage1" / "S2-99991" / "chunk-0000.npz")
    np.testing.assert_array_equal(chunk["status"], saved["status"].ravel()[:64])


def test_runner_refuses_an_existing_attempt_directory(tmp_path):
    runner = _runner()
    out = tmp_path / "attempt-001"
    out.mkdir()
    with pytest.raises(FileExistsError):
        runner.execute(_synthetic_plan(), "synthetic-plan", out, workers=1)


def test_input_hash_mismatch_fails_closed():
    runner = _runner()
    plan = _synthetic_plan()
    plan["maps"][0]["data_sha256"]["x"] = "0" * 64
    with pytest.raises(runner.RunFailure):
        runner.system_inputs(plan["maps"][0])


def test_certify_worker_verifies_a_converged_fixed_point():
    from neural_shrimp.gdmaps import STATUS_FIXED, GridMap, run_grid

    runner = _runner()
    plan = _synthetic_plan()
    spec = plan["maps"][0]
    net, x, y, theta0 = make_s2(spec["seed"])
    grid = GridMap(net, x, y, [0.5], [0.5])
    result = run_grid(grid, theta0, 20000, tail_steps=48, max_period=8, ftle_steps=100,
                      ftle_align=10, sd_horizons=())
    assert result["status"][0] == STATUS_FIXED
    task = {"spec": spec, "map_id": spec["map_id"], "kind": "fixed",
            "record": {"component": 0, "period": 1, "size": 1, "first_cell": 0,
                       "representative": 0, "borders_chaos": False},
            "lr_hidden": 0.5, "lr_output": 0.5, "seed_state": result["final_state"][0].tolist(),
            "period": 1, "certification": plan["certification"]}
    record = runner.certify_worker(task)
    assert record["certificate"]["verified"] and record["certificate"]["attracting"]
    assert record["seed_state_sha256"] == array_sha256(result["final_state"][0])


def test_analysis_summarizes_and_draws_a_synthetic_attempt(tmp_path, monkeypatch):
    runner = _runner()
    analysis = importlib.import_module("analyze_exp028")
    plan = _synthetic_plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    plan_sha = runner.sha256_bytes(plan_path.read_bytes())
    out = tmp_path / "attempt-001"
    assert runner.execute(plan, plan_sha, out, workers=2) == "complete"
    summary_path = tmp_path / "summary.json"
    monkeypatch.setattr(sys, "argv", ["analyze", "--plan", str(plan_path), "--raw", str(out),
                                      "--summary", str(summary_path),
                                      "--figures", str(tmp_path / "figures")])
    analysis.main()
    summary = json.loads(summary_path.read_text())
    stage1 = summary["maps"]["S2-99991"]
    assert sum(stage1["counts"].values()) == 256
    assert Path(stage1["figure"]["path"]).exists()
    crosstab = stage1["sd_crosstab"]["100"]
    total = sum(crosstab["sd_convergent"].values()) + sum(crosstab["sd_divergent"].values())
    assert total == 256
    assert isinstance(summary["gate"]["positive"], bool)


def test_independent_audit_agrees_with_a_synthetic_attempt(tmp_path, monkeypatch):
    runner = _runner()
    audit = importlib.import_module("audit_exp028")
    plan = _synthetic_plan()
    plan["iteration"]["steps"] = 3000
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    out = tmp_path / "attempt-001"
    assert runner.execute(plan, runner.sha256_bytes(plan_path.read_bytes()), out, workers=2) == "complete"
    audit_dir = tmp_path / "audit"
    monkeypatch.setattr(sys, "argv", ["audit", "--plan", str(plan_path), "--raw", str(out),
                                      "--out", str(audit_dir)])
    audit.main()
    report = json.loads((audit_dir / "audit.json").read_text())
    assert report["checks"]["receipt"]["mismatched"] == []
    assert report["checks"]["receipt"]["plan_hash_matches"]
    coarse = report["checks"]["maps"]["S2-99991"]
    assert coarse["inputs_hash_ok"] and coarse["zoom_rule_ok"]
    assert coarse["selection_ok"] and coarse["certificate_seeds_ok"]
    resim = coarse["resimulation"]
    assert resim["regular_label_agreement"] == resim["regular_count"]
    for item in coarse["certificate_rechecks"]:
        if item["checked"]:
            assert item["closure_ok"] and item["radius_ok"] and item["attracting_consistent"]
