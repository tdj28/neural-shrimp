import importlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from neural_shrimp.gdmaps import STATUS_FIXED, GridMap, run_grid
from neural_shrimp.trainability import array_sha256, edge_rates, make_s2

ROOT = Path(__file__).resolve().parents[1]


def _tool(name):
    # Import by name so spawned workers can re-import the module from sys.path.
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    return importlib.import_module(name)


def _runner():
    return _tool("run_exp028")


def _crash(_task):
    os._exit(17)


def _sleep(_task):
    time.sleep(30)
    return {"pid": os.getpid()}


def _synthetic_plan(seed=99991, resolution=16, steps=800):
    """A tiny artificial plan (seed outside EXP-028) exercising every stage."""
    net, x, y, theta0 = make_s2(seed)
    hidden, output = edge_rates(net, theta0, x, y)
    spec = {"map_index": 0, "map_id": f"S2-{seed}", "system": "S2", "seed": seed,
            "pipeline": "full", "zoom_map_index": 1,
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
        "experiment": "SYNTHETIC", "label": "synthetic fixture", "maps": [spec],
        "iteration": {"steps": steps, "tail_steps": 48, "max_period": 8, "min_repeats": 6,
                      "atol": 1e-8, "rtol": 1e-6, "ftle_steps": 200, "ftle_align": 50,
                      "tangent_seed": 0, "chaos_threshold": 5e-3, "drift_threshold": 1e-2,
                      "divergence_threshold": 1e8, "sd_horizons": [20, 100]},
        "zoom": {"window": 8, "stride": 4, "resolution": 16},
        "certification": {"min_component_size": 2, "max_embedded": 3, "max_other": 3,
                          "max_fixed": 2, "min_chaotic_fraction": 0.5, "min_embedded_period": 3,
                          "atol": 1e-9, "rtol": 1e-8, "newton_tolerance": 1e-10,
                          "stability_margin": 1e-7, "candidate_tolerance": 1e-3,
                          "divisor_separation": 1e-6, "max_iterations": 60},
        "ring_confirmation": {"max_cells_per_component": 4, "order": "row-major", "steps": 800,
                              "ftle_steps": 200, "ftle_align": 50, "tangent_seed": 1,
                              "tail_steps": 48, "chaos_threshold": 5e-3},
        "chaos_diagnostics": {"count": 2, "seed_base": 2810, "steps": 800, "ftle_steps": 200,
                              "ftle_align": 50, "tangent_seed": 1, "tail_steps": 48,
                              "chaos_threshold": 5e-3},
        "gate": {"min_period": 3, "requires_attracting": True, "min_confirmed_ring_fraction": 0.5,
                 "requires_audit_confirmation": True},
        "audit": {"resimulated_cells_per_coarse_map": 12, "audit_seed": 2806},
        "resources": {"workers": 2, "chunk_cells": 64, "wall_ceiling_seconds": 600},
        "output": {"raw_root": "unused", "attempt": "unused"},
    }


def _execute(tmp_path, plan, workers=2):
    runner = _runner()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    out = tmp_path / "attempt-001"
    status = runner.execute(plan, runner.sha256_bytes(plan_path.read_bytes()), out, workers=workers)
    return runner, plan_path, out, status


def test_runner_executes_all_stages_on_an_artificial_plan(tmp_path):
    runner, _, out, status = _execute(tmp_path, _synthetic_plan())
    assert status == "complete"
    end = json.loads((out / "runtime-end.json").read_text())
    assert end["status"] == "complete" and end["cleanup"]["alive_after_cleanup"] == []
    saved = np.load(out / "stage1" / "S2-99991.npz")
    assert saved["status"].shape == (16, 16)
    assert saved["final_state"].shape == (16, 16, 13)
    assert saved["lr_hidden"].shape == (16, 16)
    zoom = json.loads((out / "stage1" / "S2-99991-zoom.json").read_text())
    if zoom["window"] is not None:
        assert (out / "stage2" / "S2-99991-zoom.npz").exists()
    for name in ("certificates.json", "ring-confirmation.json", "chaos-diagnostics.json"):
        assert (out / "stage3" / name).exists()
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["status"] == "complete"
    for item in receipt["files"]:
        assert runner.sha256_file(out / item["path"]) == item["sha256"]
    chunk = np.load(out / "stage1" / "S2-99991" / "chunk-0000.npz")
    np.testing.assert_array_equal(chunk["status"], saved["status"].ravel()[:64])


def test_descriptive_maps_get_no_zoom_or_certification(tmp_path):
    plan = _synthetic_plan()
    plan["maps"][0]["pipeline"] = "descriptive"
    del plan["maps"][0]["zoom_map_index"]
    _, _, out, status = _execute(tmp_path, plan)
    assert status == "complete"
    assert not (out / "stage1" / "S2-99991-zoom.json").exists()
    assert json.loads((out / "stage3" / "certificates.json").read_text()) == []


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


def test_a_killed_worker_fails_promptly_instead_of_waiting_for_the_ceiling():
    runner = _runner()
    test_module = importlib.import_module("test_run_exp028")
    executor = runner.Executor(1, 120, time.monotonic())
    start = time.monotonic()
    with pytest.raises(Exception) as caught:
        executor.map(test_module._crash, [{}])
    assert type(caught.value).__name__ == "BrokenProcessPool"
    executor.close(terminate=True)
    assert time.monotonic() - start < 60


def test_wall_ceiling_terminates_owned_workers():
    runner = _runner()
    test_module = importlib.import_module("test_run_exp028")
    executor = runner.Executor(1, 2, time.monotonic())
    with pytest.raises(runner.RunFailure):
        executor.map(test_module._sleep, [{}])
    cleanup = executor.close(terminate=True)
    assert cleanup["alive_after_cleanup"] == []


def test_certify_worker_verifies_a_converged_fixed_point():
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
                       "representative": 0, "ring_cells": [], "ring_divisor_cells": [],
                       "ring_chaotic_cells": [], "ring_chaotic_fraction": 0.0},
            "lr_hidden": 0.5, "lr_output": 0.5, "seed_state": result["final_state"][0].tolist(),
            "period": 1, "certification": plan["certification"]}
    record = runner.certify_worker(task)
    assert record["error"] is None
    assert record["certificate"]["verified"] and record["certificate"]["attracting"]
    assert record["seed_state_sha256"] == array_sha256(result["final_state"][0])


def test_certify_worker_records_errors_per_task():
    runner = _runner()
    plan = _synthetic_plan()
    task = {"spec": {**plan["maps"][0], "data_sha256": {"x": "0", "y": "0", "theta0": "0"}},
            "map_id": "S2-99991", "kind": "fixed", "record": {"component": 0, "period": 1},
            "lr_hidden": 0.5, "lr_output": 0.5, "seed_state": [0.0] * 13, "period": 1,
            "certification": plan["certification"]}
    record = runner.certify_worker(task)
    assert record["certificate"] is None and record["error"]["type"] == "RunFailure"


def test_analysis_summarizes_and_refuses_to_overwrite(tmp_path, monkeypatch):
    analysis = _tool("analyze_exp028")
    _, plan_path, out, status = _execute(tmp_path, _synthetic_plan())
    assert status == "complete"
    summary_path = tmp_path / "summary.json"
    argv = ["analyze", "--plan", str(plan_path), "--raw", str(out), "--summary", str(summary_path),
            "--figures", str(tmp_path / "figures")]
    monkeypatch.setattr(sys, "argv", argv)
    analysis.main()
    summary = json.loads(summary_path.read_text())
    stage1 = summary["maps"]["S2-99991"]
    assert sum(stage1["counts"].values()) == 256
    assert Path(stage1["figure"]["path"]).exists()
    crosstab = stage1["sd_crosstab"]["100"]
    assert sum(crosstab["sd_convergent"].values()) + sum(crosstab["sd_divergent"].values()) == 256
    assert isinstance(summary["gate"]["producer_positive"], bool)
    with pytest.raises(SystemExit):
        analysis.main()


def test_analysis_rejects_an_incomplete_attempt(tmp_path, monkeypatch):
    analysis = _tool("analyze_exp028")
    runner, plan_path, out, _ = _execute(tmp_path, _synthetic_plan())
    receipt = json.loads((out / "receipt.json").read_text())
    receipt["status"] = "failed"
    (out / "receipt.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(sys, "argv", ["analyze", "--plan", str(plan_path), "--raw", str(out),
                                      "--summary", str(tmp_path / "s.json"),
                                      "--figures", str(tmp_path / "f")])
    with pytest.raises(SystemExit):
        analysis.main()


def test_independent_audit_agrees_with_a_synthetic_attempt(tmp_path, monkeypatch):
    audit = _tool("audit_exp028")
    _, plan_path, out, status = _execute(tmp_path, _synthetic_plan(steps=3000))
    assert status == "complete"
    audit_dir = tmp_path / "audit"
    monkeypatch.setattr(sys, "argv", ["audit", "--plan", str(plan_path), "--raw", str(out),
                                      "--out", str(audit_dir)])
    audit.main()
    report = json.loads((audit_dir / "audit.json").read_text())
    assert report["checks"]["receipt"]["mismatched"] == []
    assert report["checks"]["receipt"]["plan_hash_matches"]
    coarse = report["checks"]["maps"]["S2-99991"]
    assert coarse["inputs_hash_ok"] and coarse["zoom_rule_ok"]
    if "zoom_axes_ok" in coarse:
        assert coarse["zoom_axes_ok"]
    resim = coarse["resimulation"]
    assert resim["regular_label_agreement"] == resim["regular_count"]
    for entry in report["checks"]["maps"].values():
        for check in entry.get("components", []):
            assert check["size_ok"] and check["ring_ok"] and check["seed_ok"]
            assert check["rates_from_axes_ok"]
            if check["checked"]:
                assert check["closure_ok"] and check["radius_ok"] and check["attracting_consistent"]
    assert isinstance(report["audit_gate"]["confirmed_positive"], bool)


@pytest.mark.skipif(os.environ.get("NEURAL_SHRIMP_SLOW") != "1",
                    reason="slow end-to-end embedded-window fixture; set NEURAL_SHRIMP_SLOW=1")
def test_slow_embedded_window_path_end_to_end(tmp_path, monkeypatch):
    """Artificial S2 fixture large enough to contain embedded windows."""
    audit = _tool("audit_exp028")
    analysis = _tool("analyze_exp028")
    plan = _synthetic_plan(resolution=96, steps=20000)
    plan["iteration"].update(tail_steps=384, max_period=64, ftle_steps=4000, ftle_align=1000,
                             sd_horizons=[100, 500, 1000])
    plan["zoom"] = {"window": 32, "stride": 16, "resolution": 64}
    plan["certification"].update(min_component_size=4, max_embedded=8, max_other=4, max_fixed=2)
    plan["ring_confirmation"].update(max_cells_per_component=48, steps=30000, ftle_steps=10000,
                                     ftle_align=1000, tail_steps=384)
    plan["chaos_diagnostics"].update(count=4, steps=30000, ftle_steps=10000, ftle_align=1000,
                                     tail_steps=384)
    plan["resources"].update(workers=4, chunk_cells=512, wall_ceiling_seconds=3000)
    _, plan_path, out, status = _execute(tmp_path, plan, workers=4)
    assert status == "complete"
    monkeypatch.setattr(sys, "argv", ["analyze", "--plan", str(plan_path), "--raw", str(out),
                                      "--summary", str(tmp_path / "summary.json"),
                                      "--figures", str(tmp_path / "figures")])
    analysis.main()
    monkeypatch.setattr(sys, "argv", ["audit", "--plan", str(plan_path), "--raw", str(out),
                                      "--out", str(tmp_path / "audit")])
    audit.main()
    summary = json.loads((tmp_path / "summary.json").read_text())
    report = json.loads((tmp_path / "audit" / "audit.json").read_text())
    embedded = [c for c in summary["certificates"] if c["kind"] == "embedded"]
    assert embedded, "fixture expected to contain embedded candidates"
    producer = {(q["map_id"], q["component"]) for q in summary["gate"]["qualifying"]}
    confirmed = {(q["map_id"], q["component"]) for q in report["audit_gate"]["qualifying"]}
    assert confirmed <= producer
    for entry in report["checks"]["maps"].values():
        for check in entry.get("components", []):
            assert check["ring_ok"] and check["seed_ok"] and check["rates_from_axes_ok"]
    (tmp_path / "result.json").write_text(json.dumps(
        {"producer": sorted(producer), "confirmed": sorted(confirmed),
         "embedded": embedded}, default=str))
