import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _tool(name):
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    return importlib.import_module(name)


def _synthetic_plan():
    """Tiny artificial plan in a region outside the EXP-029 maps (negative alpha)."""
    starts = {"A": [0.1, 0.1, 0.1], "B": [0.2, -0.1, 0.3]}
    return {
        "experiment": "SYNTHETIC-029", "label": "synthetic fixture",
        "maps": [
            {"map_id": "g", "alpha": [-0.3, -0.1], "gamma": [17.0, 18.0], "resolution": [4, 4],
             "starts": starts, "start_strides": {"A": 1, "B": 1}},
            {"map_id": "f", "alpha": [-0.25, -0.2], "gamma": [17.2, 17.4], "resolution": [6, 6],
             "starts": starts, "start_strides": {"A": 1, "B": 2}},
        ],
        "integration": {"method": "RK4", "dt": 0.01, "settle": 60.0, "measure": 80.0, "max_events": 40,
                        "max_period": 8, "repeats": 4, "atol": 2e-6, "rtol": 2e-5, "ftle_align": 5.0,
                        "tangent_seed": 0, "divergence_threshold": 1e6, "equilibrium_tolerance": 1e-8},
        "labels": {"chaos_threshold_per_time": 0.005},
        "audit": {"cells_per_label_per_job": 2, "audit_seed": 2907, "solver": "scipy DOP853",
                  "rtol": 1e-10, "atol": 1e-12, "multiplier_tolerance": 1e-4},
        "resources": {"workers": 2, "chunk_cells": 8, "wall_ceiling_seconds": 600},
        "output": {"raw_root": "unused", "attempt": "unused"},
    }


def _execute(tmp_path):
    runner = _tool("run_exp029")
    plan = _synthetic_plan()
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    out = tmp_path / "attempt-001"
    status = runner.execute(plan, runner.sha256_bytes(plan_path.read_bytes()), out, workers=2)
    return runner, plan, plan_path, out, status


def test_exp029_runner_analysis_and_audit_end_to_end(tmp_path, monkeypatch):
    runner, plan, plan_path, out, status = _execute(tmp_path)
    assert status == "complete"
    end = json.loads((out / "runtime-end.json").read_text())
    assert end["cleanup"]["alive_after_cleanup"] == []
    assert end["counts"]["cells"] == 16 * 2 + 36 + 9
    focus_b = np.load(out / "maps" / "f-B.npz")
    assert focus_b["computed"].sum() == 9 and focus_b["period"].shape == (6, 6)
    assert np.all(focus_b["period"][~focus_b["computed"]] == -2)
    receipt = json.loads((out / "receipt.json").read_text())
    for item in receipt["files"]:
        assert _tool("runner_common").sha256_file(out / item["path"]) == item["sha256"]

    analysis = _tool("analyze_exp029")
    summary_path = tmp_path / "summary.json"
    argv = ["analyze", "--plan", str(plan_path), "--raw", str(out), "--summary", str(summary_path),
            "--figures", str(tmp_path / "figures")]
    monkeypatch.setattr(sys, "argv", argv)
    analysis.main()
    summary = json.loads(summary_path.read_text())
    counts = summary["jobs"]["f-B"]["counts"]
    assert counts["not_computed"] == 27 and sum(counts.values()) == 36
    assert summary["start_agreement"]["f"]["compared"] == 9
    assert Path(summary["jobs"]["g-A"]["figure"]["path"]).exists()
    with pytest.raises(SystemExit):
        analysis.main()

    audit = _tool("audit_exp029")
    monkeypatch.setattr(sys, "argv", ["audit", "--plan", str(plan_path), "--raw", str(out),
                                      "--out", str(tmp_path / "audit")])
    audit.main()
    report = json.loads((tmp_path / "audit" / "audit.json").read_text())
    assert report["receipt_mismatches"] == [] and report["receipt_plan_hash_matches"]
    for job in report["jobs"].values():
        for cell in job["cells"]:
            if cell["saved_label"] == 1 and cell["label"] == 1 and cell["period"] == cell["saved_period"]:
                assert cell["transverse_ok"]


def test_exp029_refuses_existing_attempt(tmp_path):
    runner = _tool("run_exp029")
    out = tmp_path / "attempt-001"
    out.mkdir()
    with pytest.raises(FileExistsError):
        runner.execute(_synthetic_plan(), "x", out, workers=1)
