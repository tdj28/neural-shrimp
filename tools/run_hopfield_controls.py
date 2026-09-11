#!/usr/bin/env python3
"""EXP-005: fixed independent control runs, retaining every point and step."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import threading
import time

import numpy as np

import run_hopfield as base

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "experiments/hopfield/EXP-005-plan.json"
SOURCE_FILES = ["tools/run_hopfield_controls.py", "tools/run_hopfield.py",
                "tools/hopfield_diagnostics.py", "native/hopfield.cpp",
                "experiments/hopfield/EXP-005-plan.json"]


def validate(plan):
    assert plan["experiment_id"] == "EXP-005"
    assert plan["kind"] == "hopfield_independent_control_check"
    assert [p["name"] for p in plan["points"]] == ["core", "lower", "upper", "failed-near-edge"]
    assert np.isfinite([[p["alpha"], p["gamma"]] for p in plan["points"]]).all()
    assert plan["initial_state"] == [0.1, 0.1, 0.1]
    assert plan["dt"] == [0.01, 0.005]
    assert plan["burn_time"] == 10000.0 and plan["sample_time"] == 2000.0
    assert plan["event_capacity"] == 128 and plan["max_peak_period"] == 32
    assert plan["minimum_repeats"] == 4
    assert plan["recurrence_atol"] == 2e-6 and plan["recurrence_rtol"] == 2e-5
    assert plan["irregular_ftle_threshold"] == 0.005
    assert plan["workers"] == 1 and plan["wall_clock_ceiling_seconds"] == 600
    assert plan["behavior"]["duration"] == 300 and plan["behavior"]["sample_dt"] == 0.05
    assert plan["behavior"]["perturbation_size"] == 1e-6
    assert np.allclose(plan["behavior"]["direction"], np.ones(3) / np.sqrt(3), rtol=0, atol=1e-15)
    assert plan["paid_compute"] is False and plan["paid_review"] is False
    return len(plan["points"]) * len(plan["dt"])


def cell_path(directory, name, dt):
    return directory / "cells" / f"{name}-dt-{dt:g}.npz"


def save_cell(path, point, dt, initial, result):
    final, events, stats, n, period, residual, duration = result
    payload = dict(alpha=point["alpha"], gamma=point["gamma"], dt=dt,
                   initial=initial, final=final, events=events, stats=stats,
                   n_events=n, period=period, residual=residual, period_time=duration)
    np.savez_compressed(path, **payload)
    with np.load(path, allow_pickle=False) as saved:
        assert all(np.array_equal(saved[key], value, equal_nan=True) for key, value in payload.items())


def execute_cells(fn, plan, directory):
    """Save all eight independently initialized runs before any diagnostics."""
    (directory / "cells").mkdir()
    for point in plan["points"]:
        for dt in plan["dt"]:
            initial = np.array(plan["initial_state"], dtype=float)
            base.write(directory / "progress.json", {"phase": "integration", "point": point["name"], "dt": dt})
            result = base.cell(fn, point["alpha"], point["gamma"], initial, plan, dt=dt,
                               failure_path=directory / "cells" / f"{point['name']}-dt-{dt:g}-failure.npz")
            path = cell_path(directory, point["name"], dt)
            save_cell(path, point, dt, initial, result)
            print(json.dumps({"point": point["name"], "dt": dt, "period": result[4],
                              "ftle": float(result[2][1]), "saved": str(path)}), flush=True)


def read_cell(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def common_period(cells):
    periods = [int(cell["period"]) for cell in cells]
    return periods[0] if periods and periods[0] > 0 and all(q == periods[0] for q in periods) else 0


def finite_horizon_irregular(cells, threshold):
    return bool(cells and all(int(c["period"]) == 0 and np.isfinite(c["stats"][1]) and
                              c["stats"][1] > threshold for c in cells))


def diagnose_point(point, cells, plan):
    finest = cells[int(np.argmin([float(cell["dt"]) for cell in cells]))]
    q = common_period(cells)
    if q:
        n = int(finest["n_events"])
        certificate = base.certify(point["alpha"], point["gamma"],
                                   finest["events"][n-q, 1:], float(finest["period_time"]), q)
    else:
        certificate = {"verified": False, "attracting": False,
                       "reason": "No shared positive peak period across both independent steps"}
    return {"name": point["name"], "alpha": point["alpha"], "gamma": point["gamma"],
            "step_results": [{"dt": float(cell["dt"]), "period": int(cell["period"]),
                              "residual": float(cell["residual"]), "period_time": float(cell["period_time"]),
                              "stats": cell["stats"]} for cell in cells],
            "shared_peak_period": q,
            "finite_horizon_irregular_support": finite_horizon_irregular(cells, plan["irregular_ftle_threshold"]),
            "certificate": certificate}


def execute_diagnostics(plan, directory):
    (directory / "points").mkdir()
    summary = {}
    for point in plan["points"]:
        name = point["name"]
        base.write(directory / "progress.json", {"phase": "diagnostics", "point": name})
        cells = [read_cell(cell_path(directory, name, dt)) for dt in plan["dt"]]
        item = diagnose_point(point, cells, plan)
        base.write(directory / "points" / f"{name}.json", item)
        finest = cells[int(np.argmin([float(cell["dt"]) for cell in cells]))]
        probe = base.behavior_probe(point["alpha"], point["gamma"], finest["final"],
                                    duration=plan["behavior"]["duration"],
                                    sample_dt=plan["behavior"]["sample_dt"],
                                    certificate=item["certificate"] if item["certificate"].get("verified") else None)
        arrays = {key: value for key, value in probe.items() if key != "metrics"}
        np.savez_compressed(directory / "points" / f"{name}-behavior.npz", **arrays)
        item["behavior_metrics"] = probe["metrics"]
        base.write(directory / "points" / f"{name}.json", item)
        summary[name] = item
        base.write(directory / "summary.json", summary)
        print(json.dumps({"point": name, "shared_period": item["shared_peak_period"],
                          "certified": item["certificate"].get("verified"),
                          "attracting": item["certificate"].get("attracting"),
                          "finite_horizon_irregular_support": item["finite_horizon_irregular_support"]}), flush=True)


def verify_cell(cell, point, dt, plan):
    assert float(cell["alpha"]) == point["alpha"] and float(cell["gamma"]) == point["gamma"]
    assert float(cell["dt"]) == dt
    assert np.array_equal(cell["initial"], plan["initial_state"])
    assert cell["final"].shape == (3,) and np.isfinite(cell["final"]).all()
    events, stats, n = cell["events"], cell["stats"], int(cell["n_events"])
    assert events.shape == (plan["event_capacity"], 4) and stats.shape == (20,)
    assert 0 <= n <= plan["event_capacity"] and n == min(stats[0], plan["event_capacity"])
    assert np.isfinite(events[:n]).all() and np.isnan(events[n:]).all()
    if n:
        assert np.all(np.diff(events[:n, 0]) > 0)
        assert 0 <= events[0, 0] <= events[n-1, 0] <= plan["sample_time"]
    assert np.isfinite(stats).all()
    assert stats[18] == round(plan["burn_time"] / dt)
    assert stats[19] == round(plan["sample_time"] / dt)
    assert stats[3] == plan["sample_time"]
    assert np.isclose(stats[2], plan["sample_time"] - 100.0, rtol=0, atol=1e-7)
    q, residual, duration = base.classify(events[:n], stats, plan)
    assert q == int(cell["period"])
    np.testing.assert_allclose([residual, duration], [float(cell["residual"]), float(cell["period_time"])], rtol=0, atol=0)


def verify(directory):
    assert not (directory / "failure.json").exists() and not (directory / "timeout.json").exists()
    receipt = json.loads((directory / "receipt.json").read_text())
    plan = json.loads((directory / "manifest.json").read_text())
    assert receipt["status"] == "complete" and validate(plan) == receipt["planned_cells"]
    assert base.sha(directory / "manifest.json") == receipt["manifest_sha256"]
    for entry in receipt["artifacts"]:
        assert base.sha(directory / entry["path"]) == entry["sha256"], entry["path"]
    summary = json.loads((directory / "summary.json").read_text())
    assert list(summary) == [p["name"] for p in plan["points"]]
    for point in plan["points"]:
        cells = []
        for dt in plan["dt"]:
            cell = read_cell(cell_path(directory, point["name"], dt))
            verify_cell(cell, point, dt, plan)
            cells.append(cell)
        item = summary[point["name"]]
        assert item["shared_peak_period"] == common_period(cells)
        assert item["finite_horizon_irregular_support"] == finite_horizon_irregular(cells, plan["irregular_ftle_threshold"])
        assert item == json.loads((directory / "points" / f"{point['name']}.json").read_text())
        if item["certificate"].get("verified"):
            assert common_period(cells) > 0
            assert item["certificate"]["peak_count"] == common_period(cells)
        with np.load(directory / "points" / f"{point['name']}-behavior.npz", allow_pickle=False) as probe:
            assert len(probe["times"]) == 6001
            assert probe["times"][0] == 0 and probe["times"][-1] == plan["behavior"]["duration"]
            np.testing.assert_array_equal(probe["states"][0], cells[-1]["final"])
            np.testing.assert_allclose(probe["activations"], np.tanh(probe["states"]), rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(probe["separation"], np.linalg.norm(probe["states"]-probe["perturbed_states"], axis=1), rtol=1e-12, atol=1e-14)
    print(json.dumps({"verified": True, "planned_cells": validate(plan), "source": receipt["source_commit"]}))


def run(directory):
    plan = json.loads(PLAN.read_text())
    count = validate(plan)
    assert not base.git("status", "--porcelain"), "Source must be clean before target outcomes"
    source = base.git("rev-parse", "HEAD")
    assert base.git("branch", "-r", "--contains", source), "Push source before target outcomes"
    directory.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(PLAN, directory / "manifest.json")
    started = {"experiment_id": "EXP-005", "source_commit": source, "status": "running",
               "manifest_sha256": base.sha(PLAN), "started_at": base.now(),
               "source_files": {path: base.sha(ROOT / path) for path in SOURCE_FILES},
               "planned_cells": count, "stats_columns": base.STATS,
               "platform": platform.platform(), "python": sys.version,
               "paid_compute": 0, "paid_review": 0}
    base.write(directory / "started.json", started)
    start = time.monotonic()
    def hard_stop():
        base.write(directory / "timeout.json", {"source_commit": source, "status": "hard_timeout",
                                                 "ended_at": base.now(), "ceiling_seconds": 600,
                                                 "original_attempt_preserved": True})
        os._exit(124)
    watchdog = threading.Timer(plan["wall_clock_ceiling_seconds"], hard_stop)
    watchdog.daemon = True
    watchdog.start()
    try:
        fn, build = base.build_engine(directory / "build")
        base.write(directory / "build.json", build)
        execute_cells(fn, plan, directory)
        execute_diagnostics(plan, directory)
        base.write(directory / "progress.json", {"phase": "complete"})
        artifacts = [file for file in sorted(directory.rglob("*")) if file.is_file() and
                     "build" not in file.relative_to(directory).parts]
        base.write(directory / "receipt.json", {**started, "status": "complete", "ended_at": base.now(),
                    "wall_seconds": time.monotonic()-start,
                    "artifacts": [{"path": str(file.relative_to(directory)), "sha256": base.sha(file),
                                   "bytes": file.stat().st_size} for file in artifacts]})
        verify(directory)
    except BaseException as error:
        base.write(directory / "failure.json", {"source_commit": source, "error": repr(error),
                   "ended_at": base.now(), "wall_seconds": time.monotonic()-start,
                   "original_attempt_preserved": True})
        raise
    finally:
        watchdog.cancel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["validate", "run", "verify"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "validate":
        print(json.dumps({"valid": True, "planned_cells": validate(json.loads(PLAN.read_text()))}))
    elif args.output is None:
        parser.error("--output required")
    elif args.action == "run":
        run(args.output)
    else:
        verify(args.output)


if __name__ == "__main__":
    main()
