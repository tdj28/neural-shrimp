"""Execute the frozen EXP-029 plan: autonomous survey and shrimp focus maps.

Every planned (map, start) cell is settled and measured with the batched
autonomous engine; periodic candidates also receive Floquet multipliers over
one recurrence cycle. The production entry point refuses to start unless the
worktree is clean, the tracked default plan is used, HEAD equals the live
remote branch head, every bound source matches the plan and live load and
memory are within limits. Chunks, assembled maps, errors and a SHA-256
inventory are kept; an attempt directory is never reused.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import json  # noqa: E402
import signal  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from neural_shrimp.autonomous import autonomous_map, floquet_multipliers  # noqa: E402
from neural_shrimp.capacity import hopfield_weights  # noqa: E402
from runner_common import (  # noqa: E402
    Executor, RunFailure, RunInterrupted, _json_default, _raise_interrupt, environment_record,
    git_guard, inventory, preflight, save_npz, sha256_bytes, utc_now, verify_sources, write_json,
)

DEFAULT_PLAN = ROOT / "experiments" / "track-b" / "EXP-029-plan.json"
CELL_KEYS = ("period", "cycle_time", "ftle", "equilibrium", "final_field_norm", "event_counts",
             "diverged")


def map_axes(spec):
    rows, cols = spec["resolution"]
    return (np.linspace(spec["gamma"][0], spec["gamma"][1], rows),
            np.linspace(spec["alpha"][0], spec["alpha"][1], cols))


def planned_cells(spec, start):
    rows, cols = spec["resolution"]
    stride = spec["start_strides"][start]
    r, c = np.divmod(np.arange(rows * cols), cols)
    return np.flatnonzero((r % stride == 0) & (c % stride == 0))


def chunk_worker(task):
    start_time = time.perf_counter()
    it = task["integration"]
    alpha = np.asarray(task["alpha"], dtype=np.float64)
    gamma = np.asarray(task["gamma"], dtype=np.float64)
    weights = hopfield_weights(alpha, gamma)
    x0 = np.repeat(np.asarray(task["start_state"], dtype=np.float64)[None], len(alpha), axis=0)
    result = autonomous_map(weights, x0, it["dt"], it["settle"], it["measure"],
                            max_events=it["max_events"], max_period=it["max_period"],
                            repeats=it["repeats"], atol=it["atol"], rtol=it["rtol"],
                            ftle_align=it["ftle_align"], tangent_seed=it["tangent_seed"],
                            divergence_threshold=it["divergence_threshold"],
                            equilibrium_tolerance=it["equilibrium_tolerance"])
    cells = len(alpha)
    counts = np.minimum(result["event_counts"], it["max_events"])
    last_event = np.full((cells, 3), np.nan)
    has_event = counts > 0
    last_event[has_event] = result["event_states"][np.flatnonzero(has_event), counts[has_event] - 1]
    multipliers = np.full((cells, 3), np.nan + 0j)
    closure = np.full(cells, np.nan)
    periodic = np.flatnonzero(result["period"] >= 1)
    if len(periodic):
        values, end = floquet_multipliers(weights[periodic], last_event[periodic],
                                          result["cycle_time"][periodic], it["dt"])
        multipliers[periodic] = values
        start_states = last_event[periodic]
        closure[periodic] = (np.max(np.abs(end - start_states), axis=1)
                             / np.maximum(1.0, np.max(np.abs(start_states), axis=1)))
    arrays = {key: result[key] for key in CELL_KEYS}
    arrays.update({"cell_index": np.asarray(task["cells"], dtype=np.int64), "alpha": alpha,
                   "gamma": gamma, "final_state": result["final_state"], "last_event_state": last_event,
                   "multiplier_real": multipliers.real, "multiplier_imag": multipliers.imag,
                   "floquet_closure": closure})
    save_npz(task["path"], **arrays)
    return {"job": task["job"], "chunk": task["chunk"], "path": str(task["path"]),
            "cells": cells, "seconds": time.perf_counter() - start_time, "pid": os.getpid()}


def build_tasks(plan, out_dir):
    tasks, jobs = [], []
    chunk = plan["resources"]["chunk_cells"]
    for spec in plan["maps"]:
        gamma_axis, alpha_axis = map_axes(spec)
        cols = len(alpha_axis)
        for start, state in spec["starts"].items():
            job = f"{spec['map_id']}-{start}"
            cells = planned_cells(spec, start)
            job_dir = out_dir / "chunks" / job
            job_dir.mkdir(parents=True, exist_ok=False)
            jobs.append((job, spec, start, cells))
            for number, first in enumerate(range(0, len(cells), chunk)):
                members = cells[first:first + chunk]
                r, c = np.divmod(members, cols)
                tasks.append({"job": job, "chunk": number, "cells": members.tolist(),
                              "alpha": alpha_axis[c].tolist(), "gamma": gamma_axis[r].tolist(),
                              "start_state": state, "integration": plan["integration"],
                              "path": job_dir / f"chunk-{number:04d}.npz"})
    return tasks, jobs


def assemble(job, spec, cells, records, out_dir):
    rows, cols = spec["resolution"]
    parts = [np.load(r["path"]) for r in sorted(records, key=lambda r: r["chunk"])]
    merged = {key: np.concatenate([p[key] for p in parts], axis=0) for key in parts[0].files}
    if not np.array_equal(merged["cell_index"], cells):
        raise RunFailure(f"{job}: assembled cells differ from the planned cells")
    computed = np.zeros(rows * cols, dtype=bool)
    computed[cells] = True
    shaped = {"computed": computed.reshape(rows, cols)}
    for key, value in merged.items():
        if key == "cell_index":
            continue
        trailing = value.shape[1:]
        if value.dtype == bool:
            full = np.zeros((rows * cols,) + trailing, dtype=bool)
        elif np.issubdtype(value.dtype, np.integer):
            full = np.full((rows * cols,) + trailing, -2, dtype=value.dtype)
        else:
            full = np.full((rows * cols,) + trailing, np.nan, dtype=value.dtype)
        full[cells] = value
        shaped[key] = full.reshape((rows, cols) + trailing)
    gamma_axis, alpha_axis = map_axes(spec)
    save_npz(out_dir / f"{job}.npz", gamma_axis=gamma_axis, alpha_axis=alpha_axis, **shaped)


def execute(plan, plan_sha, out_dir, guard_record=None, preflight_record=None, workers=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def log(event):
        with open(out_dir / "events.jsonl", "a") as handle:
            handle.write(json.dumps(event, default=_json_default) + "\n")

    workers = workers or plan["resources"]["workers"]
    write_json(out_dir / "runtime-start.json", {
        "experiment": plan["experiment"], "plan_sha256": plan_sha, "git": guard_record,
        "preflight": preflight_record, "environment": environment_record(), "workers": workers,
        "start_utc": utc_now()})
    previous = signal.signal(signal.SIGTERM, _raise_interrupt)
    executor = Executor(workers, plan["resources"]["wall_ceiling_seconds"], started)
    status, error, interrupted, counts = "failed", None, None, {}
    try:
        tasks, jobs = build_tasks(plan, out_dir)
        records = executor.map(chunk_worker, tasks)
        (out_dir / "maps").mkdir()
        for job, spec, start, cells in jobs:
            assemble(job, spec, cells, [r for r in records if r["job"] == job], out_dir / "maps")
            log({"event": "map_assembled", "job": job, "cells": len(cells), "time": utc_now()})
        counts = {"chunks": len(records), "cells": int(sum(len(j[3]) for j in jobs))}
        status = "complete"
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        log({"event": "failure", **error, "time": utc_now()})
        interrupted = exc if isinstance(exc, (KeyboardInterrupt, RunInterrupted)) else None
    finally:
        cleanup = executor.close(terminate=status != "complete")
        signal.signal(signal.SIGTERM, previous)
        write_json(out_dir / "runtime-end.json", {
            "status": status, "error": error, "end_utc": utc_now(), "counts": counts,
            "wall_seconds": time.monotonic() - started, "cleanup": cleanup})
        write_json(out_dir / "receipt.json", {
            "experiment": plan["experiment"], "plan_sha256": plan_sha, "status": status,
            "git": guard_record, "files": inventory(out_dir)})
    if interrupted is not None:
        raise interrupted
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    guard = git_guard(args.plan, DEFAULT_PLAN)
    plan_bytes = DEFAULT_PLAN.read_bytes()
    plan = json.loads(plan_bytes)
    plan_sha = sha256_bytes(plan_bytes)
    verify_sources(plan)
    snapshot = preflight(plan)
    out_dir = (ROOT / plan["output"]["raw_root"] / plan["output"]["attempt"]).resolve()
    print(json.dumps({"plan_sha256": plan_sha, "git": guard, "preflight": snapshot,
                      "output": str(out_dir)}, indent=2))
    if args.preflight_only:
        return
    if out_dir.exists():
        raise RunFailure(f"attempt directory already exists: {out_dir}")
    status = execute(plan, plan_sha, out_dir, guard, snapshot)
    print(json.dumps({"status": status, "output": str(out_dir)}))
    raise SystemExit(0 if status == "complete" else 2)


if __name__ == "__main__":
    main()
