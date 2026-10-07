"""Execute the frozen EXP-028 plan.

Stages: (1) coarse period-resolved maps, (2) frozen-rule zoom maps,
(3) certification of mechanically selected components and chaos reruns.
The production path refuses to start unless the worktree is clean, HEAD is
contained in a remote-tracking branch, every bound source matches the plan,
the plan hash is recorded, and live load and memory are within the plan's
limits. Every chunk, assembled map, zoom specification, certificate, rerun,
error and a SHA-256 inventory are kept; a failed attempt is never reused.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.diagnostics import certify_cycle  # noqa: E402
from neural_shrimp.gdmaps import STATUS_CHAOTIC, GridMap, run_grid  # noqa: E402
from neural_shrimp.trainability import (  # noqa: E402
    SYSTEMS, array_sha256, chaos_sample, log_axis, periodic_components, zoom_axes, zoom_window,
)

DEFAULT_PLAN = ROOT / "experiments" / "trainability" / "EXP-028-plan.json"


class RunFailure(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")
    os.replace(tmp, path)


def save_npz(path, **arrays):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"not serializable: {type(value)}")


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                          text=True).stdout.strip()


def git_guard():
    """Clean worktree and a HEAD already contained in a remote-tracking ref."""
    dirty = git("status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise RunFailure("worktree is not clean:\n" + dirty)
    head = git("rev-parse", "HEAD")
    remote = [line.strip() for line in git("branch", "-r", "--contains", head).splitlines()
              if line.strip()]
    if not remote:
        raise RunFailure(f"HEAD {head} is not contained in any remote-tracking branch")
    return {"commit": head, "remote_refs_containing_head": remote, "clean": True}


def verify_sources(plan):
    mismatches = []
    for record in plan["source_files"]:
        data = (ROOT / record["path"]).read_bytes()
        if sha256_bytes(data) != record["sha256"] or len(data) != record["bytes"]:
            mismatches.append(record["path"])
    if mismatches:
        raise RunFailure(f"bound sources differ from the plan: {mismatches}")


def available_gib():
    with open("/proc/meminfo") as handle:
        for line in handle:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024 ** 2
    raise RunFailure("MemAvailable not found")


def gpu_snapshot():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        return out.stdout.strip() or out.stderr.strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"unavailable: {error}"


def preflight(plan):
    limits = plan["resources"]
    load = os.getloadavg()
    memory = available_gib()
    snapshot = {"time": utc_now(), "loadavg": list(load), "available_gib": memory,
                "logical_cpus": os.cpu_count(), "gpu": gpu_snapshot()}
    if load[0] > limits["max_load_average"]:
        raise RunFailure(f"load average {load[0]:.2f} exceeds {limits['max_load_average']}")
    if memory < limits["min_available_gib"]:
        raise RunFailure(f"available memory {memory:.1f} GiB below {limits['min_available_gib']}")
    return snapshot


def system_inputs(spec):
    net, x, y, theta0 = SYSTEMS[spec["system"]](spec["seed"])
    hashes = {"x": array_sha256(x), "y": array_sha256(y), "theta0": array_sha256(theta0)}
    if hashes != spec["data_sha256"]:
        raise RunFailure(f"{spec['map_id']}: regenerated inputs do not match plan hashes")
    return net, x, y, theta0


def coarse_axes(spec):
    return (log_axis(spec["log10_hidden"]["center"], spec["log10_hidden"]["half_width"],
                     spec["log10_hidden"]["resolution"]),
            log_axis(spec["log10_output"]["center"], spec["log10_output"]["half_width"],
                     spec["log10_output"]["resolution"]))


def iteration_options(it, overrides=None):
    options = {"tail_steps": it["tail_steps"], "ftle_steps": it["ftle_steps"],
               "ftle_align": it["ftle_align"], "max_period": it["max_period"],
               "min_repeats": it["min_repeats"], "atol": it["atol"], "rtol": it["rtol"],
               "chaos_threshold": it["chaos_threshold"],
               "divergence_threshold": it["divergence_threshold"],
               "sd_horizons": tuple(it["sd_horizons"]), "tangent_seed": it["tangent_seed"]}
    options.update(overrides or {})
    return options


# ----------------------------------------------------------------- workers
def chunk_worker(task):
    start = time.perf_counter()
    net, x, y, theta0 = system_inputs(task["spec"])
    grid = GridMap(net, x, y, 10.0 ** np.asarray(task["log_hidden"]),
                   10.0 ** np.asarray(task["log_output"]))
    result = run_grid(grid, theta0, task["steps"], **task["options"])
    result["cell_index"] = np.asarray(task["cells"], dtype=np.int64)
    result["lr_hidden"] = 10.0 ** np.asarray(task["log_hidden"], dtype=np.float64)
    result["lr_output"] = 10.0 ** np.asarray(task["log_output"], dtype=np.float64)
    save_npz(task["path"], **result)
    return {"chunk": task["chunk"], "path": str(task["path"]), "seconds": time.perf_counter() - start,
            "cells": len(task["cells"]), "pid": os.getpid()}


def certify_worker(task):
    start = time.perf_counter()
    net, x, y, _ = system_inputs(task["spec"])
    grid = GridMap(net, x, y, [task["lr_hidden"]], [task["lr_output"]])
    map_fn, jacobian = grid.cell_map(0)
    cert = task["certification"]
    result = certify_cycle(map_fn, np.asarray(task["seed_state"]), task["period"], jacobian=jacobian,
                           atol=cert["atol"], rtol=cert["rtol"],
                           root_tolerance=cert["root_tolerance"],
                           stability_margin=cert["stability_margin"],
                           candidate_tolerance=cert["candidate_tolerance"],
                           max_evaluations=cert["max_evaluations"])
    return {**task["record"], "map_id": task["map_id"], "kind": task["kind"],
            "lr_hidden": task["lr_hidden"], "lr_output": task["lr_output"],
            "seed_state_sha256": array_sha256(task["seed_state"]),
            "certificate": result, "seconds": time.perf_counter() - start, "pid": os.getpid()}


def chaos_worker(task):
    start = time.perf_counter()
    net, x, y, theta0 = system_inputs(task["spec"])
    grid = GridMap(net, x, y, [task["lr_hidden"]], [task["lr_output"]])
    d = task["diagnostics"]
    result = run_grid(grid, theta0, d["steps"], **task["options"])
    return {"map_id": task["map_id"], "cell": task["cell"], "lr_hidden": task["lr_hidden"],
            "lr_output": task["lr_output"], "original_ftle": task["original_ftle"],
            "rerun_status": int(result["status"][0]), "rerun_period": int(result["period"][0]),
            "rerun_ftle": float(result["ftle"][0]) if np.isfinite(result["ftle"][0]) else None,
            "agrees": bool(int(result["status"][0]) == STATUS_CHAOTIC and task["original_ftle"] is not None
                           and task["original_ftle"] > d["chaos_threshold"]),
            "seconds": time.perf_counter() - start, "pid": os.getpid()}


# ----------------------------------------------------------------- execution
class Executor:
    """Owned process pool with a wall ceiling and RSS sampling."""

    def __init__(self, workers, ceiling, started):
        self.pool = mp.get_context("spawn").Pool(processes=workers)
        self.ceiling = ceiling
        self.started = started
        self.pids = sorted(p.pid for p in self.pool._pool)
        self.peak_rss_kib = 0

    def _sample_rss(self):
        total = 0
        for pid in [p.pid for p in self.pool._pool]:
            try:
                with open(f"/proc/{pid}/status") as handle:
                    for line in handle:
                        if line.startswith("VmRSS:"):
                            total += int(line.split()[1])
            except OSError:
                continue
        self.peak_rss_kib = max(self.peak_rss_kib, total)
        self.pids = sorted(set(self.pids) | {p.pid for p in self.pool._pool})

    def map(self, function, tasks):
        pending = [self.pool.apply_async(function, (task,)) for task in tasks]
        results = [None] * len(pending)
        remaining = set(range(len(pending)))
        while remaining:
            if time.monotonic() - self.started > self.ceiling:
                raise RunFailure("wall ceiling exceeded")
            for i in list(remaining):
                if pending[i].ready():
                    results[i] = pending[i].get()
                    remaining.discard(i)
            self._sample_rss()
            if remaining:
                time.sleep(0.5)
        return results

    def close(self, terminate=False):
        if terminate:
            self.pool.terminate()
        else:
            self.pool.close()
        self.pool.join()
        alive = [pid for pid in self.pids if Path(f"/proc/{pid}").exists()]
        return {"worker_pids": self.pids, "alive_after_cleanup": alive,
                "peak_sampled_rss_kib": self.peak_rss_kib}


def run_map(executor, spec, map_id, log_h, log_o, out_dir, plan, log):
    it = plan["iteration"]
    rows, cols = len(log_h), len(log_o)
    cells = np.arange(rows * cols)
    chunk = plan["resources"]["chunk_cells"]
    map_dir = out_dir / map_id
    map_dir.mkdir(parents=True, exist_ok=False)
    tasks = []
    for number, start in enumerate(range(0, len(cells), chunk)):
        members = cells[start:start + chunk]
        r, c = np.divmod(members, cols)
        tasks.append({"spec": spec, "chunk": number, "cells": members.tolist(),
                      "log_hidden": log_h[r].tolist(), "log_output": log_o[c].tolist(),
                      "steps": it["steps"], "options": iteration_options(it),
                      "path": map_dir / f"chunk-{number:04d}.npz"})
    t0 = time.perf_counter()
    chunk_records = executor.map(chunk_worker, tasks)
    seconds = time.perf_counter() - t0
    parts = [np.load(record["path"]) for record in chunk_records]
    assembled = {}
    for key in parts[0].files:
        assembled[key] = np.concatenate([part[key] for part in parts], axis=0)
    if not np.array_equal(assembled["cell_index"], cells):
        raise RunFailure(f"{map_id}: assembled cells out of order")
    shaped = {}
    for key, value in assembled.items():
        if key == "final_state":
            shaped[key] = value.reshape(rows, cols, -1)
        elif key == "cell_index":
            continue
        else:
            shaped[key] = value.reshape(rows, cols)
    save_npz(out_dir / f"{map_id}.npz", log10_hidden=log_h, log10_output=log_o, **shaped)
    log({"event": "map_complete", "map_id": map_id, "seconds": seconds,
         "chunks": len(chunk_records), "time": utc_now()})
    return shaped, {"map_id": map_id, "seconds": seconds, "chunks": chunk_records}


def certification_tasks(spec, map_id, shaped, log_h, log_o, plan):
    cert = plan["certification"]
    status, period = shaped["status"], shaped["period"]
    cols = status.shape[1]
    tasks = []
    selections = [("periodic", periodic_components(status, period, cert["min_component_size"],
                                                   cert["max_components"])),
                  ("fixed", periodic_components(status, period, cert["min_component_size"],
                                                cert["max_fixed_components"], fixed=True))]
    for kind, records in selections:
        for record in records:
            r, c = divmod(record["representative"], cols)
            tasks.append({"spec": spec, "map_id": map_id, "kind": kind, "record": record,
                          "lr_hidden": float(10.0 ** log_h[r]), "lr_output": float(10.0 ** log_o[c]),
                          "seed_state": shaped["final_state"][r, c].tolist(),
                          "period": int(record["period"]), "certification": cert})
    return tasks


def chaos_tasks(spec, map_id, map_index, shaped, log_h, log_o, plan):
    d = plan["chaos_diagnostics"]
    status = shaped["status"]
    cols = status.shape[1]
    tasks = []
    for cell in chaos_sample(status, d["count"], d["seed_base"] + map_index):
        r, c = divmod(cell, cols)
        original = shaped["ftle"][r, c]
        options = iteration_options(plan["iteration"], {
            "tail_steps": d["tail_steps"], "ftle_steps": d["ftle_steps"],
            "ftle_align": d["ftle_align"], "tangent_seed": d["tangent_seed"],
            "chaos_threshold": d["chaos_threshold"], "sd_horizons": ()})
        tasks.append({"spec": spec, "map_id": map_id, "cell": int(cell),
                      "lr_hidden": float(10.0 ** log_h[r]), "lr_output": float(10.0 ** log_o[c]),
                      "original_ftle": float(original) if np.isfinite(original) else None,
                      "diagnostics": d, "options": options})
    return tasks


def environment_record():
    return {"python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
            "processor": platform.processor(), "hostname": platform.node(),
            "blas_threads_env": {v: os.environ.get(v) for v in
                                 ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}


def inventory(out_dir):
    records = []
    for path in sorted(p for p in Path(out_dir).rglob("*") if p.is_file()):
        if path.name == "receipt.json":
            continue
        records.append({"path": str(path.relative_to(out_dir)), "bytes": path.stat().st_size,
                        "sha256": sha256_file(path)})
    return records


def execute(plan, plan_sha, out_dir, guard_record=None, preflight_record=None, workers=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    events = []

    def log(event):
        events.append(event)
        with open(out_dir / "events.jsonl", "a") as handle:
            handle.write(json.dumps(event, default=_json_default) + "\n")

    workers = workers or plan["resources"]["workers"]
    write_json(out_dir / "runtime-start.json", {
        "experiment": plan["experiment"], "plan_sha256": plan_sha, "git": guard_record,
        "preflight": preflight_record, "environment": environment_record(),
        "workers": workers, "start_utc": utc_now()})
    executor = Executor(workers, plan["resources"]["wall_ceiling_seconds"], started)
    status = "failed"
    error = None
    stage_records = {"coarse": [], "zoom": [], "certification": None, "chaos": None}
    try:
        maps = []
        for spec in plan["maps"]:
            log_h, log_o = coarse_axes(spec)
            shaped, record = run_map(executor, spec, spec["map_id"], log_h, log_o,
                                     out_dir / "stage1", plan, log)
            stage_records["coarse"].append(record)
            maps.append((spec, spec["map_index"], spec["map_id"], shaped, log_h, log_o))
            z = plan["zoom"]
            window = zoom_window(shaped["status"], shaped["period"], z["window"], z["stride"])
            zoom_spec = {"map_id": spec["map_id"], "zoom_map_index": spec["map_index"] + z["map_index_offset"],
                         "window": None if window is None else {"row0": window[0], "col0": window[1],
                                                                "count": window[2]}}
            if window is not None:
                zh, zo = zoom_axes(log_h, log_o, window[0], window[1], z["window"], z["resolution"])
                zoom_spec["log10_hidden"] = zh.tolist()
                zoom_spec["log10_output"] = zo.tolist()
            write_json(out_dir / "stage1" / f"{spec['map_id']}-zoom.json", zoom_spec)
            log({"event": "zoom_selected", **{k: v for k, v in zoom_spec.items()
                                               if k not in ("log10_hidden", "log10_output")}})
        for spec, map_index, map_id, shaped, log_h, log_o in list(maps):
            zoom_spec = json.loads((out_dir / "stage1" / f"{map_id}-zoom.json").read_text())
            if zoom_spec["window"] is None:
                log({"event": "zoom_not_run", "map_id": map_id, "reason": "no eligible window"})
                continue
            zh, zo = np.asarray(zoom_spec["log10_hidden"]), np.asarray(zoom_spec["log10_output"])
            zshaped, record = run_map(executor, spec, f"{map_id}-zoom", zh, zo,
                                      out_dir / "stage2", plan, log)
            stage_records["zoom"].append(record)
            maps.append((spec, zoom_spec["zoom_map_index"], f"{map_id}-zoom", zshaped, zh, zo))
        cert_tasks, diag_tasks = [], []
        for spec, map_index, map_id, shaped, log_h, log_o in maps:
            cert_tasks += certification_tasks(spec, map_id, shaped, log_h, log_o, plan)
            diag_tasks += chaos_tasks(spec, map_id, map_index, shaped, log_h, log_o, plan)
        (out_dir / "stage3").mkdir()
        t0 = time.perf_counter()
        certificates = executor.map(certify_worker, cert_tasks)
        write_json(out_dir / "stage3" / "certificates.json", certificates)
        diagnostics = executor.map(chaos_worker, diag_tasks)
        write_json(out_dir / "stage3" / "chaos-diagnostics.json", diagnostics)
        stage_records["certification"] = {"count": len(certificates),
                                          "seconds": time.perf_counter() - t0}
        stage_records["chaos"] = {"count": len(diagnostics)}
        status = "complete"
    except Exception as exc:  # preserve every failure with its traceback
        error = {"type": type(exc).__name__, "message": str(exc),
                 "traceback": traceback.format_exc()}
        log({"event": "failure", **error, "time": utc_now()})
    finally:
        cleanup = executor.close(terminate=status != "complete")
    write_json(out_dir / "runtime-end.json", {
        "status": status, "error": error, "end_utc": utc_now(),
        "wall_seconds": time.monotonic() - started, "cleanup": cleanup,
        "stages": {"coarse": [{k: v for k, v in r.items() if k != "chunks"} for r in stage_records["coarse"]],
                   "zoom": [{k: v for k, v in r.items() if k != "chunks"} for r in stage_records["zoom"]],
                   "certification": stage_records["certification"],
                   "chaos": stage_records["chaos"]}})
    write_json(out_dir / "receipt.json", {
        "experiment": plan["experiment"], "plan_sha256": plan_sha, "status": status,
        "git": guard_record, "files": inventory(out_dir)})
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default=str(DEFAULT_PLAN))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    plan_bytes = Path(args.plan).read_bytes()
    plan = json.loads(plan_bytes)
    plan_sha = sha256_bytes(plan_bytes)
    guard = git_guard()
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
