"""Execute the frozen EXP-028 plan.

Stages: (1) coarse period-resolved maps for every planned map, (2) frozen-rule
zoom maps for full-pipeline maps, (3a) hardened certification of mechanically
selected components, (3b) chaos reruns of every exterior-ring chaotic cell of
each certified attracting embedded candidate (capped, row-major), and (3c)
random chaos diagnostics. The production entry point refuses to start unless
the worktree is clean, the tracked default plan is used, HEAD equals the
live remote branch head, every bound source matches the plan and live load
and memory are within limits. Every chunk, map, zoom specification,
certificate, rerun, per-task error and a SHA-256 inventory are kept; an
attempt directory is never reused.
"""

import os

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_var] = "1"

import argparse  # noqa: E402
import concurrent.futures as cf  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import platform  # noqa: E402
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.gdmaps import STATUS_CHAOTIC, GridMap, run_grid  # noqa: E402
from neural_shrimp.orbits import certify_orbit  # noqa: E402
from neural_shrimp.trainability import (  # noqa: E402
    SYSTEMS, array_sha256, chaos_sample, log_axis, select_components, zoom_axes, zoom_window,
)

DEFAULT_PLAN = ROOT / "experiments" / "trainability" / "EXP-028-plan.json"


class RunFailure(RuntimeError):
    pass


class RunInterrupted(BaseException):
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


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not serializable: {type(value)}")


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


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                          text=True, timeout=60).stdout.strip()


def git_guard(plan_path):
    """Clean worktree, tracked default plan and HEAD equal to the live remote head."""
    if Path(plan_path).resolve() != DEFAULT_PLAN.resolve():
        raise RunFailure("only the tracked default plan may be executed")
    git("ls-files", "--error-unmatch", str(DEFAULT_PLAN.relative_to(ROOT)))
    dirty = git("status", "--porcelain", "--untracked-files=all")
    if dirty:
        raise RunFailure("worktree is not clean:\n" + dirty)
    head = git("rev-parse", "HEAD")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    try:
        live = git("ls-remote", "origin", f"refs/heads/{branch}")
    except subprocess.SubprocessError as error:
        raise RunFailure(f"live remote check failed: {error}") from error
    remote_head = live.split()[0] if live else None
    if remote_head != head:
        raise RunFailure(f"live remote head {remote_head} for {branch} differs from HEAD {head}")
    return {"commit": head, "branch": branch, "live_remote_head": remote_head, "clean": True}


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
    if abs(net.ridge - spec["network"]["ridge"]) > 0:
        raise RunFailure(f"{spec['map_id']}: network ridge differs from plan")
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
               "sd_horizons": tuple(it["sd_horizons"]), "tangent_seed": it["tangent_seed"],
               "drift_threshold": it["drift_threshold"]}
    options.update(overrides or {})
    return options


# ----------------------------------------------------------------- workers
def chunk_worker(task):
    start = time.perf_counter()
    net, x, y, theta0 = system_inputs(task["spec"])
    lr_hidden = 10.0 ** np.asarray(task["log_hidden"], dtype=np.float64)
    lr_output = 10.0 ** np.asarray(task["log_output"], dtype=np.float64)
    grid = GridMap(net, x, y, lr_hidden, lr_output)
    result = run_grid(grid, theta0, task["steps"], **task["options"])
    result["cell_index"] = np.asarray(task["cells"], dtype=np.int64)
    result["lr_hidden"] = lr_hidden
    result["lr_output"] = lr_output
    save_npz(task["path"], **result)
    return {"map_id": task["map_id"], "chunk": task["chunk"], "path": str(task["path"]),
            "seconds": time.perf_counter() - start, "cells": len(task["cells"]), "pid": os.getpid()}


def certify_worker(task):
    start = time.perf_counter()
    try:
        net, x, y, _ = system_inputs(task["spec"])
        grid = GridMap(net, x, y, [task["lr_hidden"]], [task["lr_output"]])
        map_fn, jacobian = grid.cell_map(0)
        c = task["certification"]
        certificate = certify_orbit(map_fn, jacobian, np.asarray(task["seed_state"]), task["period"],
                                    atol=c["atol"], rtol=c["rtol"],
                                    newton_tolerance=c["newton_tolerance"],
                                    stability_margin=c["stability_margin"],
                                    candidate_tolerance=c["candidate_tolerance"],
                                    divisor_separation=c["divisor_separation"],
                                    max_iterations=c["max_iterations"])
        error = None
    except Exception as exc:  # recorded per task; never discards other evidence
        certificate, error = None, {"type": type(exc).__name__, "message": str(exc),
                                    "traceback": traceback.format_exc()}
    return {**task["record"], "map_id": task["map_id"], "kind": task["kind"],
            "lr_hidden": task["lr_hidden"], "lr_output": task["lr_output"],
            "seed_state_sha256": array_sha256(task["seed_state"]), "certificate": certificate,
            "error": error, "seconds": time.perf_counter() - start, "pid": os.getpid()}


def chaos_worker(task):
    start = time.perf_counter()
    try:
        net, x, y, theta0 = system_inputs(task["spec"])
        grid = GridMap(net, x, y, [task["lr_hidden"]], [task["lr_output"]])
        result = run_grid(grid, theta0, task["steps"], **task["options"])
        status, period = int(result["status"][0]), int(result["period"][0])
        ftle = float(result["ftle"][0]) if np.isfinite(result["ftle"][0]) else None
        error = None
    except Exception as exc:
        status = period = ftle = None
        error = {"type": type(exc).__name__, "message": str(exc),
                 "traceback": traceback.format_exc()}
    return {"map_id": task["map_id"], "purpose": task["purpose"], "cell": task["cell"],
            "component": task.get("component"), "lr_hidden": task["lr_hidden"],
            "lr_output": task["lr_output"], "original_status": task["original_status"],
            "original_ftle": task["original_ftle"], "rerun_status": status, "rerun_period": period,
            "rerun_ftle": ftle, "confirmed_chaotic": status == STATUS_CHAOTIC,
            "error": error, "seconds": time.perf_counter() - start, "pid": os.getpid()}


# ----------------------------------------------------------------- execution
class Executor:
    """Owned process pool: detects dead workers, enforces the wall ceiling."""

    def __init__(self, workers, ceiling, started):
        self.pool = cf.ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn"))
        self.ceiling = ceiling
        self.started = started
        self.pids = set()
        self.peak_rss_kib = 0

    def _sample(self):
        processes = getattr(self.pool, "_processes", None) or {}
        total = 0
        for pid in list(processes):
            self.pids.add(pid)
            try:
                with open(f"/proc/{pid}/status") as handle:
                    for line in handle:
                        if line.startswith("VmRSS:"):
                            total += int(line.split()[1])
            except OSError:
                continue
        self.peak_rss_kib = max(self.peak_rss_kib, total)

    def map(self, function, tasks):
        futures = {self.pool.submit(function, task): i for i, task in enumerate(tasks)}
        results = [None] * len(tasks)
        pending = set(futures)
        while pending:
            remaining = self.ceiling - (time.monotonic() - self.started)
            if remaining <= 0:
                raise RunFailure("wall ceiling exceeded")
            done, pending = cf.wait(pending, timeout=min(5.0, remaining),
                                    return_when=cf.FIRST_COMPLETED)
            for future in done:
                results[futures[future]] = future.result()  # BrokenProcessPool raises here
                self.pids.add(results[futures[future]].get("pid"))
            self._sample()
        return results

    def close(self, terminate=False):
        processes = dict(getattr(self.pool, "_processes", None) or {})
        self.pids |= set(processes)
        if terminate:
            for process in processes.values():
                try:
                    process.terminate()
                except Exception:
                    pass
        self.pool.shutdown(wait=True, cancel_futures=True)
        pids = sorted(p for p in self.pids if p)
        alive = [pid for pid in pids if Path(f"/proc/{pid}").exists()]
        return {"worker_pids": pids, "alive_after_cleanup": alive,
                "peak_sampled_rss_kib": self.peak_rss_kib}


def chunk_tasks(spec, map_id, log_h, log_o, out_dir, plan):
    it = plan["iteration"]
    cols = len(log_o)
    cells = np.arange(len(log_h) * cols)
    chunk = plan["resources"]["chunk_cells"]
    map_dir = out_dir / map_id
    map_dir.mkdir(parents=True, exist_ok=False)
    tasks = []
    for number, start in enumerate(range(0, len(cells), chunk)):
        members = cells[start:start + chunk]
        r, c = np.divmod(members, cols)
        tasks.append({"spec": spec, "map_id": map_id, "chunk": number, "cells": members.tolist(),
                      "log_hidden": log_h[r].tolist(), "log_output": log_o[c].tolist(),
                      "steps": it["steps"], "options": iteration_options(it),
                      "path": map_dir / f"chunk-{number:04d}.npz"})
    return tasks


def assemble(map_id, records, log_h, log_o, out_path):
    rows, cols = len(log_h), len(log_o)
    records = sorted(records, key=lambda r: r["chunk"])
    parts = [np.load(r["path"]) for r in records]
    merged = {key: np.concatenate([p[key] for p in parts], axis=0) for key in parts[0].files}
    if not np.array_equal(merged["cell_index"], np.arange(rows * cols)):
        raise RunFailure(f"{map_id}: assembled cells out of order")
    shaped = {}
    for key, value in merged.items():
        if key == "cell_index":
            continue
        shaped[key] = value.reshape(rows, cols, -1) if key == "final_state" else value.reshape(rows, cols)
    save_npz(out_path, log10_hidden=log_h, log10_output=log_o, **shaped)
    return shaped


def run_stage(executor, jobs, out_dir, plan, log):
    """Run chunks of several maps together; returns {map_id: shaped arrays}."""
    tasks = []
    for spec, map_id, log_h, log_o in jobs:
        tasks += chunk_tasks(spec, map_id, log_h, log_o, out_dir, plan)
    t0 = time.perf_counter()
    records = executor.map(chunk_worker, tasks)
    shaped = {}
    for spec, map_id, log_h, log_o in jobs:
        mine = [r for r in records if r["map_id"] == map_id]
        shaped[map_id] = assemble(map_id, mine, log_h, log_o, out_dir / f"{map_id}.npz")
        log({"event": "map_complete", "map_id": map_id, "chunks": len(mine), "time": utc_now()})
    log({"event": "stage_complete", "maps": [j[1] for j in jobs],
         "seconds": time.perf_counter() - t0, "time": utc_now()})
    return shaped, records



def certification_tasks(spec, map_id, shaped, plan):
    c = plan["certification"]
    status, period = shaped["status"], shaped["period"]
    cols = status.shape[1]
    lists = select_components(status, period, c["min_component_size"], c["max_embedded"],
                              c["max_other"], c["max_fixed"], c["min_chaotic_fraction"],
                              c["min_embedded_period"])
    tasks = []
    for kind in ("embedded", "other", "fixed"):
        for record in lists[kind]:
            r, k = divmod(record["representative"], cols)
            tasks.append({"spec": spec, "map_id": map_id, "kind": kind, "record": record,
                          "lr_hidden": float(shaped["lr_hidden"][r, k]),
                          "lr_output": float(shaped["lr_output"][r, k]),
                          "seed_state": shaped["final_state"][r, k].tolist(),
                          "period": int(record["period"]), "certification": c})
    return tasks


def rerun_task(spec, map_id, shaped, cell, settings, plan, purpose, component=None):
    cols = shaped["status"].shape[1]
    r, k = divmod(int(cell), cols)
    original = shaped["ftle"][r, k]
    options = iteration_options(plan["iteration"], {
        "tail_steps": settings["tail_steps"], "ftle_steps": settings["ftle_steps"],
        "ftle_align": settings["ftle_align"], "tangent_seed": settings["tangent_seed"],
        "chaos_threshold": settings["chaos_threshold"], "sd_horizons": ()})
    return {"spec": spec, "map_id": map_id, "purpose": purpose, "cell": int(cell),
            "component": component, "lr_hidden": float(shaped["lr_hidden"][r, k]),
            "lr_output": float(shaped["lr_output"][r, k]),
            "original_status": int(shaped["status"][r, k]),
            "original_ftle": float(original) if np.isfinite(original) else None,
            "steps": settings["steps"], "options": options}


def environment_record():
    return {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
            "platform": platform.platform(), "processor": platform.processor(),
            "hostname": platform.node(),
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


def _raise_interrupt(signum, _frame):
    raise RunInterrupted(f"signal {signum}")


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
        "preflight": preflight_record, "environment": environment_record(),
        "workers": workers, "start_utc": utc_now()})
    previous = signal.signal(signal.SIGTERM, _raise_interrupt)
    executor = Executor(workers, plan["resources"]["wall_ceiling_seconds"], started)
    status, error, interrupted = "failed", None, None
    counts = {}
    try:
        maps = []  # (spec, map_index, map_id, shaped)
        coarse_jobs = [(spec, spec["map_id"]) + coarse_axes(spec) for spec in plan["maps"]]
        coarse, _ = run_stage(executor, coarse_jobs, out_dir / "stage1", plan, log)
        zoom_jobs = []
        z = plan["zoom"]
        for spec in plan["maps"]:
            shaped = coarse[spec["map_id"]]
            maps.append((spec, spec["map_index"], spec["map_id"], shaped))
            if spec["pipeline"] != "full":
                continue
            log_h, log_o = coarse_axes(spec)
            window = zoom_window(shaped["status"], shaped["period"], z["window"], z["stride"])
            zoom_spec = {"map_id": spec["map_id"], "zoom_map_index": spec["zoom_map_index"],
                         "window": None if window is None else {
                             "row0": window[0], "col0": window[1], "primary": window[2],
                             "secondary": window[3]}}
            if window is not None:
                zh, zo = zoom_axes(log_h, log_o, window[0], window[1], z["window"], z["resolution"])
                zoom_spec["log10_hidden"], zoom_spec["log10_output"] = zh.tolist(), zo.tolist()
                zoom_jobs.append((spec, f"{spec['map_id']}-zoom", zh, zo))
            write_json(out_dir / "stage1" / f"{spec['map_id']}-zoom.json", zoom_spec)
            log({"event": "zoom_selected", "map_id": spec["map_id"], "window": zoom_spec["window"]})
        if zoom_jobs:
            zoomed, _ = run_stage(executor, zoom_jobs, out_dir / "stage2", plan, log)
            for spec, map_id, _, _ in zoom_jobs:
                maps.append((spec, spec["zoom_map_index"], map_id, zoomed[map_id]))
        (out_dir / "stage3").mkdir()
        cert_tasks = []
        for spec, _, map_id, shaped in maps:
            if spec["pipeline"] == "full":
                cert_tasks += certification_tasks(spec, map_id, shaped, plan)
        certificates = executor.map(certify_worker, cert_tasks)
        write_json(out_dir / "stage3" / "certificates.json", certificates)
        ring = plan["ring_confirmation"]
        shaped_by_id = {map_id: (spec, shaped) for spec, _, map_id, shaped in maps}
        ring_tasks = []
        for record in certificates:
            cert = record["certificate"] or {}
            if record["kind"] != "embedded" or not (cert.get("verified") and cert.get("attracting")):
                continue
            spec, shaped = shaped_by_id[record["map_id"]]
            for cell in sorted(record["ring_chaotic_cells"])[:ring["max_cells_per_component"]]:
                ring_tasks.append(rerun_task(spec, record["map_id"], shaped, cell, ring, plan,
                                             "ring_confirmation", record["component"]))
        ring_results = executor.map(chaos_worker, ring_tasks)
        write_json(out_dir / "stage3" / "ring-confirmation.json", ring_results)
        d = plan["chaos_diagnostics"]
        diag_tasks = []
        for spec, map_index, map_id, shaped in maps:
            for cell in chaos_sample(shaped["status"], d["count"], d["seed_base"] + map_index):
                diag_tasks.append(rerun_task(spec, map_id, shaped, cell, d, plan, "random_chaos"))
        diagnostics = executor.map(chaos_worker, diag_tasks)
        write_json(out_dir / "stage3" / "chaos-diagnostics.json", diagnostics)
        counts = {"maps": len(maps), "certificates": len(certificates),
                  "certificate_errors": sum(r["error"] is not None for r in certificates),
                  "ring_reruns": len(ring_results),
                  "rerun_errors": sum(r["error"] is not None for r in ring_results + diagnostics),
                  "random_chaos_reruns": len(diagnostics)}
        status = "complete"
    except BaseException as exc:  # record every failure, including interrupts
        error = {"type": type(exc).__name__, "message": str(exc),
                 "traceback": traceback.format_exc()}
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
    guard = git_guard(args.plan)
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
