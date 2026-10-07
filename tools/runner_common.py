"""Shared execution infrastructure for frozen Program C runners.

Owned process pool with dead-worker detection and a wall ceiling, a clean
worktree / tracked-plan / live-remote guard, source-hash verification,
resource preflight, atomic JSON and NPZ writes, environment records and a
SHA-256 inventory. Runners frozen before this module existed (EXP-028) keep
their own copies; this module serves later runners.
"""

import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]


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


def git_guard(plan_path, default_plan):
    """Clean worktree, tracked default plan and HEAD equal to the live remote head."""
    default_plan = Path(default_plan).resolve()
    if Path(plan_path).resolve() != default_plan:
        raise RunFailure("only the tracked default plan may be executed")
    git("ls-files", "--error-unmatch", str(default_plan.relative_to(ROOT)))
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


