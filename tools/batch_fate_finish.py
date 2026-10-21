"""Finish the originally planned BF controls after one exactly identified run.

No restart, publication or commit is performed. A unique external attempt keeps
logs/status on every failure. Select either an exact PID wait (bounded by
--wait-seconds) or --main-complete for a run that has already ended. Both
require complete main artifacts, unchanged committed source, a free GPU,
the shared GPU lock and RAM headroom before launching each bounded control.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import batch_fate_summary as summary

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv/bin/python"
SOURCES = ("tools/batch_fate.py", "tools/batch_fate_summary.py", "tools/batch_fate_finish.py",
           "tools/explore_llm_soft_loop.py", "tools/explore_llm_attractors.py", "tools/explore_llm_phase_map.py")
MAIN_OPTIONS = ["--alpha", "0.07248", "--log-t", "-0.148", "--starts", "64", "--steps", "5000",
                "--layouts", "32,64,1", "--device", "cuda", "--threads", "8", "--out",
                "llm-soft-loop/batch-fate/bf1"]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def process_identity(pid, proc=Path("/proc")):
    try:
        stat = (proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        argv = (proc / str(pid) / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
        return int(stat[19]), [arg.decode() for arg in argv]
    except FileNotFoundError:
        return None


def check_identity(identity, expected_ticks):
    if identity is None:
        return False
    ticks, argv = identity
    summary.require(ticks == expected_ticks, "wait PID was reused; refusing to follow another process")
    expected = [str(PYTHON), "tools/batch_fate.py", *MAIN_OPTIONS]
    summary.require(argv == expected, "wait PID command is not the owned BF main run")
    return True


def source_snapshot(commit):
    summary.require(len(commit) == 40 and all(c in "0123456789abcdef" for c in commit),
                    "source commit must be a full lowercase Git SHA")
    hashes = {}
    for path in SOURCES:
        committed = subprocess.check_output(["git", "show", f"{commit}:{path}"], cwd=ROOT)
        expected = hashlib.sha256(committed).hexdigest()
        summary.require(summary.sha256(ROOT / path) == expected, f"source differs from committed snapshot: {path}")
        hashes[path] = expected
    return {"commit": commit, "sha256": hashes}


def available_gib(meminfo=Path("/proc/meminfo")):
    fields = {line.split(":")[0]: line.split(":")[1].strip().split()[0]
              for line in meminfo.read_text().splitlines()}
    return int(fields["MemAvailable"]) / 1024**2


def resource_snapshot():
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
                                         "--format=csv,noheader"], text=True).strip()
    summary.require(not processes, f"GPU compute process already present: {processes}")
    memory = available_gib()
    summary.require(memory >= 24, f"need 24 GiB available RAM before model load; found {memory:.3f}")
    free = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.free",
                                    "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    summary.require(len(free) == 1, "BF continuation expects exactly one visible GPU")
    free_mib = float(free[0])
    summary.require(free_mib >= 8192, f"need 8 GiB GPU memory free before model load; found {free_mib} MiB")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.free,utilization.gpu",
                                   "--format=csv,noheader"], text=True).strip()
    return {"time": stamp(), "available_ram_gib": memory, "free_gpu_mib": free_mib, "gpu": gpu,
            "packages": {name: version(name) for name in ("numpy", "torch", "transformers", "accelerate")},
            "thread_environment": {name: os.environ.get(name) for name in
                                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}}


@contextmanager
def gpu_lock(path, timeout=600):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("shared GPU lock wait expired")
                time.sleep(5)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def run_bounded(command, log, timeout, event, monitor_memory=False, poll_seconds=5):
    with log.open("x") as f:
        child = subprocess.Popen(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        event("child_started", pid=child.pid, command=command, log=str(log), timeout_seconds=timeout)
        try:
            deadline = time.monotonic() + timeout
            while True:
                if monitor_memory:
                    memory = available_gib()
                    if memory < 12:
                        event("resource_guard_failed", pid=child.pid, available_ram_gib=memory, minimum_gib=12)
                        raise RuntimeError(f"available RAM fell below 12 GiB during model execution: {memory:.3f}")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    code = child.wait(timeout=min(poll_seconds, remaining))
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        raise
        except BaseException:
            # Only the new process group created above belongs to this worker.
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            raise
        summary.require(code == 0, f"child {child.pid} exited {code}; preserved log: {log}")
        event("child_complete", pid=child.pid, returncode=code)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--main-complete", action="store_true", help="main run already ended; validate its complete artifacts")
    mode.add_argument("--wait-pid", type=int)
    ap.add_argument("--wait-start-ticks", type=int)
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--out-run", required=True, help="new relative path beneath DATA/explore")
    ap.add_argument("--wait-seconds", type=int, default=14400)
    ap.add_argument("--control-timeout", type=int, default=900)
    args = ap.parse_args(argv)
    if args.main_complete:
        if args.wait_start_ticks is not None:
            ap.error("--main-complete does not accept --wait-start-ticks")
    elif args.wait_pid <= 0 or args.wait_start_ticks is None or args.wait_start_ticks <= 0:
        ap.error("PID wait mode requires positive --wait-pid and --wait-start-ticks")
    return args


def main():
    args = parse_args()
    rel = Path(args.out_run)
    summary.require(not rel.is_absolute() and ".." not in rel.parts, "out-run must stay below DATA/explore")
    summary.require(0 < args.wait_seconds <= 21600 and 0 < args.control_timeout <= 1800, "invalid time bounds")
    out = summary.DATA / "explore" / rel
    out.mkdir(parents=True, exist_ok=False)
    status = {"started": stamp(), "args": vars(args), "worker_pid": os.getpid(), "status": "starting"}

    def event(state, **info):
        entry = {"time": stamp(), "state": state, **info}
        with (out / "events.jsonl").open("a") as f:
            f.write(json.dumps(entry, allow_nan=False) + "\n")
        status.update(status=state, updated=entry["time"], latest=info)
        temp = out / "status.tmp"
        temp.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")
        temp.replace(out / "status.json")
        print(json.dumps(entry), flush=True)

    def interrupted(signum, frame):
        raise InterruptedError(f"worker received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        status["source"] = source_snapshot(args.source_commit)
        if args.main_complete:
            event("validating_completed_main", process_identity="not_claimed")
        else:
            event("waiting_for_main", pid=args.wait_pid, start_ticks=args.wait_start_ticks)
            deadline = time.monotonic() + args.wait_seconds
            while check_identity(process_identity(args.wait_pid), args.wait_start_ticks):
                if time.monotonic() >= deadline:
                    raise TimeoutError("main wait deadline expired; original process left untouched")
                time.sleep(min(30, max(0, deadline - time.monotonic())))
        # Load validates all 67 chunks and their row/record identities before controls.
        _, record, _, _ = summary.load("llm-soft-loop/batch-fate/bf1")
        for name, expected in (("alpha", .07248), ("log_t", -.148), ("starts", 64), ("steps", 5000),
                               ("layouts", "32,64,1"), ("device", "cuda"), ("threads", 8),
                               ("model", "qwen0.5b"), ("layer", 12), ("concept", "philosophy"),
                               ("vectors", "llm-phase-map/vectors-l12"), ("dtype", "float64"),
                               ("spread", .05), ("seed", 100), ("max_period", 64),
                               ("tol", 1e-9), ("record_every", 100)):
            summary.require(record["args"][name] == expected, f"main args mismatch: {name}")
        main_dir = summary.DATA / "explore/llm-soft-loop/batch-fate/bf1"
        status["main_artifacts"] = [summary.artifact(main_dir / name, summary.DATA)
                                    for name in ("fate.json", "fate.npz")]
        event("main_structural_gates_passed")
        controls = []
        for rows, layout in ((64, 64), (2, 1)):
            control = str(rel / f"repeat{layout}")
            summary.require(not (summary.DATA / "explore" / control).exists(), "control directory already exists")
            with gpu_lock(summary.DATA / "render-tmp/gpu.lock"):
                source_snapshot(args.source_commit)
                event("control_resources", control=control, resources=resource_snapshot())
                command = [str(PYTHON), "tools/batch_fate.py", "--alpha", "0.07248", "--log-t", "-0.148",
                           "--starts", str(rows), "--steps", "500", "--layouts", str(layout), "--device", "cuda",
                           "--threads", "8", "--out", control]
                run_bounded(command, out / f"repeat{layout}.log", args.control_timeout, event, monitor_memory=True)
            controls.extend(["--control", control])
        source_snapshot(args.source_commit)
        run_bounded([str(PYTHON), "tools/batch_fate_summary.py", "--run", "llm-soft-loop/batch-fate/bf1",
                     *controls, "--out", str(out / "receipt.json")], out / "summary.log", 180, event)
        status["artifacts"] = [summary.artifact(p, summary.DATA) for p in
                               (out / "receipt.json", out / "summary.log", out / "repeat64.log", out / "repeat1.log")]
        event("complete", receipt=str(out / "receipt.json"), publication="not_performed")
    except BaseException as exc:
        event("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
