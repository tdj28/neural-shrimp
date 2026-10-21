"""Bounded local throughput pilot for the HR10 float64 scan, not a phase map.

Times the existing map, twin renormalization and recording at specified batch
sizes. Short trajectories provide timing evidence only, never orbit labels.
The parent retains logs on failure and guards shared RAM and GPU ownership.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("tools/benchmark_soft_scan.py", "tools/batch_fate_finish.py",
           "tools/batch_fate_summary.py", "tools/explore_llm_soft_loop.py",
           "tools/explore_llm_attractors.py", "tools/explore_llm_phase_map.py")


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def worker(args):
    import numpy as np
    import torch
    import explore_llm_soft_loop as soft

    torch.set_num_interop_threads(1)
    if args.device == "cuda":
        torch.cuda.set_per_process_memory_fraction(0.85)

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    result = {"device": args.device, "threads": args.threads, "cells": [],
              "precision": "float64", "timing_only": True,
              "protocol": {"transient": 300, "samples": 300, "record": 80,
                           "eps": 1e-6, "period_tolerance": 1e-8}}
    start = time.perf_counter()
    loop = soft.SoftLoop("qwen0.5b", 12, args.threads, "open", "Hello!", 1,
                         "float64", args.device, "native")
    sync()
    result["model_load_seconds"] = time.perf_counter() - start
    result["model_revision"] = loop.sm.engine.revision
    result["prefix_tokens"] = loop.L0
    norm = loop.model.model.norm
    x = torch.linspace(-1, 1, norm.weight.numel(), dtype=torch.float64,
                       device=loop.device)[None, None, :]
    y1, y2 = norm(x), norm(x * (1 + 1e-12))
    sensitivity = float((y2 - y1).abs().max())
    assert y1.dtype == torch.float64 and 0 < sensitivity < 1e-9
    assert hasattr(loop.model.model.rotary_emb, "_inv_freq64")
    result["precision_checks"] = {"norm_sensitive_to_1e_12": sensitivity,
                                  "float64_rotary_frequencies": True}
    vector = soft.elpm.load_vectors("llm-phase-map/vectors-l12")["philosophy"]
    result["vector_sha256"] = hashlib.sha256(vector.tobytes()).hexdigest()
    path = args.out / f"{args.device}.json"
    write(path, result)
    rng = np.random.default_rng(0)
    for n in args.cells:
        if args.device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        alphas = np.linspace(0.07284, 0.07324, n)
        temps = np.tile(10 ** np.linspace(-0.15530, -0.15395, n), 2)
        steer = alphas[:, None] * vector
        sync()
        start = time.perf_counter()
        xy = loop.prepare(np.concatenate([steer, steer]))
        sync()
        prepare = time.perf_counter() - start
        for _ in range(2):
            xy, _ = loop.step(xy, temps)
        sync()
        transient = []
        for _ in range(args.steps):
            start = time.perf_counter()
            xy, p = loop.step(xy, temps)
            sync()
            transient.append(time.perf_counter() - start)
        delta = torch.as_tensor(rng.normal(size=tuple(xy[:n].shape)),
                                dtype=loop.state_dtype, device=loop.device)
        delta = delta / delta.flatten(1).norm(dim=1)[:, None, None] * 1e-6
        xy = torch.cat([xy[:n], xy[:n] + delta])
        samples, history, tokens = [], [], []
        log_growth = np.zeros(n)
        for _ in range(args.steps):
            sync()
            start = time.perf_counter()
            xy, p = loop.step(xy, temps)
            x, y = xy[:n], xy[n:]
            diff = y - x
            dist = np.maximum(diff.flatten(1).norm(dim=1).double().cpu().numpy(), 1e-36)
            log_growth += np.log(dist / 1e-6)
            y = x + diff * torch.as_tensor(1e-6 / dist, dtype=loop.state_dtype,
                                           device=loop.device)[:, None, None]
            xy = torch.cat([x, y])
            history.append(x[:, -1, :].double().cpu().numpy().copy())
            tokens.append(p[:n].argmax(-1).cpu().numpy().copy())
            sync()
            samples.append(time.perf_counter() - start)
        assert np.isfinite(np.asarray(history)).all() and np.isfinite(log_growth).all()
        row = {"cells": n, "rows_with_twins": 2 * n, "prepare_seconds": prepare,
               "transient_step_seconds": transient, "recording_sample_step_seconds": samples,
               "finite_states": True,
               "estimated_600_step_chunk_seconds": prepare + 300 * (np.mean(transient) + np.mean(samples)),
               "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else None}
        result["cells"].append(row)
        write(path, result)
        print(json.dumps(row), flush=True)
        del xy, p, x, y, diff, delta, history, tokens
        loop.cache = None
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--cells", type=int, nargs="+", default=[64, 256, 512])
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not (1 <= args.steps <= 20 and 1 <= args.threads <= 8 and all(1 <= n <= 512 for n in args.cells)):
        parser.error("pilot resource bounds exceeded")
    if args.worker:
        worker(args)
        return
    args.source_commit = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.source_commit}^{{commit}}"],
        cwd=ROOT, text=True).strip()
    import batch_fate_finish as guard
    args.out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for source in SOURCES:
        blob = subprocess.check_output(["git", "show", f"{args.source_commit}:{source}"], cwd=ROOT)
        digest = hashlib.sha256(blob).hexdigest()
        if digest != hashlib.sha256((ROOT / source).read_bytes()).hexdigest():
            raise ValueError(f"uncommitted source: {source}")
        hashes[source] = digest
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(args.threads)
    status = {"started": datetime.now(timezone.utc).isoformat(), "pid": os.getpid(),
              "source_commit": args.source_commit, "source_sha256": hashes, "status": "running"}

    def event(state, **info):
        with (args.out / "events.jsonl").open("a") as f:
            f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "state": state, **info}) + "\n")

    try:
        with guard.gpu_lock(guard.summary.DATA / "render-tmp/gpu.lock", timeout=1):
            status["resources"] = guard.resource_snapshot()
            write(args.out / "status.json", status)
            for device in ([args.device] if args.device else ["cuda", "cpu"]):
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--device", device,
                           "--out", str(args.out), "--source-commit", args.source_commit,
                           "--steps", str(args.steps), "--threads", str(args.threads),
                           "--cells", *map(str, args.cells)]
                guard.run_bounded(command, args.out / f"{device}.log", 900, event, monitor_memory=True)
        status["status"] = "complete"
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["finished"] = datetime.now(timezone.utc).isoformat()
        write(args.out / "status.json", status)


if __name__ == "__main__":
    main()
