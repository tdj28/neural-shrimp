"""Cross-GPU parity and timing for the gdmap CUDA engine (RunPod Phase 0).

  record   run fixed workloads on the reference GPU (the qualified RTX 2000 Ada,
           sm_89) and save every output array with its SHA-256;
  compare  run the same workloads with another build on another GPU and compare
           every array bit for bit (NaN positions included), reporting the first
           mismatching key, mismatch counts and the largest finite difference.

Workloads use the artificial qualification fixtures of tools/qualify_gdmap_cuda.py
(seeds 99990-99991) and two exploratory systems already used in exploration
(S2-28011, S1R-28002); no reserved seed (29000-29999) is touched. The timing-only
workload measures throughput on a 256 x 256 grid of the 72-parameter system.
A different GPU needs NEURAL_SHRIMP_CUDA_QUALIFYING (for example 9.0) to run;
the receipt records that it was a qualification run.

Usage:
  python tools/crossgpu_parity.py record  --library LIB --out DIR
  python tools/crossgpu_parity.py compare --library LIB --reference DIR --out DIR
"""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from neural_shrimp import gdmaps_cuda as gc  # noqa: E402
from neural_shrimp.gdmaps import GridMap  # noqa: E402
from neural_shrimp.trainability import SYSTEMS  # noqa: E402
import qualify_gdmap_cuda as q  # noqa: E402

EXP028 = dict(tail_steps=384, ftle_steps=4000, ftle_align=1000, max_period=64, min_repeats=6, atol=1e-8,
              rtol=1e-6, chaos_threshold=0.005, drift_threshold=0.01, divergence_threshold=1e8,
              tangent_seed=0, sd_horizons=(100, 500, 1000))


def log_grid(net, x, y, hidden, output, side):
    lh = np.linspace(*hidden, side)
    lo = np.linspace(*output, side)
    r, c = np.divmod(np.arange(side * side), side)
    return GridMap(net, x, y, 10.0 ** lh[r], 10.0 ** lo[c])


def workloads():
    net, x, y, theta0 = q.s1_like()
    yield "s1_like_24", q.rate_grid(net, x, y, theta0, (-1.5, 1.0), (-1.5, 0.5), 24), theta0, 20000, q.PRODUCTION, True
    net, x, y, theta0 = q.s2_like()
    yield "s2_like_24", q.rate_grid(net, x, y, theta0, (-1.0, 1.0), (-1.0, 1.0), 24), theta0, 20000, q.PRODUCTION, True
    net, x, y, theta0 = SYSTEMS["S2"](28011)
    yield "S2-28011_64", log_grid(net, x, y, (1.70, 1.90), (-0.20, -0.06), 64), theta0, 20000, EXP028, True
    net, x, y, theta0 = SYSTEMS["S1R"](28002)
    yield "S1R-28002_32", log_grid(net, x, y, (1.25, 1.45), (0.90, 0.98), 32), theta0, 20000, EXP028, True
    yield "S1R-28002_256_timing", log_grid(net, x, y, (1.25, 1.45), (0.90, 0.98), 256), theta0, 2000, \
        dict(EXP028, ftle_steps=400, ftle_align=100, sd_horizons=()), False


def digest(array):
    a = np.ascontiguousarray(array)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def environment(backend):
    def run(args):
        try:
            return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"host": platform.node(), "cpu": platform.processor() or run(["uname", "-p"]),
            "python": platform.python_version(), "numpy": np.__version__,
            "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                               "--format=csv,noheader"]),
            "backend": backend.metadata()}


def execute(library):
    backend = gc.load(library)
    results = {}
    for name, grid, theta0, steps, options, compare in workloads():
        started = time.perf_counter()
        out, meta = backend.run_grid(grid, theta0, steps, return_metadata=True, **options)
        results[name] = {"arrays": out, "compare": compare, "wall_seconds": time.perf_counter() - started,
                         "kernel_seconds": meta["kernel_seconds"], "cells": meta["cells"], "steps": steps,
                         "us_per_cell_step": meta["microseconds_per_cell_step"],
                         "qualifying_run": meta.get("qualifying_run")}
        print(json.dumps({"workload": name, "cells": meta["cells"], "steps": steps,
                          "us_per_cell_step": round(meta["microseconds_per_cell_step"], 5)}), flush=True)
    return backend, results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["record", "compare"])
    parser.add_argument("--library", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    backend, results = execute(args.library)
    receipt = {"mode": args.mode, "environment": environment(backend), "workloads": {}}
    for name, r in results.items():
        entry = {k: v for k, v in r.items() if k != "arrays"}
        entry["sha256"] = {k: digest(v) for k, v in r["arrays"].items()}
        if args.mode == "record" and r["compare"]:
            np.savez(args.out / f"{name}.npz", **r["arrays"])
        if args.mode == "compare" and r["compare"]:
            ref = np.load(args.reference / f"{name}.npz")
            diffs = {}
            for key, value in r["arrays"].items():
                other = ref[key]
                same = value.shape == other.shape and value.dtype == other.dtype and \
                    np.ascontiguousarray(value).tobytes() == np.ascontiguousarray(other).tobytes()
                info = {"bitwise_equal": bool(same)}
                if not same and value.shape == other.shape and np.issubdtype(value.dtype, np.number):
                    a, b = value.astype(np.float64), other.astype(np.float64)
                    mismatch = ~((a == b) | (np.isnan(a) & np.isnan(b)))
                    finite = np.isfinite(a) & np.isfinite(b)
                    info["mismatched_entries"] = int(mismatch.sum())
                    info["max_abs_finite_difference"] = float(np.max(np.abs(a - b)[finite])) if finite.any() else None
                diffs[key] = info
            entry["comparison"] = diffs
            entry["all_bitwise_equal"] = all(d["bitwise_equal"] for d in diffs.values())
        receipt["workloads"][name] = entry
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=1, default=str))
    if args.mode == "compare":
        verdict = {n: w.get("all_bitwise_equal") for n, w in receipt["workloads"].items() if "comparison" in w}
        print(json.dumps({"bitwise_equal_by_workload": verdict}))


if __name__ == "__main__":
    main()
