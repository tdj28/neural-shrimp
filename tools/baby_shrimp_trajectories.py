"""Bounded exploratory trajectories for the baby/mother shrimp comparison.

Save full float64 states, finite-time twin growth and six-step direction signs.
Recurrence candidates are not Newton-verified orbits. The local parent owns one
GPU worker, retains partial evidence and enforces time/RAM limits.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("tools/baby_shrimp_trajectories.py", "tools/batch_fate_finish.py", "tools/batch_fate_summary.py",
           "tools/explore_llm_soft_loop.py", "tools/explore_llm_attractors.py", "tools/explore_llm_phase_map.py")


def write(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def recurrence(states, max_period=192, tol=1e-9):
    """Require every one of the last max_period states to return, at each lag."""
    import numpy as np
    if len(states) < 2 * max_period or not np.isfinite(states).all():
        raise ValueError("finite trajectory with at least two maximum periods required")
    tail = states[-max_period:]
    scale = np.maximum(np.linalg.norm(tail, axis=-1), 1.0)
    distances = [float(np.max(np.linalg.norm(tail - states[-max_period-q:-q], axis=-1) / scale))
                 for q in range(1, max_period+1)]
    return {"period_candidate": next((q for q, d in enumerate(distances, 1) if d < tol), 0),
            "max_relative_return_by_lag": distances, "tolerance": tol, "phases_checked": max_period}


def worker(args):
    import numpy as np
    import torch
    import explore_llm_soft_loop as soft

    torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(.85)
    plan = json.loads(args.cells.read_text())
    cells = plan["cells"]
    n = len(cells)
    loop = soft.SoftLoop("qwen0.5b", 12, 8, "open", "Hello!", 1, "float64", "cuda", "native")
    if loop.sm.engine.revision != "7ae557604adf67be50417f59c2c2f167def9a775":
        raise ValueError("unexpected model revision")
    norm = loop.model.model.norm
    test = torch.linspace(-1, 1, norm.weight.numel(), dtype=torch.float64, device=loop.device)[None,None,:]
    sensitivity = float((norm(test * (1+1e-12))-norm(test)).abs().max())
    if not (0 < sensitivity < 1e-9 and hasattr(loop.model.model.rotary_emb, "_inv_freq64")):
        raise ValueError("effective precision check failed")
    vector = soft.elpm.load_vectors("llm-phase-map/vectors-l12")["philosophy"]
    steering = np.array([c["alpha"] * vector for c in cells])
    temperatures = np.tile([10.0**c["log10_temperature"] for c in cells], 2)
    xy = loop.prepare(np.concatenate([steering, steering]))
    started = time.monotonic()
    protocol = {"model_revision": loop.sm.engine.revision, "dtype": str(xy.dtype), "device": str(xy.device),
                "prefix_tokens": loop.L0, "norm_sensitivity_to_1e_12": sensitivity,
                "rotary_float64": True, "vector_sha256": hashlib.sha256(vector.tobytes()).hexdigest(),
                "transient": args.transient, "record": args.record, "eps": 1e-6,
                "phase_period": 6, "max_period": 192, "batch_cells": n, "seed": 0, "cells": cells}
    write(args.out / "protocol.json", protocol)

    def progress(phase, step):
        row = {"phase": phase, "step": step, "seconds": round(time.monotonic()-started, 2)}
        write(args.out / "progress.json", row)
        print(json.dumps(row), flush=True)

    for k in range(args.transient):
        xy, _ = loop.step(xy, temperatures)
        if k % 100 == 99:
            progress("transient", k+1)
    rng = np.random.default_rng(0)
    delta = torch.as_tensor(rng.normal(size=tuple(xy[:n].shape)), dtype=xy.dtype, device=xy.device)
    delta = delta / delta.flatten(1).norm(dim=1)[:,None,None] * 1e-6
    xy = torch.cat([xy[:n], xy[:n]+delta])
    states = np.lib.format.open_memmap(args.out / "states.npy", mode="w+", dtype=np.float64,
                                      shape=(args.record, n, xy.shape[-1]))
    tokens = np.zeros((args.record,n), dtype=np.int32)
    growth = np.zeros((args.record,n), dtype=np.float64)
    cosine = np.full((args.record,n), np.nan)
    wbuf = [None] * 6
    for k in range(args.record):
        xy, probs = loop.step(xy, temperatures)
        x, y = xy[:n], xy[n:]
        diff = y-x
        dist = diff.flatten(1).norm(dim=1).double().cpu().numpy()
        if not np.isfinite(dist).all() or not torch.isfinite(x).all():
            raise ValueError(f"nonfinite trajectory at recorded step {k}")
        dist = np.maximum(dist, 1e-36)
        growth[k] = np.log(dist/1e-6)
        w = diff * torch.as_tensor(1/dist, dtype=xy.dtype, device=xy.device)[:,None,None]
        if wbuf[k%6] is not None:
            cosine[k] = (w * wbuf[k%6]).flatten(1).sum(1).double().cpu().numpy()
        wbuf[k%6] = w
        states[k] = x[:,0,:].double().cpu().numpy()
        tokens[k] = probs[:n].argmax(-1).cpu().numpy()
        y = x + diff * torch.as_tensor(1e-6/dist, dtype=xy.dtype, device=xy.device)[:,None,None]
        xy = torch.cat([x,y])
        if k % 100 == 99:
            states.flush()
            np.savez_compressed(args.out / "diagnostics-partial.npz", recorded=k+1,
                                tokens=tokens[:k+1], log_growth=growth[:k+1], word_cos=cosine[:k+1])
            progress("record", k+1)
    states.flush()
    results = []
    for i, cell in enumerate(cells):
        r = {**cell, **recurrence(states[:,i,:]), "lyapunov": float(growth[:,i].mean()),
             "lyapunov_halves": [float(v.mean()) for v in np.array_split(growth[:,i], 2)]}
        text = tokens[:,i]
        r["text_period"] = next((q for q in range(1,193) if np.array_equal(text[q:],text[:-q])), 0)
        if r["text_period"]:
            r["text_token_ids"] = text[:r["text_period"]].tolist()
            r["text"] = [loop.sm.tok.decode([int(t)]) for t in text[:r["text_period"]]]
        results.append(r)
    write(args.out / "trajectories.json", {"protocol": protocol, "cells": results,
                                           "seconds": time.monotonic()-started,
                                           "verified_orbits": False})
    np.savez_compressed(args.out / "diagnostics.npz", tokens=tokens, log_growth=growth, word_cos=cosine)
    print(json.dumps({"status": "complete", "candidates": [{k:r[k] for k in ("name", "period_candidate", "text_period", "lyapunov")}
                                                              for r in results]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--transient", type=int, default=1000)
    parser.add_argument("--record", type=int, default=3000)
    parser.add_argument("--timeout", type=int, default=2700)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cells = json.loads(args.cells.read_text())["cells"]
    if not (1 <= len(cells) <= 16 and 600 <= args.transient <= 3000 and 600 <= args.record <= 6000
            and args.record % 6 == 0 and 0 < args.timeout <= 5400):
        parser.error("resource bounds exceeded")
    if args.worker:
        worker(args)
        return
    import batch_fate_finish as guard
    args.out.mkdir(parents=True, exist_ok=False)
    commit = subprocess.check_output(["git", "rev-parse", args.source_commit], cwd=ROOT, text=True).strip()
    hashes = {}
    for source in SOURCES:
        blob = subprocess.check_output(["git", "show", commit+":"+source], cwd=ROOT)
        hashes[source] = hashlib.sha256(blob).hexdigest()
        if hashes[source] != hashlib.sha256((ROOT/source).read_bytes()).hexdigest():
            raise ValueError(f"uncommitted source: {source}")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "8"
    os.environ["HF_HUB_OFFLINE"] = "1"
    status = {"source_commit": commit, "source_sha256": hashes, "cells_sha256": hashlib.sha256(args.cells.read_bytes()).hexdigest(),
              "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}

    def event(state, **kw):
        with (args.out / "events.jsonl").open("a") as f:
            f.write(json.dumps({"time": datetime.now(timezone.utc).isoformat(), "state": state, **kw})+"\n")

    def interrupted(signum, frame):
        raise InterruptedError(f"trajectory parent received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        with guard.gpu_lock(guard.summary.DATA / "render-tmp/gpu.lock", timeout=1):
            status["resources"] = guard.resource_snapshot()
            write(args.out / "status.json", status)
            command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--cells", str(args.cells.resolve()),
                       "--out", str(args.out.resolve()), "--source-commit", commit, "--transient", str(args.transient),
                       "--record", str(args.record), "--timeout", str(args.timeout)]
            guard.run_bounded(command, args.out/"worker.log", args.timeout, event, monitor_memory=True)
        status["status"] = "complete"
    except BaseException as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        status["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write(args.out / "status.json", status)


if __name__ == "__main__":
    main()
