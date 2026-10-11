"""Exploratory GPU rotation-number map (A3) on the qualified CUDA backend.

Stage 1: the backend's run_grid (EXP-028 screening options) gives status,
period, FTLE and the final state of every cell.
Stage 2: the final states are uploaded as per-cell initial states and advanced
two updates at a time with all recording windows disabled. The first ``pca``
two-step samples accumulate a per-cell mean and covariance (principal plane);
the next ``samples`` two-step samples accumulate the unwrapped angle in that
plane, giving the rotation number in turns per two steps.

Writes DATA/explore/<name>/rotation.npz with the keys of
tools/explore_a3_rotation_map.py, plus final_state and metadata.json.
"""

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp import gdmaps_cuda as gc  # noqa: E402

try:  # optional: GPU-side bookkeeping for the rotation stage
    import torch
    if not torch.cuda.is_available():
        torch = None
except ImportError:
    torch = None
from neural_shrimp.gdmaps import GridMap  # noqa: E402
from neural_shrimp.trainability import SYSTEMS  # noqa: E402

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
PLAN = ROOT / "experiments/trainability/EXP-028-plan.json"
NEVER = 2 ** 30 - 1


def rotation_stage(backend, grid, final_state, alive, pca, samples, chunk, device=0):
    net = grid.net
    n, p, cells = len(grid.x), net.n_parameters, grid.cells
    cfg = gc._config(net, n, capacity=chunk, tail_capacity=1, sd_max=0, tail_steps=1, device=device,
                     max_bytes=12 * 1024 ** 3)
    schedule = gc._Schedule(ftle_start=NEVER, ftle_log_start=NEVER, tail_start=NEVER, sd_max=0,
                            norm_start=NEVER, divergence_threshold=1e8)
    x, y = np.ascontiguousarray(grid.x), np.ascontiguousarray(grid.y)
    rho = np.full(cells, np.nan)
    share = np.full(cells, np.nan)
    handle = ctypes.c_void_p()
    dp, ip, lp = ctypes.c_double, ctypes.c_int, ctypes.c_longlong
    backend.device_info(device)
    backend._call("gdmap_cuda_create", ctypes.byref(cfg), gc._pointer(x, dp), gc._pointer(y, dp),
                  ctypes.byref(handle))
    try:
        for lo in range(0, cells, chunk):
            hi = min(cells, lo + chunk)
            m = hi - lo
            theta_soa = np.full((p, chunk), np.nan)
            theta_soa[:, :m] = final_state[lo:hi].T
            tangent_soa = np.zeros((p, chunk))
            lr_h = np.zeros(chunk)
            lr_o = np.zeros(chunk)
            lr_h[:m], lr_o[:m] = grid.rates[lo:hi, 0], grid.rates[lo:hi, -1]
            alive_in = np.zeros(chunk, dtype=np.intc)
            alive_in[:m] = alive[lo:hi]
            diverged_in = np.full(chunk, -1, dtype=np.longlong)
            backend._call("gdmap_cuda_upload", handle, m, gc._pointer(theta_soa, dp),
                          gc._pointer(tangent_soa, dp), gc._pointer(lr_h, dp), gc._pointer(lr_o, dp),
                          gc._pointer(alive_in, ip), gc._pointer(diverged_in, lp))
            out = {"theta": np.empty((p, chunk)), "alive": np.empty(chunk, dtype=np.intc),
                   "diverged": np.empty(chunk, dtype=np.longlong), "ftle_ok": np.empty(chunk, dtype=np.intc),
                   "log": np.empty(chunk), "min": np.empty(chunk), "last": np.empty(chunk),
                   "tsum": np.empty(chunk), "tcount": np.empty(chunk, dtype=np.longlong),
                   "wnorm": np.empty(chunk), "early": np.empty((1, chunk))}
            seconds = ctypes.c_double()
            step = 0

            def advance_two():
                nonlocal step
                backend._call("gdmap_cuda_advance", handle, 0, m, step, step + 2, ctypes.byref(schedule),
                              0, 0, ctypes.byref(seconds))
                step += 2
                backend._call("gdmap_cuda_fetch", handle, gc._pointer(out["theta"], dp),
                              gc._pointer(out["alive"], ip), gc._pointer(out["diverged"], lp),
                              gc._pointer(out["ftle_ok"], ip), gc._pointer(out["log"], dp),
                              gc._pointer(out["min"], dp), gc._pointer(out["last"], dp),
                              gc._pointer(out["tsum"], dp), gc._pointer(out["tcount"], lp),
                              gc._pointer(out["wnorm"], dp), gc._pointer(out["early"], dp))
                return out["theta"][:, :m].T

            # Moments, principal planes and angle bookkeeping run in torch on the
            # GPU when available (the numpy path is host-bound for 10^6 cells).
            xp = torch if torch is not None else None
            dev = "cuda" if torch is not None else None

            def as_array(a):
                return torch.as_tensor(np.nan_to_num(a), device=dev) if xp is not None else np.nan_to_num(a)

            t = as_array(final_state[lo:hi])
            mean = t * 0.0
            second = (t[:, :, None] * t[:, None, :]) * 0.0
            for _ in range(pca):
                mean += t
                second += t[:, :, None] * t[:, None, :]
                t = as_array(advance_two())
            mean /= pca
            cov = second / pca - mean[:, :, None] * mean[:, None, :]
            del second
            # Batched eigh of many small matrices: cuSOLVER's batched workspace grows
            # far beyond device memory (232 GB requested for 409,600 13x13 blocks),
            # so the decomposition runs on the host and the planes return to the GPU.
            vals, vecs = np.linalg.eigh(cov.cpu().numpy() if xp is not None else cov)
            del cov
            if xp is not None:
                vals, vecs = torch.as_tensor(vals, device=dev), torch.as_tensor(vecs, device=dev)
            plane = vecs[:, :, -2:]
            ratio = vals[:, -2:].sum(1) / vals.sum(1).clamp(min=1e-300) if xp is not None else \
                vals[:, -2:].sum(1) / np.maximum(vals.sum(1), 1e-300)
            share[lo:hi] = ratio.cpu().numpy() if xp is not None else ratio
            two_pi = 2 * np.pi

            def angle(state):
                z = ((state - mean)[:, :, None] * plane).sum(1)
                return torch.atan2(z[:, 1], z[:, 0]) if xp is not None else np.arctan2(z[:, 1], z[:, 0])

            previous = angle(t)
            total = previous * 0.0
            last = None
            for _ in range(samples):
                last = advance_two()
                current = angle(as_array(last))
                total += (current - previous + np.pi) % two_pi - np.pi
                previous = current
            total = total.cpu().numpy() if xp is not None else total
            ok = (out["alive"][:m] != 0) & alive[lo:hi] & np.all(np.isfinite(last), axis=1)
            rho[lo:hi] = np.where(ok, np.abs(total) / (2 * np.pi * samples), np.nan)
            share[lo:hi] = np.where(ok, share[lo:hi], np.nan)
            print(json.dumps({"chunk": [lo, hi], "of": cells}), flush=True)
    finally:
        backend._lib.gdmap_cuda_destroy(handle)
    return rho, share


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--hidden", type=float, nargs=2, required=True, help="log10 range")
    parser.add_argument("--output", type=float, nargs=2, required=True, help="log10 range")
    parser.add_argument("--resolution", type=int, nargs=2, default=(1024, 1024), help="rows, cols")
    parser.add_argument("--pca", type=int, default=256)
    parser.add_argument("--samples", type=int, default=6000)
    parser.add_argument("--chunk", type=int, default=0, help="cells per rotation chunk (0: automatic)")
    parser.add_argument("--library", required=True)
    args = parser.parse_args()
    out = DATA / "explore" / args.name
    out.mkdir(parents=True, exist_ok=False)
    it = json.loads(PLAN.read_text())["iteration"]
    options = {k: it[k] for k in ("tail_steps", "ftle_steps", "ftle_align", "max_period", "min_repeats",
                                  "atol", "rtol", "chaos_threshold", "drift_threshold",
                                  "divergence_threshold", "tangent_seed")}
    net, x, y, theta0 = SYSTEMS[args.system](args.seed)
    rows, cols = args.resolution
    log_h = np.linspace(*args.hidden, rows)
    log_o = np.linspace(*args.output, cols)
    r, c = np.divmod(np.arange(rows * cols), cols)
    grid = GridMap(net, x, y, 10.0 ** log_h[r], 10.0 ** log_o[c])
    backend = gc.load(args.library)
    started = time.perf_counter()
    labels, meta = backend.run_grid(grid, theta0, it["steps"], sd_horizons=(), return_metadata=True,
                                    **options)
    stage1 = time.perf_counter() - started
    print(json.dumps({"stage1_seconds": round(stage1, 1)}), flush=True)
    p = net.n_parameters
    chunk = args.chunk or int(max(1024, min(grid.cells, 1.5e9 // (p * p * 8))))
    alive = (labels["status"] != 0) & np.all(np.isfinite(labels["final_state"]), axis=1)
    rho, share = rotation_stage(backend, grid, labels["final_state"], alive, args.pca, args.samples, chunk)
    seconds = time.perf_counter() - started
    shape = (rows, cols)
    np.savez(out / "rotation.npz", log10_hidden=log_h, log10_output=log_o, rho=rho.reshape(shape),
             planar_share=share.reshape(shape), status=labels["status"].reshape(shape),
             period=labels["period"].reshape(shape), ftle=labels["ftle"].reshape(shape),
             final_state=labels["final_state"].reshape(rows, cols, -1))
    (out / "metadata.json").write_text(json.dumps({"args": vars(args), "steps": it["steps"], "options": options,
                                                   "stage1_seconds": stage1, "seconds": seconds,
                                                   "backend": meta, "chunk": chunk}, indent=1, default=str))
    print(json.dumps({"out": str(out), "seconds": round(seconds, 1)}))


if __name__ == "__main__":
    main()
