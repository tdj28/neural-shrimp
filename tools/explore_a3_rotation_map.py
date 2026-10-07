"""Exploratory A3: two-dimensional rotation-number map of the two-step map.

Every cell of a learning-rate rectangle is iterated from the shared initial
parameters with the EXP-028 screening options (status, period, FTLE), then
continued from its final state: the first ``pca`` two-step samples fix a
principal plane and centroid per cell, and the unwrapped angle of later
two-step samples in that plane gives the rotation number (turns per two
steps). Locked cells give m/q with period 2q; quasi-periodic cells give
irrational values with FTLE near zero; chaotic cells give noisy values.

Cells are split into row blocks processed in parallel. Output: rotation.npz.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.gdmaps import GridMap, run_grid  # noqa: E402
from neural_shrimp.trainability import SYSTEMS  # noqa: E402

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
PLAN = ROOT / "experiments/trainability/EXP-028-plan.json"


def block(task):
    system, seed, lh, lo, pca, samples = task
    it = json.loads(PLAN.read_text())["iteration"]
    options = {k: it[k] for k in ("tail_steps", "ftle_steps", "ftle_align", "max_period", "min_repeats",
                                  "atol", "rtol", "chaos_threshold", "drift_threshold",
                                  "divergence_threshold", "tangent_seed")}
    net, x, y, theta0 = SYSTEMS[system](seed)
    grid = GridMap(net, x, y, 10.0 ** lh, 10.0 ** lo)
    labels = run_grid(grid, theta0, it["steps"], sd_horizons=(), **options)
    theta = labels["final_state"].copy()
    alive = np.all(np.isfinite(theta), axis=1) & (labels["status"] != 0)
    theta[~alive] = 0.0
    lobe = np.empty((pca, len(lh), theta.shape[1]))
    with np.errstate(over="ignore", invalid="ignore"):
        for k in range(pca):
            lobe[k] = theta
            theta = grid.step(grid.step(theta))
        centroid = lobe.mean(0)
        centered = np.transpose(lobe - centroid[None], (1, 0, 2))
        _, singular, vt = np.linalg.svd(np.nan_to_num(centered), full_matrices=False)
        plane = vt[:, :2, :]
        share = (singular[:, :2] ** 2).sum(1) / np.maximum((singular ** 2).sum(1), 1e-300)

        def angle(t):
            planar = np.einsum("cp,cjp->cj", t - centroid, plane)
            return np.arctan2(planar[:, 1], planar[:, 0])

        previous = angle(theta)
        total = np.zeros(len(lh))
        for _ in range(samples):
            theta = grid.step(grid.step(theta))
            current = angle(theta)
            total += (current - previous + np.pi) % (2 * np.pi) - np.pi
            previous = current
    alive &= np.all(np.isfinite(theta), axis=1)
    rho = np.where(alive, np.abs(total) / (2 * np.pi * samples), np.nan)
    return {"rho": rho, "planar_share": np.where(alive, share, np.nan), "status": labels["status"],
            "period": labels["period"], "ftle": labels["ftle"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--hidden", type=float, nargs=2, required=True, help="log10 range")
    parser.add_argument("--output", type=float, nargs=2, required=True, help="log10 range")
    parser.add_argument("--resolution", type=int, nargs=2, default=(192, 192), help="hidden rows, output cols")
    parser.add_argument("--pca", type=int, default=256, help="two-step samples fixing the plane")
    parser.add_argument("--samples", type=int, default=6000, help="two-step samples for the rotation")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    rows, cols = args.resolution
    log_h = np.linspace(*args.hidden, rows)
    log_o = np.linspace(*args.output, cols)
    r, c = np.divmod(np.arange(rows * cols), cols)
    chunks = np.array_split(np.arange(rows * cols), args.workers * 4)
    tasks = [(args.system, args.seed, log_h[r[i]], log_o[c[i]], args.pca, args.samples) for i in chunks]
    start = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        parts = list(pool.map(block, tasks, chunksize=1))
    merged = {k: np.concatenate([p[k] for p in parts]).reshape(rows, cols) for k in parts[0]}
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "rotation.npz", log10_hidden=log_h, log10_output=log_o, **merged)
    (out / "metadata.json").write_text(json.dumps({"args": vars(args),
                                                   "seconds": time.perf_counter() - start}, indent=1))
    print(json.dumps({"out": str(out), "seconds": round(time.perf_counter() - start, 1)}))


if __name__ == "__main__":
    main()
