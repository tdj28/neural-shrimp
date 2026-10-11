"""Exploratory A3: rotation-number staircase along a one-dimensional rate slice.

Iterates many cells of a slice (one learning rate varied, the other fixed) on
CPU, keeps the final ``keep`` states, and for every cell estimates the
rotation number of the two-step map from its own principal-axis projection.
Locked (periodic) cells give rational plateaus; quasi-periodic cells give
irrational values; chaotic cells give noisy estimates. Also records the
fraction of variance in the first two axes (a crude circle-quality measure).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.gdmaps import GridMap, run_grid  # noqa: E402
from neural_shrimp.trainability import SYSTEMS  # noqa: E402


def rotation(states):
    lobe = states[0::2]
    centered = lobe - lobe.mean(0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    planar = centered @ vt[:2].T
    angles = np.unwrap(np.arctan2(planar[:, 1], planar[:, 0]))
    rho = abs(angles[-1] - angles[0]) / (2 * np.pi * (len(angles) - 1))
    share = float((singular[:2] ** 2).sum() / max((singular ** 2).sum(), 1e-300))
    return float(rho), share


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--vary", choices=["hidden", "output"], required=True)
    parser.add_argument("--fixed", type=float, required=True, help="log10 of the fixed rate")
    parser.add_argument("--range", type=float, nargs=2, required=True, help="log10 range of the varied rate")
    parser.add_argument("--cells", type=int, default=2000)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--keep", type=int, default=4000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    net, x, y, theta0 = SYSTEMS[args.system](args.seed)
    varied = np.linspace(args.range[0], args.range[1], args.cells)
    fixed = np.full(args.cells, args.fixed)
    lh, lo = (varied, fixed) if args.vary == "hidden" else (fixed, varied)
    grid = GridMap(net, x, y, 10.0 ** lh, 10.0 ** lo)
    labels = run_grid(grid, theta0, 20000, tail_steps=384, ftle_steps=4000, ftle_align=1000, sd_horizons=())
    theta = np.repeat(theta0[None], args.cells, axis=0)
    kept = np.empty((args.keep, args.cells, len(theta0)))
    alive = np.ones(args.cells, dtype=bool)
    with np.errstate(over="ignore", invalid="ignore"):
        for step in range(args.steps):
            theta = grid.step(theta)
            alive &= np.all(np.isfinite(theta), axis=1) & (np.max(np.abs(theta), axis=1) < 1e8)
            theta[~alive] = 0.0
            if step >= args.steps - args.keep:
                kept[step - (args.steps - args.keep)] = theta
    rows = []
    for i in range(args.cells):
        rho, share = rotation(kept[:, i, :]) if alive[i] else (None, None)
        rows.append({"log10_varied": float(varied[i]), "rho": rho, "planar_share": share,
                     "status": int(labels["status"][i]), "period": int(labels["period"][i]),
                     "ftle": None if not np.isfinite(labels["ftle"][i]) else float(labels["ftle"][i])})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "staircase.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    colors = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#a3a29b", 5: "#a3a29b"}
    for code, label in ((2, "periodic"), (3, "chaotic candidate"), (4, "unresolved"), (5, "drift")):
        pts = [(r["log10_varied"], r["rho"]) for r in rows if r["status"] == code and r["rho"] is not None]
        if pts:
            a = np.array(pts)
            ax.plot(a[:, 0], a[:, 1], ".", markersize=2.5, color=colors[code], label=label)
    for frac, text in ((1 / 5, "1/5"), (2 / 9, "2/9"), (1 / 4, "1/4"), (2 / 7, "2/7"), (1 / 3, "1/3"),
                       (3 / 13, "3/13"), (3 / 14, "3/14"), (3 / 11, "3/11")):
        ax.axhline(frac, color="#e1e0d9", linewidth=0.6, zorder=0)
        ax.text(varied[-1], frac, f" {text}", va="center", fontsize=8, color="#898781")
    ax.set_xlabel(f"log10 {args.vary}-layer learning rate (other fixed at {args.fixed:.4f})", fontsize=9)
    ax.set_ylabel("rotation number of the two-step map (turns per step)", fontsize=9)
    ax.set_title(f"{args.system}-{args.seed}: rotation-number staircase (exploratory)", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.savefig(out / "staircase.png", dpi=160)
    print(json.dumps({"out": str(out)}))


if __name__ == "__main__":
    main()
