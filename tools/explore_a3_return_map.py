"""Exploratory A3: principal-axis return maps of chaotic training near a window."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from neural_shrimp.gdmaps import GridMap  # noqa: E402
from neural_shrimp.trainability import SYSTEMS  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--log-hidden", type=float, required=True)
    parser.add_argument("--log-output", type=float, required=True)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--keep", type=int, default=20000)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    net, x, y, theta0 = SYSTEMS[args.system](args.seed)
    grid = GridMap(net, x, y, [10.0 ** args.log_hidden], [10.0 ** args.log_output])
    theta = theta0[None].copy()
    states = []
    for step in range(args.steps):
        theta = grid.step(theta)
        if step >= args.steps - args.keep:
            states.append(theta[0].copy())
    states = np.array(states)
    centered = states - states.mean(axis=0)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    xi = centered @ vt[0]
    explained = (singular ** 2 / np.sum(singular ** 2))[:5]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, lag in zip(axes, (1, 2, 4)):
        ax.plot(xi[:-lag], xi[lag:], ",", color="#2a78d6", alpha=0.6)
        ax.set_xlabel("first principal coordinate at t", fontsize=9)
        ax.set_ylabel(f"... at t + {lag}", fontsize=9)
        ax.set_title(f"return map, lag {lag}", loc="left", fontsize=10)
    fig.suptitle(f"{args.system}-{args.seed} at log10 rates ({args.log_hidden:.4f}, {args.log_output:.4f}); "
                 f"explained variance of first 5 axes {np.round(explained, 3).tolist()} (exploratory)",
                 fontsize=10, x=0.01, ha="left")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "return-maps.png", dpi=150)
    np.savez(out / "trajectory.npz", xi=xi, explained=explained, components=vt[:5])
    print(json.dumps({"explained": explained.tolist()}))


if __name__ == "__main__":
    main()
