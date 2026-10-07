"""Phase portraits of the two-step map at a few learning rates (exploratory, real data).

Starts each rate from a saved state, settles, then records ``--keep`` updates and
projects every second state on its first two principal axes (the plane the
rotation number is measured in). Points are colored by the parity of their
two-step index, so a period-doubled circle (two loops visited alternately)
shows as two interleaved colors; a single invariant circle shows as one loop.
Also prints the rotation number of the two-step and four-step maps.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

import explore_realdata as tool  # noqa: E402
from neural_shrimp.gdmaps import GridMap  # noqa: E402

SURFACE, INK, MUTED, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#898781", "#2a78d6", "#eb6834"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", required=True)
    parser.add_argument("--rates", type=float, nargs="+", required=True, help="log10 learning rates")
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=28101)
    parser.add_argument("--steps", type=int, default=40000)
    parser.add_argument("--keep", type=int, default=8192)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    config = {"seed": args.seed, "width": 8, "n": 1024, "task": "halves", "ridge": args.ridge}
    net, x, y, _ = tool.network(config)
    fig, axes = plt.subplots(1, len(args.rates), figsize=(4.2 * len(args.rates), 4.2), constrained_layout=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)
    for ax, le in zip(axes, args.rates):
        grid = GridMap(net, x, y, [10.0 ** le], [10.0 ** le])
        state = np.load(args.init)[None]
        for _ in range(args.steps):
            state = grid.step(state)
        record = np.empty((args.keep, net.n_parameters))
        for k in range(args.keep):
            record[k] = state[0]
            state = grid.step(state)
        lobe = record[0::2]
        centered = lobe - lobe.mean(axis=0)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        planar = centered @ vt[:2].T
        rho2 = tool.rotation(lobe)
        rho4 = tool.rotation(record[0::4])
        ax.plot(planar[0::2, 0], planar[0::2, 1], ".", markersize=1.5, color=BLUE, label="even two-step index")
        ax.plot(planar[1::2, 0], planar[1::2, 1], ".", markersize=1.5, color=ORANGE, label="odd two-step index")
        ax.set_title(f"log10 eta {le:+.4f}\nrho2 {rho2[0]:.5f} (fwd {rho2[1]:.2f}), rho4 {rho4[0]:.5f} "
                     f"(fwd {rho4[1]:.2f})", fontsize=8, color=INK, loc="left")
        ax.tick_params(labelsize=7, colors=MUTED)
        ax.set_facecolor(SURFACE)
        print(f"log10 eta {le:+.5f}: rho2 {rho2[0]:.6f} fwd {rho2[1]:.3f} share {rho2[2]:.3f} | "
              f"rho4 {rho4[0]:.6f} fwd {rho4[1]:.3f} share {rho4[2]:.3f}", flush=True)
    axes[0].legend(fontsize=7, frameon=False, loc="best", markerscale=6)
    fig.savefig(args.out, dpi=160, facecolor=SURFACE)


if __name__ == "__main__":
    main()
