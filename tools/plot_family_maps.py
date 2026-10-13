"""Figure: coarse float64 soft-token maps of one model along several steering directions (one row per map): largest
Lyapunov exponent (clipped) and the period of the attractor reached from the fixed start (white: none).

Usage: python tools/plot_family_maps.py OUT.png CONCEPT=SCAN_DIR [CONCEPT=SCAN_DIR ...]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import explore_llm_attractors as ela

    out, specs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
    fig, axes = plt.subplots(len(specs), 2, figsize=(10.5, 3.8 * len(specs)), squeeze=False)
    for row, (concept, folder) in enumerate(specs):
        z = np.load(ela.DATA / "explore" / folder / "scan.npz")
        lt, al, L, P = z["log10_temperature"], z["alpha"], z["lyapunov"], z["period"]
        ext = [al[0], al[-1], lt[0], lt[-1]]
        print(concept, "chaotic share", round(float((L > 1e-3).mean()), 4), "cells", L.size)
        im = axes[row, 0].imshow(np.clip(L, -0.5, 0.5), origin="lower", extent=ext, aspect="auto", cmap="RdBu_r",
                                 vmin=-0.5, vmax=0.5, interpolation="nearest")
        fig.colorbar(im, ax=axes[row, 0], label="largest Lyapunov exponent (clipped)")
        cmap = plt.get_cmap("tab20").copy()
        cmap.set_bad("white")
        im = axes[row, 1].imshow(np.where(P > 0, P, np.nan), origin="lower", extent=ext, aspect="auto", cmap=cmap,
                                 vmin=0.5, vmax=40.5, interpolation="nearest")
        fig.colorbar(im, ax=axes[row, 1], label="period")
        for ax in axes[row]:
            ax.set_xlabel(f"steering along '{concept}'")
            ax.set_ylabel("log10 temperature")
    fig.tight_layout()
    fig.savefig(out, dpi=200)


if __name__ == "__main__":
    main()
