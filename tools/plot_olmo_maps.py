"""Figure: OLMo-2-1124-7B-Instruct soft-token maps (run S8, layer 16 of 32, 64 x 64 float64 cells), steered along the
philosophy direction (top) and the hostility-vs-kindness direction (bottom): largest Lyapunov exponent (clipped) and
the period of the attractor reached from the fixed start (white: none up to the maximum period).

Usage: python tools/plot_olmo_maps.py OUT.png
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

MAPS = [("philosophy", "llm-soft-loop/pod64/scalex-olmo2-7b/coarse-64-philosophy"),
        ("hostility-vs-kindness", "llm-soft-loop/pod64/scalex-olmo2-7b/coarse-64-hostility")]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import explore_llm_attractors as ela

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6))
    for row, (concept, folder) in enumerate(MAPS):
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
    fig.savefig(sys.argv[1], dpi=200)


if __name__ == "__main__":
    main()
