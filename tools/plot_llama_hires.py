"""Figure: 128 x 128 float64 soft-token maps of Llama-3.1-8B-Instruct (run L14), each merged from four 64 x 64 tiles
whose axes are the halves of the 128-point grid: largest Lyapunov exponent (clipped) and the period of the attractor
reached from the fixed start (white: none). Missing tiles are left blank and reported.

Usage: python tools/plot_llama_hires.py OUT.png PLANE=TILE_PREFIX [PLANE=TILE_PREFIX ...]
       (e.g. hostility=llm-soft-loop/pod64/llama8b-hires/host)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def merge(prefix):
    """Merge tiles PREFIX-{a}{t} (a: alpha half, t: temperature half) into one grid; checks that the tiles' axes
    are the halves of one linspace."""
    import explore_llm_attractors as ela

    tiles, axes = {}, {}
    for a in (0, 1):
        for t in (0, 1):
            f = ela.DATA / "explore" / f"{prefix}-{a}{t}" / "scan.npz"
            if f.exists():
                tiles[a, t] = np.load(f)
    if not tiles:
        raise SystemExit(f"no tiles for {prefix}")
    n = next(iter(tiles.values()))["lyapunov"].shape[0]
    L = np.full((2 * n, 2 * n), np.nan)
    P = np.zeros((2 * n, 2 * n), dtype=int)
    for (a, t), z in tiles.items():
        L[t * n:(t + 1) * n, a * n:(a + 1) * n] = z["lyapunov"]
        P[t * n:(t + 1) * n, a * n:(a + 1) * n] = z["period"]
        axes.setdefault("alpha", {})[a] = z["alpha"]
        axes.setdefault("logT", {})[t] = z["log10_temperature"]
    full = {}
    for key in ("alpha", "logT"):
        halves = axes[key]
        if len(halves) == 2:
            grid = np.concatenate([halves[0], halves[1]])
            step = np.diff(grid)
            if not np.allclose(step, step[0], rtol=1e-9):
                raise SystemExit(f"{prefix}: {key} tiles are not halves of one grid")
            full[key] = grid
        else:                                      # one half only: extend with its own spacing
            (k, h), = halves.items()
            d = h[1] - h[0]
            full[key] = np.concatenate([h - n * d, h]) if k == 1 else np.concatenate([h, h + n * d])
    return L, P, full["alpha"], full["logT"], sorted(tiles)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out, specs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
    fig, axes = plt.subplots(len(specs), 2, figsize=(11, 4.4 * len(specs)), squeeze=False)
    for row, (plane, prefix) in enumerate(specs):
        L, P, al, lt, present = merge(prefix)
        done = ~np.isnan(L)
        chaotic = (L > 1e-3) & (P <= 0) & done
        print(plane, "tiles", present, "cells", int(done.sum()), "chaotic share",
              round(float(chaotic.sum() / max(done.sum(), 1)), 4))
        ext = [al[0], al[-1], lt[0], lt[-1]]
        im = axes[row, 0].imshow(np.clip(L, -0.5, 0.5), origin="lower", extent=ext, aspect="auto", cmap="RdBu_r",
                                 vmin=-0.5, vmax=0.5, interpolation="nearest")
        fig.colorbar(im, ax=axes[row, 0], label="largest Lyapunov exponent (clipped)")
        cmap = plt.get_cmap("tab20").copy()
        cmap.set_bad("white")
        im = axes[row, 1].imshow(np.where(P > 0, P, np.nan), origin="lower", extent=ext, aspect="auto", cmap=cmap,
                                 vmin=0.5, vmax=40.5, interpolation="nearest")
        fig.colorbar(im, ax=axes[row, 1], label="period (white: none)")
        for ax in axes[row]:
            ax.set_xlabel(f"steering strength along '{plane}'")
            ax.set_ylabel("log10 temperature")
        axes[row, 0].set_title(f"Llama-3.1-8B, {plane}: 128 x 128 float64 cells", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=300)


if __name__ == "__main__":
    main()
