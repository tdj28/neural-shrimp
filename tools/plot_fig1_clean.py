"""Single-column version of the article's opening map (human, 19 Oct 2026: "try single column ... create a figure like
figure 1 but cleaner and simpler, perhaps with the zoom in area we are getting now").

One Lyapunov map of a float64 scan, colored as in the shrimp literature and as explore_llm_soft_loop.py render
--style lyapunov does:
- chaotic cells (largest exponent above --chaos-level and no detected period) in color by their exponent;
- every other cell in gray by how strongly it contracts (darker: more strongly contracting; the dark curves inside
  periodic windows are where the loop is most stable).
One color bar (the chaotic cells' exponent), large fonts for a 3.4-inch column, optional labels with arrows.

Usage: python tools/plot_fig1_clean.py --scan llm-soft-loop/pod64/hr9 --out paper/llm/figures/fig-opening-map.png \
         --label 0.0753 -0.1552 0.0738 -0.1585 "shrimp (period 9)"
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
DATA = ROOT.parent / "neural-shrimp-data"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scan", required=True, help="explore-relative directory with scan.npz")
    ap.add_argument("--window", type=float, nargs=4, default=None, metavar=("A0", "A1", "T0", "T1"),
                    help="crop to alpha in [A0, A1] and log10 T in [T0, T1]")
    ap.add_argument("--chaos-level", type=float, default=1e-3)
    ap.add_argument("--width", type=float, default=3.375, help="figure width in inches (REVTeX column: 3.375)")
    ap.add_argument("--height", type=float, default=3.55)
    ap.add_argument("--cmap", default="plasma")
    ap.add_argument("--label", nargs=5, action="append", default=[], metavar=("X", "Y", "TX", "TY", "TEXT"),
                    help="arrow from (TX, TY) to (X, Y) in data coordinates, with TEXT at (TX, TY)")
    ap.add_argument("--box", type=float, nargs=4, default=None, metavar=("A0", "A1", "T0", "T1"),
                    help="draw a thin white rectangle (e.g. the region of a zoom)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    d = np.load(DATA / "explore" / args.scan / "scan.npz", allow_pickle=True)
    lt, al, per, ly = d["log10_temperature"], d["alpha"], d["period"], d["lyapunov"]
    if args.window:
        a0, a1, t0, t1 = args.window
        rows = (lt >= t0) & (lt <= t1)
        cols = (al >= a0) & (al <= a1)
        if not rows.any() or not cols.any():
            raise SystemExit("the window contains no cells of this scan")
        lt, al = lt[rows], al[cols]
        per, ly = per[np.ix_(rows, cols)], ly[np.ix_(rows, cols)]
    chaos = (ly > args.chaos_level) & (per <= 0)
    hot_norm = Normalize(0.0, float(np.max(ly[chaos])) if chaos.any() else 1.0)
    img = np.zeros(ly.shape + (4,))
    neg = np.clip(-ly, 0, None)
    cap = max(float(np.percentile(neg[~chaos], 99)) if (~chaos).any() else 1.0, 1e-12)
    grey = 1.0 - 0.9 * np.clip(neg / cap, 0, 1) ** 0.5
    img[...] = np.stack([grey, grey, grey, np.ones_like(grey)], axis=-1)
    img[chaos] = plt.get_cmap(args.cmap)(hot_norm(ly[chaos]))

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})
    fig = plt.figure(figsize=(args.width, args.height))
    ax = fig.add_axes([0.185, 0.275, 0.785, 0.70])
    da, dt = (al[1] - al[0]) / 2, (lt[1] - lt[0]) / 2
    ax.imshow(img, origin="lower", aspect="auto", interpolation="nearest",
              extent=[al[0] - da, al[-1] + da, lt[0] - dt, lt[-1] + dt])
    ax.set_xlabel(r"steering strength $\alpha$")
    ax.set_ylabel(r"$\log_{10}$ temperature")
    ax.locator_params(axis="x", nbins=5)
    ax.locator_params(axis="y", nbins=6)
    if args.box:
        b0, b1, c0, c1 = args.box
        ax.add_patch(plt.Rectangle((b0, c0), b1 - b0, c1 - c0, fill=False, ec="white", lw=0.9))
    for x, y, tx, ty, text in args.label:
        ax.annotate(text, xy=(float(x), float(y)), xytext=(float(tx), float(ty)), fontsize=7.5, color="white",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", fc=(0, 0, 0, 0.55), ec="none"),
                    arrowprops=dict(arrowstyle="-|>", color="white", lw=0.9, shrinkA=2, shrinkB=1))
    cax = fig.add_axes([0.185, 0.125, 0.785, 0.026])
    cb = fig.colorbar(ScalarMappable(norm=hot_norm, cmap=args.cmap), cax=cax, orientation="horizontal")
    cb.set_label(r"chaotic cells: largest Lyapunov exponent $\lambda$ per step", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=600, bbox_inches="tight", pad_inches=0.03)
    stats = {"scan": args.scan, "window": args.window, "cells": int(ly.size), "chaotic": int(chaos.sum()),
             "chaotic_share": round(float(chaos.mean()), 4), "max_lambda": float(np.max(ly[chaos])) if chaos.any() else None,
             "periodic": int((per > 0).sum()), "neither": int(((per <= 0) & ~chaos).sum()),
             "out": str(out)}
    out.with_suffix(".json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
