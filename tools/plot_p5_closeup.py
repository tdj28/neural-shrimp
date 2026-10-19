"""Run HP5 figure: the period-5 shrimp of the two-block plane (gratitude at block 4, hostility at block 16, log10 T
-0.1; Qwen2.5-0.5B-Instruct, float64) on a fine fixed-start map (human, 19 Oct 2026: "Fix anything wrong with the
original shrimp we had in the paper too ... the earlier one looks mangled").

Fig. 18 shows the window only through its followed domain on a 225 x 293 grid. Here the stitched 512 x 512 map of
tools/two_block_map.py (run HP5) is drawn as the period-6 shrimp's map is (plot_p6_shrimp.py):
- cells with a period that is a multiple of five (the window's family: 5, 10, 20, 40) in gray by contraction;
- other periodic cells (other attractors reached from the fixed start, such as the coexisting period 4) in a pale tint;
- chaotic cells by their largest exponent.
Panel (a) is the map alone; panel (b) adds the verified skeleton of plot_real_shrimp.py (fold curves black, flip curves
red and dashed, located cusps as gold stars, self-crossings of the fold curves in magenta, the near-nilpotent head as a
white circle) and the edge of the followed domain (run FA) in cyan.

Usage: python tools/plot_p5_closeup.py --map llm-soft-loop/pod64/hp5 --out paper/llm/figures/fig-p5-closeup.png
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
sys.path.insert(0, str(ROOT / "tools"))
import plot_fm5 as pf  # noqa: E402

HEAD = (-0.1184836, 0.1017217)      # near-nilpotent point (tools/nilpotent_search.py)
EXTRA = ("llm-soft-loop/pod64/fm5c", "llm-soft-loop/pod64/fm5d", "llm-soft-loop/pod64/fm5x", "llm-soft-loop/pod64/fm5l-a")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--map", default="llm-soft-loop/pod64/hp5")
    ap.add_argument("--fa", default="llm-soft-loop/pod64/fa/t-0.100")
    ap.add_argument("--chaos-level", type=float, default=1e-3)
    ap.add_argument("--window", type=float, nargs=4, default=None, metavar=("A10", "A11", "A20", "A21"))
    ap.add_argument("--json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    z = np.load(DATA / "explore" / args.map / "map.npz")
    a1, a2, lam, per = z["alpha1"], z["alpha2"], z["lyapunov"], z["period"]
    s1, s2 = float(a1[1] - a1[0]), float(a2[1] - a2[0])
    ext = [a1[0] - s1 / 2, a1[-1] + s1 / 2, a2[0] - s2 / 2, a2[-1] + s2 / 2]
    chaos = lam > args.chaos_level
    fam = (per > 0) & (per % 5 == 0) & ~chaos
    other = (per > 0) & (per % 5 != 0) & ~chaos
    neither = ~chaos & (per <= 0)
    _, curves, located = pf.load("llm-soft-loop/pod64/fm5", [r for r in EXTRA if (DATA / "explore" / r).exists()])
    fa = np.load(DATA / "explore" / args.fa / "follow.npz", allow_pickle=True)
    pale = LinearSegmentedColormap.from_list("pale", ["#9db7c9", "#e6eef4"])
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 6.2), dpi=200)

    def base(ax):
        ax.imshow(np.where(other, np.clip(lam, -1.2, 0), np.nan), origin="lower", extent=ext, aspect="auto", cmap=pale,
                  vmin=-1.2, vmax=0, interpolation="nearest")
        ax.imshow(np.where(neither, 1.0, np.nan), origin="lower", extent=ext, aspect="auto",
                  cmap=LinearSegmentedColormap.from_list("w", ["#f4f4f4", "#f4f4f4"]), interpolation="nearest")
        ax.imshow(np.where(fam, np.clip(lam, -1.2, 0), np.nan), origin="lower", extent=ext, aspect="auto", cmap="gray",
                  vmin=-1.2, vmax=0, interpolation="nearest")
        im = ax.imshow(np.where(chaos, lam, np.nan), origin="lower", extent=ext, aspect="auto", cmap="plasma",
                       vmin=0, vmax=max(0.3, float(np.nanmax(np.where(chaos, lam, np.nan))) if chaos.any() else 0.3),
                       interpolation="nearest")
        ax.set_xlabel(r"$\alpha_1$ (gratitude, block 4)")
        return im

    def limits(ax):                     # the map's extent (or --window); curves beyond it are clipped
        w = args.window or ext
        ax.set_xlim(w[0], w[1])
        ax.set_ylim(w[2], w[3])

    axA, axB = axes
    im = base(axA)
    axA.set_ylabel(r"$\alpha_2$ (hostility, block 16)")
    axA.set_title(f"(a) from the fixed start, {lam.shape[1]} x {lam.shape[0]} float64 cells", fontsize=9.5)
    limits(axA)
    base(axB)
    for rec in curves.values():
        pts = rec["points"]
        if not pts:
            continue
        fold = rec["kind"] == "fold"
        axB.plot([q["alpha"] for q in pts], [q["alpha2"] for q in pts], "-" if fold else "--", color="k" if fold else "r",
                 lw=1.0 if fold else 0.9, zorder=4)
    seen = []
    for r in located.values():
        if r["located"]:
            x, y = r["final"]["alpha"], r["final"]["alpha2"]
            if not any(abs(x - u) < 1e-6 and abs(y - w) < 1e-6 for u, w in seen):
                seen.append((x, y))
                axB.plot(x, y, "*", color="gold", ms=11, mec="k", mew=0.6, zorder=6)
    crossings = []
    for rec in curves.values():
        if rec["kind"] == "fold" and rec["points"]:
            for _, _, x in pf.self_crossings(np.array([[q["alpha"], q["alpha2"]] for q in rec["points"]])):
                if not any(abs(x[0] - c[0]) < 1e-6 and abs(x[1] - c[1]) < 1e-6 for c in crossings):
                    crossings.append(x)
                    axB.plot(*x, "x", color="m", ms=8, mew=1.6, zorder=7)
    axB.plot(*HEAD, "o", color="w", mec="k", mew=1.0, ms=6, zorder=8)
    member = (fa["label"] >= 1).astype(float)
    axB.contour(fa["alpha1"], fa["alpha2"], member, levels=[0.5], colors="cyan", linewidths=0.8, zorder=5)
    axB.set_title("(b) with the verified skeleton; cyan: edge of the followed domain", fontsize=9.5)
    limits(axB)
    handles = [Patch(color="0.35", label="period 5, 10, 20, 40 (gray: contraction)"),
               Patch(color="#c2d4e1", label="other periodic attractors"),
               Line2D([], [], color="k", label="fold curve"), Line2D([], [], color="r", ls="--", label="flip curve"),
               Line2D([], [], marker="*", ls="", color="gold", mec="k", ms=9, label="located cusp"),
               Line2D([], [], marker="x", ls="", color="m", ms=7, label="fold self-crossing"),
               Line2D([], [], marker="o", ls="", mfc="w", mec="k", ms=5, label="near-nilpotent head")]
    axB.legend(handles=handles, fontsize=6.5, loc="lower right", framealpha=0.85)
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.01)
    cb.set_label("chaotic cells: largest exponent")
    fig.savefig(args.out, bbox_inches="tight")
    summary = {"map": args.map, "shape": list(lam.shape), "chaotic_share": round(float(chaos.mean()), 4),
               "family_share": round(float(fam.mean()), 4), "other_periodic_share": round(float(other.mean()), 4),
               "family_periods": {int(q): int(((per == q) & fam).sum()) for q in np.unique(per[fam])},
               "curves": len(curves), "located_cusps": len(seen), "fold_self_crossings": len(crossings)}
    print(json.dumps(summary))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(summary, indent=1) + "\n")


if __name__ == "__main__":
    main()
