"""Figure of the hook inside the kite of the period-6 shrimp (run HR12; human, 19 Oct 2026: "that hook at the top of
the kite is jagged ... we need to get a higher resolution of that"; "Make that box a little bigger so we can cover the
striations a bit below").

(a) The 640 x 640 fixed-start map of box H2 (gray: periodic cells, darker where more contracting; colors: chaotic
    cells), with the kite's fold curve (black) and the body's two flip curves (red, dashed) where they cross the box,
    and the two pairs of adjacent cells checked by Newton continuation (tools/kite_basin_check.py): 1 on the light
    wedge's lower edge, 2 in a stippled tongue.
(b) The box drawn in (a): a stippled tongue at the map's own cells.

Usage: python tools/plot_p6_hook.py --out paper/llm/figures/fig-p6-hook.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import plot_p6_kite as pk  # noqa: E402

PAIRS = {"1": (0.07297495107632094, -0.1535315068493151, -0.15352876712328767),
         "2": (0.07292258215962441, -0.15382910798122065, -0.15382801251956182)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scan", default="llm-soft-loop/pod64/hr12")
    ap.add_argument("--zoom", type=float, nargs=4, default=[0.07291, 0.07297, -0.15390, -0.15372])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    d = np.load(pk.DATA / "explore" / args.scan / "scan.npz")
    a, t, lam = d["alpha"], d["log10_temperature"], d["lyapunov"]
    ext = [a[0], a[-1], t[0], t[-1]]
    gray = np.clip(np.where(lam <= 1e-3, lam, np.nan), -0.6, 0)
    chaos = np.where(lam > 1e-3, lam, np.nan)
    _, adp, _, _ = pk.curve(pk.POD / "p6s/p6s-ad-curve-plus/curve.json")
    _, adm, _, _ = pk.curve(pk.POD / "p6s/p6s-ad-curve-minus/curve.json")
    _, tdp, _, _ = pk.curve(pk.POD / "p6s/p6s-td-curve-plus/curve.json")
    _, tdm, _, _ = pk.curve(pk.POD / "p6s/p6s-td-curve-minus/curve.json")
    _, kp, _, _ = pk.curve(pk.POD / "kite/kite-la-curve-plus/curve.json")
    _, km, _, _ = pk.curve(pk.POD / "kite/kite-la-curve-minus/curve.json")
    ad, td, kite = np.vstack([adp[::-1], adm]), np.vstack([tdp[::-1], tdm]), np.vstack([kp[::-1], km])
    plt.rcParams.update({"font.size": 8})
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(7.2, 4.6), gridspec_kw={"width_ratios": [1.15, 1.0]})
    for ax, sub in ((axa, None), (axb, args.zoom)):
        if sub is None:
            ax.imshow(gray, origin="lower", extent=ext, aspect="auto", cmap="gray", vmin=-0.6, vmax=0, interpolation="nearest")
            ax.imshow(chaos, origin="lower", extent=ext, aspect="auto", cmap="plasma", vmin=0, vmax=0.4, interpolation="nearest")
            for xy, ls, col in ((ad, "--", "r"), (td, "--", "r"), (kite, "-", "k")):
                ax.plot(xy[:, 0], xy[:, 1], ls, color=col, lw=0.9)
            for k, (al, t0, t1) in PAIRS.items():
                ax.plot([al], [0.5 * (t0 + t1)], "o", ms=5, mfc="none", mec="cyan", mew=1.2)
                ax.text(al + 0.0000025, 0.5 * (t0 + t1), k, color="cyan", fontsize=8, fontweight="bold", va="center")
            a0, a1, b0, b1 = args.zoom
            ax.add_patch(Rectangle((a0, b0), a1 - a0, b1 - b0, fill=False, ec="w", lw=0.8))
            ax.set_xlim(a[0], a[-1])
            ax.set_ylim(t[0], t[-1])
            ax.set_title("(a) box H2, 640 x 640 cells", fontsize=8)
        else:
            a0, a1, b0, b1 = sub
            ci = np.nonzero((a >= a0) & (a <= a1))[0]
            ri = np.nonzero((t >= b0) & (t <= b1))[0]
            e2 = [a[ci[0]], a[ci[-1]], t[ri[0]], t[ri[-1]]]
            ax.imshow(gray[np.ix_(ri, ci)], origin="lower", extent=e2, aspect="auto", cmap="gray", vmin=-0.6, vmax=0,
                      interpolation="nearest")
            ax.imshow(chaos[np.ix_(ri, ci)], origin="lower", extent=e2, aspect="auto", cmap="plasma", vmin=0, vmax=0.4,
                      interpolation="nearest")
            al, t0, t1 = PAIRS["2"]
            ax.plot([al], [0.5 * (t0 + t1)], "o", ms=6, mfc="none", mec="cyan", mew=1.2)
            ax.set_title(f"(b) the white box: {len(ci)} x {len(ri)} cells", fontsize=8)
        ax.set_xlabel(r"steering strength $\alpha$")
        ax.set_ylabel(r"$\log_{10}T$")
        ax.ticklabel_format(useOffset=False)
        ax.tick_params(labelsize=6.5)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    rec = {"scan": args.scan, "zoom": args.zoom, "pairs": PAIRS, "cells": int(lam.size),
           "chaotic_share": float((lam > 1e-3).mean())}
    out.with_suffix(".json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
