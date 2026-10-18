"""Figure fig-bimodal: band return maps with two or more turning points in the two-block planes (run TPF, projections
saved by tools/turning_scan.py --save). Qwen2.5-0.5B-Instruct, float64, log10 T = -0.1.

(a) hostility at block 4 + gratitude at block 16, alpha1 0.0568, alpha2 -0.0189: a 20-band chaotic attractor; phase 0's
    20-step return map is N-shaped (a maximum and a minimum);
(b), (c) the same plane at alpha1 -0.0366, alpha2 -0.0467: a 12-band attractor whose phases are N-shaped maps on two
    intervals (phases 2 and 9);
(d) gratitude at block 4 + hostility at block 16, alpha1 -0.1099, alpha2 0.0998: phase 4 of a 9-band attractor, three
    maxima.
Each panel: the visits of one phase projected on their leading principal component s (explained variance in the
title), points (s_n, s_(n+1)) one band period apart, the diagonal, and a cobweb of consecutive visits (red: 25 visits
in (a), 12 in the others).

Usage: python tools/plot_bimodal.py [--out paper/llm/figures/fig-bimodal.png]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data")) / "explore/llm-soft-loop/pod64/tp"
PANELS = [("f-tbhm", 0, 0, "(a) hostility@4 + gratitude@16\n$\\alpha_1=0.0568$, $\\alpha_2=-0.0189$, 20 bands, phase 0", 25),
          ("f-tbhm", 1, 2, "(b) same plane, $\\alpha_1=-0.0366$, $\\alpha_2=-0.0467$\n12 bands, phase 2", 12),
          ("f-tbhm", 1, 9, "(c) as (b), phase 9", 12),
          ("f-tbgm", 0, 4, "(d) gratitude@4 + hostility@16\n$\\alpha_1=-0.1099$, $\\alpha_2=0.0998$, 9 bands, phase 4", 12)]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "paper/llm/figures/fig-bimodal.png"))
    args = ap.parse_args()
    fig, axes = plt.subplots(1, len(PANELS), figsize=(3.3 * len(PANELS), 3.6), dpi=170)
    report = []
    for ax, (run, cell, ph, title, web) in zip(axes, PANELS):
        d = json.loads((DATA / run / "turning.json").read_text())
        c = d["cells"][cell]
        meta = next(p for p in c["phases"] if p["phase"] == ph)
        s = np.load(DATA / run / "projections.npz")[f"c{cell}_p{ph}"]
        x, y = s[:-1], s[1:]
        lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
        pad = 0.06 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ":", color="0.4", lw=0.8)
        ax.plot(x, y, ".", ms=2, color="k", zorder=2)
        if web:                                       # cobweb of `web` consecutive visits from visit 100 on
            cx, cy = [s[100]], [s[100]]
            for k in range(100, 100 + web):
                cx += [s[k], s[k + 1]]
                cy += [s[k + 1], s[k + 1]]
            ax.plot(cx, cy, "-", color="tab:red", lw=0.7, zorder=1)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(r"$s_n$", fontsize=8)
        ax.set_title(title + f"\nexplained {meta['explained']:.4f}, turning points counted {meta['turning']}", fontsize=7)
        report.append({"run": run, "cell": c["cell"], "q": c["q"], "phase": ph, "explained": meta["explained"],
                       "turning": meta["turning"], "points": int(len(s))})
    axes[0].set_ylabel(r"$s_{n+1}$ (one band period later)", fontsize=8)
    fig.tight_layout()
    fig.savefig(args.out)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
