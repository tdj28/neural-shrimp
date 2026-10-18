"""The period-5 shrimp at two temperatures (runs FA-100b and FA-105, with the fold and flip curves of FM5/FM5X at
log10 T = -0.100 and FM5T at -0.105, and the near-nilpotent point continued in temperature, run NT).

(a), (b): the followed domain at each temperature, coloured by period, with that temperature's curves. (c): the outlines
of both domains and the path of the head between log10 T = -0.104 and -0.094 (gated rows of NT; tools/
plot_nilpotent_track.py), so that the drift of the whole window can be compared with the drift of its head.

Usage: python tools/plot_shrimp_temperature.py [--out paper/llm/figures/fig-shrimp-temperature.png]
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
import plot_nilpotent_track as pnt  # noqa: E402

RUNS = {-0.100: ("llm-soft-loop/pod64/fa/t-0.100", "llm-soft-loop/pod64/fm5",
                 ["llm-soft-loop/pod64/fm5c", "llm-soft-loop/pod64/fm5d", "llm-soft-loop/pod64/fm5x"]),
        -0.105: ("llm-soft-loop/pod64/fa/t-0.105", "llm-soft-loop/pod64/fm5t-t105", [])}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--json")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    import period_colors as pc

    fas = {lt: np.load(DATA / "explore" / fa / "follow.npz", allow_pickle=True) for lt, (fa, _, _) in RUNS.items()}
    curves = {}
    for lt, (_, run, extra) in RUNS.items():
        extra = [e for e in extra if (DATA / "explore" / e).exists()]
        _, c, _ = pf.load(run, extra)
        curves[lt] = c
    P = {lt: np.where(z["label"] == 1, z["period"].astype(float), np.nan) for lt, z in fas.items()}
    cats = pc.categorize(list(P.values()))
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.0), dpi=180)
    allx, ally = [], []
    for ax, lt in zip(axes[:2], sorted(RUNS, reverse=True)):
        z = fas[lt]
        a1, a2, lab = z["alpha1"], z["alpha2"], z["label"]
        st = float(a1[1] - a1[0])
        ext = [a1[0] - st / 2, a1[-1] + st / 2, a2[0] - st / 2, a2[-1] + st / 2]
        ax.imshow(np.where(lab == 0, 1.0, np.nan), origin="lower", extent=ext, aspect="auto",
                  cmap=ListedColormap(["0.88"]), interpolation="nearest")
        im = ax.imshow(cats.index(P[lt]), origin="lower", extent=ext, aspect="auto", cmap=cats.cmap, norm=cats.norm,
                       interpolation="nearest")
        ax.imshow(np.where(lab == 2, 1.0, np.nan), origin="lower", extent=ext, aspect="auto", cmap=ListedColormap(["k"]),
                  interpolation="nearest")
        ax.imshow(np.where(lab == 3, 1.0, np.nan), origin="lower", extent=ext, aspect="auto",
                  cmap=ListedColormap(["0.55"]), interpolation="nearest")
        for rec in curves[lt].values():
            if rec["points"]:
                fold = rec["kind"] == "fold"
                ax.plot([q["alpha"] for q in rec["points"]], [q["alpha2"] for q in rec["points"]], "-" if fold else "--",
                        color="k" if fold else "r", lw=1.0 if fold else 0.9)
        ii, jj = np.where(lab >= 1)
        allx += [a1[jj.min()], a1[jj.max()]]
        ally += [a2[ii.min()], a2[ii.max()]]
        ax.set_title(f"({'a' if lt == -0.100 else 'b'}) the shrimp at $\\log_{{10}}T={lt:.3f}$", fontsize=10)
    cats.colorbar(im, ax=axes[1], label="period of the followed attractor")
    pad = 0.0006
    for ax in axes:
        ax.set_xlim(min(allx) - pad, max(allx) + pad)
        ax.set_ylim(min(ally) - pad, max(ally) + pad)
        ax.set_xlabel(r"$\alpha_1$ (gratitude, block 4)")
        ax.set_ylabel(r"$\alpha_2$ (hostility, block 16)")
        ax.ticklabel_format(useOffset=False)
    ax = axes[2]
    cols = {-0.100: "tab:blue", -0.105: "tab:red"}
    summary = {}
    for lt, z in fas.items():
        a1, a2, lab = z["alpha1"], z["alpha2"], z["label"]
        m = (lab == 1) & (z["period"] == 5)
        ax.contour(a1, a2, m.astype(float), levels=[0.5], colors=cols[lt], linewidths=1.0)
        ax.contour(a1, a2, (lab >= 1).astype(float), levels=[0.5], colors=cols[lt], linewidths=0.6, linestyles=":")
        ii, jj = np.where(m)
        summary[f"{lt:.3f}"] = {"period5_cells": int(m.sum()), "members": int((lab >= 1).sum()),
                                "period5_centroid": [float(a1[jj].mean()), float(a2[ii].mean())]}
        ax.plot([], [], "-", color=cols[lt], label=f"period 5 at {lt:.3f} (dotted: whole domain)")
    rows = []
    for run in ("llm-soft-loop/pod64/nt/down", "llm-soft-loop/pod64/nt/up"):
        if (DATA / "explore" / run / "track.json").exists():
            rows += [r for r in pnt.load(run) if r["passed"]]
    rows.sort(key=lambda r: r["log10_T"])
    if rows:
        sc = ax.scatter([r["a1"] for r in rows], [r["a2"] for r in rows], c=[r["log10_T"] for r in rows],
                        cmap="viridis", s=18, zorder=5, label="head (near-nilpotent point)")
        fig.colorbar(sc, ax=ax, label=r"$\log_{10}T$ of the head")
        summary["head_path"] = [[r["log10_T"], r["a1"], r["a2"]] for r in rows]
    c0, c1 = summary["-0.100"]["period5_centroid"], summary["-0.105"]["period5_centroid"]
    summary["period5_centroid_shift"] = [c1[0] - c0[0], c1[1] - c0[1]]
    ax.legend(fontsize=7, loc="lower right")
    ax.set_title("(c) both domains and the head's path in temperature", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out)
    text = json.dumps(summary, indent=1)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")


if __name__ == "__main__":
    main()
