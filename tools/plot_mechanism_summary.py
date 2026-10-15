"""Figure fig-mechanism-summary: the three measurements behind "steering makes tokens compete", one panel each
(human request: a graph that makes the mechanism paragraph easy to follow). Same data as fig-mechanism (runs MECH,
XO, XO7, XOG, XOR); the crossover effects and their intervals are read from paper/llm/figures/fig-mechanism.json,
which tools/plot_mechanism.py writes, so the figure and the text quote the same numbers.

(a) Gain split on the block-12 alpha map: per temperature row, mean log readout and network factors of the chaotic
    cells and of the nearly unsteered cells (|alpha| <= 0.01); rows with at least three cells in both groups, as in
    the paper's row-matched statistics.
(b) Chaotic share per temperature row: residual steering at block 12 against a logit bias alone (alpha = 0).
(c) Orbit-preserving crossover: readout effect (arm 10 - arm 00) and network effect (arm 01 - arm 00), mean over
    held-out chaotic cells with 95 % bootstrap intervals.

Usage: python tools/plot_mechanism_summary.py [--out PNG] [--json OUT]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import plot_mechanism as pm  # noqa: E402

RED, GRAY, PURPLE, ORANGE = "#c0392b", "0.45", "#7b3294", "#e08214"


def row_profiles(A, min_n=3):
    near0 = np.abs(A["alpha"]) <= 0.01
    out = []
    for t in np.unique(np.round(A["log10_temperature"], 6)):
        m = np.isclose(A["log10_temperature"], t) & np.isfinite(A["mean_log_g_read"]) & np.isfinite(A["mean_log_g_net"])
        a, b = m & A["chaotic"], m & near0
        if a.sum() >= min_n and b.sum() >= min_n:
            out.append({"log10T": float(t), "chaotic": int(a.sum()), "near0": int(b.sum()),
                        "read_chaotic": float(A["mean_log_g_read"][a].mean()),
                        "read_near0": float(A["mean_log_g_read"][b].mean()),
                        "net_chaotic": float(A["mean_log_g_net"][a].mean()),
                        "net_near0": float(A["mean_log_g_net"][b].mean())})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "paper/llm/figures/fig-mechanism-summary.png"))
    ap.add_argument("--json", default=str(ROOT / "paper/llm/figures/fig-mechanism-summary.json"))
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = json.loads((ROOT / "paper/llm/figures/fig-mechanism.json").read_text())
    amap = pm.arrays(pm.load_cells("amap-a", "amap-b"))
    bmap = pm.arrays(pm.load_cells("bmap-a", "bmap-b"))
    prof = row_profiles(amap)
    summary = {"rows": prof}

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45), gridspec_kw={"width_ratios": [1.05, 1.0, 1.0], "wspace": 0.34})
    # (a) gain split per temperature row
    ax = axes[0]
    t = np.array([r["log10T"] for r in prof])
    for key, col, ls, lab in (("read_chaotic", RED, "-", "readout, chaotic cells"),
                              ("read_near0", GRAY, "-", r"readout, $|\alpha|\leq0.01$"),
                              ("net_chaotic", RED, (0, (3, 1.6)), "network, chaotic cells"),
                              ("net_near0", GRAY, (0, (3, 1.6)), r"network, $|\alpha|\leq0.01$")):
        ax.plot(t, [r[key] for r in prof], ls=ls, color=col, lw=1.1, marker="o", ms=2.2, label=lab)
    ax.axhline(0, color="black", lw=0.6)
    ax.text(-0.195, 1.45, r"expands $\uparrow$", fontsize=6.3, va="top", color="0.3")
    ax.text(-0.195, -0.95, r"contracts $\downarrow$", fontsize=6.3, va="top", color="0.3")
    ax.set_xlabel(r"$\log_{10}T$", fontsize=7.5)
    ax.set_ylabel("mean log factor per step (nats)", fontsize=7.5)
    ax.set_title("(a) gain split along the orbits", fontsize=8)
    ax.legend(fontsize=5.6, loc="lower right", frameon=False, handlelength=2.4)
    ax.set_ylim(-6.0, 1.6)
    # (b) chaotic share per temperature row
    ax = axes[1]
    for (A, lab, col), ytext in zip(((amap, "residual steering", "black"), (bmap, "logit bias only", PURPLE)), (33, 23)):
        rows = np.unique(np.round(A["log10_temperature"], 6))
        share = np.array([A["chaotic"][np.isclose(A["log10_temperature"], r)].mean() for r in rows]) * 100
        n, k = int(len(A["chaotic"])), int(A["chaotic"].sum())
        ax.plot(rows, share, "-", color=col, lw=1.1)
        ax.text(-0.58, ytext, f"{lab}\n{100 * k / n:.1f} % of cells", color=col, fontsize=6.3, va="top")
        summary[lab] = {"chaotic": k, "cells": n}
    ax.set_xlim(-0.6, 0.31)
    ax.set_xlabel(r"$\log_{10}T$", fontsize=7.5)
    ax.set_ylabel("chaotic cells in the row (%)", fontsize=7.5)
    ax.set_title("(b) token competition alone", fontsize=8)
    # (c) crossover effects
    ax = axes[2]
    groups = (("XO", "0.5B, philosophy"), ("XO_gratitude", "0.5B, gratitude"), ("XO_random1", "0.5B, random"),
              ("XO7", "7B, philosophy"))
    y = 0.0
    ticks, labels = [], []
    summary["crossover"] = {}
    for key, name in groups:
        s = stats[key]
        for eff, col, mk, dy in (("readout_effect (10-00)", ORANGE, "o", 0.17), ("network_effect (01-00)", GRAY, "s", -0.17)):
            v = s[eff]
            ax.errorbar([v["mean"]], [y + dy], xerr=[[v["mean"] - v["ci95"][0]], [v["ci95"][1] - v["mean"]]], fmt=mk,
                        color=col, ms=3.6, capsize=2, lw=1.0, label=None)
            summary["crossover"].setdefault(name, {})[eff] = v
        n = s["readout_effect (10-00)"]["cells"]
        ax.text(-1.76, y + 0.36, f"{name} ({n} cells)", fontsize=6.2, va="bottom", ha="left", zorder=3,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
        y -= 1.0
    ax.errorbar([], [], xerr=[], fmt="o", color=ORANGE, ms=3.6, label="steer only the readout")
    ax.errorbar([], [], xerr=[], fmt="s", color=GRAY, ms=3.6, label="steer only the network")
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks([])
    ax.set_ylim(y + 0.5, 1.75)
    ax.set_xlim(-1.8, 1.45)
    ax.set_xlabel("change in growth (nats per step)", fontsize=7.5)
    ax.set_title("(c) crossover at fixed states", fontsize=8)
    leg = ax.legend(fontsize=6, loc="upper right", frameon=True, framealpha=1.0, handletextpad=0.3)
    leg.get_frame().set_edgecolor("none")
    for a in axes:
        a.tick_params(labelsize=6.5)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", pad_inches=0.03)
    Path(args.json).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1)[:1500])


if __name__ == "__main__":
    main()
