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

# one color per factor of DF = (1/T) Sigma_p J_h in every panel: readout Sigma_p/T orange, network J_h blue
READ, NET = "#D55E00", "#0072B2"          # vermillion and blue (Okabe-Ito): distinct for colorblind readers


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
    ap.add_argument("--panel-a", choices=("single", "split"), default="split",
                    help="(a) in one panel, or split into the network factor (top) and the readout factor (bottom)")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stats = json.loads((ROOT / "paper/llm/figures/fig-mechanism.json").read_text())
    amap = pm.arrays(pm.load_cells("amap-a", "amap-b"))
    bmap = pm.arrays(pm.load_cells("bmap-a", "bmap-b"))
    prof = row_profiles(amap)
    summary = {"rows": prof}

    # little text in the figure: colors and line styles are explained in the caption (orange: readout factor
    # Sigma_p/T, blue: network factor J_h; in (a) solid: chaotic cells, dashed: nearly unsteered cells)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3), gridspec_kw={"width_ratios": [1.05, 1.0, 1.0], "wspace": 0.58})
    dashed = (0, (3, 1.6))
    # (a) gain split per temperature row: chaotic cells solid with filled markers, the nearly unsteered baseline dashed
    # without markers; the network factor in blue squares, the readout factor in vermillion circles (as in (c))
    t = np.array([r["log10T"] for r in prof])
    if args.panel_a == "split":
        sub = axes[0].get_subplotspec().subgridspec(2, 1, hspace=0.14)
        axes[0].remove()
        axn = fig.add_subplot(sub[0])
        axr = fig.add_subplot(sub[1], sharex=axn)
        pairs = ((axn, "net", NET, "s"), (axr, "read", READ, "o"))
        axn.tick_params(labelbottom=False)
        axn.set_ylim(-0.75, 1.35)
        axr.set_ylim(-6.0, 1.35)
        axn.set_ylabel("network", fontsize=7.5)
        axr.set_ylabel("readout", fontsize=7.5)
        axn.set_title("(a)", fontsize=8, loc="left")
        axr.set_xlabel(r"$\log_{10}T$", fontsize=7.5)
        panel_a = (axn, axr)
    else:
        ax = axes[0]
        pairs = ((ax, "net", NET, "s"), (ax, "read", READ, "o"))
        ax.set_ylim(-6.0, 1.6)
        ax.set_ylabel("mean log gain per step (nats)", fontsize=7.5)
        ax.set_title("(a)", fontsize=8, loc="left")
        ax.set_xlabel(r"$\log_{10}T$", fontsize=7.5)
        panel_a = (ax,)
    for ax, k, col, mk in pairs:
        ax.plot(t, [r[f"{k}_near0"] for r in prof], ls=dashed, color=col, lw=1.2)
        ax.plot(t, [r[f"{k}_chaotic"] for r in prof], ls="-", color=col, lw=1.4, marker=mk, ms=3.0)
        ax.axhline(0, color="black", lw=0.6)
    # (b) chaotic share per temperature row
    ax = axes[1]
    for A, lab, col in ((amap, "residual steering", "black"), (bmap, "logit bias only", READ)):
        rows = np.unique(np.round(A["log10_temperature"], 6))
        share = np.array([A["chaotic"][np.isclose(A["log10_temperature"], r)].mean() for r in rows]) * 100
        n, k = int(len(A["chaotic"])), int(A["chaotic"].sum())
        ax.plot(rows, share, "-", color=col, lw=1.1, label=lab)
        summary[lab] = {"chaotic": k, "cells": n}
    ax.legend(fontsize=6, frameon=False, loc="upper left", handlelength=1.2, handletextpad=0.4, borderaxespad=0.3)
    ax.set_xlim(-0.6, 0.31)
    ax.set_xticks([-0.6, -0.4, -0.2, 0.0, 0.2])
    ax.set_xlabel(r"$\log_{10}T$", fontsize=7.5)
    ax.set_ylabel("chaotic cells in the row (%)", fontsize=7.5)
    ax.set_title("(b)", fontsize=8, loc="left")
    # (c) crossover effects
    ax = axes[2]
    groups = (("XO", "0.5B, philosophy", "philosophy"), ("XO_gratitude", "0.5B, gratitude", "gratitude"),
              ("XO_random1", "0.5B, random", "random"), ("XO7", "7B, philosophy", "7B philosophy"))
    y = 0.0
    ticks, labels = [], []
    summary["crossover"] = {}
    for key, name, short in groups:
        s = stats[key]
        for eff, col, mk, dy in (("readout_effect (10-00)", READ, "o", 0.17), ("network_effect (01-00)", NET, "s", -0.17)):
            v = s[eff]
            ax.errorbar([v["mean"]], [y + dy], xerr=[[v["mean"] - v["ci95"][0]], [v["ci95"][1] - v["mean"]]], fmt=mk,
                        color=col, ms=3.6, capsize=2, lw=1.0)
            summary["crossover"].setdefault(name, {})[eff] = v
        ticks.append(y)
        labels.append(short)
        y -= 1.0
    ax.axvline(0, color="black", lw=0.6)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_ylim(y + 0.5, 0.6)
    ax.set_xlim(-1.8, 1.45)
    ax.set_xlabel("change in growth (nats per step)", fontsize=7.5)
    ax.set_title("(c)", fontsize=8, loc="left")
    for a in list(axes[1:]) + list(panel_a):
        a.tick_params(labelsize=6.5)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", pad_inches=0.03)
    Path(args.json).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=1)[:1500])


if __name__ == "__main__":
    main()
