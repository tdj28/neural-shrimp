"""Figure and statistics of run DLG (tools/soft_dialogue.py): pair maps of two agents in one soft-token conversation.

For a pair map with one direction (alpha_A, alpha_B on one grid), the diagonal holds the agents alone (an agent
talking with itself). Every off-diagonal cell (i, j) is compared with its two agents, the diagonal cells (i, i) and
(j, j): counts of
  - entrained pairs: the pair is periodic, both agents alone are chaotic (visible or hidden chaos);
  - pairs made chaotic: the pair is chaotic, both agents alone are periodic;
  - inherited: the pair's class equals both agents' class;
with the same counts for the null model that draws each off-diagonal class at random from the map's off-diagonal
class frequencies (expected counts). Labels: hidden_chaos.classify; "chaotic" = visible or hidden chaos.

Panels per map: classes (periodic cells coloured by the period of the exchange map), the leading exponent per
exchange, and the entrained and chaos-made cells marked over the class map.

Usage: python tools/plot_dialogue.py --runs dialogue/NAME [...] --out PNG [--stats JSON]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data")) / "explore/llm-soft-loop/pod64"
CHAOS = ("visible chaos", "hidden chaos")


def grid(cells):
    """Cells of one pair map at one temperature as (alpha_A values, alpha_B values, cell index grid)."""
    aa = np.array(sorted({round(c["alpha_a"], 12) for c in cells}))
    ab = np.array(sorted({round(c["alpha_b"], 12) for c in cells}))
    idx = -np.ones((len(ab), len(aa)), dtype=int)
    for k, c in enumerate(cells):
        idx[np.searchsorted(ab, round(c["alpha_b"], 12)), np.searchsorted(aa, round(c["alpha_a"], 12))] = k
    return aa, ab, idx


def pair_stats(cells, idx):
    """Off-diagonal comparison with the diagonal agents (square grids with equal axes only)."""
    n = idx.shape[0]
    cls = lambda k: "chaotic" if cells[k]["class"] in CHAOS else ("periodic" if cells[k]["class"] == "periodic" else "other")
    diag = [cls(idx[i, i]) for i in range(n)]
    counts = {"entrained": 0, "made_chaotic": 0, "inherited": 0, "off_diagonal": 0, "both_chaotic_agents": 0,
              "both_periodic_agents": 0}
    off = []
    marks = []
    for i in range(n):          # row: alpha_B index
        for j in range(n):      # column: alpha_A index
            if i == j:
                continue
            c = cls(idx[i, j])
            a, b = diag[j], diag[i]
            off.append(c)
            counts["off_diagonal"] += 1
            if a == b == "chaotic":
                counts["both_chaotic_agents"] += 1
                if c == "periodic":
                    counts["entrained"] += 1
                    marks.append((i, j, "entrained"))
            if a == b == "periodic":
                counts["both_periodic_agents"] += 1
                if c == "chaotic":
                    counts["made_chaotic"] += 1
                    marks.append((i, j, "made_chaotic"))
            if c == a == b:
                counts["inherited"] += 1
    freq = {k: off.count(k) / len(off) for k in ("chaotic", "periodic", "other")}
    counts["null_expected_entrained"] = counts["both_chaotic_agents"] * freq["periodic"]
    counts["null_expected_made_chaotic"] = counts["both_periodic_agents"] * freq["chaotic"]
    counts["diagonal"] = {k: diag.count(k) for k in ("chaotic", "periodic", "other")}
    counts["off_diagonal_classes"] = {k: off.count(k) for k in ("chaotic", "periodic", "other")}
    return counts, marks


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", required=True, help="run folders under explore/llm-soft-loop/pod64")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stats", default=None)
    args = ap.parse_args()
    maps = []
    for run in args.runs:
        d = json.loads((DATA / run / "dialogue.json").read_text())
        by_t = {}
        for c in d["cells"]:
            by_t.setdefault(round(c["log10_temperature"], 9), []).append(c)
        for t, cells in sorted(by_t.items()):
            maps.append((run, d["args"], t, cells))
    fig, axes = plt.subplots(len(maps), 2, figsize=(9.5, 4.3 * len(maps)), dpi=150, squeeze=False)
    report = []
    for (run, a, t, cells), (ax1, ax2) in zip(maps, axes):
        aa, ab, idx = grid(cells)
        per = np.full(idx.shape, np.nan)
        lam = np.full(idx.shape, np.nan)
        cls = np.zeros(idx.shape)
        for (i, j), k in np.ndenumerate(idx):
            if k < 0:
                continue
            c = cells[k]
            lam[i, j] = min(c["lyapunov"])
            if c["class"] == "periodic":
                per[i, j] = c["period"]
                cls[i, j] = 1
            elif c["class"] in CHAOS:
                cls[i, j] = 2
        ext = [aa[0], aa[-1], ab[0], ab[-1]]
        ax1.imshow(cls, origin="lower", extent=ext, aspect="auto", cmap=ListedColormap(["0.85", "tab:blue", "tab:red"]),
                   vmin=0, vmax=2, interpolation="nearest")
        if np.isfinite(per).any():
            ax1.imshow(np.where(np.isfinite(per), np.log2(per), np.nan), origin="lower", extent=ext, aspect="auto",
                       cmap="viridis", interpolation="nearest")
        title = f"{run.split('/')[-1]} ({a['frame']}), log10 T {t:g}: A {a['concept_a']}, B {a['concept_b']}"
        rec = {"run": run, "log10_temperature": t, "frame": a["frame"], "concept_a": a["concept_a"],
               "concept_b": a["concept_b"], "cells": len(cells),
               "classes": {k: sum(1 for c in cells if c["class"] == k) for k in sorted({c["class"] for c in cells})},
               "periods": {str(q): int(np.sum(per == q)) for q in sorted(set(per[np.isfinite(per)].astype(int)))}}
        if len(aa) == len(ab) and np.allclose(aa, ab) and a["concept_a"] == a["concept_b"]:
            counts, marks = pair_stats(cells, idx)
            rec["pair_comparison"] = counts
            for i, j, kind in marks:
                ax1.plot(aa[j], ab[i], "x" if kind == "entrained" else "+", ms=3,
                         color="white" if kind == "entrained" else "k")
            ax1.plot(ext[:2], ext[2:], ":", color="k", lw=0.8)
            title += f"\nentrained {counts['entrained']} (null {counts['null_expected_entrained']:.0f}), " \
                     f"made chaotic {counts['made_chaotic']} (null {counts['null_expected_made_chaotic']:.0f})"
        ax1.set_title(title, fontsize=7)
        ax1.set_xlabel(r"$\alpha_A$", fontsize=8)
        ax1.set_ylabel(r"$\alpha_B$", fontsize=8)
        im = ax2.imshow(lam, origin="lower", extent=ext, aspect="auto", cmap="RdBu_r",
                        vmin=-np.nanmax(np.abs(lam)), vmax=np.nanmax(np.abs(lam)), interpolation="nearest")
        fig.colorbar(im, ax=ax2, fraction=0.046, label="smaller twin exponent per exchange")
        ax2.set_xlabel(r"$\alpha_A$", fontsize=8)
        ax2.set_title("leading exponent (red: expanding)", fontsize=7)
        report.append(rec)
    fig.tight_layout()
    fig.savefig(args.out)
    if args.stats:
        Path(args.stats).write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
