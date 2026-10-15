"""Figure fig-dialogue: two loops writing one text together (runs DLGa, DLGv1, DLGx; tools/soft_dialogue.py).

(a) Agents A and B (Qwen2.5-0.5B-Instruct, philosophy direction at layer 12, strengths alpha_A and alpha_B) alternate
    soft tokens at log10 T = -0.10: class of the exchange map G = F_B o F_A per cell (periodic cells coloured by the
    period of G, chaotic cells black). The dotted diagonal holds each agent alone (equal agents reproduce the single loop
    exactly); white crosses: entrained pairs (periodic, although both agents alone are chaotic); circles: the twelve
    entrained pairs verified by Newton's method (DLGv1).
(b) The smaller twin exponent of G per exchange on the same cells.
(c) Two different models, unsteered: the smaller twin exponent per exchange against log10 T for Qwen2.5-0.5B writing
    with itself, Qwen2.5-1.5B writing with itself, and the two writing together (messages passed as distributions over
    their shared vocabulary); dots mark periodic cells.

Usage: python tools/plot_dialogue_paper.py [--out paper/llm/figures/fig-dialogue.png]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data")) / "explore/llm-soft-loop/pod64/dialogue"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import period_colors as pc
    import plot_dialogue as pdl

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "paper/llm/figures/fig-dialogue.png"))
    args = ap.parse_args()
    cells = json.loads((DATA / "cw-t-0.10" / "dialogue.json").read_text())["cells"]
    aa, ab, idx = pdl.grid(cells)
    counts, marks = pdl.pair_stats(cells, idx)
    P = np.full(idx.shape, np.nan)
    C = np.zeros(idx.shape, dtype=bool)
    L = np.full(idx.shape, np.nan)
    for (i, j), k in np.ndenumerate(idx):
        c = cells[k]
        L[i, j] = min(c["lyapunov"])
        if c["class"] == "periodic":
            P[i, j] = c["period"]
        elif c["class"] in pdl.CHAOS:
            C[i, j] = True
    ext = [aa[0], aa[-1], ab[0], ab[-1]]
    fig = plt.figure(figsize=(14, 4.6), dpi=170)
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    cats = pc.categorize([P], max_blocks=9)
    from matplotlib.colors import ListedColormap
    ax1.imshow(np.where(C, 1.0, np.nan), origin="lower", extent=ext, aspect="auto", cmap=ListedColormap(["0.12"]),
               interpolation="nearest")
    cmap = cats.cmap.copy()
    cmap.set_bad((0, 0, 0, 0))                     # cells without a period stay transparent over the chaos layer
    im = ax1.imshow(cats.index(P), origin="lower", extent=ext, aspect="auto", cmap=cmap, norm=cats.norm,
                    interpolation="nearest")
    cats.colorbar(im, ax=ax1, label="period of the exchange map (black: chaotic)")
    ax1.plot(ext[:2], ext[2:], ":", color="k", lw=0.8)
    for i, j, kind in marks:
        if kind == "entrained":
            ax1.plot(aa[j], ab[i], "x", ms=2.6, color="white", mew=0.7)
    v = json.loads((ROOT / "docs/dlg/dlgv1-verify.json").read_text())["cells"]
    ax1.plot([c["alpha_a"] for c in v], [c["alpha_b"] for c in v], "o", ms=5, mfc="none", mec="k", mew=1.0)
    ax1.set_xlabel(r"$\alpha_A$", fontsize=8)
    ax1.set_ylabel(r"$\alpha_B$", fontsize=8)
    ax1.set_title(f"(a) two writers, log$_{{10}}T=-0.10$: {counts['entrained']} of {counts['both_chaotic_agents']} "
                  f"pairs of chaotic agents are periodic", fontsize=7.5)
    lim = np.nanmax(np.abs(L))
    im2 = ax2.imshow(L, origin="lower", extent=ext, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim,
                     interpolation="nearest")
    fig.colorbar(im2, ax=ax2, fraction=0.046, label="smaller twin exponent per exchange")
    ax2.plot(ext[:2], ext[2:], ":", color="k", lw=0.8)
    ax2.set_xlabel(r"$\alpha_A$", fontsize=8)
    ax2.set_title("(b) exponent of the exchange (red: expanding)", fontsize=7.5)
    lines = {}
    for tag in ("x-cw-05", "x-cw-15", "x-cw-pair"):
        d = json.loads((DATA / tag / "dialogue.json").read_text())["cells"]
        lines[tag] = sorted(((c["log10_temperature"], min(c["lyapunov"]), c["class"] == "periodic") for c in d))
    for tag, name, col, lw in (("x-cw-05", "0.5B with itself", "tab:blue", 1.0), ("x-cw-15", "1.5B with itself",
                                                                                  "tab:orange", 1.0),
                               ("x-cw-pair", "0.5B and 1.5B together", "k", 1.4)):
        t = np.array([r[0] for r in lines[tag]])
        y = np.array([max(r[1], -20.0) for r in lines[tag]])
        per = np.array([r[2] for r in lines[tag]])
        ax3.plot(t, y, "-", color=col, lw=lw, label=name)
        ax3.plot(t[per], y[per], "o", ms=1.8, color=col)
    ax3.axhline(0, color="0.5", lw=0.6, ls=":")
    ax3.set_ylim(-20, 3)
    ax3.set_xlabel(r"$\log_{10}T$", fontsize=8)
    ax3.set_ylabel("smaller twin exponent per exchange (clipped at -20)", fontsize=7)
    ax3.set_title("(c) two different models, unsteered (dots: periodic)", fontsize=7.5)
    ax3.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out)
    print(json.dumps(counts, indent=1))


if __name__ == "__main__":
    main()
