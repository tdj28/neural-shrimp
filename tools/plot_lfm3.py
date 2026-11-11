"""Figure fig-lfm3: the soft-token loop of two LFM3-1.7B checkpoints (run LF3, exploratory, 28 Oct 2026; float64;
tools/lfm3_float64.py, tools/lf3_tk_summary.py).

Columns: the base model (LiquidAI/LFM3-1.7B-Base) and the post-trained model (LiquidAI/LFM3-1.7B).
(a, b) No steering: largest Lyapunov exponent at 160 temperatures (1000 transient and 2000 measured steps), clipped
       at -1.5. Red, "chaotic": exponent above 1e-3 and no detected period; blue, "periodic": a detected period;
       black cross, "unresolved": no detected period and no exponent above 1e-3.
(c, d) Steering along the philosophy direction at layer 12: 32 x 32 cells, 300 transient and 300 measured steps,
       exponent clipped at +-0.5 on the color scale of the paper's maps.
(e, f) Steering along the hostility-versus-kindness direction, as (c, d).

Every label rests on one twin offset (1e-6). The JSON written beside the figure holds the counts of each panel; they
are recomputed from the arrays and must equal the summary that tools/lf3_tk_summary.py stored with the scans.

Usage: python tools/plot_lfm3.py [OUT.png] [OUT.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
MODELS = [("lfm3-1.7b-base", "LFM3-1.7B-Base"), ("lfm3-1.7b", "LFM3-1.7B (post-trained)")]
MAPS = [("phil", "philosophy"), ("host", "hostility")]
RED, BLUE = "#b2182b", "#2166ac"                       # the two poles of the maps' color scale
INK, MUTED, HAIR = "#0b0b0b", "#52514e", "#d9d8d3"
CLIP = -1.5


def labels(L, per):
    chaotic = (L > 1e-3) & (per <= 0)
    unresolved = (per <= 0) & ~chaotic
    return chaotic, unresolved


def check(found, stored, what):
    for key, value in found.items():
        ref = stored[key]
        same = abs(ref - value) <= 1e-12 * max(1.0, abs(ref)) if isinstance(value, float) else ref == value
        if not same:
            raise SystemExit(f"{what}: {key} is {value} in the arrays and {ref} in the stored summary")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    import explore_llm_attractors as ela

    out_png = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "paper/llm/figures/fig-lfm3.png")
    out_json = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "paper/llm/figures/fig-lfm3.json")
    P = ela.DATA / "explore" / "llm-soft-loop" / "pod64" / "lfm3"
    stored = json.loads((P / "summary.json").read_text())
    fig = plt.figure(figsize=(7.8, 7.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.72, 1.0, 1.0], hspace=0.5, wspace=0.24, left=0.095, right=0.885,
                          top=0.925, bottom=0.07)
    summary, im, letters = {}, None, iter("abcdef")
    axes = {}
    for col, (tag, name) in enumerate(MODELS):
        summary[tag] = {}
        # no steering: one exponent per temperature
        z = np.load(P / f"{tag}-alpha0" / "scan.npz")
        lt, L, per = z["log10_temperature"], z["lyapunov"][:, 0], z["period"][:, 0]
        if z["alpha"].tolist() != [0.0]:
            raise SystemExit(f"{tag}-alpha0 is not a sweep without steering")
        chaotic, unresolved = labels(L, per)
        periodic = per > 0
        found = {"temperatures_all": int(L.size), "chaotic_all": int(chaotic.sum()),
                 "unresolved_all": int(unresolved.sum()), "max_exponent": float(L.max())}
        check(found, stored[tag]["sweep"], f"{tag} sweep")
        ax = axes[(0, col)] = fig.add_subplot(gs[0, col])
        ax.axhline(0, color=MUTED, lw=0.6)
        y = np.clip(L, CLIP, None)
        ax.scatter(lt[periodic], y[periodic], s=7, color=BLUE, lw=0)
        ax.scatter(lt[unresolved], y[unresolved], s=16, marker="x", color=INK, lw=0.8)
        ax.scatter(lt[chaotic], y[chaotic], s=11, color=RED, lw=0)
        ax.set_ylim(CLIP - 0.08, 0.75)
        ax.set_xlim(lt[0] - 0.04, lt[-1] + 0.04)
        ax.set_xlabel(r"$\log_{10}$ temperature", fontsize=8)
        if col == 0:
            ax.set_ylabel("largest exponent\n(clipped at $-1.5$)", fontsize=8)
        ax.set_title(f"({next(letters)}) no steering: chaotic at {found['chaotic_all']} of {L.size} temperatures",
                     fontsize=8.5, loc="left")
        ax.tick_params(labelsize=7, length=2.5, color=MUTED)
        ax.grid(axis="y", color=HAIR, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.text(0.5, 1.2, name, fontsize=10, color=INK, ha="center", va="bottom", transform=ax.transAxes)
        summary[tag]["no_steering"] = {
            **found, "periodic": int(periodic.sum()), "clipped_below": int((L < CLIP).sum()),
            "chaotic_log10_temperature": [float(v) for v in lt[chaotic]],
            "chaotic_exponents": [float(v) for v in L[chaotic]],
            "log10_temperature_range": [float(lt[0]), float(lt[-1])]}
    for row, (key, concept) in enumerate(MAPS, start=1):
        for col, (tag, name) in enumerate(MODELS):
            z = np.load(P / f"{tag}-{key}" / "scan.npz")
            al, lt, L, per = z["alpha"], z["log10_temperature"], z["lyapunov"], z["period"]
            if L.shape != (lt.size, al.size):
                raise SystemExit(f"{tag}-{key}: exponents have shape {L.shape}")
            chaotic, unresolved = labels(L, per)
            found = {"cells": int(L.size), "chaotic": int(chaotic.sum()), "unresolved": int(unresolved.sum()),
                     "share": float(chaotic.mean()), "max_exponent": float(L.max())}
            check(found, stored[tag][key], f"{tag} {key}")
            da, dt = al[1] - al[0], lt[1] - lt[0]
            ax = axes[(row, col)] = fig.add_subplot(gs[row, col])
            im = ax.imshow(np.clip(L, -0.5, 0.5), origin="lower", aspect="auto", cmap="RdBu_r", vmin=-0.5, vmax=0.5,
                           interpolation="nearest",
                           extent=[al[0] - da / 2, al[-1] + da / 2, lt[0] - dt / 2, lt[-1] + dt / 2])
            ax.set_xlabel(rf"steering strength $\alpha$ ({concept})", fontsize=8)
            if col == 0:
                ax.set_ylabel(r"$\log_{10}$ temperature", fontsize=8)
            ax.set_title(f"({next(letters)}) {concept}: chaotic at {found['chaotic']} of {L.size} cells",
                         fontsize=8.5, loc="left")
            ax.tick_params(labelsize=7, length=2.5, color=MUTED)
            for s in ax.spines.values():
                s.set_color(MUTED)
            summary[tag][concept] = {**found, "periodic": int((per > 0).sum()),
                                     "alpha_range": [float(al[0]), float(al[-1])],
                                     "log10_temperature_range": [float(lt[0]), float(lt[-1])],
                                     "share_of_cells_below_minus_0.5": float((L < -0.5).mean())}
    handles = [Line2D([], [], color=RED, marker="o", ms=3.6, lw=0, label="chaotic"),
               Line2D([], [], color=BLUE, marker="o", ms=3.0, lw=0, label="periodic"),
               Line2D([], [], color=INK, marker="x", ms=4.2, mew=0.8, lw=0, label="unresolved")]
    axes[(0, 0)].legend(handles=handles, fontsize=6.5, frameon=False, loc="center left", bbox_to_anchor=(0.03, 0.5),
                        handletextpad=0.2, borderaxespad=0.3)
    top, bottom = axes[(1, 1)].get_position(), axes[(2, 1)].get_position()
    cax = fig.add_axes([0.905, bottom.y0, 0.013, top.y1 - bottom.y0])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("largest Lyapunov exponent (per step, clipped)", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_edgecolor(MUTED)
    fig.savefig(out_png, dpi=200)
    Path(out_json).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
