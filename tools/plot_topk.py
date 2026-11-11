"""Figure fig-topk: the soft-token loop of Qwen2.5-0.5B-Instruct when only the k most probable tokens are fed back
(run TK, exploratory, 28 Oct 2026; tools/topk_loop.py; float64).

(a, b) Largest Lyapunov exponent at 128 temperatures on the steered line (philosophy direction at layer 12,
       alpha = 0.0743) and on the unsteered line, one strip per k (all tokens, 50, 20, 10, 5, 3, 2, 1). The color
       scale is that of the paper's maps, clipped at +-0.5. A black bar under a cell marks a positive exponent without
       a detected period whose twin crossed a top-k set boundary at least once ("irregular": the exponent can contain
       jumps).
(c, d) Separation of two copies at the steered cell that is chaotic with all tokens (log10 T = -0.0476): median over
       32 repeats of the distance between the copies after each of 200 steps, for five initial offsets (the point at
       step 0), with all tokens (c) and with the 20 most probable tokens (d). No renormalization; the dotted line is
       the threshold, 0.1 of the state norm, used for the separation counts.

The JSON written beside the figure holds the counts per strip and, for all three cells of the separation runs, the
counts of repeats that reach the threshold and the median step at which they do. These are recomputed from the arrays
and must equal the summary that tools/topk_loop.py stored with them.

Usage: python tools/plot_topk.py [OUT.png] [OUT.json]
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
KS = [0, 50, 20, 10, 5, 3, 2, 1]                       # top to bottom; 0 = the whole vocabulary
LINES = [(1, "(a) steered line"), (0, "(b) unsteered line")]   # column of the scan arrays, title
RAMP = ["#0d366b", "#1c5cab", "#2a78d6", "#6da7ec", "#b7d3f6"]  # one hue, dark for the largest offset
INK, MUTED, HAIR = "#0b0b0b", "#52514e", "#d9d8d3"


def load_strip(P, k):
    z, t = np.load(P / f"soft-k{k}" / "scan.npz"), np.load(P / f"soft-k{k}" / "topk.npz")
    L, per, sw = z["lyapunov"], z["period"], t["set_switch_steps"]
    positive = (L > 1e-3) & (per <= 0)
    irregular = positive & (sw > 0)
    return z["log10_temperature"], z["alpha"], L, per, positive & ~irregular, irregular


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import explore_llm_attractors as ela

    out_png = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "paper/llm/figures/fig-topk.png")
    out_json = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "paper/llm/figures/fig-topk.json")
    P = ela.DATA / "explore" / "llm-soft-loop" / "pod64" / "topk"
    fig = plt.figure(figsize=(7.8, 5.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.52, wspace=0.26, left=0.085, right=0.9,
                          top=0.94, bottom=0.085)
    summary = {"strips": {}, "separation": {}}
    im = None
    for col, (j, title) in enumerate(LINES):
        ax = fig.add_subplot(gs[0, col])
        for row, k in enumerate(KS):
            lt, alpha, L, per, chaotic, irregular = load_strip(P, k)
            step = lt[1] - lt[0]
            y = len(KS) - 1 - row
            im = ax.imshow(np.clip(L[:, j], -0.5, 0.5)[None, :], origin="lower", aspect="auto", cmap="RdBu_r",
                           vmin=-0.5, vmax=0.5, interpolation="nearest",
                           extent=[lt[0] - step / 2, lt[-1] + step / 2, y - 0.36, y + 0.36])
            for x in lt[irregular[:, j]]:
                ax.plot([x - step / 2, x + step / 2], [y - 0.46, y - 0.46], color=INK, lw=1.6, solid_capstyle="butt")
            summary["strips"].setdefault(f"alpha={alpha[j]:.4f}", {})["all" if k == 0 else str(k)] = {
                "temperatures": int(L.shape[0]), "chaotic": int(chaotic[:, j].sum()),
                "irregular": int(irregular[:, j].sum()), "periodic": int((per[:, j] > 0).sum()),
                "unresolved": int(((per[:, j] <= 0) & ~(L[:, j] > 1e-3)).sum()),
                "largest_exponent": float(L[:, j].max())}
        ax.set_ylim(-0.62, len(KS) - 0.5)
        ax.set_yticks(range(len(KS)))
        ax.set_yticklabels(["all" if k == 0 else str(k) for k in KS][::-1], fontsize=7)
        ax.set_xlabel(r"$\log_{10}$ temperature", fontsize=8)
        if col == 0:
            ax.set_ylabel("tokens kept, $k$", fontsize=8)
        ax.set_title(title, fontsize=9, loc="left")
        if col == 1:                                  # key of the mark, on the title line
            ax.plot([0.47, 0.5], [1.055, 1.055], color=INK, lw=1.6, solid_capstyle="butt", transform=ax.transAxes,
                    clip_on=False)
            ax.text(0.515, 1.055, "positive, and the kept set changed", fontsize=6.5, color=MUTED, va="center",
                    transform=ax.transAxes)
        ax.tick_params(labelsize=7, length=2.5, color=MUTED)
        ax.tick_params(axis="y", length=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
    cax = fig.add_axes([0.915, 0.585, 0.012, 0.355])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("largest Lyapunov exponent\n(per step, clipped)", fontsize=7)
    cb.ax.tick_params(labelsize=6.5, length=2)
    cb.outline.set_edgecolor(MUTED)
    for col, (name, title) in enumerate([("soft", "(c) two copies, all tokens kept"),
                                         ("soft-k20", "(d) two copies, 20 tokens kept")]):
        ax = fig.add_subplot(gs[1, col])
        d = np.load(P / f"divergence-{name}" / "divergence.npz")
        info = json.loads((P / f"divergence-{name}" / "divergence.json").read_text())
        S, deltas, cells = d["log10_separation"], d["deltas"], d["cells"]     # S[s] is the separation after step s + 1
        thr = float(info["threshold_log10"])
        stored = {(tuple(r["cell"]), r["delta"]): r for r in info["summary"]}
        rows = []
        for c in range(len(cells)):
            hit = S[:, c, :, :] > thr                                         # steps x repeats x offsets
            first = np.where(hit.any(axis=0), hit.argmax(axis=0) + 1, -1)
            med = np.median(S[:, c, :, :], axis=1)                            # steps x offsets
            for i, delta in enumerate(deltas):
                got = first[:, i][first[:, i] > 0]
                row = {"cell_log10T_alpha": [float(v) for v in cells[c]], "offset": float(delta),
                       "repeats": int(S.shape[2]), "reached_threshold": int(got.size),
                       "median_steps_of_those": float(np.median(got)) if got.size else None,
                       "median_log10_distance_after_200_steps": float(med[-1, i]),
                       "largest_median_log10_distance": float(med.max(axis=0)[i])}
                ref = stored[(tuple(float(v) for v in cells[c]), float(delta))]
                if (ref["reached"], ref["median_steps"], ref["of"]) != (row["reached_threshold"],
                                                                        row["median_steps_of_those"], row["repeats"]):
                    raise SystemExit(f"separation counts differ from the stored summary: {row} against {ref}")
                rows.append(row)
        summary["separation"]["all" if name == "soft" else "20"] = {
            "state_norm": float(info["state_norm"]), "threshold_log10": thr, "steps": int(S.shape[0]), "rows": rows}
        c = 0                                                                 # the steered cell is drawn
        if abs(cells[c][0] + 0.0476) > 1e-9 or abs(cells[c][1] - 0.0743) > 1e-9:
            raise SystemExit(f"unexpected first cell {cells[c]}")
        med = np.median(S[:, c, :, :], axis=1)
        steps = np.arange(S.shape[0] + 1)
        ax.axhline(thr, color=MUTED, lw=0.7, ls=":")
        for i, delta in enumerate(deltas):
            ax.plot(steps, np.concatenate([[np.log10(delta)], med[:, i]]), color=RAMP[i], lw=1.3,
                    label=rf"$10^{{{int(round(np.log10(delta)))}}}$")
        ax.set_xlim(0, 200)
        ax.set_ylim(-15.5, 0.6)
        ax.set_xlabel("steps", fontsize=8)
        if col == 0:
            ax.set_ylabel(r"$\log_{10}$ distance between the copies", fontsize=8)
            ax.text(197, thr - 0.3, "threshold", fontsize=6.5, color=MUTED, ha="right", va="top")
        ax.set_title(title, fontsize=9, loc="left")
        ax.tick_params(labelsize=7, length=2.5, color=MUTED)
        ax.grid(axis="y", color=HAIR, lw=0.5)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        if col == 1:
            ax.legend(title="initial offset", fontsize=6.5, title_fontsize=6.5, frameon=False, loc="center left",
                      bbox_to_anchor=(1.01, 0.5), handlelength=1.4)
    fig.savefig(out_png, dpi=200)
    Path(out_json).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
