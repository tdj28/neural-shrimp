"""Figure fig-real-inference: ordinary generation at settings where the soft map is periodic or chaotic (run RI1,
exploratory, 22 Oct 2026; tools/real_inference_twins.py; Qwen2.5-0.5B-Instruct, float64). Reads the committed receipt
docs/ri/ri1-receipt.json, so the figure needs no raw data.

Two copies of an ordinary sampled generation share their random numbers; the first prompt embedding of one copy is
nudged by a relative 1e-3. Fifteen settings lie on a line of steering strengths at log10 T = -0.153 that crosses the
right edge of the period-6 body; 32 pairs per setting, 300 new tokens.

(a, c) Rates per step along the line for the paper's prompt (a) and for a second prompt (c): the soft map's largest
       Lyapunov exponent, and for ordinary generation the median growth rate of the total-variation distance between
       the two copies' next-token distributions while they still agree on every token (plain sampling and the
       model's default sampler).
(b, d) Number of the 32 pairs whose texts come to differ, for the two samplers. The gray band is the central 95 %
       range of a binomial count of 32 at the fraction pooled over the panel's 15 settings (plain sampling).
(e)    The four entropy-matched pairs of cells: pairs whose texts differ at the periodic and at the chaotic cell.

The JSON written beside the figure holds the plotted numbers, the pooled counts, the band and a chi-square test of
equal fractions along the line.

Usage: python tools/plot_real_inference.py [OUT.png] [OUT.json]
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / "docs" / "ri" / "ri1-receipt.json"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"     # soft map; ordinary, plain sampling; ordinary, default sampler
INK, MUTED, HAIR, BAND = "#0b0b0b", "#52514e", "#d9d8d3", "#ecebe6"
PROMPTS = [("open", "the paper's prompt"), ("p08", "a second prompt")]
NUDGE, PAIRS = 1e-3, 32


def binomial_range(n, p, level=0.95):
    """Smallest and largest count of the central `level` range of Binomial(n, p)."""
    pmf = [math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)]
    cdf = np.cumsum(pmf)
    tail = (1 - level) / 2
    lo = int(np.searchsorted(cdf, tail))                 # first k with cdf >= tail
    hi = int(np.searchsorted(cdf, 1 - tail))             # first k with cdf >= 1 - tail
    return lo, hi


def chi_square_equal_fractions(counts, n):
    """Pearson statistic for equal success fractions in len(counts) groups of n, with its upper tail probability."""
    counts = np.asarray(counts, dtype=float)
    p = counts.sum() / (n * counts.size)
    stat = float(((counts - n * p) ** 2 / (n * p * (1 - p))).sum())
    dof = counts.size - 1
    from scipy.stats import chi2
    return {"statistic": stat, "degrees_of_freedom": dof, "upper_tail_probability": float(chi2.sf(stat, dof))}


def line_rows(rows, prompt, sampler):
    sel = sorted((r for r in rows if r["group"] == "line" and r["prompt"] == prompt and r["sampler"] == sampler
                  and r["nudge"] == NUDGE), key=lambda r: r["position"])
    if [r["position"] for r in sel] != list(range(1, 16)):
        raise SystemExit(f"line {prompt}/{sampler}: expected positions 1..15, found {[r['position'] for r in sel]}")
    return sel


def splits(r):
    n = round(r["split_rate"] * PAIRS)
    if abs(n - r["split_rate"] * PAIRS) > 1e-9 or sum(r["counts"].values()) != PAIRS:
        raise SystemExit(f"{r['task']}: split rate {r['split_rate']} is not a count of {PAIRS} pairs")
    return int(n)


def style(ax):
    ax.tick_params(labelsize=7, length=2.5, color=MUTED)
    ax.grid(axis="y", color=HAIR, lw=0.5)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    out_png = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "paper/llm/figures/fig-real-inference.png")
    out_json = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "paper/llm/figures/fig-real-inference.json")
    rec = json.loads(RECEIPT.read_text())
    rows = rec["rows"]
    if not (rec["all_gates_ok"] and rec["complete"] and rec["float64"] and all(r["gate_ok"] for r in rows)):
        raise SystemExit("the receipt reports a failed gate; refusing to draw")
    fig = plt.figure(figsize=(7.8, 4.5))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.62], hspace=0.42, wspace=0.42, left=0.085, right=0.985,
                          top=0.93, bottom=0.105)
    summary = {"receipt": "docs/ri/ri1-receipt.json", "run_json_sha256": rec["run_json_sha256"], "nudge": NUDGE,
               "pairs_per_setting": PAIRS, "line": {}, "matched": {}}
    for row, (prompt, name) in enumerate(PROMPTS):
        plain, qwen = line_rows(rows, prompt, "plain"), line_rows(rows, prompt, "qwen")
        alpha = np.array([r["alpha"] for r in plain])
        soft = np.array([r["soft_lyapunov"] for r in plain])
        if any(abs(a["soft_lyapunov"] - b["soft_lyapunov"]) > 0 or a["alpha"] != b["alpha"]
               for a, b in zip(plain, qwen)):
            raise SystemExit("the two samplers do not share the line's settings")
        chaotic = np.array([r["soft_lyapunov"] > 1e-3 and r["soft_period"] <= 0 for r in plain])
        edge = None
        if chaotic.any() and not chaotic.all():
            k = int(np.argmax(chaotic))
            if chaotic[:k].any() or not chaotic[k:].all():
                raise SystemExit("the soft map's regime changes more than once along the line")
            edge = 0.5 * (alpha[k - 1] + alpha[k])
        ax = fig.add_subplot(gs[row, 0])
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.plot(alpha, soft, color=BLUE, lw=1.3, marker="o", ms=3.2)
        ax.plot(alpha, [r["median_tv_slope"] for r in plain], color=ORANGE, lw=1.3, marker="o", ms=3.2)
        ax.plot(alpha, [r["median_tv_slope"] for r in qwen], color=AQUA, lw=1.1, ls=(0, (3, 2)), marker="o", ms=3.6,
                mfc="white", mew=0.9)
        ax.set_ylim(-0.88, 0.34)
        ax.set_ylabel("rate per step", fontsize=8)
        ax.set_title(f"({'ac'[row]}) rates, {name}", fontsize=9, loc="left")
        style(ax)
        bx = fig.add_subplot(gs[row, 1])
        n_plain, n_qwen = [splits(r) for r in plain], [splits(r) for r in qwen]
        pooled = sum(n_plain) / (PAIRS * len(n_plain))
        lo, hi = binomial_range(PAIRS, pooled)
        bx.axhspan(lo, hi, color=BAND, lw=0)
        bx.plot(alpha, n_plain, color=ORANGE, lw=1.3, marker="o", ms=3.2)
        bx.plot(alpha, n_qwen, color=AQUA, lw=1.1, ls=(0, (3, 2)), marker="o", ms=3.6, mfc="white", mew=0.9)
        bx.set_ylim(-1.2, 33)
        bx.set_yticks([0, 8, 16, 24, 32])
        bx.set_ylabel(f"pairs of {PAIRS}", fontsize=8)
        bx.set_title(f"({'bd'[row]}) pairs whose texts differ", fontsize=9, loc="left")
        style(bx)
        for a in (ax, bx):
            a.set_xticks([0.0734, 0.0735, 0.0736])
            a.set_xlim(alpha[0] - 1.2e-5, alpha[-1] + 1.2e-5)
            if row == 1:
                a.set_xlabel(r"steering strength $\alpha$", fontsize=8)
            if edge is not None:
                a.axvline(edge, color=MUTED, lw=0.7, ls=":")
        top = 0.955
        if edge is not None:
            ax.text(edge - 6e-6, top, "soft map periodic", fontsize=6.3, color=MUTED, ha="right", va="top",
                    transform=ax.get_xaxis_transform())
            ax.text(edge + 6e-6, top, "chaotic", fontsize=6.3, color=MUTED, ha="left", va="top",
                    transform=ax.get_xaxis_transform())
        else:
            ax.text(0.5, top, "soft map periodic at every setting", fontsize=6.3, color=MUTED, ha="center", va="top",
                    transform=ax.transAxes)
        summary["line"][prompt] = {
            "alpha": [float(a) for a in alpha], "log10_temperature": float(plain[0]["log_t"]),
            "soft_lyapunov": [float(v) for v in soft], "soft_period": [int(r["soft_period"]) for r in plain],
            "soft_chaotic": [bool(v) for v in chaotic],
            "plain": {"pairs_differ": n_plain, "median_growth_rate": [r["median_tv_slope"] for r in plain],
                      "pooled": [int(sum(n_plain)), PAIRS * len(n_plain)],
                      "binomial_95_range": [lo, hi],
                      "where_soft_periodic": [int(np.array(n_plain)[~chaotic].sum()), int(PAIRS * (~chaotic).sum())],
                      "where_soft_chaotic": [int(np.array(n_plain)[chaotic].sum()), int(PAIRS * chaotic.sum())],
                      "equal_fractions_test": chi_square_equal_fractions(n_plain, PAIRS)},
            "default_sampler": {"pairs_differ": n_qwen, "median_growth_rate": [r["median_tv_slope"] for r in qwen],
                                "pooled": [int(sum(n_qwen)), PAIRS * len(n_qwen)],
                                "where_soft_periodic": [int(np.array(n_qwen)[~chaotic].sum()),
                                                        int(PAIRS * (~chaotic).sum())],
                                "where_soft_chaotic": [int(np.array(n_qwen)[chaotic].sum()),
                                                       int(PAIRS * chaotic.sum())]}}
    # (e) the entropy-matched pairs of cells
    ex = fig.add_subplot(gs[0, 2])
    for sampler, color, kw in (("plain", ORANGE, {"lw": 1.2, "ms": 3.2}),
                               ("qwen", AQUA, {"lw": 1.0, "ls": (0, (3, 2)), "ms": 3.6, "mfc": "white", "mew": 0.9})):
        m = [r for r in rows if r["group"] == "matched" and r["sampler"] == sampler and r["nudge"] == NUDGE]
        pairs = {}
        for r in m:
            key, kind = r["task"].split("--")[0].rsplit("-", 1)
            if (kind == "chaotic") != (r["soft_lyapunov"] > 1e-3 and r["soft_period"] <= 0):
                raise SystemExit(f"{r['task']}: the soft map's label does not match the task name")
            pairs.setdefault(key, {})[kind] = r
        if len(pairs) != 4 or any(set(p) != {"periodic", "chaotic"} for p in pairs.values()):
            raise SystemExit("expected four matched pairs of a periodic and a chaotic cell")
        out = []
        for key in sorted(pairs):
            p = pairs[key]
            ex.plot([0, 1], [splits(p["periodic"]), splits(p["chaotic"])], color=color, marker="o", **kw)
            out.append({"pair": key, "soft_entropy": p["chaotic"]["soft_entropy"],
                        "periodic": {"alpha": p["periodic"]["alpha"], "log10_temperature": p["periodic"]["log_t"],
                                     "soft_lyapunov": p["periodic"]["soft_lyapunov"],
                                     "soft_period": p["periodic"]["soft_period"],
                                     "pairs_differ": splits(p["periodic"])},
                        "chaotic": {"alpha": p["chaotic"]["alpha"], "log10_temperature": p["chaotic"]["log_t"],
                                    "soft_lyapunov": p["chaotic"]["soft_lyapunov"],
                                    "pairs_differ": splits(p["chaotic"])}})
        summary["matched"]["plain" if sampler == "plain" else "default_sampler"] = {
            "pairs": out, "periodic_total": [sum(o["periodic"]["pairs_differ"] for o in out), PAIRS * len(out)],
            "chaotic_total": [sum(o["chaotic"]["pairs_differ"] for o in out), PAIRS * len(out)]}
    ex.set_xlim(-0.35, 1.35)
    ex.set_xticks([0, 1])
    ex.set_xticklabels(["soft map\nperiodic", "soft map\nchaotic"], fontsize=7)
    ex.set_ylim(-1.2, 33)
    ex.set_yticks([0, 8, 16, 24, 32])
    ex.set_ylabel(f"pairs of {PAIRS}", fontsize=8)
    ex.set_title("(e) matched cells", fontsize=9, loc="left")
    style(ex)
    # key, in the free cell under (e)
    kx = fig.add_subplot(gs[1, 2])
    kx.axis("off")
    handles = [Line2D([], [], color=BLUE, lw=1.3, marker="o", ms=3.2, label="soft map:\nLyapunov exponent"),
               Line2D([], [], color=ORANGE, lw=1.3, marker="o", ms=3.2, label="ordinary generation,\nplain sampling"),
               Line2D([], [], color=AQUA, lw=1.1, ls=(0, (3, 2)), marker="o", ms=3.6, mfc="white", mew=0.9,
                      label="ordinary generation,\nthe model's default sampler"),
               Patch(facecolor=BAND, edgecolor="none", label="range expected for equal\nfractions (95 %)")]
    kx.legend(handles=handles, fontsize=6.6, frameon=False, loc="center left", bbox_to_anchor=(-0.42, 0.5),
              handlelength=2.4, labelspacing=0.9)
    fig.savefig(out_png, dpi=200)
    Path(out_json).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
