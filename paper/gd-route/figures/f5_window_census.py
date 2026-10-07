"""Figure 5: census over the 61 audit-confirmed EXP-028 windows.

Evidence: exploratory analyses of frozen certificates. Panel A: boundary type
along 64 rays per window (tools/explore_a2_survey.py rays). Panel B: share of
orbits that are well-ordered (Birkhoff) circle orbits, against the summed
random-angle null expectation (tools/explore_a3_well_ordering.py, with and
without the tour-order and period-doubled refinements). Panel C: window period
against the denominator q of its tongue rotation m/q for windows passing the
refined test on at least 5 points.
"""

import json
from collections import Counter

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import common as c

RAYS = c.EXPLORE / "a2-survey/rays.json"
ORDER = c.EXPLORE / "a3-well-ordering/well-ordering.json"
TYPES = ["fold", "flip", "neimark_sacker", "other"]
OTHER = "#e1e0d9"


def main():
    c.style()
    rays = json.loads(RAYS.read_text())
    rows = json.loads(ORDER.read_text())
    fig = c.plt.figure(figsize=(7.1, 5.6), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
    ax = fig.add_subplot(grid[0, :])
    windows = []
    for key, w in rays.items():
        counts = {t: 0 for t in TYPES}
        for r in w["rays"]:
            counts[r["type"] if r["type"] in TYPES else "other"] += 1
        windows.append((w["system"], -counts["fold"] / len(w["rays"]), key, w["period"], counts, len(w["rays"])))
    windows.sort()
    x = np.arange(len(windows))
    bottom = np.zeros(len(windows))
    colors = {"fold": c.FOLD, "flip": c.FLIP, "neimark_sacker": c.NS, "other": OTHER}
    labels = {"fold": "fold", "flip": "flip", "neimark_sacker": "Neimark-Sacker",
              "other": "no loss within range / unresolved"}
    for t in TYPES:
        values = np.array([w[4][t] / w[5] for w in windows])
        ax.bar(x, values, bottom=bottom, width=0.86, color=colors[t], edgecolor=c.SURFACE, linewidth=0.4,
               label=labels[t])
        bottom += values
    split = next(i for i, w in enumerate(windows) if w[0] == "S2")
    ax.axvline(split - 0.5, color=c.INK, linewidth=0.8)
    ax.text(split / 2, 1.04, "S1R (72 parameters)", ha="center", fontsize=7, color=c.INK)
    ax.text(split + (len(windows) - split) / 2, 1.04, "S2 (13 parameters)", ha="center", fontsize=7, color=c.INK)
    ax.set_xticks(x, [str(w[3]) for w in windows], fontsize=5.5)
    ax.set_xlabel("confirmed windows, labeled by period (sorted by system, then fold share)")
    ax.set_ylabel("share of 64 rays")
    ax.set_ylim(0, 1.12)
    ax.set_xlim(-0.6, len(windows) - 0.4)
    ax.set_title("A  How each window's periodic orbit loses stability (64 rays per window)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=4, frameon=False)
    bx = fig.add_subplot(grid[1, 0])
    groups = [
        ("certified orbits,\n>= 5 points", [r for r in rows if r["points"] >= 5], "plane"),
        ("confirmed\nwindows", [r for r in rows if r["confirmed"]], "plane"),
        ("confirmed,\nrefined", [r for r in rows if r["confirmed"]
                                 and any(t["q"] >= 5 for t in r["refined"]["tests"])], "refined"),
        ("confirmed,\nodd period", [r for r in rows if r["confirmed"] and r["period"] % 2], "plane"),
    ]
    group_values = []
    for i, (label, sub, kind) in enumerate(groups):
        if kind == "plane":
            observed = sum(r["well_ordered"] for r in sub)
            expected = sum(r["null_pass_rate"] for r in sub)
        else:
            observed = sum(r["refined"]["passed_q5"] for r in sub)
            expected = sum(r["refined"]["null_q5"] for r in sub)
        group_values.append((label, {"observed": observed, "n": len(sub), "expected": expected}))
        bx.bar(i, observed / len(sub), width=0.62, color=c.SECONDARY)
        bx.plot([i - 0.31, i + 0.31], [expected / len(sub)] * 2, color=c.FLIP, linewidth=2.0)
        bx.text(i, observed / len(sub) + 0.03, f"{observed}/{len(sub)}", ha="center", fontsize=6.5, color=c.INK)
    bx.set_xticks(range(len(groups)), [g[0] for g in groups], fontsize=6.3)
    bx.set_ylim(0, 1)
    bx.set_ylabel("share well-ordered (Birkhoff circle orbit)")
    bx.set_title("B  Well-ordering versus random-angle null")
    bx.legend(handles=[Patch(color=c.SECONDARY, label="observed"),
                       Line2D([], [], color=c.FLIP, linewidth=2, label="null expectation")],
              loc="upper right", frameon=False)
    cx = fig.add_subplot(grid[1, 1])
    confirmed = [r for r in rows if r["confirmed"]]
    passed = [r for r in confirmed if r.get("refined", {}).get("passed_q5")]
    for level, marker in ((0, "o"), (1, "D"), (2, "s"), (3, "^")):
        sub = Counter((r["refined"]["base"]["q"], r["period"]) for r in passed
                      if r["refined"]["base"]["level"] == level)
        if not sub:
            continue
        cx.plot([q for q, _ in sub], [p for _, p in sub], marker, color=c.FOLD, markersize=4.5,
                markeredgecolor=c.SURFACE, markeredgewidth=0.5, linestyle="",
                label=f"period = 2q x {2 ** level}" if level else "period = 2q")
        for (q, p), n in sub.items():
            if n > 1:
                cx.annotate(f"x{n}", (q, p), textcoords="offset points", xytext=(4, -2), fontsize=6,
                            color=c.SECONDARY)
    qs = np.arange(2, 31)
    for k in range(3):
        cx.plot(qs, 2 * qs * 2 ** k, color="#e1e0d9", linewidth=0.8, zorder=0)
        cx.text(qs[-1] / 2 ** k + 0.3, 2 * qs[-1], f"2q x {2 ** k}" if k else "2q", fontsize=6, color=c.MUTED)
    cx.set_xlim(0, 31)
    cx.set_ylim(0, 64)
    cx.set_xlabel("tongue denominator q (rotation m/q of the two-step map)")
    cx.set_ylabel("certified window period")
    cx.set_title(f"C  Period versus tongue denominator ({len(passed)} windows)")
    cx.legend(loc="lower right", frameon=False)
    census = {t: sum(w[4][t] for w in windows) for t in TYPES}
    derived = {
        "windows": len(windows), "rays_per_window": sorted({w[5] for w in windows}), "ray_type_totals": census,
        "windows_fold_and_flip_at_least_8": sum(w[4]["fold"] >= 8 and w[4]["flip"] >= 8 for w in windows),
        "windows_neimark_sacker_at_least_8": sum(w[4]["neimark_sacker"] >= 8 for w in windows),
        "well_ordering": {label.replace(chr(10), " "): value for label, value in group_values},
        "refined_passing_windows": len(passed),
        "period_vs_q": sorted([r["refined"]["base"]["q"], r["period"], r["refined"]["base"]["level"]] for r in passed),
    }
    alt = (f"Top: for each of the {len(windows)} confirmed windows, the share of 64 continuation rays ending at a "
           f"fold, flip or Neimark-Sacker crossing. Fold dominates many S1R windows; Neimark-Sacker and flip dominate "
           f"most S2 windows. Bottom left: share of orbits that are well-ordered circle orbits, against the summed "
           f"random-angle null (orange): " + "; ".join(f"{k} {v['observed']}/{v['n']} versus {v['expected']:.2f} "
                                                       f"expected" for k, v in derived["well_ordering"].items())
           + f". Bottom right: for the {len(passed)} windows passing the refined test, certified period against the "
           f"tongue denominator q: every point lies on period = 2q or period = 4q.")
    c.save(fig, "f5_window_census", [RAYS, ORDER], __file__, "exploratory", alt, derived,
           "Only tests on at least 5 points count toward the refined share; orbits with 2 or 3 points pass trivially.")


if __name__ == "__main__":
    main()
