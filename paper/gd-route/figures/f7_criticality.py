"""Figure 7: criticality of the complex-pair crossing of the period-2 orbit (S2-28011).

Evidence: exploratory (docs/explore-criticality-2026-10-07.md). Panel A: first
Lyapunov coefficient d of the normal form at every point of the merged curve
(Kuznetsov's formula for n-dimensional maps; d < 0 supercritical, d > 0
subcritical), on a symmetric-log axis, with the sign changes (candidate generalized
Neimark-Sacker points) marked; the strip below shows the label of the fixed-start
rotation maps four cells beyond the curve. Panel B: supercritical points: the
measured amplitude of the circle reached from near the 2-cycle against the
normal-form prediction sqrt(log|mu| / -d) at three offsets beyond the curve.
Panel C: subcritical points: the escape threshold before the curve against the
offset, with the predicted repelling-circle amplitude.

Consistency rule used for the dynamical checks (also in build_numbers.py): a
supercritical point is consistent when the measured amplitude matches the
prediction within 6% at all three offsets; a subcritical point is consistent
when, beyond the curve, the state jumps to an amplitude of at least 0.1 that
changes by less than 10% across offsets, and, before it, finite escape
thresholds grow by a factor between 1.4 and 2.8 per fourfold offset.
"""

import json

import numpy as np
from matplotlib.lines import Line2D

import common as c

CURVES = [c.EXPLORE / "criticality/s2-28011-curve/curve.json", c.EXPLORE / "criticality/s2-28011-steep/curve.json"]
BOUNDARY = c.EXPLORE / "criticality/boundary.json"
DYNAMICS = c.EXPLORE / "criticality/s2-28011-dynamics/dynamics2.json"
LABEL_COLORS = {"quasi-periodic": c.INK, "periodic": c.CLASS_COLORS[2], "chaotic": c.CLASS_COLORS[3],
                "divergent": "#a3a29b", "unresolved": "#e1e0d9"}


def curve_points():
    rows = {}
    for path in CURVES:
        for r in json.loads(path.read_text())["rows"]:
            if r.get("type") == "neimark_sacker" and r.get("d") is not None:
                rows[round(r["log10_output"], 4)] = r
    return [rows[k] for k in sorted(rows)]


def sign_changes(points):
    out = []
    for a, b in zip(points[:-1], points[1:]):
        if np.sign(a["d"]) != np.sign(b["d"]):
            out.append(0.5 * (a["log10_output"] + b["log10_output"]))
    return out


def consistent(row):
    if row["d"] < 0:
        errors = [abs(b["w_median"] / b["predicted_w"] - 1) for b in row["beyond"]]
        return max(errors) <= 0.06, max(errors)
    amps = [b["w_median"] for b in row["beyond"]]
    jump = min(amps) >= 0.1 and max(amps) / min(amps) < 1.1
    thresholds = [b["threshold"] for b in row["before"]]
    finite = all(isinstance(t, (int, float)) for t in thresholds)
    growth = finite and all(1.4 <= t2 / t1 <= 2.8 for t1, t2 in zip(thresholds[:-1], thresholds[1:]))
    return bool(jump and growth), None


def main():
    c.style()
    points = curve_points()
    changes = sign_changes(points)
    boundary = {round(b["log10_output"], 4): b for b in json.loads(BOUNDARY.read_text())}
    dyn = json.loads(DYNAMICS.read_text())["rows"]
    fig = c.plt.figure(figsize=(7.1, 5.3), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.95])
    ax = fig.add_subplot(grid[0, :])
    x = np.array([p["log10_output"] for p in points])
    d = np.array([p["d"] for p in points])
    ax.axhline(0, color=c.MUTED, linewidth=0.6)
    ax.plot(x[d < 0], d[d < 0], "o", color=c.FOLD, markersize=3.5, label="d < 0: supercritical (soft onset)")
    ax.plot(x[d > 0], d[d > 0], "^", color=c.FLIP, markersize=4, label="d > 0: subcritical (hard onset)")
    for z in changes:
        ax.axvline(z, color=c.SECONDARY, linewidth=0.7, linestyle="--")
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_ylabel("first Lyapunov coefficient d")
    ax.set_xlabel("log10 output-layer rate (point on the curve)")
    ax.set_title("A  Criticality along the curve; dashed: sign changes (generalized NS candidates)",
                 fontsize=8)
    strip_y = d.min() * 1.6
    for p in points:
        b = boundary.get(round(p["log10_output"], 4))
        if b is None:
            continue
        label = b["labels_beyond_cells"].get("4")
        ax.plot(p["log10_output"], strip_y, "s", markersize=3.2, color=LABEL_COLORS.get(label, c.MUTED))
    ax.text(0.01, 0.13, "strip: fixed-start map label 4 cells beyond the curve", transform=ax.transAxes,
            fontsize=6.3, color=c.SECONDARY)
    handles = ax.get_legend_handles_labels()[0]
    handles += [Line2D([], [], marker="s", linestyle="", color=col, label=f"map: {lab}")
                for lab, col in LABEL_COLORS.items()]
    ax.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=6)
    bx = fig.add_subplot(grid[1, 0])
    results = []
    for row in dyn:
        ok, worst = consistent(row)
        results.append({"log10_output": row["log10_output"], "d": row["d"], "consistent": ok, "max_amplitude_error": worst})
        if row["d"] < 0:
            pred = [b["predicted_w"] for b in row["beyond"]]
            meas = [b["w_median"] for b in row["beyond"]]
            bx.plot(pred, meas, "o-", color=c.FOLD, markersize=3.5, linewidth=0.6)
    lim = [0.005, 0.12]
    bx.plot(lim, lim, color=c.MUTED, linewidth=0.8, zorder=0)
    bx.set_xscale("log")
    bx.set_yscale("log")
    bx.set_xlim(lim)
    bx.set_ylim(lim)
    bx.set_xlabel("predicted circle amplitude sqrt(log|mu| / -d)")
    bx.set_ylabel("measured amplitude")
    bx.set_title("B  Supercritical points: soft onset", fontsize=8)
    cx = fig.add_subplot(grid[1, 1])
    for row in dyn:
        if row["d"] > 0:
            off = [-b["delta"] for b in row["before"]]
            cx.plot(off, [b["threshold"] for b in row["before"]], "^-", color=c.FLIP, markersize=4, linewidth=0.6)
            cx.plot(off, [b["predicted_unstable_circle_w"] for b in row["before"]], "v:", color=c.SECONDARY,
                    markersize=3.5, linewidth=0.6)
    cx.set_xscale("log")
    cx.set_yscale("log")
    cx.set_xlabel("offset before the curve (decades)")
    cx.set_ylabel("escape threshold")
    cx.set_title("C  Subcritical points: hard onset", fontsize=8)
    cx.legend(handles=[Line2D([], [], marker="^", color=c.FLIP, label="measured escape threshold"),
                       Line2D([], [], marker="v", linestyle=":", color=c.SECONDARY,
                              label="predicted repelling circle")], frameon=False, fontsize=6.3, loc="upper left")
    derived = {"points": len(points), "supercritical": int((d < 0).sum()), "subcritical": int((d > 0).sum()),
               "sign_changes": changes, "most_negative_d": float(d.min()),
               "most_negative_at": float(x[int(np.argmin(d))]), "dynamical_checks": results}
    n_ok = sum(r["consistent"] for r in results)
    alt = (f"First Lyapunov coefficient d at {len(points)} points of the curve where the period-2 orbit of S2-28011 "
           f"acquires complex multipliers: {derived['supercritical']} supercritical (d < 0) and "
           f"{derived['subcritical']} subcritical (d > 0), with {len(changes)} sign changes. The fixed-start maps "
           f"show quasi-periodic training beyond the supercritical arc. In {n_ok} of {len(results)} dynamical checks "
           f"the behaviour matches the sign: circle amplitudes follow the square-root prediction at supercritical "
           f"points, and escape thresholds before the curve grow like the square root of the offset at "
           f"subcritical points.")
    c.save(fig, "f7_criticality", [*CURVES, BOUNDARY, DYNAMICS], __file__, "exploratory", alt, derived,
           "Sign changes located to the trace spacing only.")


if __name__ == "__main__":
    main()
