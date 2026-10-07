"""Figure 2: the Neimark-Sacker curve of the period-2 orbit and its predicted tongue roots.

Evidence: exploratory. Background: the 1,024 x 1,024 period map of the S2-28011
zoom region (CUDA float64 backend, EXP-028 screening options). Curve: the
period-2 orbit continued along hidden-rate slices to its loss of stability
(tools/explore_a3_ns_curve.py, bisection 1e-10); on every traced slice whose
start was period 2 it is a complex pair crossing the unit circle. Orange points: where the crossing angle
theta / 2 pi equals a rational m / q with q <= 13 (the predicted root of the
period-2q Arnold tongue). Panel B: theta / 2 pi along the curve.
"""

import json
from fractions import Fraction

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle

import common as c

MAP = c.EXPLORE / "s2-28011-zoom-1024/map.npz"
CURVE = c.EXPLORE / "a3-ns-curve/s2-28011/ns-curve.json"
MAX_DEN = 13


def main():
    c.style()
    saved = np.load(MAP)
    lh, lo = saved["log10_hidden"], saved["log10_output"]
    status, period = saved["status"], saved["period"]
    runs = c.ns_segments(CURVE)
    rows = [r for run in runs for r in run]
    fig = c.plt.figure(figsize=(7.1, 4.6), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.05, 0.62])
    ax = fig.add_subplot(grid[0, 0])
    periodic = (status == 2) & (period >= 2)
    levels = np.log2(np.where(periodic, period, 1).astype(float))
    cmap = ListedColormap(c.PERIOD_RAMP).with_extremes(bad=c.NEUTRAL)
    image = ax.imshow(np.ma.masked_where(~periodic, levels), origin="lower", aspect="auto",
                      extent=[lo[0], lo[-1], lh[0], lh[-1]], interpolation="nearest", cmap=cmap,
                      norm=BoundaryNorm(np.linspace(1, 6, len(c.PERIOD_RAMP) + 1), cmap.N), rasterized=True)
    bottom = min(r["log10_hidden_boundary"] for r in rows) - 0.012
    if bottom < lh[0]:
        ax.add_patch(Rectangle((lo[0], bottom), lo[-1] - lo[0], lh[0] - bottom, facecolor=c.SURFACE,
                               edgecolor=c.MUTED, hatch="////", linewidth=0.0, alpha=0.6))
        ax.text(lo[0] + 0.01, bottom + 0.004, "not mapped", fontsize=6.5, color=c.MUTED)
    for i, run in enumerate(runs):
        ax.plot([r["log10_output"] for r in run], [r["log10_hidden_boundary"] for r in run], "-", color=c.INK,
                linewidth=1.1, label="Neimark-Sacker curve of the period-2 orbit" if i == 0 else None)
        ax.plot([r["log10_output"] for r in run], [r["log10_hidden_boundary"] for r in run], ".", color=c.INK,
                markersize=2.5)
    found = c.ns_roots(runs, max_den=MAX_DEN)
    for r in found:
        ax.plot(r["log10_output"], r["log10_hidden"], "o", color=c.FLIP, markersize=5, markeredgecolor=c.SURFACE,
                markeredgewidth=0.8, zorder=5)
        f = r["fraction"]
        ax.annotate(f"{f.numerator}/{f.denominator}", (r["log10_output"], r["log10_hidden"]), textcoords="offset points",
                    xytext=(4, -9), fontsize=6.5, color=c.INK)
    ax.set_xlim(lo[0], lo[-1])
    ax.set_ylim(bottom, lh[-1])
    ax.set_xlabel("log10 output-layer learning rate")
    ax.set_ylabel("log10 hidden-layer learning rate")
    ax.set_title("A  Periodic cells, Neimark-Sacker curve, predicted roots")
    ax.legend(loc="upper right", frameon=False, fontsize=6.5)
    cb = fig.colorbar(image, cax=fig.add_subplot(grid[0, 1]))
    cb.set_ticks([1, 2, 3, 4, 5, 6])
    cb.set_ticklabels(["2", "4", "8", "16", "32", "64"])
    cb.set_label("screened period (periodic cells)", fontsize=7)
    cb.outline.set_visible(False)
    bx = fig.add_subplot(grid[0, 2])
    x = [r["log10_output"] for r in rows]
    for run in runs:
        bx.plot([r["log10_output"] for r in run], [r["theta_over_2pi"] for r in run], "-", color=c.INK, linewidth=1.0)
        bx.plot([r["log10_output"] for r in run], [r["theta_over_2pi"] for r in run], ".", color=c.INK, markersize=3)
    shown = sorted({r["fraction"] for r in found})
    for f in shown:
        bx.axhline(float(f), color="#e1e0d9", linewidth=0.7, zorder=0)
        bx.text(max(x) + 0.004, float(f), f"{f.numerator}/{f.denominator} (period {2 * f.denominator})",
                va="center", fontsize=6.5, color=c.SECONDARY)
    for r in found:
        bx.plot(r["log10_output"], float(r["fraction"]), "o", color=c.FLIP, markersize=4.5,
                markeredgecolor=c.SURFACE, markeredgewidth=0.7)
    bx.set_xlabel("log10 output-layer learning rate")
    bx.set_ylabel("crossing angle theta / 2 pi (turns per two steps)")
    bx.set_title("B  Birth rotation number")
    bx.set_xlim(min(x) - 0.01, max(x) + 0.13)
    traced = json.loads(CURVE.read_text())["rows"]
    theta = [r["theta_over_2pi"] for r in rows]
    derived = {
        "slices_traced": len(traced), "slices_neimark_sacker": len(rows),
        "slices_other": {t: sum(r.get("type") == t for r in traced) for t in sorted({r.get("type") for r in traced})},
        "theta_over_2pi_min": min(theta), "theta_over_2pi_max": max(theta),
        "lowest_boundary_log10_hidden": min(r["log10_hidden_boundary"] for r in rows),
        "roots": [{"fraction": f"{r['fraction'].numerator}/{r['fraction'].denominator}",
                   "log10_output": r["log10_output"], "log10_hidden": r["log10_hidden"]} for r in found],
    }
    names = ", ".join(sorted({d["fraction"] for d in derived["roots"]}, key=Fraction))
    alt = (f"Left: period map of the S2-28011 zoom region with periodic cells shaded by period. A black curve marks "
           f"where the period-2 orbit loses stability; on all {len(rows)} traced slices the crossing multipliers are "
           f"a complex pair (Neimark-Sacker). The curve is U-shaped with its lowest point at log10 hidden rate "
           f"{derived['lowest_boundary_log10_hidden']:.3f}. Orange points mark where the crossing angle equals a "
           f"rational m/q (predicted tongue roots): {names}. Right: the crossing angle theta/2pi along the curve "
           f"ranges from {min(theta):.4f} to {max(theta):.4f} turns per two steps.")
    c.save(fig, "f2_ns_tongue_roots", [MAP, CURVE], __file__, "exploratory", alt, derived,
           f"{len(rows)} traced Neimark-Sacker points; roots for denominators up to {MAX_DEN}; slices whose start "
           "was not period 2 break the curve.")


if __name__ == "__main__":
    main()
