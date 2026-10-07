"""Figure 2: rotation-number maps of S2-28011 with the merged complex-pair crossing curve.

Evidence: exploratory. Two GPU float64 maps (tools/explore_gpu_rotation_map.py):
1,024 x 1,024 cells over the main region and 1,024 x 400 cells over the right
wing, each cell screened with the EXP-028 options and then followed for 6,000
two-step samples; the rotation number is measured in each cell's own principal
plane. Overlay: the merged trace of where the period-2 orbit loses stability by a
complex pair of two-step multipliers (Neimark-Sacker candidates) and the
resonance points where the crossing angle is m/q (tongue-roots.json; filled when
locked cells of that rotation lie closer to the point than in 95% of random
placements, hollow otherwise). Panel C: rotation 0.002 decades beyond the curve
against the crossing angle at the same output rate (analysis.json).
"""

import json

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

import common as c

MAPS = [("main region", c.EXPLORE / "a3-gpu-rotation/s2-28011-1024"),
        ("right wing", c.EXPLORE / "a3-gpu-rotation/s2-28011-right")]
CURVE = c.EXPLORE / "a3-ns-curve/s2-28011-merged/ns-curve.json"
NULL_LEVEL = 0.05
RHO_RANGE = (0.12, 0.26)


def draw_map(ax, saved, rows, roots):
    lh, lo = saved["log10_hidden"], saved["log10_output"]
    status, period, ftle, rho = saved["status"], saved["period"], saved["ftle"], saved["rho"]
    extent = [lo[0], lo[-1], lh[0], lh[-1]]
    chaotic = (status == 3) | ((status != 0) & (ftle >= 5e-3))
    base = np.where(status == 0, 0, np.where(chaotic, 1, 2))
    ax.imshow(base, origin="lower", aspect="auto", extent=extent, interpolation="nearest",
              cmap=ListedColormap(["#ffffff", "#c9c8c2", c.NEUTRAL]), vmin=-0.5, vmax=2.5, rasterized=True)
    show = np.ma.masked_where(chaotic | (status == 0) | ~np.isfinite(rho) | ((status == 2) & (period <= 2)), rho)
    image = ax.imshow(show, origin="lower", aspect="auto", extent=extent, interpolation="nearest", cmap="viridis",
                      vmin=RHO_RANGE[0], vmax=RHO_RANGE[1], rasterized=True)
    inside = [r for r in rows if lo[0] <= r["log10_output"] <= lo[-1]]
    ax.plot([r["log10_output"] for r in inside], [r["log10_hidden_boundary"] for r in inside], "-", color=c.INK,
            linewidth=1.0)
    for r in roots:
        if r.get("outside_map") or not (lo[0] <= r["root_log10_output"] <= lo[-1]
                                         and lh[0] <= r["root_log10_hidden"] <= lh[-1]):
            continue
        near = r["null_fraction_at_least_as_close"] <= NULL_LEVEL
        ax.plot(r["root_log10_output"], r["root_log10_hidden"], "o", markersize=5,
                markerfacecolor=c.FLIP if near else c.SURFACE, markeredgecolor=c.FLIP, markeredgewidth=1.0, zorder=5)
        ax.annotate(r["rotation"], (r["root_log10_output"], r["root_log10_hidden"]), textcoords="offset points",
                    xytext=(4, -9), fontsize=6.3, color=c.INK)
    ax.set_xlim(lo[0], lo[-1])
    ax.set_ylim(lh[0], lh[-1])
    return image


def main():
    c.style()
    rows = sorted(json.loads(CURVE.read_text())["rows"], key=lambda r: r["log10_output"])
    rows = [r for r in rows if r.get("type") == "neimark_sacker"]
    fig = c.plt.figure(figsize=(7.1, 6.4), layout="constrained")
    grid = fig.add_gridspec(2, 3, width_ratios=[1.0, 0.55, 0.05], height_ratios=[1.0, 0.62])
    derived, inputs = {"curve_points": len(rows)}, [CURVE]
    image = None
    for col, (label, folder) in enumerate(MAPS):
        saved = np.load(folder / "rotation.npz")
        roots = json.loads((folder / "tongue-roots.json").read_text())
        analysis = json.loads((folder / "analysis.json").read_text())
        inputs += [folder / "rotation.npz", folder / "tongue-roots.json", folder / "analysis.json"]
        ax = fig.add_subplot(grid[0, col])
        image = draw_map(ax, saved, rows, roots)
        ax.set_title(f"{'AB'[col]}  {label}: rotation number", fontsize=8)
        ax.set_xlabel("log10 output-layer rate")
        if col == 0:
            ax.set_ylabel("log10 hidden-layer rate")
        first = analysis["continuity"][0]
        inside = [r for r in roots if not r.get("outside_map")
                  and saved["log10_output"][0] <= r["root_log10_output"] <= saved["log10_output"][-1]
                  and saved["log10_hidden"][0] <= r["root_log10_hidden"] <= saved["log10_hidden"][-1]]
        derived[label] = {
            "cells": int(saved["rho"].size),
            "continuity_offset": first["offset_log10_hidden"], "continuity_points": first["points"],
            "continuity_median_abs_diff": first["median_abs_diff"], "continuity_max_abs_diff": first["max_abs_diff"],
            "roots_inside": len(inside),
            "roots_near": sum(r["null_fraction_at_least_as_close"] <= NULL_LEVEL for r in inside),
            "roots": [{k: r.get(k) for k in ("rotation", "root_log10_output", "root_log10_hidden",
                                             "root_distance_cells", "null_fraction_at_least_as_close")}
                      for r in inside]}
    cb = fig.colorbar(image, cax=fig.add_subplot(grid[0, 2]))
    cb.set_label("rotation number (turns per two steps)", fontsize=7)
    cb.outline.set_visible(False)
    cx = fig.add_subplot(grid[1, :2])
    markers = {"main region": ("o", c.FOLD), "right wing": ("s", c.NS)}
    for label, folder in MAPS:
        first = json.loads((folder / "analysis.json").read_text())["continuity"][0]
        pairs = np.array([p[:3] for p in first["pairs"]], dtype=float)
        marker, color = markers[label]
        cx.plot(pairs[:, 1], pairs[:, 2], marker, color=color, markersize=3.5, markeredgecolor=c.SURFACE,
                markeredgewidth=0.4, linestyle="",
                label=f"{label}: {first['points']} points, median |difference| {first['median_abs_diff']:.1e}")
    lims = [0.16, 0.245]
    cx.plot(lims, lims, color=c.MUTED, linewidth=0.8, zorder=0)
    cx.set_xlim(lims)
    cx.set_ylim(lims)
    cx.set_xlabel("crossing angle of the period-2 multipliers (turns per two steps)")
    cx.set_ylabel("rotation 0.002 decades beyond")
    cx.set_title("C  Rotation just beyond the curve against the crossing angle", fontsize=8)
    cx.legend(loc="upper left", frameon=False, fontsize=6.5)
    handles = [Line2D([], [], color=c.INK, linewidth=1.0, label="complex-pair crossing (Neimark-Sacker candidate)"),
               Line2D([], [], marker="o", linestyle="", markerfacecolor=c.FLIP, markeredgecolor=c.FLIP,
                      label="resonance point, same-rotation locked cells nearby (null fraction <= 0.05)"),
               Line2D([], [], marker="o", linestyle="", markerfacecolor=c.SURFACE, markeredgecolor=c.FLIP,
                      label="resonance point, none nearby")]
    fig.legend(handles=handles, loc="outside lower center", ncol=1, frameon=False, fontsize=6.5)
    main, right = derived["main region"], derived["right wing"]
    alt = (f"Two rotation-number maps of the 13-parameter network S2-28011 with the traced curve where its period-2 "
           f"orbit acquires complex multipliers ({len(rows)} points). Beyond the curve the rotation number varies "
           f"continuously and periodic windows appear as constant-color regions; chaotic cells are gray. At 0.002 "
           f"decades beyond the curve the rotation number matches the crossing angle with median absolute difference "
           f"{main['continuity_median_abs_diff']:.1e} ({main['continuity_points']} points, main map) and "
           f"{right['continuity_median_abs_diff']:.1e} ({right['continuity_points']} points, right wing). Of the "
           f"resonance points inside each map, {main['roots_near']} of {main['roots_inside']} and "
           f"{right['roots_near']} of {right['roots_inside']} have locked cells of the same rotation nearby.")
    c.save(fig, "f2_rotation_map", inputs, __file__, "exploratory", alt, derived,
           "Resonance-point nearness uses the receipt's random-placement null; the 0.05 level is a descriptive cut.")


if __name__ == "__main__":
    main()
