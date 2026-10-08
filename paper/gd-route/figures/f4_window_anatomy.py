"""Figure 4: anatomy of the period-18 window S2-28011-zoom:587.

Evidence: exploratory (A2 survey). Panel A: 512 x 512 local period map from the
shared initial parameters (CUDA float64 backend, EXP-028 screening options),
with the window's numerically validated orbit continued along 64 rays from its center
(tools/explore_a2_survey.py rays; bisection to 1e-9) and each ray's loss of
stability typed by the crossing multiplier. Panel B: the dominant multiplier
at the last attracting point of every ray, on the unit circle.
"""

import json

import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import common as c

KEY = "S2-28011-zoom:587"
MAP = c.EXPLORE / "s2-28011-w587-512/map.npz"
RAYS = c.EXPLORE / "a2-survey/rays.json"
SURVEY = c.EXPLORE / "a2-survey/survey.json"


def main():
    c.style()
    window = json.loads(RAYS.read_text())[KEY]
    survey = next(r for r in json.loads(SURVEY.read_text()) if r["key"] == KEY)
    saved = np.load(MAP)
    lh, lo = saved["log10_hidden"], saved["log10_output"]
    status, period = saved["status"], saved["period"]
    p = window["period"]
    layer = np.zeros(status.shape, dtype=int)
    layer[(status == 2) & (period != p)] = 1
    layer[(status == 2) & (period == 2 * p)] = 2
    layer[(status == 2) & (period == p)] = 3
    fig = c.plt.figure(figsize=(7.1, 3.9), layout="constrained")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.45, 1])
    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(layer, origin="lower", extent=[lo[0], lo[-1], lh[0], lh[-1]], aspect="auto", interpolation="nearest",
              cmap=ListedColormap([c.SURFACE, "#e1e0d9", "#a3a29b", c.SECONDARY]), vmin=-0.5, vmax=3.5,
              rasterized=True)
    center, scale = np.array(window["center"]), np.array(window["scale"])
    counts, ends = {}, []
    for ray in window["rays"]:
        counts[ray["type"]] = counts.get(ray["type"], 0) + 1
        d = np.array(ray["direction"]) * scale
        s = ray["s_boundary"] if ray["s_boundary"] is not None else 1.0
        end = center + s * d
        ends.append(end)
        ax.plot([center[1], end[1]], [center[0], end[0]], color=c.MUTED, linewidth=0.35)
        if ray["type"] in c.TYPE_COLORS:
            ax.plot(end[1], end[0], c.TYPE_MARKERS[ray["type"]], markersize=4.2, color=c.TYPE_COLORS[ray["type"]],
                    markeredgecolor=c.SURFACE, markeredgewidth=0.5)
    ax.plot(center[1], center[0], "+", color=c.INK, markersize=8)
    ends = np.array(ends)
    span = ends.max(0) - ends.min(0)
    low, high = ends.min(0) - 0.45 * span, ends.max(0) + 0.45 * span
    ax.set_xlim(max(lo[0], low[1]), min(lo[-1], high[1]))
    ax.set_ylim(max(lh[0], low[0]), min(lh[-1], high[0]))
    ax.set_xlabel("log10 output-layer learning rate")
    ax.set_ylabel("log10 hidden-layer learning rate")
    ax.set_title(f"A  Window {KEY} (period {p}): typed stability boundaries")
    handles = [Line2D([], [], marker=c.TYPE_MARKERS[k], linestyle="", color=c.TYPE_COLORS[k],
                      label=f"{c.TYPE_LABELS[k]}: {counts.get(k, 0)} rays") for k in c.TYPE_COLORS]
    handles += [Patch(color=c.SECONDARY, label=f"period {p}"), Patch(color="#a3a29b", label=f"period {2 * p}"),
                Patch(color="#e1e0d9", label="other periodic")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=6.5)
    metrics = survey["metrics"]
    a, compared = metrics["edge_agreement"]
    f2, ft = metrics["flip_to_2p"]
    ax.text(0.02, 0.98, f"ray and map edges within 2 cells: {a}/{compared} rays\n"
            f"flip rays opening onto period {2 * p}: {f2}/{ft}\n(local map at 256 x 256)",
            transform=ax.transAxes, va="top", fontsize=6.3, color=c.INK,
            bbox={"facecolor": c.SURFACE, "edgecolor": "none", "alpha": 0.85, "pad": 1.5})
    bx = fig.add_subplot(grid[0, 1])
    t = np.linspace(0, 2 * np.pi, 400)
    bx.plot(np.cos(t), np.sin(t), color=c.MUTED, linewidth=0.7)
    bx.axhline(0, color="#e1e0d9", linewidth=0.5, zorder=0)
    bx.axvline(0, color="#e1e0d9", linewidth=0.5, zorder=0)
    for ray in window["rays"]:
        if ray["type"] not in c.TYPE_COLORS or not ray.get("dominant"):
            continue
        re, im = ray["dominant"]
        for sign in ((1, -1) if ray["type"] == "neimark_sacker" else (1,)):
            bx.plot(re, sign * im, c.TYPE_MARKERS[ray["type"]], markersize=4.5, color=c.TYPE_COLORS[ray["type"]],
                    markeredgecolor=c.SURFACE, markeredgewidth=0.5)
    bx.set_aspect("equal")
    bx.set_xlim(-1.25, 1.25)
    bx.set_ylim(-1.25, 1.25)
    bx.set_xlabel("Re of dominant multiplier")
    bx.set_ylabel("Im of dominant multiplier")
    bx.set_title("B  Crossing multipliers (unit circle)")
    derived = {"window": KEY, "period": p, "rays": len(window["rays"]), "ray_types": counts,
               "edge_agreement": [a, compared], "flip_to_2p": [f2, ft],
               "boundary_points": [{"direction": r["direction"], "type": r["type"], "s_boundary": r["s_boundary"]}
                                   for r in window["rays"]]}
    alt = (f"Local period map around the numerically validated period-{p} window {KEY}. The window's region (dark) has a "
           f"fold-bounded body: {counts.get('fold', 0)} of {len(window['rays'])} rays end where a multiplier "
           f"crosses +1. Along its upper edge {counts.get('flip', 0)} rays end at a flip (-1), opening onto "
           f"period-{2 * p} bands (medium gray), and {counts.get('neimark_sacker', 0)} at a complex pair. Ray and "
           f"map edges agree within two cells on {a} of {compared} rays. Right: the crossing multipliers on the "
           f"unit circle cluster at +1 and -1.")
    c.save(fig, "f4_window_anatomy", [MAP, RAYS, SURVEY], __file__, "exploratory", alt, derived,
           "Rays from tools/explore_a2_survey.py; edge metrics from its analyze stage on the 256 x 256 local map. "
           "Window selected as the clearest shrimp-like example (one of two passing the draft criteria); not "
           "representative of all windows (see Figure 5).")


if __name__ == "__main__":
    main()
