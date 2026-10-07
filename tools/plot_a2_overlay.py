"""Overlay typed ray boundaries on an exploratory local period map (A2).

Background: cells with the window's period (dark gray), other periodic cells
(light gray), everything else (surface). Markers: ray boundary points colored
by type with the validated categorical slots. Rays are drawn as thin lines.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

TYPE_COLORS = {"fold": "#2a78d6", "flip": "#eb6834", "neimark_sacker": "#1baf7a"}
SURFACE, INK, MUTED = "#fcfcfb", "#0b0b0b", "#898781"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True, help="explore map.npz")
    parser.add_argument("--rays", required=True, help="rays.json from explore_a2_rays.py")
    parser.add_argument("--target", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    saved = np.load(args.map)
    target = next(t for t in json.loads(Path(args.rays).read_text()) if t["target"] == args.target)
    period, status = saved["period"], saved["status"]
    lh, lo = saved["log10_hidden"], saved["log10_output"]
    layer = np.zeros(period.shape, dtype=int)
    layer[(status == 2) & (period != target["period"])] = 1
    layer[(status == 2) & (period == target["period"])] = 2
    layer[(status == 2) & (period == 2 * target["period"])] = 3
    fig, ax = plt.subplots(figsize=(7.5, 6.5), constrained_layout=True)
    fig.patch.set_facecolor(SURFACE)
    ax.imshow(layer, origin="lower", extent=[lo[0], lo[-1], lh[0], lh[-1]], aspect="auto",
              interpolation="nearest", cmap=ListedColormap([SURFACE, "#e1e0d9", "#52514e", "#a3a29b"]),
              vmin=-0.5, vmax=3.5)
    center = np.array(target["center_log10"])
    scale = np.array(target["scale_log10"])
    for ray in target["rays"]:
        direction = np.array(ray["direction"]) * scale
        s = ray["s_boundary"] if ray["s_boundary"] is not None else 1.0
        end = center + s * direction
        ax.plot([center[1], end[1]], [center[0], end[0]], color=MUTED, linewidth=0.4)
        if ray["type"] in TYPE_COLORS:
            ax.plot(end[1], end[0], "o", markersize=4, color=TYPE_COLORS[ray["type"]],
                    markeredgecolor=SURFACE, markeredgewidth=0.6)
    ax.plot(center[1], center[0], "+", color=INK, markersize=9)
    handles = [Line2D([], [], marker="o", linestyle="", color=c, label=k.replace("_", "-"))
               for k, c in TYPE_COLORS.items()]
    handles += [Line2D([], [], marker="s", linestyle="", color="#52514e", label=f"period {target['period']}"),
                Line2D([], [], marker="s", linestyle="", color="#a3a29b", label=f"period {2 * target['period']}"),
                Line2D([], [], marker="s", linestyle="", color="#e1e0d9", label="other periodic")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=8,
              frameon=False)
    ax.set_xlabel("log10 output-layer learning rate", fontsize=9, color=MUTED)
    ax.set_ylabel("log10 hidden-layer learning rate", fontsize=9, color=MUTED)
    ax.tick_params(labelsize=8, colors=MUTED)
    ax.set_title(f"{args.target}: typed stability boundaries along {len(target['rays'])} rays "
                 "(exploratory)", loc="left", fontsize=10, color=INK)
    fig.savefig(args.out, dpi=170, facecolor=SURFACE)


if __name__ == "__main__":
    main()
