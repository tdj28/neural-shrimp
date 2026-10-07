"""Figure 1: the published binary convergence score versus period-resolved dynamics.

Evidence: prospectively frozen EXP-028 maps (S2-28011 coarse and zoom) and the
frozen analysis summary (docs/EXP-028-summary.json). Every panel shows the same
cells: (A, D) the Sohl-Dickstein convergence score after 1,000 updates, (B, E)
the dynamical class after 20,000 updates, (C, F) the screened period of periodic
cells. Panel G: what the score-convergent cells of every coarse map actually are.
"""

import json

import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch, Rectangle

import common as c

MAPS = [("S2-28011", c.RAW028 / "stage1/S2-28011.npz"), ("S2-28011-zoom", c.RAW028 / "stage2/S2-28011-zoom.npz")]
COARSE = ["S1-28001", "S1-28002", "S1R-28001", "S1R-28002", "S2-28011", "S2-28012"]
HORIZON = "1000"
ORDER = [1, 2, 3, 5, 4, 0]  # fixed, periodic, chaotic, drift, unresolved, divergent
NAMES = {0: "divergent", 1: "fixed_point_candidate", 2: "periodic_candidate", 3: "chaotic_candidate",
         4: "unresolved", 5: "drift_candidate"}


def pct(fraction):
    """Percent with two decimals near 0 or 100 %, so rounding never reads as all or none."""
    value = 100 * fraction
    return f"{value:.2f}%" if value > 99.5 or value < 0.5 else f"{value:.1f}%"


def panel_binary(ax, saved, title):
    extent = [saved["log10_output"][0], saved["log10_output"][-1], saved["log10_hidden"][0], saved["log10_hidden"][-1]]
    conv = saved[f"sd_convergent_{HORIZON}"].astype(int)
    ax.imshow(conv, origin="lower", extent=extent, aspect="auto", interpolation="nearest",
              cmap=ListedColormap([c.CLASS_COLORS[0], c.NEUTRAL]), vmin=-0.5, vmax=1.5)
    ax.set_title(title)
    return extent


def panel_class(ax, saved, extent, title):
    status = saved["status"]
    index = np.zeros_like(status)
    for i, code in enumerate(ORDER):
        index[status == code] = i
    ax.imshow(index, origin="lower", extent=extent, aspect="auto", interpolation="nearest",
              cmap=ListedColormap([c.CLASS_COLORS[k] for k in ORDER]), vmin=-0.5, vmax=len(ORDER) - 0.5)
    ax.set_title(title)


def panel_period(fig, ax, saved, extent, title, bar=True):
    status, period = saved["status"], saved["period"]
    periodic = (status == 2) & (period >= 2)
    levels = np.log2(np.where(periodic, period, 1).astype(float))
    cmap = ListedColormap(c.PERIOD_RAMP).with_extremes(bad=c.NEUTRAL)
    image = ax.imshow(np.ma.masked_where(~periodic, levels), origin="lower", extent=extent, aspect="auto",
                      interpolation="nearest", cmap=cmap,
                      norm=BoundaryNorm(np.linspace(1, 6, len(c.PERIOD_RAMP) + 1), cmap.N))
    ax.set_title(title)
    if not bar:
        return image
    if bar:
        cb = fig.colorbar(image, ax=ax, shrink=0.9, pad=0.02)
        cb.set_ticks([1, 2, 3, 4, 5, 6])
        cb.set_ticklabels(["2", "4", "8", "16", "32", "64"])
        cb.set_label("screened period", fontsize=7)
        cb.outline.set_visible(False)


def main():
    c.style()
    summary = json.loads((c.ROOT / "docs/EXP-028-summary.json").read_text())
    fig = c.plt.figure(figsize=(7.1, 7.0), layout="constrained")
    grid = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.78], width_ratios=[1, 1, 1, 0.06])
    letters = iter("ABCDEF")
    period_axes = []
    derived = {"score_convergent_share": {}, "crosstab_horizon_updates": int(HORIZON), "coarse_maps": {}}
    zoom = np.load(MAPS[1][1])
    zoom_box = (zoom["log10_output"][0], zoom["log10_hidden"][0],
                zoom["log10_output"][-1] - zoom["log10_output"][0], zoom["log10_hidden"][-1] - zoom["log10_hidden"][0])
    for row, (name, path) in enumerate(MAPS):
        saved = np.load(path)
        label = "coarse" if row == 0 else "zoom"
        a = fig.add_subplot(grid[row, 0])
        extent = panel_binary(a, saved, f"{next(letters)}  Published score")
        share = float(saved[f"sd_convergent_{HORIZON}"].mean())
        derived["score_convergent_share"][name] = share
        a.text(0.04, 0.04, f"{pct(share)} scored\nconvergent", transform=a.transAxes, fontsize=6.5,
               color=c.INK, va="bottom")
        b = fig.add_subplot(grid[row, 1])
        panel_class(b, saved, extent, f"{next(letters)}  Dynamical class")
        d = fig.add_subplot(grid[row, 2])
        image = panel_period(fig, d, saved, extent, f"{next(letters)}  Period of periodic cells", bar=False)
        period_axes.append(d)
        for ax in (a, b, d):
            ax.set_xlabel("log10 output-layer rate")
            if row == 0:
                ax.add_patch(Rectangle(zoom_box[:2], zoom_box[2], zoom_box[3], fill=False, edgecolor=c.INK,
                                       linewidth=0.8))
        a.set_ylabel(f"{name} ({label})\nlog10 hidden-layer rate")
    cb = fig.colorbar(image, cax=fig.add_subplot(grid[0:2, 3]))
    cb.set_ticks([1, 2, 3, 4, 5, 6])
    cb.set_ticklabels(["2", "4", "8", "16", "32", "64"])
    cb.set_label("screened period", fontsize=7)
    cb.outline.set_visible(False)
    handles = [Patch(facecolor=c.NEUTRAL, edgecolor=c.MUTED, label="published score: convergent"),
               Patch(color=c.CLASS_COLORS[0], label="published score: divergent (and class divergent)")]
    ax = fig.add_subplot(grid[2, :3])
    left = np.zeros(len(COARSE))
    for code in ORDER:
        values = []
        for m in COARSE:
            conv = summary["maps"][m]["sd_crosstab"][HORIZON]["sd_convergent"]
            total = sum(conv.values())
            values.append(conv[NAMES[code]] / total)
        values = np.array(values)
        ax.barh(np.arange(len(COARSE)), values, left=left, height=0.62, color=c.CLASS_COLORS[code],
                edgecolor=c.SURFACE, linewidth=1.0,
                label="late divergence" if code == 0 else c.CLASS_LABELS[code])
        left += values
    for i, m in enumerate(COARSE):
        conv = summary["maps"][m]["sd_crosstab"][HORIZON]["sd_convergent"]
        total = sum(conv.values())
        share = 1 - conv["fixed_point_candidate"] / total
        derived["coarse_maps"][m] = {"score_convergent": total, "by_class": conv, "share_not_fixed": share}
        ax.text(1.005, i, f"{total:,} cells; {pct(share)} not fixed points",
                va="center", fontsize=6.5, color=c.SECONDARY)
    ax.set_yticks(np.arange(len(COARSE)), COARSE)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of cells the published score calls convergent (1,000 updates)")
    ax.set_title("G  What the score-convergent cells of every coarse map are after 20,000 updates")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=6, frameon=False, fontsize=6.5)
    fig.legend(handles=handles[:2], loc="outside upper center", ncol=2, frameon=False, fontsize=6.5)
    shares = [v["share_not_fixed"] for v in derived["coarse_maps"].values()]
    alt = (f"Maps of the per-layer learning-rate plane of the 13-parameter network S2-28011. The published binary "
           f"convergence score marks {pct(derived['score_convergent_share']['S2-28011'])} of the coarse plane and "
           f"{pct(derived['score_convergent_share']['S2-28011-zoom'])} of the zoom region convergent, while the same "
           f"cells after 20,000 updates are fixed points, periodic orbits of many periods, chaos, drift or unresolved. "
           f"Stacked bars for six coarse maps of three networks: {pct(min(shares))} to {pct(max(shares))} of the "
           f"cells the score calls convergent are not fixed points.")
    c.save(fig, "f1_binary_vs_period", [MAPS[0][1], MAPS[1][1], c.ROOT / "docs/EXP-028-summary.json"], __file__,
           "prospectively frozen (EXP-028), independently audited", alt, derived,
           "S1 rows are the published ridge-free setting; S1R and S2 add ridge 1e-3.")


if __name__ == "__main__":
    main()
