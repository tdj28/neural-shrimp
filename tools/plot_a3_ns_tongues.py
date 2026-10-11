"""Overlay the traced Neimark-Sacker curve and its rational roots on a period map (exploratory)."""

import argparse
import json
from fractions import Fraction

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402

RAMP = ["#b7d3f6", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
        "#184f95", "#104281", "#0d366b"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--ns", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-den", type=int, default=13)
    args = parser.parse_args()
    m = np.load(args.map)
    rows = [r for r in json.load(open(args.ns))["rows"] if r.get("type") == "neimark_sacker"]
    lo = np.array([r["log10_output"] for r in rows])
    lh = np.array([r["log10_hidden_boundary"] for r in rows])
    rho = np.array([r["theta_over_2pi"] for r in rows])
    status, period = m["status"], m["period"]
    periodic = (status == 2) & (period >= 2)
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    levels = np.log2(np.where(periodic, period, 1).astype(float))
    cmap = ListedColormap(RAMP).with_extremes(bad="#f0efec")
    image = ax.imshow(np.ma.masked_where(~periodic, levels), origin="lower", aspect="auto",
                      extent=[m["log10_output"][0], m["log10_output"][-1], m["log10_hidden"][0], m["log10_hidden"][-1]],
                      interpolation="nearest", cmap=cmap, norm=BoundaryNorm(np.linspace(1, 6, len(RAMP) + 1), cmap.N))
    bar = fig.colorbar(image, ax=ax, shrink=0.8)
    bar.set_ticks([1, 2, 3, 4, 5, 6])
    bar.set_ticklabels(["2", "4", "8", "16", "32", "64"])
    bar.set_label("screened period of periodic cells", fontsize=9)
    order = np.argsort(lo)
    ax.plot(lo[order], lh[order], "-", color="#0b0b0b", linewidth=1.4, label="Neimark-Sacker curve of the period-2 orbit")
    shown = set()
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        lo_r, hi_r = sorted((rho[a], rho[b]))
        for q in range(5, args.max_den + 1):
            for p in range(1, q):
                f = Fraction(p, q)
                if f.denominator != q or not (lo_r <= float(f) <= hi_r):
                    continue
                t = (float(f) - rho[a]) / (rho[b] - rho[a])
                x, y = lo[a] + t * (lo[b] - lo[a]), lh[a] + t * (lh[b] - lh[a])
                ax.plot(x, y, "o", color="#eb6834", markersize=5, markeredgecolor="#fcfcfb")
                ax.annotate(f"{p}/{q} (period {2 * q})", (x, y), textcoords="offset points", xytext=(4, -10),
                            fontsize=7, color="#52514e")
                shown.add(str(f))
    ax.set_xlim(m["log10_output"][0], m["log10_output"][-1])
    ax.set_ylim(min(m["log10_hidden"][0], lh.min() - 0.01), m["log10_hidden"][-1])
    ax.set_xlabel("log10 output-layer learning rate", fontsize=9, color="#898781")
    ax.set_ylabel("log10 hidden-layer learning rate", fontsize=9, color="#898781")
    ax.set_title("Predicted tongue roots (orange: birth rotation m/q on the Neimark-Sacker curve) "
                 "and periodic cells (exploratory)", loc="left", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.savefig(args.out, dpi=170)
    print("roots shown:", sorted(shown, key=lambda s: float(Fraction(s))))


if __name__ == "__main__":
    main()
