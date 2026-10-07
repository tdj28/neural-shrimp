"""Analyze an exploratory rotation-number map against the Neimark-Sacker curve.

1. Continuity test: at points just beyond the traced Neimark-Sacker curve of
   the period-2 orbit (offset along the hidden-rate axis), the measured
   rotation number should equal the birth angle arg(mu)/2pi of the crossing
   multiplier at that point of the curve.
2. Locking test: periodic cells with even period 2q should have rotation m/q
   with denominator exactly q (Arnold-tongue locking); odd periods cannot be
   two-step circle locking and are reported separately.
3. Figure: rotation number (continuous) for non-chaotic cells, chaotic cells
   in gray, the Neimark-Sacker curve and its predicted rational roots.
"""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def best_fraction(value, max_den):
    f = Fraction(value).limit_denominator(max_den)
    return f, abs(float(f) - value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--map", required=True)
    parser.add_argument("--ns", required=True)
    parser.add_argument("--offsets", type=float, nargs="+", default=[0.004, 0.008, 0.016])
    parser.add_argument("--chaos", type=float, default=5e-3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    m = np.load(args.map)
    lh, lo = m["log10_hidden"], m["log10_output"]
    rho, status, period, ftle = m["rho"], m["status"], m["period"], m["ftle"]
    rows = [r for r in json.load(open(args.ns))["rows"] if r.get("type") == "neimark_sacker"]
    report = {"continuity": [], "locking": {}}
    for off in args.offsets:
        diffs = []
        for r in rows:
            h = r["log10_hidden_boundary"] + off
            if not (lo[0] <= r["log10_output"] <= lo[-1] and lh[0] <= h <= lh[-1]):
                continue
            i, j = int(np.argmin(np.abs(lh - h))), int(np.argmin(np.abs(lo - r["log10_output"])))
            if np.isfinite(rho[i, j]) and status[i, j] != 0 and abs(ftle[i, j]) < args.chaos:
                diffs.append((r["log10_output"], r["theta_over_2pi"], float(rho[i, j]), int(status[i, j]),
                              int(period[i, j])))
        d = np.array([x[2] - x[1] for x in diffs]) if diffs else np.array([])
        report["continuity"].append({"offset_log10_hidden": off, "points": len(diffs),
                                     "median_abs_diff": float(np.median(np.abs(d))) if len(d) else None,
                                     "max_abs_diff": float(np.max(np.abs(d))) if len(d) else None,
                                     "pairs": diffs})
    periodic = (status == 2) & (period >= 3)
    consistent, total, odd, by_q = 0, 0, 0, {}
    for i, j in np.argwhere(periodic):
        p = int(period[i, j])
        if p % 2:
            odd += 1
            continue
        q = p // 2
        if not np.isfinite(rho[i, j]):
            continue
        total += 1
        f, err = best_fraction(float(rho[i, j]), 64)
        ok = f.denominator == q and err < 1e-6
        consistent += ok
        key = f"{f.numerator}/{f.denominator}" if ok else "inconsistent"
        by_q.setdefault(p, {}).setdefault(key, 0)
        by_q[p][key] += 1
    report["locking"] = {"even_period_cells": total, "rotation_m_over_q": consistent, "odd_period_cells": odd,
                         "by_period": {str(k): v for k, v in sorted(by_q.items())}}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(report, indent=1))
    for c in report["continuity"]:
        print("offset", c["offset_log10_hidden"], "points", c["points"], "median |rho - birth|",
              c["median_abs_diff"], "max", c["max_abs_diff"])
    print("even-period periodic cells:", total, "with rotation m/q of denominator period/2:", consistent,
          "| odd-period cells:", odd)
    fig, ax = plt.subplots(figsize=(9.5, 7.5), constrained_layout=True)
    extent = [lo[0], lo[-1], lh[0], lh[-1]]
    chaotic = (status == 3) | ((status != 0) & (ftle >= args.chaos))
    base = np.where(status == 0, 0, np.where(chaotic, 1, 2))
    from matplotlib.colors import ListedColormap
    ax.imshow(base, origin="lower", aspect="auto", extent=extent, interpolation="nearest",
              cmap=ListedColormap(["#ffffff", "#c9c8c2", "#f0efec"]), vmin=-0.5, vmax=2.5)
    show = np.ma.masked_where(chaotic | (status == 0) | ~np.isfinite(rho) | ((status == 2) & (period <= 2)), rho)
    image = ax.imshow(show, origin="lower", aspect="auto", extent=extent, interpolation="nearest",
                      cmap="viridis", vmin=0.15, vmax=0.26)
    bar = fig.colorbar(image, ax=ax, shrink=0.8)
    bar.set_label("rotation number of the two-step map (turns per two steps)", fontsize=9)
    ns = sorted(rows, key=lambda r: r["log10_output"])
    ax.plot([r["log10_output"] for r in ns], [r["log10_hidden_boundary"] for r in ns], "-", color="#0b0b0b",
            linewidth=1.2, label="Neimark-Sacker curve of the period-2 orbit")
    for a, b in zip(ns[:-1], ns[1:]):
        lo_r, hi_r = sorted((a["theta_over_2pi"], b["theta_over_2pi"]))
        for q in range(4, 14):
            for p in range(1, q):
                f = Fraction(p, q)
                if f.denominator != q or not (lo_r < float(f) <= hi_r):
                    continue
                t = (float(f) - a["theta_over_2pi"]) / (b["theta_over_2pi"] - a["theta_over_2pi"])
                x = a["log10_output"] + t * (b["log10_output"] - a["log10_output"])
                y = a["log10_hidden_boundary"] + t * (b["log10_hidden_boundary"] - a["log10_hidden_boundary"])
                ax.plot(x, y, "o", color="#eb6834", markersize=4.5, markeredgecolor="#ffffff")
                ax.annotate(f"{p}/{q}", (x, y), textcoords="offset points", xytext=(3, -9), fontsize=7)
    ax.set_xlabel("log10 output-layer learning rate", fontsize=9)
    ax.set_ylabel("log10 hidden-layer learning rate", fontsize=9)
    ax.set_title("Rotation number beyond the edge of stability (gray: chaotic; light: period 2 / other; "
                 "white: divergent); exploratory", loc="left", fontsize=9)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.savefig(out / "rotation-map.png", dpi=170)


if __name__ == "__main__":
    main()
