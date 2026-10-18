"""Run FS figure: the fixed-shape control and the precision dose (tools/repro_noise.py modes repeat, fixedperm,
batch64, dose; docs/fs-design.md).

(a) Largest relative distance between the paired runs per cell: identical shapes (repeat, permuted rows) against a
changed shape (batch64), by the map's label. (b) Dose: first step at which an injected displacement d exceeds 1e-2,
against ln(1e-2/d) / lambda, for the chaotic cells; medians and quartiles per dose. (c) Share of periodic cells whose
displaced run passes 1e-2, per dose.

Usage: python tools/plot_fixed_shape.py --run llm-soft-loop/pod64/repro/fs --out paper/llm/figures/fig-fixed-shape.png
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    r = json.loads((DATA / "explore" / args.run / "repro.json").read_text())
    labels = np.array(r["labels"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=170)
    ax = axes[0]
    cols = {"chaotic": "tab:red", "periodic": "tab:blue", "other": "0.5"}
    summary = {}
    for k, comp in enumerate(("repeat", "fixedperm", "batch64")):
        md = np.array([x["max_dist"] for x in r["comparisons"][comp]])
        summary[comp] = {"max_dist_max": float(md.max()), "cells_exactly_zero": int(np.sum(md == 0))}
        for lab, c in cols.items():
            m = labels == lab
            y = np.where(md[m] > 0, md[m], 1e-18)
            ax.scatter(np.full(m.sum(), k) + np.random.default_rng(k).uniform(-0.25, 0.25, m.sum()), y, s=4, color=c,
                       label=lab if k == 2 else None)
    ax.set_yscale("log")
    ax.set_ylim(3e-19, 30)
    ax.axhline(1e-2, color="k", lw=0.6, ls=":")
    ax.set_xticks([0, 1, 2], ["same shape,\nsame rows", "same shape,\nrows permuted", "shape changed"])
    ax.set_ylabel("largest relative distance between the runs")
    ax.text(0, 1e-17, "exactly 0\nin all 448", ha="center", fontsize=8)
    ax.text(1, 1e-17, "exactly 0\nin all 448", ha="center", fontsize=8)
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("(a) only a change of matrix shape separates the runs", fontsize=9)
    ax = axes[1]
    rows = r["dose_passages"]
    doses = sorted({x["dose"] for x in rows})
    med, q1, q3, pred_med, ratio = [], [], [], [], []
    for d0 in doses:
        ch = [x for x in rows if x["dose"] == d0 and x["label"] == "chaotic" and x["predicted"] is not None
              and x["predicted"] <= 1500 and x["first_step_above_eps"] is not None]
        t = np.array([x["first_step_above_eps"] for x in ch])
        tp = np.array([x["predicted"] for x in ch])
        med.append(np.median(t)); q1.append(np.percentile(t, 25)); q3.append(np.percentile(t, 75))
        pred_med.append(np.median(tp)); ratio.append(float(np.median(t / tp)))
    x = np.log(1e-2 / np.array(doses))
    ax.errorbar(x, med, yerr=[np.array(med) - np.array(q1), np.array(q3) - np.array(med)], fmt="o", color="tab:red",
                capsize=3, label="measured (median, quartiles)")
    ax.plot(x, pred_med, "k--", label=r"predicted $\ln(10^{-2}/d)/\lambda$ (median)")
    for xi, top, rr in zip(x, q3, ratio):
        ax.annotate(f"{rr:.2f}", (xi, top), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=7)
    ax.set_xlabel(r"$\ln(10^{-2}/d)$; injected relative displacement $d$ from $10^{-3}$ (left) to $10^{-15}$")
    ax.set_ylabel("steps until the distance exceeds $10^{-2}$")
    ax.legend(fontsize=7, loc="upper left")
    ax.set_title("(b) chaotic cells: growth on the Lyapunov schedule (labels: median ratio)", fontsize=9)
    ax = axes[2]
    share = []
    for d0 in doses:
        pe = [x for x in rows if x["dose"] == d0 and x["label"] == "periodic"]
        share.append(np.mean([x["first_step_above_eps"] is not None for x in pe]))
    ax.semilogx(doses, share, "o-", color="tab:blue")
    ax.axhline(0.1, color="k", lw=0.6, ls=":")
    ax.set_xlabel("injected relative displacement $d$")
    ax.set_ylabel("share of periodic cells whose runs separate")
    ax.set_title("(c) periodic cells: separation only by crossing a basin boundary", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out)
    summary.update({"doses": doses, "median_ratio": ratio, "periodic_share": [float(v) for v in share]})
    text = json.dumps(summary, indent=1)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")


if __name__ == "__main__":
    main()
