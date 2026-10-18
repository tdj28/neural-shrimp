"""Run NB analysis (human-relayed proposal, 18 Oct 2026, test B): do the norm bounds of the factorized derivative
capture where chaos lives?

Along each cell's own trajectory, tools/norm_bounds.py measured the chain
  lambda <= (1/k) <log ||DF^k||> <= <log ||DF||> <= <log(||C_p||/T)> + <log ||J_h||>,
with lambda the tangent exponent of the same orbit, k = 2..5 blocks and the one-step and sum bounds. The norms are
power-iteration estimates (warm-started along the orbit, --iters per step), which can only fall short of the true
norms: every estimated bound is a lower estimate of the true bound, so a chain that holds for the estimates holds for
the truth, while a negative estimate does not by itself exclude chaos. Gates, from every row before anything is
summarized: rows with a non-finite value are excluded; the rows whose worst one-step power iteration still changed by
more than 1e-3 are scored separately ("converged" set); a bound below its own cell's tangent exponent (more than 1e-6)
is a chain violation and is reported, never dropped. Scored per bound and map (stripes subsample, cross-512 subsample),
for all rows and for the converged rows:
- violations: cells labelled chaotic by the map whose bound is negative (the bound would exclude chaos the map saw);
- recall: share of chaotic cells with a positive bound (must be 1 for a valid bound);
- precision: share of positive-bound cells that are chaotic (tightness);
- certified share: share of the map's periodic cells whose bound is negative (where the bound alone excludes chaos).

Usage: python tools/plot_norm_bounds.py --runs llm-soft-loop/pod64/nb/cells2 llm-soft-loop/pod64/nb/cross-a \
         llm-soft-loop/pod64/nb/cross-b [--json docs/nb/nb-summary.json] [--out paper/llm/figures/fig-norm-bounds.png]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
BOUNDS = ["sum", "one_step", "k2", "k3", "k4", "k5"]
LABELS = {"sum": "readout + network", "one_step": "one step", "k2": "k = 2", "k3": "k = 3", "k4": "k = 4",
          "k5": "k = 5"}


def load(runs):
    rows, seen = [], set()
    for run in runs:
        path = DATA / "explore" / run / "nb.json"
        if not path.exists():
            continue
        for r in json.loads(path.read_text())["cells"]:
            key = (r["source"], round(r["log10_temperature"], 9), round(r["alpha"], 9))
            if key in seen:                  # a cell measured twice (overlapping runs): keep the first
                continue
            seen.add(key)
            rows.append({**r, "run": run})
    return rows


def gate(r):
    vals = [r[k] for k in BOUNDS] + [r["tangent_exponent"]]
    if not all(np.isfinite(v) for v in vals):
        return "non-finite"
    return None


def converged(r):
    return max(r.get("power_rel_change_J", 0), r.get("power_rel_change_DF", 0)) <= 1e-3


def score(rows, only_converged=False):
    out = {}
    for src in sorted({r["source"] for r in rows}):
        R = [r for r in rows if r["source"] == src and gate(r) is None and (converged(r) or not only_converged)]
        lab = np.array([r["map_label"] for r in R])
        lam = np.array([r["tangent_exponent"] for r in R])
        ch, pe = lab == "chaotic", lab == "periodic"
        s = {"cells": len(R), "excluded_by_gate": sum(1 for r in rows if r["source"] == src and gate(r)),
             "chaotic": int(ch.sum()), "periodic": int(pe.sum()), "other": int((~ch & ~pe).sum()),
             "tangent_positive_among_chaotic": int((lam[ch] > 0).sum()), "bounds": {}}
        for k in BOUNDS:
            b = np.array([r[k] for r in R])
            pos = b > 0
            s["bounds"][k] = {
                "chain_violations": int((b < lam - 1e-6).sum()),
                "chaotic_with_negative_bound": int((ch & ~pos).sum()),
                "chaotic_with_negative_bound_and_positive_tangent": int((ch & ~pos & (lam > 0)).sum()),
                "recall": float((ch & pos).sum() / max(ch.sum(), 1)),
                "precision": float((ch & pos).sum() / max(pos.sum(), 1)),
                "periodic_certified": float((pe & ~pos).sum() / max(pe.sum(), 1)),
                "median_chaotic": float(np.median(b[ch])) if ch.any() else None,
                "median_periodic": float(np.median(b[pe])) if pe.any() else None}
        out[src] = s
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--json")
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = load(args.runs)
    summary = {"runs": args.runs, "rows": len(rows), "by_map": score(rows),
               "by_map_converged_rows": score(rows, only_converged=True)}
    text = json.dumps(summary, indent=1)
    print(text)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text + "\n")
    if not args.out:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    srcs = sorted({r["source"] for r in rows})
    fig, axes = plt.subplots(len(srcs), 3, figsize=(15, 4.6 * len(srcs)), dpi=160)
    axes = np.atleast_2d(axes)
    for row, src in enumerate(srcs):
        R = [r for r in rows if r["source"] == src and gate(r) is None]
        lt = np.array([r["log10_temperature"] for r in R])
        al = np.array([r["alpha"] for r in R])
        lab = np.array([r["map_label"] for r in R])
        ch = lab == "chaotic"
        for col, k in enumerate(["one_step", "k5"]):
            ax = axes[row, col]
            b = np.array([r[k] for r in R])
            vmax = float(np.percentile(np.abs(b), 98))
            sc = ax.scatter(al, lt, c=b, cmap="RdBu_r", vmin=-vmax, vmax=vmax, s=7, marker="s", linewidths=0)
            ax.scatter(al[ch], lt[ch], s=3.5, c="k", marker=".", linewidths=0, alpha=0.75)
            ax.locator_params(axis="x", nbins=5)
            fig.colorbar(sc, ax=ax, label=f"bound ({LABELS[k]}), per step")
            ax.set_xlabel(r"$\alpha$")
            ax.set_ylabel(r"$\log_{10}T$")
            ax.set_title(f"{src}: {LABELS[k]} bound (red positive; black dots: chaotic cells)", fontsize=9)
        ax = axes[row, 2]
        s = summary["by_map"][src]["bounds"]
        xs = np.arange(len(BOUNDS))
        ax.plot(xs, [s[k]["median_chaotic"] for k in BOUNDS], "o-", color="tab:red", label="median, chaotic cells")
        ax.plot(xs, [s[k]["median_periodic"] for k in BOUNDS], "o-", color="tab:blue", label="median, periodic cells")
        lam = np.array([r["tangent_exponent"] for r in R])
        ax.axhline(float(np.median(lam[ch])), color="tab:red", ls=":", lw=1, label="median exponent, chaotic")
        ax.axhline(0, color="k", lw=0.6)
        ax2 = ax.twinx()
        ax2.plot(xs, [s[k]["precision"] for k in BOUNDS], "s--", color="0.4", label="precision")
        ax2.plot(xs, [s[k]["periodic_certified"] for k in BOUNDS], "^--", color="tab:green",
                 label="periodic cells certified")
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("share")
        ax.set_xticks(xs, [LABELS[k] for k in BOUNDS], rotation=30, fontsize=7)
        ax.set_ylabel("bound per step")
        ax.legend(fontsize=6.5, loc="upper right")
        ax2.legend(fontsize=6.5, loc="center right")
        ax.set_title(f"{src}: the ladder of bounds", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out)


if __name__ == "__main__":
    main()
