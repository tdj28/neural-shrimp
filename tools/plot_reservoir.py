"""Figure fig-reservoir and statistics of run RC1 (tools/reservoir_capacity.py): memory capacity of the soft-token loop
of Qwen2.5-0.5B-Instruct (philosophy, block 12, float64) across its regimes.

(a) Along the cascade line log10 T = -0.14798: memory capacity against alpha for each input size, the twin exponent of
    the input-free map on a second axis, the float64 flips of the doubling sequence (Table tab:cascade) and the
    accumulation point alpha_inf (dotted).
(b) The 448 labeled cells: capacity against the map exponent, coloured by label, with means in bins of the exponent
    and bootstrap intervals over cells.
Statistics: capacity in fixed exponent bins (deep periodic lambda <= -0.3, near zero |lambda| <= 0.05, chaotic
lambda >= 0.1) with bootstrap intervals, the bin with the largest mean, and the Spearman correlation of capacity with
|lambda|.

Usage: python tools/plot_reservoir.py [--out paper/llm/figures/fig-reservoir.png] [--stats docs/rc1-stats.json]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data")) / "explore/llm-soft-loop/pod64/rc"
FLIPS = [0.0654807769499, 0.0659055406325, 0.0660549041785, 0.0661071412523, 0.0661209563174, 0.0661241213732]
ALPHA_INF = 0.0661250
EDGES = [-1.6, -0.6, -0.3, -0.15, -0.05, 0.05, 0.15, 0.3, 0.6]


def boot_mean(v, rng, n=4000):
    v = np.asarray(v, dtype=float)
    if len(v) < 2:
        return [float("nan"), float("nan")]
    b = [rng.choice(v, len(v)).mean() for _ in range(n)]
    return [float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))]


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "paper/llm/figures/fig-reservoir.png"))
    ap.add_argument("--stats", default=str(ROOT / "docs/rc1-stats.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    feig = json.loads((DATA / "feig" / "capacity.json").read_text())["cells"]
    h2 = json.loads((DATA / "h2" / "capacity.json").read_text())["cells"]
    eps_keys = sorted(feig[0]["capacity"], key=float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), dpi=160)
    al = np.array([c["alpha"] for c in feig])
    for e, col in zip(eps_keys, ("tab:blue", "tab:green", "tab:purple")):
        ax1.plot(al, [c["capacity"][e]["mc"] for c in feig], ".-", ms=3, lw=0.8, color=col, label=f"MC, input {e}")
    axl = ax1.twinx()
    lam = np.array([min(c["lyapunov"]) if "lyapunov" in c else np.nan for c in feig])
    axl.plot(al, lam, "-", color="0.55", lw=0.8)
    axl.axhline(0, color="0.55", lw=0.5, ls=":")
    axl.set_ylabel("input-free twin exponent (gray)", fontsize=8)
    for f in FLIPS:
        ax1.axvline(f, color="tab:red", lw=0.5, alpha=0.6)
    ax1.axvline(ALPHA_INF, color="k", lw=0.8, ls=":")
    ax1.set_xlabel(r"$\alpha$ (log$_{10}T=-0.148$)", fontsize=8)
    ax1.set_ylabel("memory capacity (sum of held-out $R^2$)", fontsize=8)
    ax1.legend(fontsize=7, loc="upper left")
    ax1.set_title("(a) along the doubling cascade: flips (red), onset of chaos (dotted)", fontsize=8)
    stats = {"feig": {e: {"max_mc": float(max(c["capacity"][e]["mc"] for c in feig)),
                          "alpha_at_max": float(al[int(np.argmax([c["capacity"][e]["mc"] for c in feig]))])}
                      for e in eps_keys}, "h2": {}}
    ex = np.array([c["map_exponent"] for c in h2], dtype=float)
    labels = [c["label"] for c in h2]
    colours = {"chaotic": "tab:red", "periodic": "tab:blue", "other": "0.6"}
    e = eps_keys[0]
    mc = np.array([c["capacity"][e]["mc"] for c in h2])
    for lab, col in colours.items():
        m = np.array([l == lab for l in labels])
        ax2.plot(ex[m], mc[m], ".", ms=3, color=col, label=lab)
    centres, means, lows, highs = [], [], [], []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        m = (ex >= lo) & (ex < hi)
        if m.sum() >= 3:
            ci = boot_mean(mc[m], rng)
            centres.append(0.5 * (lo + hi)), means.append(mc[m].mean()), lows.append(ci[0]), highs.append(ci[1])
    ax2.errorbar(centres, means, yerr=[np.array(means) - lows, np.array(highs) - means], fmt="s-", color="k", ms=4,
                 lw=1, capsize=2, label="bin mean, 95 % CI")
    ax2.axvline(0, color="k", lw=0.5, ls=":")
    ax2.set_xlabel("map exponent $\\lambda$", fontsize=8)
    ax2.set_title(f"(b) 448 labeled cells, input {e}", fontsize=8)
    ax2.legend(fontsize=7)
    for e in eps_keys:
        mc = np.array([c["capacity"][e]["mc"] for c in h2])
        rec = {}
        for name, m in (("deep_periodic", ex <= -0.3), ("near_zero", np.abs(ex) <= 0.05), ("chaotic", ex >= 0.1)):
            rec[name] = {"n": int(m.sum()), "mean_mc": float(mc[m].mean()), "ci95": boot_mean(mc[m], rng)}
        rec["spearman_mc_vs_abs_lambda"] = spearman(mc, np.abs(ex))
        rec["bins"] = [{"lambda": [lo, hi], "n": int(((ex >= lo) & (ex < hi)).sum()),
                        "mean_mc": float(mc[(ex >= lo) & (ex < hi)].mean()) if ((ex >= lo) & (ex < hi)).any() else None}
                       for lo, hi in zip(EDGES[:-1], EDGES[1:])]
        stats["h2"][e] = rec
    fig.tight_layout()
    fig.savefig(args.out)
    Path(args.stats).write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
