"""Figure: Huginn-0125 GSM8K chain-of-thought accuracy against the relaxation strength lam (32 recurrence steps per
token, 8-shot, greedy), for the development problems (train 0-119, run H1b) and the frozen test (test 0-199, design
9805ed32), with the spectral thresholds of the development GSM8K prompts. Left: accuracy with Wilson 95 % intervals.
Right: paired difference from native lam = 1 with the paired bootstrap 95 % interval of tools/huginn_arc_analysis.py.

Usage: python tools/plot_huginn_gsm8k.py OUT.png
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import huginn_arc_analysis as ha  # noqa: E402

HUGINN = Path("/data2/neural-shrimp-data/explore/llm-soft-loop/pod64/huginn")
SETS = {"development (train 0-119)": [HUGINN / "gsm8k-dev-cot8-r32"],
        "frozen test (test 0-199)": [HUGINN / "partB-0", HUGINN / "partB-96", HUGINN / "partB-100"]}
LAM_C = (1.074, 1.126)          # flip / Neimark-Sacker thresholds of five development GSM8K prompts (Section 6)


def wilson(k, n, z=1.959964):
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def series(folders):
    rows = [json.loads(line) for f in folders for line in open(f / "rows.jsonl")]
    arms = sorted({r["arm"] for r in rows}, key=float)
    by = {a: {r["index"]: r for r in rows if r["arm"] == a} for a in arms}
    common = sorted(set.intersection(*[set(v) for v in by.values()]))
    base = np.array([bool(by["1.0"][i]["correct"]) for i in common])
    out = []
    for a in arms:
        x = np.array([bool(by[a][i]["correct"]) for i in common])
        mean, boots = ha.paired_bootstrap(base, x)
        out.append(dict(lam=float(a), acc=x.mean(), wilson=wilson(x.sum(), len(x)), diff=mean,
                        ci=tuple(np.percentile(boots, [2.5, 97.5])), n=len(common)))
    return out


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for ax in axes:
        ax.axvspan(*LAM_C, color="0.85", lw=0, label="spectral thresholds $\\lambda_c$ (development prompts)")
        ax.axvline(1.0, color="0.6", lw=0.8, ls=":")
    for (name, folders), colour, marker, shift in zip(SETS.items(), ("C0", "C3"), ("o", "s"), (-0.003, 0.003)):
        s = series(folders)
        print(name, [(r["lam"], round(r["acc"], 3), round(r["diff"], 3), tuple(round(v, 3) for v in r["ci"]))
                     for r in s])
        lam = np.array([r["lam"] for r in s]) + shift
        acc = np.array([r["acc"] for r in s])
        lo, hi = np.array([r["wilson"] for r in s]).T
        axes[0].errorbar(lam, acc, yerr=[acc - lo, hi - acc], color=colour, marker=marker, capsize=3, lw=1.2,
                         label=f"{name}, n = {s[0]['n']}")
        d = np.array([r["diff"] for r in s])
        clo, chi = np.array([r["ci"] for r in s]).T
        axes[1].errorbar(lam, d, yerr=[d - clo, chi - d], color=colour, marker=marker, capsize=3, lw=1.2)
    axes[0].set_ylabel("GSM8K accuracy (flexible extraction)")
    axes[1].set_ylabel("paired difference from native $\\lambda=1$")
    axes[1].axhline(0, color="k", lw=0.8)
    for ax in axes:
        ax.set_xlabel("relaxation $\\lambda$ in $h \\leftarrow (1-\\lambda)h + \\lambda F(h)$")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=200)


if __name__ == "__main__":
    main()
