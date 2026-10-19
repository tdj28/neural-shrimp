"""Figure fig-repro: incidental numerical differences against the dynamical regime (run REPRO1, tools/repro_noise.py).

The 448 labelled cells of docs/h2-horizon-cells.json (Qwen2.5-0.5B, philosophy at block 12) were run twice in
lockstep from the same start, differing only in the batch layout (n rows against 2n rows) or the precision.
(a) float64, batch layout: the relative distance between the two runs for every chaotic and periodic cell (thin
    lines), with the median and interquartile range per label.
(b) the same for float32 and bfloat16 batch layouts (chaotic and periodic cells).
(c) float64 chaotic cells: the step at which the distance first exceeds 1e-2 against ln(1e-2 / d_1) / lambda, with
    d_1 the first nonzero distance and lambda the map exponent of docs/h2-horizon-cells.json.

Usage: python tools/plot_repro.py [--out PNG]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data")) / \
    "explore/llm-soft-loop/pod64/repro/h2cells"


def band(ax, d, color, label):
    t = np.arange(1, d.shape[0] + 1)
    q = np.quantile(np.maximum(d, 1e-17), [0.25, 0.5, 0.75], axis=1)
    ax.fill_between(t, q[0], q[2], color=color, alpha=0.25, lw=0)
    ax.plot(t, q[1], color=color, lw=1.4, label=label)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "paper/llm/figures/fig-repro.png"))
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r = json.loads((RUN / "repro.json").read_text())
    z = np.load(RUN / "repro.npz")
    lab = np.array(r["labels"])
    lam = np.array(json.loads((ROOT / "docs/h2-horizon-cells.json").read_text())["map_exponent"], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=170)
    ax = axes[0]
    d = z["batch64_dist"]
    t = np.arange(1, d.shape[0] + 1)
    # Okabe-Ito blue (periodic) and orange (chaotic): told apart under colour-vision deficiencies and in grey print
    for L, col in (("periodic", "#0072B2"), ("chaotic", "#E69F00")):     # every cell's run, under the summaries
        for k in np.nonzero(lab == L)[0]:
            ax.plot(t, np.maximum(d[:, k], 1e-17), color=col, alpha=0.07, lw=0.35, rasterized=True)
    for L, col in (("chaotic", "#8C5A00"), ("periodic", "navy")):
        band(ax, d[:, lab == L], col, f"{L} ({int((lab == L).sum())} cells): median and quartiles")
    o = lab == "other"                     # neither label: its median only (its quartiles span most of the range)
    ax.plot(t, np.median(np.maximum(d[:, o], 1e-17), axis=1), color="tab:gray", lw=1.0,
            label=f"other ({int(o.sum())} cells): median")
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 3)
    ax.set_xlabel("step")
    ax.set_ylabel("relative distance between the two runs")
    ax.set_title("(a) float64: the same cells in a batch of n or 2n rows (one line per cell)", fontsize=9)
    ax.legend(fontsize=7, loc="center right")
    ax = axes[1]
    for comp, ls in (("batch32", "-"), ("bf16", "--")):
        d = z[f"{comp}_dist"]
        for L, col in (("chaotic", "#E69F00"), ("periodic", "#0072B2")):
            t = np.arange(1, d.shape[0] + 1)
            ax.plot(t, np.median(np.maximum(d[:, lab == L], 1e-17), axis=1), ls, color=col,
                    label=f"{'float32' if comp == 'batch32' else 'bfloat16'}, {L}")
    ax.set_yscale("log")
    ax.set_ylim(1e-8, 3)
    ax.set_xscale("log")
    ax.set_xlabel("step")
    ax.set_title("(b) float32 and bfloat16 batch layouts (medians)", fontsize=9)
    ax.legend(fontsize=7)
    ax = axes[2]
    rows = r["comparisons"]["batch64"]
    obs, pred = [], []
    for i in np.nonzero(lab == "chaotic")[0]:
        x = rows[i]
        if x["first_step_dist_above_1e-2"] is not None and lam[i] > 0:
            obs.append(x["first_step_dist_above_1e-2"])
            pred.append(np.log(1e-2 / max(x["min_nonzero_dist"], 1e-17)) / lam[i])
    obs, pred = np.array(obs), np.array(pred)
    ax.loglog(pred, obs, "o", ms=3, color="#E69F00", alpha=0.6)
    lim = [min(pred.min(), obs.min()) * 0.8, max(pred.max(), obs.max()) * 1.2]
    ax.plot(lim, lim, "k:", lw=1)
    rho = np.corrcoef(np.log(obs), np.log(pred))[0, 1]
    ax.set_xlabel(r"predicted step $\ln(10^{-2}/d_1)/\lambda$")
    ax.set_ylabel("observed step where the distance passes $10^{-2}$")
    ax.set_title("(c) float64 chaotic cells: the exponent predicts when (n=%d,\nmedian ratio %.2f, log-log r=%.2f)"
                 % (len(obs), np.median(obs / pred), rho), fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out)
    print(json.dumps({"n": int(len(obs)), "median_ratio": float(np.median(obs / pred)), "loglog_r": float(rho)}))


if __name__ == "__main__":
    main()
