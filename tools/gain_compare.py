"""Compare the gain split (tools/gain_split.py) across models at matched next-token entropy (runs G1, G2).

For each model: the network factor log g_net and the readout factor log g_read against the mean entropy of the
next-token distribution, chaotic cells (positive exponent, no period) marked. Also printed: per model and entropy bin
(below 1, 1-3, 3-6, 6-10, above 10 nats), the number of cells and the median of each factor.

Usage: python tools/gain_compare.py OUT.png LABEL=GAIN_DIR [LABEL=GAIN_DIR ...]   (GAIN_DIR explore-relative)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
BINS = ((0, 1), (1, 3), (3, 6), (6, 10), (10, 99))


def load(folder):
    import explore_llm_attractors as ela

    cells = json.loads((ela.DATA / "explore" / folder / "gain-split.json").read_text())["cells"]
    H = np.array([c["entropy_mean"] for c in cells])
    net = np.array([c["log_g_net"] for c in cells])
    read = np.array([c["log_g_read"] for c in cells])
    lam = np.array([c["lyapunov"] for c in cells])
    chaotic = (lam > 1e-3) & (np.array([c["period"] for c in cells]) == 0)
    return H, net, read, lam, chaotic


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out, specs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharex=True)
    colours = plt.get_cmap("tab10")
    summary = {}
    for i, (label, folder) in enumerate(specs):
        H, net, read, lam, chaotic = load(folder)
        rows = {}
        for lo, hi in BINS:
            m = (H >= lo) & (H < hi)
            if m.any():
                rows[f"{lo}-{hi}"] = {"cells": int(m.sum()), "median_log_g_net": float(np.median(net[m])),
                                      "median_log_g_read": float(np.median(read[m])), "chaotic": int(chaotic[m].sum())}
        summary[label] = rows
        for ax, y in zip(axes, (net, read)):
            ax.scatter(H[~chaotic], y[~chaotic], color=colours(i), s=14, label=label)
            ax.scatter(H[chaotic], y[chaotic], facecolors="none", edgecolors=colours(i), s=60, linewidths=1.2)
    for ax, name in zip(axes, ("network factor  log g_net", "readout factor  log g_read")):
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_xlabel("mean entropy of the next-token distribution (nats)")
        ax.set_ylabel(name + " (per step)")
    axes[0].legend(fontsize=7, frameon=False)
    axes[1].set_title("open circles: chaotic cells", fontsize=8)
    fig.suptitle("Unsteered soft-token maps: where the loop's gain comes from (exponent = network + readout factor)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=250)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
