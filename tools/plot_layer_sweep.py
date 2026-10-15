"""Figure fig-layer-sweep: the soft-token map of Qwen2.5-0.5B-Instruct steered along the philosophy direction at
different blocks (run LS): largest Lyapunov exponent on the same 128 x 128 float64 grid (alpha -0.12 to 0.12, log10 T
-1.0 to 0.3) for steering after blocks 4, 8, 12, 16 and 20 of 24. Block 12 is the same-architecture map
pairmap-qwen0.5b/map128-philosophy (vectors built with the same recipe); the others are scan64/ls-l{4,8,16,20}-{lo,hi}
(two halves of the grid each). Vectors: mean concept-minus-neutral last-token state at the block, scaled to the mean
residual norm there, so alpha is a fraction of the residual norm at every block.

Usage: python tools/plot_layer_sweep.py [OUT.png]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data")) / "explore/llm-soft-loop/pod64"


def load(block):
    if block == 12:
        z = np.load(DATA / "pairmap-qwen0.5b/map128-philosophy/scan.npz", allow_pickle=True)
        return z["log10_temperature"], z["alpha"], z["lyapunov"].astype(float), z["period"].astype(float)
    parts = [np.load(DATA / f"scan64/ls-l{block}-{h}/scan.npz", allow_pickle=True) for h in ("lo", "hi")]
    lt = np.concatenate([p["log10_temperature"] for p in parts])
    return (lt, parts[0]["alpha"], np.concatenate([p["lyapunov"] for p in parts]).astype(float),
            np.concatenate([p["period"] for p in parts]).astype(float))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "paper/llm/figures/fig-layer-sweep.png")
    blocks = (4, 8, 12, 16, 20)
    fig, axes = plt.subplots(1, 5, figsize=(14, 3.4), dpi=200, sharey=True)
    summary = {}
    for ax, b in zip(axes, blocks):
        lt, al, L, P = load(b)
        chaotic = (L > 1e-3) & (P == 0)
        share = float(chaotic.mean())
        rows = np.nonzero(chaotic.any(axis=1))[0]
        summary[b] = {"chaotic_share": share, "cells": int(L.size),
                      "chaotic_log10T": [float(lt[rows.min()]), float(lt[rows.max()])] if len(rows) else None,
                      "share_abs_alpha_below_0.02": float(chaotic[:, np.abs(al) < 0.02].mean())}
        im = ax.imshow(np.clip(L, -0.4, 0.4), origin="lower", aspect="auto", cmap="RdBu_r", vmin=-0.4, vmax=0.4,
                       extent=[al[0], al[-1], lt[0], lt[-1]], interpolation="nearest")
        ax.set_title(f"block {b}: {100 * share:.1f} % chaotic", fontsize=9)
        ax.set_xlabel(r"$\alpha$ (philosophy)")
    axes[0].set_ylabel(r"$\log_{10}T$")
    fig.colorbar(im, ax=axes, label="largest Lyapunov exponent (clipped)", fraction=0.02, pad=0.01)
    fig.savefig(out, bbox_inches="tight")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
