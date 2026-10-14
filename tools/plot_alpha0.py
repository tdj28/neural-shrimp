"""Figure: the unsteered soft-token map (alpha = 0) along temperature, one panel per model (run C1): largest
Lyapunov exponent from the long-horizon float64 sweep, cells without a period up to the maximum tested in red,
periodic cells coloured by period.

Usage: python tools/plot_alpha0.py OUT.png LABEL=SCAN_DIR [LABEL=SCAN_DIR ...]
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import explore_llm_attractors as ela

    out, specs = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
    fig, axes = plt.subplots(len(specs), 1, figsize=(8.5, 2.3 * len(specs)), sharex=True, squeeze=False)
    cmap = plt.get_cmap("viridis")
    for ax, (label, folder) in zip(axes[:, 0], specs):
        z = np.load(ela.DATA / "explore" / folder / "scan.npz")
        if z["alpha"].size != 1 or float(z["alpha"][0]) != 0.0:
            raise SystemExit(f"{folder}: not an alpha = 0 sweep")
        lt, L, P = z["log10_temperature"], z["lyapunov"][:, 0], z["period"][:, 0]
        chaotic = (L > 1e-3) & (P <= 0)
        ax.axhline(0, color="0.6", lw=0.6)
        per = P > 0
        sc = ax.scatter(lt[per], np.clip(L[per], -1.5, 1), c=np.log2(P[per]), cmap=cmap, s=5, vmin=0, vmax=6)
        ax.scatter(lt[chaotic], L[chaotic], color="crimson", s=6, label="no period, exponent > 0")
        other = (P <= 0) & ~chaotic
        if other.any():
            ax.scatter(lt[other], np.clip(L[other], -1.5, 1), color="k", marker="x", s=8, label="no period, exponent <= 0")
        ax.set_ylim(-1.55, 1.0)
        ax.set_ylabel("largest exponent\n(clipped at -1.5)")
        ax.set_title(f"{label}: {int(chaotic.sum())} of {lt.size} temperatures without a period and with a positive "
                     f"exponent", fontsize=9)
        ax.legend(loc="lower left", fontsize=7, frameon=False)
    axes[-1, 0].set_xlabel("log10 temperature (steering strength 0)")
    cb = fig.colorbar(sc, ax=axes[:, 0], shrink=0.8, pad=0.01)
    cb.set_label("log2 period (periodic cells)")
    fig.savefig(out, dpi=250, bbox_inches="tight")


if __name__ == "__main__":
    main()
