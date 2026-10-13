"""Figure: bifurcation diagrams through the hostile hub's period-7 body (run S9a, job set hcascade). For each float64
scan line with attractor projections (scan --projections), top: the recorded attractor states projected on the unit
steering vector against the line's parameter; bottom: the largest Lyapunov exponent, with the cells' periods.

Usage: python tools/plot_hcascade.py OUT.png [LINE ...]
  LINE: explore-relative scan folder (default: the S9a temperature and alpha lines)
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

LINES = ["llm-soft-loop/pod64/hcascade/tline-a0.0655", "llm-soft-loop/pod64/hcascade/aline-t0.1830"]


def load(folder):
    import explore_llm_attractors as ela

    z = np.load(ela.DATA / "explore" / folder / "scan.npz")
    lt, al = z["log10_temperature"], z["alpha"]
    if len(al) == 1:
        x, label, fixed = lt, "log10 temperature", f"alpha = {al[0]:.4f}"
        sl = (slice(None), 0)
    else:
        x, label, fixed = al, "steering strength alpha", f"log10 T = {lt[0]:.4f}"
        sl = (0, slice(None))
    return dict(x=x, label=label, fixed=fixed, lyap=z["lyapunov"][sl], period=z["period"][sl],
                proj=z["projection"][sl][..., 0])


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = sys.argv[1]
    lines = sys.argv[2:] or LINES
    fig, axes = plt.subplots(2, len(lines), figsize=(6.2 * len(lines), 6.4), sharex="col",
                             gridspec_kw={"height_ratios": [2.2, 1]}, squeeze=False)
    for k, folder in enumerate(lines):
        d = load(folder)
        x, proj, lyap, per = d["x"], d["proj"], d["lyap"], d["period"]
        chaotic = lyap > 1e-3
        xs = np.repeat(x, proj.shape[1]).reshape(proj.shape)
        ax = axes[0, k]
        ax.plot(xs[~chaotic].ravel(), proj[~chaotic].ravel(), ",", color="k", alpha=0.6)
        ax.plot(xs[chaotic].ravel(), proj[chaotic].ravel(), ",", color="C3", alpha=0.35)
        ax.set_ylabel("attractor on the steering direction")
        ax.set_title(f"{d['fixed']} (black: non-chaotic cells, red: positive exponent)", fontsize=9)
        ax = axes[1, k]
        ax.axhline(0, color="0.5", lw=0.8)
        ax.plot(x, np.clip(lyap, -0.4, None), color="k", lw=0.7)
        for q, colour in ((7, "C0"), (14, "C1"), (28, "C2"), (21, "C4"), (35, "C5"), (42, "C6")):
            sel = per == q
            if sel.any():
                ax.plot(x[sel], np.full(sel.sum(), 0.33), "|", color=colour, ms=6, label=f"period {q}")
        ax.set_ylabel("largest Lyapunov exponent")
        ax.set_xlabel(d["label"])
        ax.legend(fontsize=7, ncol=3, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=220)


if __name__ == "__main__":
    main()
