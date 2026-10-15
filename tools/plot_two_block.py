"""Figure fig-two-block: steering two directions at two blocks (run TB). Largest Lyapunov exponent over (alpha1,
alpha2) at two temperatures for gratitude after block 4 with hostility after block 16 (tb/grat4-host16) and the
reverse (tb/host4-grat16); Qwen2.5-0.5B-Instruct, float64. The axes alpha2 = 0 and alpha1 = 0 are single-block
steering; the rest of the plane combines the two.

Usage: python tools/plot_two_block.py [OUT.png]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data")) / "explore/llm-soft-loop/pod64/tb"
RUNS = [("grat4-host16", "gratitude @ block 4", "hostility @ block 16"),
        ("host4-grat16", "hostility @ block 4", "gratitude @ block 16")]


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "paper/llm/figures/fig-two-block.png")
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 8.6), dpi=200)
    summary = {}
    for r, (run, lab1, lab2) in enumerate(RUNS):
        for c, lt in enumerate((-0.1, 0.1)):
            z = np.load(DATA / run / f"tb-t{lt:+.3f}.npz")
            a1, a2, L, chaotic = z["alpha1"], z["alpha2"], z["lyapunov"], z["chaotic"]
            ax = axes[r, c]
            im = ax.imshow(np.clip(L, -0.4, 0.4), origin="lower", aspect="auto", cmap="RdBu_r", vmin=-0.4, vmax=0.4,
                           extent=[a1[0], a1[-1], a2[0], a2[-1]], interpolation="nearest")
            ax.axhline(0, color="k", lw=0.5, ls=":")
            ax.axvline(0, color="k", lw=0.5, ls=":")
            i0, j0 = int(np.argmin(abs(a2))), int(np.argmin(abs(a1)))
            share = float(chaotic.mean())
            summary[f"{run} {lt}"] = {"chaotic_share": share, "axis_alpha2_0": float(chaotic[i0].mean()),
                                      "axis_alpha1_0": float(chaotic[:, j0].mean())}
            ax.set_title(f"log10 T = {lt}: {100 * share:.0f} % chaotic", fontsize=9)
            ax.set_xlabel(r"$\alpha_1$, " + lab1, fontsize=8)
            ax.set_ylabel(r"$\alpha_2$, " + lab2, fontsize=8)
    fig.colorbar(im, ax=axes, label="largest Lyapunov exponent (clipped)", fraction=0.025, pad=0.02)
    fig.savefig(out, bbox_inches="tight")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
