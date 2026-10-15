"""Figure fig-return-maps-body: 7-step return maps per phase (s_{n+7} against s_n, s the leading principal component
of that phase's points) for the chaotic cells of run RM (retmap/body-edge, tools/return_map.py): three cells at the
right edge of the hostile period-7 body, next to the avoided crossing of its superstable curves, and one cell of the
hostile temperature line alpha 0.0655.

Usage: python tools/plot_return_maps.py [OUT.png]
"""
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data")) / \
    "explore/llm-soft-loop/pod64/retmap/body-edge/return-map.npz"
ROWS = (0, 1, 2, 5)          # cell indices of the run: the three body-edge cells and alpha 0.0655, log10 T 0.18480


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "paper/llm/figures/fig-return-maps-body.png")
    z = np.load(RUN)
    cells, q = z["cells"], int(z["period"])
    fig, axes = plt.subplots(len(ROWS), q, figsize=(16, 9.5), dpi=90)
    for r, c in enumerate(ROWS):
        for ph in range(q):
            s = z[f"c{c}_p{ph}"]
            ax = axes[r, ph]
            ax.plot(s[:-1], s[1:], ",", color="k", alpha=0.6)
            ax.set_xticks([])
            ax.set_yticks([])
            if ph == 0:
                ax.set_ylabel("a %.5f\nT %.5f" % tuple(cells[c]), fontsize=7)
            ax.set_title(f"phase {ph} (ptp {np.ptp(s):.1e})", fontsize=7)
    fig.suptitle("7-step return maps per phase: s_{n+7} vs s_n (s: leading principal component of that phase)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    main()
