"""Plot loss-only AR estimates of the two-step multiplier against Jacobian values (exploratory).

One row per precursor run: modulus |mu| (left) and angle / 2 pi (right) along
the slice. Jacobian values exist for period-2 cells; AR estimates from the
noisy even-step training loss are shown for every cell (beyond the
Neimark-Sacker point they estimate the rotation number).
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", "/data2/neural-shrimp-data"))
JACOBIAN, LOW, HIGH = "#0b0b0b", "#2a78d6", "#eb6834"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    fig, axes = plt.subplots(len(args.runs), 2, figsize=(11, 3.2 * len(args.runs)), constrained_layout=True,
                             squeeze=False)
    fig.patch.set_facecolor("#fcfcfb")
    for k, run in enumerate(args.runs):
        rows = json.loads((DATA / "explore" / "consequence" / run / "precursor.json").read_text())["rows"]
        noises = [key for key in rows[0] if key.startswith("noise_")]
        x = np.array([r["log10_hidden"] for r in rows])
        p2 = np.array([r["period2"] and "complex_modulus" in r for r in rows])
        for j, (field, jac, title) in enumerate((("ar_modulus", "complex_modulus", "|mu| (two-step multiplier)"),
                                                 ("ar_frequency", "predicted_frequency",
                                                  "angle / 2 pi (turns per two steps)"))):
            ax = axes[k, j]
            jv = np.array([r.get(jac, np.nan) for r in rows])
            ax.plot(x[p2], jv[p2], "-", color=JACOBIAN, linewidth=2, label="Jacobian (period-2 orbit)")
            for noise, color in zip(noises, (LOW, HIGH)):
                est = np.array([r.get(noise, {}).get(field, np.nan) for r in rows])
                if field == "ar_frequency":
                    est = np.where(est > 0.3, np.nan, est)  # drop second-harmonic picks for display
                ax.plot(x, est, "o", color=color, markersize=4, markeredgecolor="#fcfcfb", markeredgewidth=0.6,
                        label=f"AR(6) from loss, noise {noise.split('_')[1]}")
            boundary = x[p2].max() if p2.any() else None
            if boundary is not None:
                ax.axvline(boundary, color="#a3a29b", linewidth=0.8, linestyle="--")
            ax.set_title(f"{args.labels[k] if args.labels else run}: {title}", loc="left", fontsize=9,
                         color="#0b0b0b")
            ax.tick_params(labelsize=8, colors="#52514e")
            ax.grid(True, color="#f0efec", linewidth=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_xlabel("log10 hidden-layer learning rate", fontsize=8, color="#52514e")
        axes[k, 0].legend(fontsize=7, frameon=False, loc="lower right")
    fig.savefig(args.out, dpi=150, facecolor="#fcfcfb")


if __name__ == "__main__":
    main()
