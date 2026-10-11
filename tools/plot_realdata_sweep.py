"""Plot an exploratory adiabatic sweep along a Neimark-Sacker branch (real data).

Top: two-step rotation number against log10 learning rate, with recurrence-
periodic rates (locked orbits) and FTLE-positive rates distinguished, and the
multiplier-angle prediction at the bifurcation. Bottom: loss traces at a few
rates (first 2,048 recorded updates), showing the slow modulation of the loss
that the invariant circle produces.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", required=True, help="sweep directory")
    parser.add_argument("--ns", required=True, help="continuation JSON with the Neimark-Sacker point")
    parser.add_argument("--traces", type=int, default=3)
    args = parser.parse_args()
    sweep = Path(args.sweep)
    rows = json.loads((sweep / "sweep.json").read_text())["rows"]
    saved = np.load(sweep / "sweep-losses.npz")
    ns = json.loads(Path(args.ns).read_text())
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.3, 1.0]})
    fig.patch.set_facecolor(SURFACE)
    ax = axes[0]
    finite = [r for r in rows if r["finite"]]
    groups = {"quasi-periodic candidate (FTLE <= 0, monotone)": [], "locked (recurrent period)": [],
              "irregular (FTLE > 0.002 or non-monotone)": []}
    for r in finite:
        if r["period"] > 0:
            groups["locked (recurrent period)"].append(r)
        elif r["ftle"] > 0.002 or r["rho2_forward"] < 0.99:
            groups["irregular (FTLE > 0.002 or non-monotone)"].append(r)
        else:
            groups["quasi-periodic candidate (FTLE <= 0, monotone)"].append(r)
    for (name, members), color in zip(groups.items(), (AQUA, BLUE, ORANGE)):
        if members:
            ax.plot([r["log10_eta"] for r in members], [r["rho2"] for r in members], "o", markersize=4,
                    color=color, markeredgewidth=0, label=f"{name}: {len(members)}")
    ax.plot([ns["log10_eta_boundary"]], [ns["theta_over_2pi"]], "*", markersize=12, color=INK,
            label=f"arg(mu)/2pi at the Neimark-Sacker point = {ns['theta_over_2pi']:.5f}")
    ax.axvline(ns["log10_eta_boundary"], color=MUTED, linestyle="--", linewidth=0.8)
    ax.set_ylabel("rotation number of the two-step map (turns per two updates)", fontsize=9, color=MUTED)
    ax.set_xlabel("log10 global learning rate", fontsize=9, color=MUTED)
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.set_title(f"{sweep.parent.name}: adiabatic sweep along the Neimark-Sacker branch (exploratory)",
                 loc="left", fontsize=10, color=INK)
    ax = axes[1]
    log_eta, losses = saved["log10_eta"], saved["losses"]
    picks = np.unique(np.linspace(0, len(log_eta) - 1, args.traces).round().astype(int))
    for i, color in zip(picks, (BLUE, AQUA, ORANGE, INK)):
        ax.plot(np.arange(2048), losses[i, :2048], linewidth=0.8, color=color,
                label=f"log10 eta {log_eta[i]:+.4f}")
    ax.set_xlabel("update (after settling)", fontsize=9, color=MUTED)
    ax.set_ylabel("training loss", fontsize=9, color=MUTED)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    for a in axes:
        a.tick_params(labelsize=8, colors=MUTED)
        a.set_facecolor(SURFACE)
    fig.savefig(sweep / "sweep.png", dpi=160, facecolor=SURFACE)
    for r in finite:
        print(f"{r['log10_eta']:+.5f} rho2 {r['rho2']:.6f} fwd {r['rho2_forward']:.3f} share {r['rho2_share']:.3f} "
              f"period {r['period']:5d} ftle {r['ftle']:+.5f} loss {r['loss_min']:.4f}-{r['loss_max']:.4f}")


if __name__ == "__main__":
    main()
