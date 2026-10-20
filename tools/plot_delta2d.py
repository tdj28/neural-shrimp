"""Figure of run DELTA2D: the leading multiplier and the determinant of the tracked leading two-dimensional block over
the top of the period-6 body (paper/llm/figures/fig-delta2d.png; run card and amendment in docs/research-log.md).

(a) The real leading multiplier mu_1 (diverging colors, zero white; cells whose leading multiplier is a complex pair
    in gray), its zero contour, and its critical points (saddle: x).
(b) delta, the determinant of the tracked block (diverging colors), its zero contour, its critical points, and cells
    where the tracked pair mixes with the third direction (overlap below 0.9; dots).
The outline marks the DITCHFINE patch.

Usage: python tools/plot_delta2d.py --analysis docs/delta2d/analysis.json [--out paper/llm/figures/fig-delta2d.png]
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FINE = (0.0731674, 0.0731926, -0.152048, -0.151992)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--analysis", default=str(ROOT / "docs" / "delta2d" / "analysis.json"))
    ap.add_argument("--out", default=str(ROOT / "paper" / "llm" / "figures" / "fig-delta2d.png"))
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    rec = json.loads(Path(args.analysis).read_text())
    z = np.load(Path(args.analysis).with_suffix(".npz"))
    al, lt, mu, mi, ok, delta, ov = z["alpha"], z["log10_temperature"], z["mu"], z["mu_imag"], z["ok"], z["delta"], z["overlap"]
    real = ok & (mi == 0)
    da, dt = al[1] - al[0], lt[1] - lt[0]
    ext = [al[0] - da / 2, al[-1] + da / 2, lt[0] - dt / 2, lt[-1] + dt / 2]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, field, key, title in ((axes[0], np.where(real, mu, np.nan), "mu1", "(a)"),
                                  (axes[1], delta, "delta", "(b)")):
        lim = np.nanmax(np.abs(field))
        im = ax.imshow(field, origin="lower", aspect="auto", extent=ext, cmap="RdBu_r",
                       norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim), interpolation="nearest")
        if key == "mu1":
            cpx = np.where(ok & (mi != 0), 1.0, np.nan)
            ax.imshow(cpx, origin="lower", aspect="auto", extent=ext, cmap="Greys", vmin=0, vmax=2.5, interpolation="nearest")
        else:
            bad = np.argwhere(np.isfinite(ov) & (ov < 0.9))
            if len(bad):
                ax.plot(al[bad[:, 1]], lt[bad[:, 0]], ".", color="black", ms=1.5)
        A, L = np.meshgrid(al, lt)
        if np.any(np.isfinite(field)):
            ax.contour(A, L, np.nan_to_num(field, nan=np.nanmax(field) if key == "delta" else 0.0), levels=[0.0],
                       colors="black", linewidths=0.8)
        for c in rec["critical_points"][key]:
            mk = {"saddle": "x", "minimum": "v", "maximum": "^"}[c["type"]]
            ax.plot(c["alpha"], c["log10T"], mk, color="black", ms=6, mew=1.2)
        a0, a1, t0, t1 = FINE
        ax.plot([a0, a1, a1, a0, a0], [t0, t0, t1, t1, t0], "-", color="0.25", lw=0.7)
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xlabel(r"steering strength $\alpha$", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02)
        cb.ax.tick_params(labelsize=6)
    axes[0].set_ylabel(r"$\log_{10}T$", fontsize=7.5)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", pad_inches=0.03)
    print(args.out)


if __name__ == "__main__":
    main()
