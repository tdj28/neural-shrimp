"""Figure of run DELTA2D: the leading multiplier and the determinant of the tracked leading two-dimensional block over
the top of the period-6 body (paper/llm/figures/fig-delta2d.png; run card and amendment in docs/research-log.md).

(a) tau, the trace of the tracked block (diverging colors, zero white), its zero contour (black), its saddle (x), and
    the measured sign changes of the leading multiplier (white dots: midpoints of neighbouring cells whose real leading
    multipliers have opposite signs; gray squares: cells whose leading multiplier is a complex pair).
(b) delta, the determinant of the tracked block, with its zero contour and the same measured sign changes; cells where
    the tracked pair mixes with a third direction (overlap below 0.9) are crossed out.
The outline marks the DITCHFINE patch (DELTA2D amendment 2: the trace panel replaces the mu_1 panel).

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
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), sharey=True, gridspec_kw={"wspace": 0.38})
    tau = z["tau"]
    pts, cpx = [], np.argwhere(ok & (mi != 0))
    nt, na = mu.shape
    for i in range(nt):
        for j in range(na):
            for k, l in ((i, j + 1), (i + 1, j)):
                if k < nt and l < na and real[i, j] and real[k, l] and np.sign(mu[i, j]) != np.sign(mu[k, l]):
                    pts.append(((al[j] + al[l]) / 2, (lt[i] + lt[k]) / 2))
    crit = {"tau": rec["critical_points"]["tau"], "delta": rec["critical_points"]["delta"]}
    for ax, field, key, title in ((axes[0], tau, "tau", "(a)"), (axes[1], delta, "delta", "(b)")):
        lim = 0.2 if key == "tau" else np.nanmax(np.abs(field))     # tau clipped so that the channel's values show
        cmap = matplotlib.colormaps["RdBu_r"].copy()
        cmap.set_bad("0.82")                                          # unsolved cells (beyond the fold): gray
        im = ax.imshow(np.ma.masked_invalid(field), origin="lower", aspect="auto", extent=ext, cmap=cmap,
                       norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim), interpolation="nearest")
        A, L = np.meshgrid(al, lt)
        ax.contour(A, L, np.where(np.isfinite(field), field, np.nan), levels=[0.0], colors="black", linewidths=0.9)
        if len(cpx):
            ax.plot(al[cpx[:, 1]], lt[cpx[:, 0]], "s", ms=2.2, mfc="0.55", mec="none")
        if pts:
            x, y = zip(*pts)
            ax.plot(x, y, "o", ms=2.0, mfc="white", mec="black", mew=0.3, ls="none")
        if key == "delta":
            bad = np.argwhere(np.isfinite(ov) & (ov < 0.9))
            if len(bad):
                ax.plot(al[bad[:, 1]], lt[bad[:, 0]], "x", color="0.2", ms=2.5, mew=0.5)
        for c in crit[key]:
            ax.contour(A, L, np.where(np.isfinite(field), field, np.nan), levels=[c["value"]], colors="0.15",
                       linewidths=0.6, linestyles="dashed")                   # the level set through the saddle
            ax.plot(c["alpha"], c["log10T"], "X", color="black", mfc="gold", ms=8, mew=0.8)
        a0, a1, t0, t1 = FINE
        ax.plot([a0, a1, a1, a0, a0], [t0, t0, t1, t1, t0], "-", color="0.25", lw=0.7)
        ax.set_title(title, fontsize=8, loc="left")
        ax.set_xlabel(r"steering strength $\alpha$", fontsize=7.5)
        ax.set_xticks([0.07310, 0.07320, 0.07330])
        ax.tick_params(labelsize=6.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02, extend="both" if key == "tau" else "neither")
        cb.ax.tick_params(labelsize=6)
    axes[0].set_ylabel(r"$\log_{10}T$", fontsize=7.5)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", pad_inches=0.03)
    print(args.out)


if __name__ == "__main__":
    main()
