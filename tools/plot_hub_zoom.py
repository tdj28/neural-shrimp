"""Figure: the lower part of the hostile hub's period-7 body at higher resolution (run S9b, 160 x 160 float64 cells):
left, the period of the attractor reached from the fixed start (black: positive largest exponent; white: no period),
right, the largest Lyapunov exponent; both with the float64 fold (black) and flip (red, dashed) curves of the period-7
orbit (hcurves/hub7-*), the two bifurcation lines of run S9a (grey) and the Newton-verified orbits (stars).

Usage: python tools/plot_hub_zoom.py OUT.png
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

ZOOM = "llm-soft-loop/pod64/scan64/hcascade-zoom"
ORBITS = {"14": (0.18250, 0.065518), "28": (0.18470, 0.065644), "21": (0.17980, 0.065267)}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import explore_llm_attractors as ela

    base = ela.DATA / "explore" / "llm-soft-loop" / "pod64"
    path = ela.DATA / "explore" / ZOOM
    z = np.load(path / "scan.npz") if (path / "scan.npz").exists() else np.load(path / "scan-partial.npz")
    lt, al = z["log10_temperature"], z["alpha"]
    P, L = z["period"].astype(float), z["lyapunov"].astype(float)
    if P.ndim == 1:                                           # partial checkpoint: flat arrays
        done = int(z["done"])
        P[done:], L[done:] = np.nan, np.nan
        P, L = P.reshape(len(lt), len(al)), L.reshape(len(lt), len(al))
    chaotic = L > 1e-3
    print("cells", P.size, "chaotic share", round(float(np.nanmean(chaotic)), 3),
          "periods", {int(q): int((P == q).sum()) for q in np.unique(P[~np.isnan(P)])})
    ext = [al[0], al[-1], lt[0], lt[-1]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    cmap = plt.get_cmap("tab20").copy()
    cmap.set_bad("white")
    Pm = np.where((P > 0) & ~chaotic, P, np.nan)
    im = axes[0].imshow(Pm, origin="lower", extent=ext, aspect="auto", cmap=cmap, vmin=0.5, vmax=40.5,
                        interpolation="nearest")
    axes[0].imshow(np.where(chaotic, 1.0, np.nan), origin="lower", extent=ext, aspect="auto", cmap="gray_r",
                   vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(im, ax=axes[0], label="period (black: positive exponent; white: none)")
    im = axes[1].imshow(np.clip(L, -0.4, 0.4), origin="lower", extent=ext, aspect="auto", cmap="RdBu_r",
                        vmin=-0.4, vmax=0.4, interpolation="nearest")
    fig.colorbar(im, ax=axes[1], label="largest Lyapunov exponent (clipped)")
    for ax in axes:
        for c in sorted((base / "hcurves").glob("hub7-*")):
            d = json.loads((c / "curve.json").read_text())
            style = dict(color="k", lw=1.3) if d["kind"] == "fold" else dict(color="C3", lw=1.3, ls="--")
            ax.plot([p["alpha"] for p in d["points"]], [p["log10_temperature"] for p in d["points"]], **style)
        ax.axvline(0.0655, color="0.5", lw=0.8)
        ax.axhline(0.1830, color="0.5", lw=0.8)
        for q, (t, a) in ORBITS.items():
            ax.plot(a, t, "*", ms=11, mfc="gold", mec="k")
            ax.annotate(q, (a, t), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xlim(al[0], al[-1])
        ax.set_ylim(lt[0], lt[-1])
        ax.set_xlabel("steering along 'hostility-vs-kindness'")
        ax.set_ylabel("log10 temperature")
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=200)


if __name__ == "__main__":
    main()
