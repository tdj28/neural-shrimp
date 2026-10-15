"""Figure: the hostile hub's period-7 body. Float64 attractor periods of the partial zoom (scan64/hub-arm) with the
float64 fold and flip curves of the period-7 orbit (hcurves/hub7-*) and the two cusps where the quadratic
normal-form coefficient b changes sign (the cubic coefficient c is nonzero there: cusp-cubic.jsonl, run V11).

Usage: python tools/plot_hub_shrimp.py OUT.png
"""
import json
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
    import period_colors as pc

    base = ela.DATA / "explore" / "llm-soft-loop" / "pod64"
    z = np.load(base / "scan64" / "hub-arm" / "scan-partial.npz")
    done = int(z["done"])
    lt, al = z["log10_temperature"], z["alpha"]
    P = z["period"].astype(float)
    P[done:] = np.nan
    P = P.reshape(len(lt), len(al))
    Pm = np.where(P > 0, P, np.nan)
    xlim, ylim = (0.0600, 0.0700), (0.172, 0.204)
    # colour-bar categories: the periods of the cells drawn inside the axes (the scan extends beyond them)
    xe, ye = np.linspace(al[0], al[-1], len(al) + 1), np.linspace(lt[0], lt[-1], len(lt) + 1)
    shown = (((ye[1:] > ylim[0]) & (ye[:-1] < ylim[1]))[:, None]
             & ((xe[1:] > xlim[0]) & (xe[:-1] < xlim[1]))[None, :])
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    cats = pc.categorize(np.where(shown, Pm, np.nan), max_blocks=pc.blocks_that_fit(ax))
    im = ax.imshow(cats.index(Pm, valid=shown), origin="lower", extent=[al[0], al[-1], lt[0], lt[-1]],
                   aspect="auto", cmap=cats.cmap, norm=cats.norm, interpolation="nearest", alpha=0.55)
    cats.colorbar(im, ax=ax, label="period of the attractor (white: none or not computed)")
    styles = {"fold": dict(color="k", lw=1.6), "flip": dict(color="C3", lw=1.6, ls="--")}
    for c in sorted((base / "hcurves").glob("hub7-*")):
        d = json.loads((c / "curve.json").read_text())
        pts = d["points"]
        ax.plot([p["alpha"] for p in pts], [p["log10_temperature"] for p in pts], **styles[d["kind"]])
    for c in ("hub7-foldU-plus", "hub7-foldS-plus"):
        rows = [json.loads(l) for l in open(base / "hcurves" / c / "cusp-check.jsonl")]
        for a, b in zip(rows, rows[1:]):
            if np.sign(a["b_oriented"]) != np.sign(b["b_oriented"]):
                ax.plot((a["alpha"] + b["alpha"]) / 2, (a["log10_T"] + b["log10_T"]) / 2, "o", ms=9, mfc="gold",
                        mec="k", zorder=5)
    ax.plot([], [], **styles["fold"], label="fold curves (period 7)")
    ax.plot([], [], **styles["flip"], label="flip curves (period 7 to 14)")
    ax.plot([], [], "o", ms=8, mfc="gold", mec="k", ls="none", label="nondegenerate cusps ($b=0$, $c\\neq0$)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("steering along 'hostility-vs-kindness'")
    ax.set_ylabel("log10 temperature")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=200)


if __name__ == "__main__":
    main()
