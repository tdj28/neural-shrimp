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

    base = ela.DATA / "explore" / "llm-soft-loop" / "pod64"
    z = np.load(base / "scan64" / "hub-arm" / "scan-partial.npz")
    done = int(z["done"])
    lt, al = z["log10_temperature"], z["alpha"]
    P = z["period"].astype(float)
    P[done:] = np.nan
    P = P.reshape(len(lt), len(al))
    Pm = np.where(P > 0, P, np.nan)
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    cmap = plt.get_cmap("tab20").copy()
    cmap.set_bad("white")
    im = ax.imshow(Pm, origin="lower", extent=[al[0], al[-1], lt[0], lt[-1]], aspect="auto", cmap=cmap,
                   interpolation="nearest", vmin=0.5, vmax=40.5, alpha=0.55)
    plt.colorbar(im, ax=ax, label="period of the attractor (white: none or not computed)")
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
    ax.set_xlim(0.0600, 0.0700)
    ax.set_ylim(0.172, 0.204)
    ax.set_xlabel("steering along 'hostility-vs-kindness'")
    ax.set_ylabel("log10 temperature")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=200)


if __name__ == "__main__":
    main()
