"""Figure fig-hostile-cascade (regenerated 15 Oct 2026; the original generator was not kept): the hostility-steered
soft-token map of Qwen2.5-0.5B-Instruct (layer 12).

Left: period of the attractor in the float32 zoom scan32/hzoom-B (192 x 192 cells, alpha 0.06-0.10, log10 T -0.20 to
0.06; black: chaos, i.e. exponent above 1e-3 and no period), one colour per period (tools/period_colors.py); dashed,
the cut alpha = 0.083; dotted, the float64 fold of the '!Yes' 2-cycle (hnormal/curve-p2-fold, log10 T_c).
Right: float64 fold edges of the period-adding windows along that cut (edges/hadd-a0.083, edges/hadd2-a0.083) as
1 / sqrt(log10 T_c - log10 T_fold) against the laps of the 2-cycle (period / 2); the least-squares line over periods
16-20, and the normal-form line for full passages, slope sqrt|ab| / pi from hnormal/normal-form-p2, anchored at the
period-20 window.

Usage: python tools/plot_hostile_cascade.py [OUT.png]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data")) / "explore/llm-soft-loop/pod64"
CUT = 0.083


def window_folds():
    folds = {}
    for run in ("hadd-a0.083", "hadd2-a0.083"):
        for w in json.loads((DATA / "edges" / run / "edges.json").read_text())["windows"]:
            for e in w.get("edges", []):
                if e["kind"] == "fold" and e.get("lambda") is not None and w["period"] >= 14:
                    folds[int(w["period"])] = float(e["lambda"])
    return dict(sorted(folds.items()))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import period_colors as pc

    out = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "paper/llm/figures/fig-hostile-cascade.png")
    z = np.load(DATA / "scan32" / "hzoom-B" / "scan.npz", allow_pickle=True)
    lt, al, P, L = z["log10_temperature"], z["alpha"], z["period"].astype(float), z["lyapunov"].astype(float)
    tc = json.loads((DATA / "hnormal" / "curve-p2-fold" / "curve.json").read_text())["points"][0]["log10_temperature"]
    nf = json.loads((DATA / "hnormal" / "normal-form-p2" / "normal-form.json").read_text())
    slope_nf = float(np.sqrt(abs(nf["ab"])) / np.pi)
    folds = window_folds()

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), dpi=200, gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    chaotic = (L > 1e-3) & (P == 0)
    Pm = np.where(P > 0, P, np.nan)
    cats = pc.categorize(Pm, max_blocks=pc.blocks_that_fit(ax))
    ext = [al[0], al[-1], lt[0], lt[-1]]
    im = ax.imshow(cats.index(Pm), origin="lower", extent=ext, aspect="auto", cmap=cats.cmap, norm=cats.norm,
                   interpolation="nearest")
    ax.imshow(np.where(chaotic, 1.0, np.nan), origin="lower", extent=ext, aspect="auto", cmap="gray_r", vmin=0,
              vmax=1.25, interpolation="nearest")
    cats.colorbar(im, ax=ax, label="period (black: chaos)")
    ax.axvline(CUT, color="white", ls="--", lw=0.9)
    ax.axhline(tc, color="white", ls=":", lw=0.9)
    ax.text(0.0785, tc + 0.016, "P2 '!Yes'", fontsize=7)
    ax.text(0.0755, -0.148, "P6 'Yes, that's crap'", fontsize=7, color="white")
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel(r"steering along hostility $-$ kindness, $\alpha$")
    ax.set_ylabel(r"$\log_{10}T$")
    ax.set_title("period of the attractor (black: chaos); float32 zoom", fontsize=8)

    bx = axes[1]
    laps = np.array([p / 2 for p in folds])
    y = np.array([1 / np.sqrt(tc - t) for t in folds.values()])
    bx.plot(laps, y, "o", color="C3", label="window folds (float64)")
    sel = laps >= 8
    k, b = np.polyfit(laps[sel], y[sel], 1)
    xs = np.linspace(laps.min() - 0.5, laps.max() + 0.5, 50)
    bx.plot(xs, k * xs + b, color="C3", lw=1, label=f"fit over P16-P20: slope {k:.3f}")
    x20, y20 = laps[-1], y[-1]
    bx.plot(xs, y20 + slope_nf * (xs - x20), "k--", lw=1,
            label=r"normal form, full passage: $\sqrt{|ab|}/\pi=%.3f$" % slope_nf)
    bx.set_xlabel("laps of the '!Yes' 2-cycle (period / 2)")
    bx.set_ylabel(r"$1/\sqrt{\log_{10}T_c-\log_{10}T_{\rm fold}}$")
    bx.set_title(r"accumulation on the 2-cycle fold ($\alpha=0.083$)", fontsize=8)
    bx.legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(out)
    print(json.dumps({"log10_Tc": tc, "folds": folds, "fit_slope": k, "normal_form_slope": slope_nf}))


if __name__ == "__main__":
    main()
