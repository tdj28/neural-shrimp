"""Figure: the tip of the hook (runs HOOKTIP and the fold-point check).

(a) The fixed start's landing coordinate u0 on the saddle-node orbit at Newton-converged points of the kite's fold
    curve (tools/hook_tip_check.py): negative where the start then converges onto that orbit (it reaches the weaker
    orbit just inside the coexistence region), positive where it escapes. The sign change is the predicted tip.
(b), (c) Largest exponent from the map's window (steps 300-600) and from a long window (steps 3000-3400) along rows
    at fixed offsets from the fold curve (tools/hook_tip_strip.py): S has exponent ln(0.18)/6 = -0.28, the weaker
    orbit a small negative one near the fold. Shaded: the predicted tip interval.

Usage: python tools/plot_hook_tip.py --tip llm-soft-loop/kite/hook-tip --strip llm-soft-loop/pod64/kite/hook-tip-strip \
    --out paper/llm/figures/fig-p6-hook-tip.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tip", required=True)
    ap.add_argument("--strip", required=True)
    ap.add_argument("--split", type=float, default=-0.15, help="exponent between S and the weaker orbit")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import explore_llm_soft_loop as esl

    T = json.loads((esl.DATA / "explore" / args.tip / "hook-tip.json").read_text())
    pts = sorted(T["points"], key=lambda r: r["alpha"])
    tips = T["predicted_tips"]
    lo_hi = None
    for x, y in zip(pts[:-1], pts[1:]):
        if x["u0"] * y["u0"] < 0:
            lo_hi = (x["alpha"], y["alpha"])
    St = json.loads((esl.DATA / "explore" / args.strip / "hook-tip-strip.json").read_text())
    cells = St["cells"]
    offs = sorted({c["rows_below_fold"] for c in cells})
    summary = {"predicted_interval": lo_hi, "linear_tip": tips[0]["alpha_tip"] if tips else None, "rows": []}
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.4))
    a = ax[0]
    al = np.array([p["alpha"] for p in pts])
    u0 = np.array([p["u0"] for p in pts])
    a.plot(al, u0 * 1e3, "o-", color="k", ms=4)
    for x, u in zip(al, u0):
        a.plot(x, u * 1e3, "o", ms=5, color="tab:blue" if u < 0 else "tab:red")
    a.axhline(0, color="0.5", lw=0.6)
    if lo_hi:
        a.axvspan(*lo_hi, color="gold", alpha=0.3, lw=0)
    a.set_xlabel(r"$\alpha$ of the fold point", fontsize=7)
    a.set_ylabel(r"landing coordinate $u_0$ ($10^{-3}$)", fontsize=7)
    a.tick_params(labelsize=6)
    a.ticklabel_format(useOffset=False, axis="x")
    a.set_title("(a) where the fixed start lands on the fold orbit", fontsize=8)
    cmap = plt.get_cmap("coolwarm")
    for j, (key, axx, title) in enumerate((("lambda_short", ax[1], "(b) the map's window (steps 300-600)"),
                                           ("lambda_long", ax[2], "(c) a long window (steps 3000-3400)"))):
        for i, o in enumerate(offs):
            cs = sorted([c for c in cells if c["rows_below_fold"] == o], key=lambda c: c["alpha"])
            x = np.array([c["alpha"] for c in cs])
            y = np.array([c[key] for c in cs])
            axx.plot(x, y, ".-", ms=2.5, lw=0.7, color=cmap(i / max(1, len(offs) - 1)),
                     label=f"{o:+g} rows" if j == 1 else None)
            weak = x[y > args.split]
            if j == 0:
                summary["rows"].append({"rows_below_fold": o})
            summary["rows"][i][key + "_weak_alpha_range"] = [float(weak.min()), float(weak.max())] if len(weak) else None
        if lo_hi:
            axx.axvspan(*lo_hi, color="gold", alpha=0.3, lw=0)
        axx.axhline(args.split, color="0.5", lw=0.5, ls=":")
        axx.set_xlabel(r"steering strength $\alpha$", fontsize=7)
        axx.set_ylabel("largest exponent (per step)", fontsize=7)
        axx.tick_params(labelsize=6)
        axx.ticklabel_format(useOffset=False, axis="x")
        axx.set_title(title, fontsize=8)
    ax[2].legend(fontsize=5.5, title="offset below the fold", title_fontsize=6, loc="lower left")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=250)
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
