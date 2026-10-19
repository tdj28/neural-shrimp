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
    ap.add_argument("--folds", default=None, help="hook_tip_fold_points.py output: fold points inside the bracket")
    ap.add_argument("--strip2", default=None, help="a finer strip near the tip (hook_tip_strip.py)")
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
    extra = []
    if args.folds:
        F = json.loads((esl.DATA / "explore" / args.folds / "hook-tip-folds.json").read_text())
        extra = sorted([r for r in F["rows"] if r.get("located")], key=lambda r: r["alpha"])
        summary["located_fold_points"] = [{k: r[k] for k in ("alpha", "log10T", "fold_residual", "u0",
                                                               "final_distance_to_orbit")} for r in extra]
        allp = sorted(pts + extra, key=lambda r: r["alpha"])
        for x, y in zip(allp[:-1], allp[1:]):
            if x["u0"] * y["u0"] < 0:
                f = x["u0"] / (x["u0"] - y["u0"])
                summary["tip_from_u0"] = {"between": [x["alpha"], y["alpha"]],
                                          "alpha_linear": x["alpha"] + f * (y["alpha"] - x["alpha"])}
    fig, ax = plt.subplots(1, 4 if args.strip2 else 3, figsize=(14.0 if args.strip2 else 10.5, 3.4))
    a = ax[0]
    al = np.array([p["alpha"] for p in pts])
    u0 = np.array([p["u0"] for p in pts])
    a.plot(al, u0 * 1e3, "o-", color="k", ms=4)
    for x, u in zip(al, u0):
        a.plot(x, u * 1e3, "o", ms=5, color="tab:blue" if u < 0 else "tab:red")
    if extra:
        a.plot([r["alpha"] for r in extra], [r["u0"] * 1e3 for r in extra], "s", ms=3.5, mfc="none", mec="k",
               label="located inside the bracket")
        a.legend(fontsize=5.5, loc="upper right")
        ins = a.inset_axes([0.42, 0.33, 0.55, 0.42])
        ex = np.array([r["alpha"] for r in extra])
        eu = np.array([r["u0"] for r in extra])
        ins.plot(ex, eu * 1e3, "s-", ms=3, mfc="none", color="k", lw=0.7)
        ins.axhline(0, color="0.5", lw=0.5)
        if "tip_from_u0" in summary:
            ins.axvline(summary["tip_from_u0"]["alpha_linear"], color="tab:blue", lw=0.8)
        ins.tick_params(labelsize=4.5)
        ins.ticklabel_format(useOffset=False, axis="x")
        ins.set_xticks([0.072964, 0.072966, 0.072968])
        ins.set_title("located fold points near the zero", fontsize=5.5)
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
        if "tip_from_u0" in summary:
            axx.axvline(summary["tip_from_u0"]["alpha_linear"], color="tab:blue", lw=0.8,
                        label="predicted tip" if j == 0 else None)
        elif lo_hi:
            axx.axvspan(*lo_hi, color="gold", alpha=0.3, lw=0)
        axx.axhline(args.split, color="0.5", lw=0.5, ls=":")
        axx.set_xticks([0.072955, 0.072965, 0.072975, 0.072985])
        axx.set_xlabel(r"steering strength $\alpha$", fontsize=7)
        axx.set_ylabel("largest exponent (per step)", fontsize=7)
        axx.tick_params(labelsize=6)
        axx.ticklabel_format(useOffset=False, axis="x")
        axx.set_title(title, fontsize=8)
    ax[2].legend(fontsize=5.5, title="offset below the fold", title_fontsize=6, loc="lower left")
    if "tip_from_u0" in summary:
        ax[1].legend(fontsize=5.5, loc="lower right")
    if args.strip2:
        S2 = json.loads((esl.DATA / "explore" / args.strip2 / "hook-tip-strip.json").read_text())["cells"]
        d = ax[3]
        bnd = []
        for o in sorted({c["rows_below_fold"] for c in S2}):
            cs = sorted([c for c in S2 if c["rows_below_fold"] == o], key=lambda c: c["alpha"])
            x = np.array([c["alpha"] for c in cs])
            y = np.array([c["lambda_long"] for c in cs])
            k = np.nonzero(y > args.split)[0]
            if len(k) and k[0] > 0:
                bnd.append((o, x[k[0] - 1], x[k[0]]))
        if bnd:
            o = np.array([b[0] for b in bnd])
            mid = np.array([(b[1] + b[2]) / 2 for b in bnd])
            half = np.array([(b[2] - b[1]) / 2 for b in bnd])
            d.errorbar(np.sqrt(o), mid, yerr=half, fmt="o", ms=3.5, color="tab:red", capsize=2,
                       label="boundary inside the fold (long window)")
            pf = np.polyfit(np.sqrt(o), mid, 1)
            xx = np.linspace(0, np.sqrt(o).max(), 20)
            d.plot(xx, np.polyval(pf, xx), "k--", lw=0.8, label=f"linear in the square root: tip {pf[1]:.7f}")
            summary["strip2_boundaries"] = [{"rows_below_fold": float(b[0]), "between": [float(b[1]), float(b[2])]}
                                            for b in bnd]
            summary["strip2_tip_extrapolated"] = float(pf[1])
            if "tip_from_u0" in summary:
                d.axhline(summary["tip_from_u0"]["alpha_linear"], color="tab:blue", lw=0.8,
                          label=f"zero of $u_0$: {summary['tip_from_u0']['alpha_linear']:.7f}")
        d.set_xlabel("square root of the offset below the fold (rows)", fontsize=7)
        d.set_ylabel(r"steering strength $\alpha$", fontsize=7)
        d.ticklabel_format(useOffset=False, axis="y")
        d.tick_params(labelsize=6)
        d.legend(fontsize=5.5)
        d.set_title("(d) the boundary approaches the tip", fontsize=8)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=250)
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
