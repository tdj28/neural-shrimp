"""Fold and flip curves of the period-5 window of the two-block plane (run FM5), gated and drawn over the classifier
map (run TBZg2) and the continued orbit's grid (run DSS5).

Gates (AGENTS.md verification gates), read from every raw point before anything is summarized:
- residual and test function |g| at every curve point (maxima reported per curve);
- a fold curve must keep a real critical multiplier at +1 and a flip curve at -1: the largest |mu - target| and
  |Im mu| over the curve are reported, and a curve whose multiplier leaves the target by more than 1e-4 is marked
  failed;
- tangent reversals are listed as cusp candidates with the critical multiplier and the next modulus there; they are
  candidates until located and checked (b sign a = 0, c != 0, det J != 0);
- codimension-two approaches: points where another multiplier comes within 0.05 of the unit circle.

Usage:
  python tools/plot_fm5.py [--run llm-soft-loop/pod64/fm5] [--json docs/fm5/fm5-summary.json] [--out PNG]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
TB = "llm-soft-loop/pod64/tb/zg2-grat4-host16-t-0.1/tb-t-0.100.npz"
DSS = "llm-soft-loop/pod64/dss/p5-grat4-host16"
TOL_TARGET = 1e-4


def branch_summary(rec):
    pts = rec["points"]
    turns = [i for i, r in enumerate(pts) if r.get("turned_since_previous")]
    return {"free": rec["free"], "period": rec["period"], "points": len(pts), "stop": rec["stop"],
            "start": rec["points"][0]["lambda"] if pts else None,
            "max_residual": max(r["residual"] for r in pts) if pts else None,
            "first_turn_estimate": (pts[turns[0]].get("turn_lambda_estimate", pts[turns[0]]["lambda"])
                                    if turns else None),
            "crossings": [{"type": c["type"], "between": c["between_log10_T"],
                           "crossing_multiplier": c["crossing_multiplier"][:2],
                           "next_largest_modulus": c["next_largest_modulus"]} for c in rec["crossings"]]}


def curve_summary(rec, near=0.05):
    pts = rec["points"]
    kind = rec["kind"]
    target = 1.0 if kind == "fold" else -1.0
    crit = np.array([p["critical_multiplier"] for p in pts])
    dev = np.abs(crit[:, 0] - target) if len(pts) else np.array([])
    imag = np.abs(crit[:, 1]) if len(pts) else np.array([])
    cusps = [{"index": p["index"], "alpha": p["alpha"], "alpha2": p["alpha2"],
              "critical_multiplier": p["critical_multiplier"], "other_spectral_radius": p["other_spectral_radius"]}
             for p in pts if p["cusp_candidate"]]
    codim2 = [{"index": p["index"], "alpha": p["alpha"], "alpha2": p["alpha2"],
               "other_nearest_plus_one": p["other_nearest_plus_one"],
               "other_nearest_minus_one": p["other_nearest_minus_one"],
               "largest_complex_modulus": p["largest_complex_modulus"]}
              for p in pts if p["other_spectral_radius"] is not None and p["other_spectral_radius"] > 1 - near]
    a1 = [p["alpha"] for p in pts]
    a2 = [p["alpha2"] for p in pts]
    failed = bool(len(pts) and (dev.max() > TOL_TARGET or imag.max() > TOL_TARGET))
    return {"kind": kind, "period": rec["period"], "points": len(pts), "stop": rec["stop"], "seconds": rec["seconds"],
            "start": rec["start"], "direction": rec["args"]["direction"],
            "max_residual": max(p["residual"] for p in pts) if pts else None,
            "max_abs_g": max(p["g"] for p in pts) if pts else None,
            "max_critical_deviation": float(dev.max()) if len(pts) else None,
            "max_critical_imag": float(imag.max()) if len(pts) else None,
            "gate_failed": failed,
            "max_other_spectral_radius": max((p["other_spectral_radius"] or 0) for p in pts) if pts else None,
            "alpha_range": [min(a1), max(a1)] if pts else None, "alpha2_range": [min(a2), max(a2)] if pts else None,
            "cusp_candidates": cusps, "codim2_approaches": codim2}


def load(run):
    base = DATA / "explore" / run
    branches, curves = {}, {}
    for d in sorted(base.glob("branch-*")):
        if (d / "branch.json").exists():
            branches[d.name] = json.loads((d / "branch.json").read_text())
    for d in sorted(base.glob("curve-*")):
        if (d / "curve.json").exists():
            curves[d.name] = json.loads((d / "curve.json").read_text())
    return branches, curves


def figure(branches, curves, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    sys.path.insert(0, str(ROOT / "tools"))
    import dss_window as dw
    import period_colors as pc

    tb = np.load(DATA / "explore" / TB, allow_pickle=True)
    z = np.load(DATA / "explore" / DSS / "grid.npz", allow_pickle=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=170)

    def overlay(ax, label=True):
        for name, rec in sorted(curves.items()):
            pts = rec["points"]
            if not pts:
                continue
            a1 = [p["alpha"] for p in pts]
            a2 = [p["alpha2"] for p in pts]
            fold = rec["kind"] == "fold"
            ax.plot(a1, a2, "-" if fold else "--", color="k" if fold else "r", lw=1.3 if fold else 1.1,
                    label=None)
            cz = [(p["alpha"], p["alpha2"]) for p in pts if p["cusp_candidate"]]
            if cz:
                ax.plot(*zip(*cz), "*", color="gold", ms=9, mec="k", mew=0.5, zorder=5)
        for name, rec in sorted(branches.items()):
            ta = rec["args"]
            lam = [p["lambda"] for p in rec["points"]]
            if rec["free"] == "alpha":
                ax.plot(lam, [ta["alpha2"]] * len(lam), ":", color="w", lw=0.8)
            else:
                ax.plot([ta["alpha"]] * len(lam), lam, ":", color="w", lw=0.8)
        if branches:
            rec = next(iter(branches.values()))
            ax.plot(rec["args"]["alpha"], rec["args"]["alpha2"], "o", color="w", mec="k", ms=5, zorder=6)

    ax = axes[0]
    P = np.where(tb["chaotic"], np.nan, tb["period"].astype(float))
    P[P <= 0] = np.nan
    b1, b2 = tb["alpha1"], tb["alpha2"]
    e2 = [b1[0], b1[-1], b2[0], b2[-1]]
    ax.imshow(np.where(tb["chaotic"], 1.0, np.nan), origin="lower", extent=e2, aspect="auto",
              cmap=ListedColormap(["0.15"]), interpolation="nearest")
    cats = pc.categorize([P], max_blocks=9)
    cmap = cats.cmap.copy()
    cmap.set_bad((0, 0, 0, 0))
    im = ax.imshow(cats.index(P), origin="lower", extent=e2, aspect="auto", cmap=cmap, norm=cats.norm,
                   interpolation="nearest")
    cats.colorbar(im, ax=ax, label="period (black: chaotic)")
    overlay(ax)
    a1, a2 = z["alpha"], z["log10_temperature"]
    ax.plot([a1[0], a1[-1], a1[-1], a1[0], a1[0]], [a2[0], a2[0], a2[-1], a2[-1], a2[0]], "-", color="w", lw=0.8)
    ax.set_xlim(e2[0], e2[1])
    ax.set_ylim(e2[2], e2[3])
    ax.set_title("(a) fold (black) and flip (red) curves of the period-5 orbit over the attractor map", fontsize=7.5)

    ax = axes[1]
    mu1, mu2 = dw.leading_pair(z)
    ok = z["ok"]
    stable = ok & (np.abs(mu1) < 1)
    t = np.real(mu1 + mu2)
    da1, da2 = a1[1] - a1[0], a2[1] - a2[0]
    ext = [a1[0] - da1 / 2, a1[-1] + da1 / 2, a2[0] - da2 / 2, a2[-1] + da2 / 2]
    im = ax.imshow(np.where(stable, t, np.nan), origin="lower", extent=ext, aspect="auto", cmap="RdBu_r", vmin=-0.6,
                   vmax=0.6, interpolation="nearest")
    ax.imshow(np.where(ok & ~stable, 1.0, np.nan), origin="lower", extent=ext, aspect="auto",
              cmap=ListedColormap(["0.75"]), interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, label=r"trace $\mu_1+\mu_2$ (continued orbit, DSS5)")
    ax.contour(a1, a2, np.where(ok, t, np.nan), levels=[0], colors="0.3", linewidths=0.6)
    overlay(ax)
    pad1, pad2 = 4 * da1, 4 * da2
    ax.set_xlim(ext[0] - pad1, ext[1] + pad1)
    ax.set_ylim(ext[2] - pad2, ext[3] + pad2)
    ax.set_title("(b) the same curves over the continued orbit's trace (grey: repelling)", fontsize=7.5)
    for ax in axes:
        ax.set_xlabel(r"$\alpha_1$ (gratitude, block 4)", fontsize=8)
        ax.set_ylabel(r"$\alpha_2$ (hostility, block 16)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="llm-soft-loop/pod64/fm5")
    ap.add_argument("--json")
    ap.add_argument("--out")
    args = ap.parse_args()
    branches, curves = load(args.run)
    if not branches and not curves:
        raise SystemExit(f"no branch or curve results under {args.run}")
    summary = {"run": args.run, "branches": {k: branch_summary(v) for k, v in branches.items()},
               "curves": {k: curve_summary(v) for k, v in curves.items()}}
    text = json.dumps(summary, indent=1)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text + "\n")
    print(text)
    if args.out:
        figure(branches, curves, args.out)


if __name__ == "__main__":
    main()
