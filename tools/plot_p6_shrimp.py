"""Run P6S figure: the period-6 shrimp of the crossing window (Fig. 1's plane: philosophy steering at layer 12, alpha
against log10 T) with its verified skeleton, drawn as the period-5 shrimp of the two-block plane is (plot_real_shrimp.py):
fold curves black and solid, flip curves red and dashed, cusp candidates as gold stars.

Panels:
(a) the 512 x 512 float64 map HR9 (3x finer than Fig. 1), colored as Fig. 1 (chaotic cells by their largest exponent,
    periodic cells in gray by contraction), with the continued curves;
(b) the attractor followed from the body (tools/follow_attractor.py --plane alpha-temperature): members of the
    period-6 family by period (6, 12, 24, ...), chaos behind the text, slow cells, and cells where the attractor was
    lost, with the same curves.

Gate: every curve point drawn has residual <= --max-residual and a critical multiplier of the curve's kind (real +1 for
a fold, real -1 for a flip, within --kind-tol); rejected points are counted and reported, never drawn.

Usage: python tools/plot_p6_shrimp.py --scan llm-soft-loop/pod64/hr9 --curves llm-soft-loop/pod64/p6s \
         --follow llm-soft-loop/pod64/p6s/follow --out paper/llm/figures/fig-p6-shrimp.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "neural-shrimp-data"


def load_curves(folder, max_residual, kind_tol):
    """Every curve-*/curve.json under folder, gated point by point."""
    out, report = [], []
    for f in sorted((DATA / "explore" / folder).glob("*curve-*/curve.json")):
        c = json.loads(f.read_text())
        kind = c.get("kind")
        target = 1.0 if kind == "fold" else -1.0
        keep, bad = [], 0
        for q in c["points"]:
            mu = q.get("critical_multiplier_refined", q.get("critical_multiplier"))
            mu_re = mu[0] if isinstance(mu, (list, tuple)) else mu
            mu_im = mu[1] if isinstance(mu, (list, tuple)) and len(mu) > 1 else 0.0
            ok = (q.get("residual", 1.0) <= max_residual and mu_re is not None
                  and abs(mu_re - target) <= kind_tol and abs(mu_im) <= kind_tol and not q.get("rejected", False))
            if ok:
                keep.append(q)
            else:
                bad += 1
        out.append({"name": f.parent.name, "kind": kind, "points": keep})
        report.append({"curve": f.parent.name, "kind": kind, "points": len(c["points"]), "drawn": len(keep),
                       "failed_gate": bad, "stop": c.get("stop")})
    return out, report


def boundary_gate(fa, ft, lab, fper, curves, period):
    """As plot_real_shrimp.boundary_gate, in the (alpha, log10 T) plane with each axis in its own grid steps: for every
    pair of 8-neighbouring cells with (i) period p next to period 2p, the distance from the pair's midpoint to the
    nearest flip curve, and (ii) period p next to a cell where the attractor was lost, the distance to the nearest
    fold curve."""
    scale = np.array([abs(fa[1] - fa[0]), abs(ft[1] - ft[0])])
    polys = {"fold": [], "flip": []}
    for rec in curves:
        if len(rec["points"]) >= 2:
            polys[rec["kind"]].append(np.array([[q["alpha"], q["log10_temperature"]] for q in rec["points"]]) / scale)

    def dist(mid, kind):
        q = np.asarray(mid) / scale
        best = np.inf
        for P in polys[kind]:
            A, B = P[:-1], P[1:]
            d = B - A
            t = np.clip(((q - A) * d).sum(1) / np.maximum((d ** 2).sum(1), 1e-30), 0, 1)
            best = min(best, float(np.sqrt(((A + t[:, None] * d - q) ** 2).sum(1)).min()))
        return best

    pairs = {f"{period}|{2 * period} (flip)": [], f"{period}|lost (fold)": []}
    n2, n1 = lab.shape
    for i in range(n2):
        for j in range(n1):
            if not (lab[i, j] == 1 and fper[i, j] == period):
                continue
            for di, dj in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                ii, jj = i + di, j + dj
                if not (0 <= ii < n2 and 0 <= jj < n1) or lab[ii, jj] < 0:
                    continue
                mid = ((fa[j] + fa[jj]) / 2, (ft[i] + ft[ii]) / 2)
                if lab[ii, jj] == 1 and fper[ii, jj] == 2 * period and polys["flip"]:
                    pairs[f"{period}|{2 * period} (flip)"].append(dist(mid, "flip"))
                elif lab[ii, jj] == 0 and polys["fold"]:
                    pairs[f"{period}|lost (fold)"].append(dist(mid, "fold"))
    out = {}
    for k, d in pairs.items():
        d = np.array(d)
        out[k] = {"pairs": int(len(d)), "within_1_step": int((d <= 1).sum()), "within_2_steps": int((d <= 2).sum()),
                  "median_steps": float(np.median(d)) if len(d) else None, "max_steps": float(d.max()) if len(d) else None}
    return out


def draw_located(ax, located, ms=9):
    """Gold stars at cusps located and checked by tools/cusp_locate.py (located: true)."""
    for f in located:
        rec = json.loads(f.read_text())
        if rec.get("located"):
            fin = rec["final"]
            ax.plot(fin["alpha"], fin["log10_temperature"], "*", color="gold", ms=ms, mec="k", mew=0.6, zorder=7)


def draw_curves(ax, curves, lw=1.1, ms=7):
    from matplotlib import patheffects as pe
    for rec in curves:
        pts = rec["points"]
        if len(pts) < 2:
            continue
        x = [q["alpha"] for q in pts]
        y = [q["log10_temperature"] for q in pts]
        fold = rec["kind"] == "fold"
        ax.plot(x, y, "-" if fold else "--", color="k" if fold else "r", lw=lw if fold else lw * 0.9, zorder=4,
                path_effects=[pe.Stroke(linewidth=lw + 1.2, foreground="w", alpha=0.7), pe.Normal()])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scan", required=True)
    ap.add_argument("--curves", required=True, help="explore-relative folder holding the *curve-*/curve.json runs")
    ap.add_argument("--follow", default=None, help="explore-relative folder with follow.npz")
    ap.add_argument("--window", type=float, nargs=4, default=None, metavar=("A0", "A1", "T0", "T1"))
    ap.add_argument("--max-residual", type=float, default=1e-8)
    ap.add_argument("--kind-tol", type=float, default=1e-3)
    ap.add_argument("--chaos-level", type=float, default=1e-3)
    ap.add_argument("--period", type=int, default=6, help="period of the body (for the boundary gate)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    d = np.load(DATA / "explore" / args.scan / "scan.npz", allow_pickle=True)
    lt, al, per, ly = d["log10_temperature"], d["alpha"], d["period"], d["lyapunov"]
    if args.window:
        a0, a1, t0, t1 = args.window
        rows, cols = (lt >= t0) & (lt <= t1), (al >= a0) & (al <= a1)
        lt, al, per, ly = lt[rows], al[cols], per[np.ix_(rows, cols)], ly[np.ix_(rows, cols)]
    chaos = (ly > args.chaos_level) & (per <= 0)
    hot_norm = Normalize(0.0, float(np.max(ly[chaos])) if chaos.any() else 1.0)
    neg = np.clip(-ly, 0, None)
    cap = max(float(np.percentile(neg[~chaos], 99)) if (~chaos).any() else 1.0, 1e-12)
    grey = 1.0 - 0.9 * np.clip(neg / cap, 0, 1) ** 0.5
    img = np.stack([grey, grey, grey, np.ones_like(grey)], axis=-1)
    img[chaos] = plt.get_cmap("plasma")(hot_norm(ly[chaos]))
    da, dt = (al[1] - al[0]) / 2, (lt[1] - lt[0]) / 2
    ext = [al[0] - da, al[-1] + da, lt[0] - dt, lt[-1] + dt]

    curves, report = load_curves(args.curves, args.max_residual, args.kind_tol)
    located = sorted((DATA / "explore" / args.curves).glob("locate-*/locate.json"))
    report_located = [{"locate": f.parent.name, "located": json.loads(f.read_text()).get("located")} for f in located]
    panels = 2 if args.follow else 1
    plt.rcParams.update({"font.size": 8})
    fig, axes = plt.subplots(1, panels, figsize=(7.0 if panels == 2 else 3.6, 3.5), squeeze=False)
    axA = axes[0, 0]
    axA.imshow(img, origin="lower", aspect="auto", interpolation="nearest", extent=ext)
    draw_curves(axA, curves)
    draw_located(axA, located)
    axA.set_xlim(ext[0], ext[1]); axA.set_ylim(ext[2], ext[3])
    axA.set_xlabel(r"steering strength $\alpha$"); axA.set_ylabel(r"$\log_{10}T$")
    axA.set_title("(a) Lyapunov map, 512 x 512 float64 cells", fontsize=8)
    cb = fig.colorbar(ScalarMappable(norm=hot_norm, cmap="plasma"), ax=axA, orientation="horizontal", fraction=0.05,
                      pad=0.16)
    cb.set_label(r"chaotic cells: largest exponent $\lambda$", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    summary = {"scan": args.scan, "cells": int(ly.size), "chaotic": int(chaos.sum()), "curves": report,
               "located_cusps": report_located}

    if args.follow:
        z = np.load(DATA / "explore" / args.follow / "follow.npz", allow_pickle=True)
        if str(z["plane"]) != "alpha-temperature":
            raise SystemExit("the follow run is not in the alpha-temperature plane")
        fa, ft, lab, fper = z["alpha1"], z["log10_temperature"], z["label"], z["period"]
        fda, fdt = (fa[1] - fa[0]) / 2, (ft[1] - ft[0]) / 2
        fext = [fa[0] - fda, fa[-1] + fda, ft[0] - fdt, ft[-1] + fdt]
        axB = axes[0, 1]
        sys.path.insert(0, str(ROOT / "tools"))
        import period_colors as pc
        axB.imshow(np.where(lab == 0, 1.0, np.nan), origin="lower", extent=fext, aspect="auto",
                   cmap=ListedColormap(["0.88"]), interpolation="nearest")
        P = np.where(lab == 1, fper.astype(float), np.nan)
        cats = pc.categorize([P], max_blocks=7)
        im = axB.imshow(cats.index(P), origin="lower", extent=fext, aspect="auto", cmap=cats.cmap, norm=cats.norm,
                        interpolation="nearest")
        axB.imshow(np.where(lab == 2, 1.0, np.nan), origin="lower", extent=fext, aspect="auto",
                   cmap=ListedColormap(["k"]), interpolation="nearest")
        axB.imshow(np.where(lab == 3, 1.0, np.nan), origin="lower", extent=fext, aspect="auto",
                   cmap=ListedColormap(["0.55"]), interpolation="nearest")
        cats.colorbar(im, ax=axB, label="period of the followed attractor")
        periods = sorted({int(p) for p in np.unique(fper[lab == 1]) if p > 0})
        handles = [Patch(color="k", label="chaos behind the text"), Patch(color="0.55", label="slow (no period)"),
                   Patch(color="0.88", label="attractor lost"),
                   Line2D([], [], color="k", lw=1.1, label="fold curve"),
                   Line2D([], [], color="r", lw=1.0, ls="--", label="flip curve")]
        draw_curves(axB, curves)
        draw_located(axB, located)
        axB.set_xlim(ext[0], ext[1]); axB.set_ylim(ext[2], ext[3])
        axB.set_xlabel(r"steering strength $\alpha$")
        axB.set_title("(b) the attractor followed from the body", fontsize=8)
        axB.legend(handles=handles, fontsize=5.8, loc="lower left", framealpha=0.85)
        summary["boundary_gate"] = boundary_gate(fa, ft, lab, fper, curves, args.period)
        summary["follow"] = {"members": int((lab == 1).sum()), "chaos_behind_text": int((lab == 2).sum()),
                             "slow": int((lab == 3).sum()), "lost": int((lab == 0).sum()),
                             "periods": {str(p): int(((lab == 1) & (fper == p)).sum()) for p in periods}}
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.03)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary)[:1500])


if __name__ == "__main__":
    main()
