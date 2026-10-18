"""Figure and checks: the lower leg's own fold and flip curves (run FM5L) against the followed domain (FA-100b).

The leg's curves come from the period-5 orbit marched by Newton into the leg (68 cells of the follow run's parent
chain) and continued there (tools/explore_llm_soft_loop.py curve; two of the four runs were stopped by their time limit
and rebuilt from their progress logs by tools/curve_from_progress.py). The leg is bounded on its left by a flip curve
(multiplier -1) and on its right by a fold curve (+1). Checks, written to --json:
- strip width: the alpha1 distance from the flip arm to the fold arm at each alpha2, in grid steps;
- membership: every grid cell whose centre lies inside the strip, by its follow-map label, and every period-5 cell of
  the leg (below the cusp at alpha2 0.0961), inside or outside the strip;
- the boundary gate of tools/plot_real_shrimp.py with and without the FM5L curves, in both forms (distance of (5|lost)
  pairs to a fold curve, and to the nearer of a fold or flip curve).

Panels: (a) the leg in the follow map with all computed curves; (b) a zoom with the strip shaded; (c) the strip width
against alpha2; (d) the leading multiplier along the two alpha1 branches across the leg at alpha2 = 0.094.

Usage: python tools/plot_lower_leg.py --json docs/fm5l/lower-leg.json --out paper/llm/figures/fig-lower-leg.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import plot_fm5 as pf  # noqa: E402
import plot_real_shrimp as prs  # noqa: E402

RUN = "llm-soft-loop/pod64/fm5l-a"
OLD = ["llm-soft-loop/pod64/fm5c", "llm-soft-loop/pod64/fm5d", "llm-soft-loop/pod64/fm5x"]
CUSP_TOP = 0.0961            # the located cusp at the top of the leg (FM5d, alpha2 0.096151)


def arr(pts):
    return np.array([[p["alpha"], p["alpha2"]] for p in pts])


def arms(c):
    """The leg's two arms as alpha1(alpha2), increasing in alpha2: each minus run reversed, then its plus run up to the
    first point where alpha2 stops increasing (the bend of the flip curve, the cusp of the fold curve)."""
    out = {}
    for kind, (minus, plus) in (("flip", ("curve-a1-down-minus", "curve-a1-down-plus")),
                                ("fold", ("curve-a1-up-minus", "curve-a1-up-plus"))):
        P = np.vstack([arr(c[minus]["points"])[::-1], arr(c[plus]["points"])[1:]])
        d = np.diff(P[:, 1])
        stop = int(np.argmax(d <= 0)) if np.any(d <= 0) else len(d)
        out[kind] = P[:stop + 1]
    return out


def gate_both(fa, curves):
    a1, a2, lab, per = fa["alpha1"], fa["alpha2"], fa["label"], fa["period"]
    step = float(a1[1] - a1[0])
    folds, flips = prs.curve_points(curves, ("fold",)), prs.curve_points(curves, ("flip",))
    lost, ten = [], []
    n2, n1 = lab.shape
    for i in range(n2):
        for j in range(n1):
            if not (lab[i, j] == 1 and per[i, j] == 5):
                continue
            for di, dj in prs.pf_neighbours():
                ii, jj = i + di, j + dj
                if not (0 <= ii < n2 and 0 <= jj < n1):
                    continue
                mid = ((a1[j] + a1[jj]) / 2, (a2[i] + a2[ii]) / 2)
                if lab[ii, jj] == 0:
                    lost.append((prs.dist_to_polylines(mid, folds, step), prs.dist_to_polylines(mid, flips, step)))
                elif lab[ii, jj] == 1 and per[ii, jj] == 10:
                    ten.append(prs.dist_to_polylines(mid, flips, step))
    lost, ten = np.array(lost), np.array(ten)
    either = lost.min(1)
    return {"5|lost pairs": int(len(lost)), "5|lost within 1 step of a fold": int((lost[:, 0] <= 1).sum()),
            "5|lost within 2 steps of a fold": int((lost[:, 0] <= 2).sum()),
            "5|lost within 1 step of a fold or flip": int((either <= 1).sum()),
            "5|lost within 2 steps of a fold or flip": int((either <= 2).sum()),
            "5|lost farthest from a fold or flip (steps)": float(either.max()),
            "5|10 pairs": int(len(ten)), "5|10 within 1 step of a flip": int((ten <= 1).sum()),
            "5|10 within 2 steps of a flip": int((ten <= 2).sum()), "5|10 farthest (steps)": float(ten.max())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fa", default="llm-soft-loop/pod64/fa/t-0.100")
    ap.add_argument("--json")
    ap.add_argument("--out")
    args = ap.parse_args()
    fa = np.load(pf.DATA / "explore" / args.fa / "follow.npz", allow_pickle=True)
    a1, a2, lab, per = fa["alpha1"], fa["alpha2"], fa["label"], fa["period"]
    st = float(a1[1] - a1[0])
    base = pf.DATA / "explore" / RUN
    c = {d.name: json.loads((d / "curve.json").read_text()) for d in sorted(base.glob("curve-*"))
         if (d / "curve.json").exists()}
    br = {d.name: json.loads((d / "branch.json").read_text()) for d in sorted(base.glob("branch-*"))}
    A = arms(c)
    ff = lambda y: np.interp(y, A["flip"][:, 1], A["flip"][:, 0])        # noqa: E731
    fd = lambda y: np.interp(y, A["fold"][:, 1], A["fold"][:, 0])        # noqa: E731
    lo_a2 = max(A["flip"][0, 1], A["fold"][0, 1])
    # per-point checks of every curve
    checks = {}
    for n, rec in c.items():
        pts = rec["points"]
        target = 1.0 if rec["kind"] == "fold" else -1.0
        checks[n] = {"kind": rec["kind"], "points": len(pts), "reconstructed": bool(rec.get("reconstructed_from_progress")),
                     "stop": rec["stop"], "max_residual": max(p["residual"] for p in pts),
                     "max_abs_g": max(abs(p["g"]) for p in pts),
                     "max_multiplier_offset": max(abs(p["critical_multiplier"][0] - target) + abs(p["critical_multiplier"][1])
                                                  for p in pts),
                     "max_other_spectral_radius": max(p["other_spectral_radius"] or 0.0 for p in pts),
                     "gmres_unconverged": int(sum(p["gmres_unconverged"] for p in pts)),
                     "cusp_candidates": [[p["index"], p["alpha"], p["alpha2"]] for p in pts if p.get("cusp_candidate")],
                     "alpha_range": [min(p["alpha"] for p in pts), max(p["alpha"] for p in pts)],
                     "alpha2_range": [min(p["alpha2"] for p in pts), max(p["alpha2"] for p in pts)]}
    march = br["branch-a1-down"]["march"]
    # strip widths
    widths = {f"{y:.4f}": float((fd(y) - ff(y)) / st) for y in (0.080, 0.082, 0.084, 0.086, 0.088, 0.090, 0.0917, 0.094,
                                                                0.096)}
    # membership
    m5 = (lab == 1) & (per == 5)
    names = {-1: "not computed", 0: "lost", 1: "member", 2: "chaotic, same text", 3: "converging slowly"}
    inside, leg5_out, leg5_in = {}, [], 0
    rows = [i for i in range(len(a2)) if lo_a2 <= a2[i] <= CUSP_TOP - 1e-4]
    for i in rows:
        lo, hi = ff(a2[i]), fd(a2[i])
        for j in range(len(a1)):
            if lo < a1[j] < hi:
                key = names[int(lab[i, j])] + (f", period {int(per[i, j])}" if lab[i, j] == 1 else "")
                inside[key] = inside.get(key, 0) + 1
        for j in np.nonzero(m5[i])[0]:
            if a1[j] > hi + 8 * st or a1[j] < lo - 8 * st:
                continue                                    # cells of the body or another leg, not this leg
            if lo < a1[j] < hi:
                leg5_in += 1
            else:
                leg5_out.append([float(a1[j]), float(a2[i]), float(min(abs(a1[j] - lo), abs(a1[j] - hi)) / st)])
    leg_cells = np.array([[a1[j], a2[i]] for i in rows for j in np.nonzero(m5[i])[0] if ff(a2[i]) - 8 * st <= a1[j] <= fd(a2[i]) + 8 * st])
    lowest = leg_cells[np.argmin(leg_cells[:, 1])].tolist()
    yy = np.linspace(lo_a2, CUSP_TOP, 20001)
    ww = (fd(yy) - ff(yy)) / st
    one_step = float(yy[np.argmax(ww >= 1.0)])                      # lowest alpha2 where the strip is a grid step wide
    _, old, located = pf.load("llm-soft-loop/pod64/fm5", OLD)
    # the returning arms (past the bend of the flip curve and the cusp of the fold curve) against the body's curves of
    # the other kind, alpha2 0.087 to 0.0995 (below the bend, above the end of the shorter returning arm)
    back = {}
    for n, other_kind in (("curve-a1-down-plus", "fold"), ("curve-a1-up-plus", "flip")):
        pts = c[n]["points"]
        k0 = int(np.argmax([p["alpha2"] for p in pts])) if n.endswith("down-plus") else \
            [p["index"] for p in pts if p.get("cusp_candidate")][0]
        Q = arr(pts[k0 + 1:])
        Q = Q[(Q[:, 1] >= 0.087) & (Q[:, 1] <= 0.0995)]
        polys = [arr(r["points"]) for r in old.values() if r["kind"] == other_kind and r["points"]]
        d = np.array([min(prs.dist_to_polylines(tuple(q), [P], st) for P in polys) for q in Q])
        back[n] = {"kind": c[n]["kind"], "compared_with": other_kind, "points": int(len(Q)),
                   "median_steps": float(np.median(d)), "max_steps": float(d.max()), "min_steps": float(d.min())}
    _, new, _ = pf.load("llm-soft-loop/pod64/fm5", OLD + [RUN])
    summary = {"run": RUN, "fa": args.fa, "grid_step": st, "curves": checks,
               "march": {"waypoints": len(march), "max_residual": max(m["residual"] for m in march),
                         "from": [march[0]["alpha"], march[0]["alpha2"]], "to": [march[-1]["alpha"], march[-1]["alpha2"]]},
               "branches": {n: {"crossings": [{k: v for k, v in x.items()} for x in b["crossings"]], "stop": b["stop"],
                                "alpha_range": [b["points"][0]["lambda"], b["points"][-1]["lambda"]]} for n, b in br.items()},
               "strip_width_steps": widths, "rows_checked_alpha2": [float(a2[rows[0]]), float(a2[rows[-1]])],
               "cells_inside_strip": inside, "leg_period5_cells_inside": leg5_in, "leg_period5_cells_outside": leg5_out,
               "lowest_leg_period5_cell": lowest, "strip_one_step_wide_from_alpha2": one_step,
               "returning_arms_vs_body_curves": back,
               "gate_before": gate_both(fa, old), "gate_with_fm5l": gate_both(fa, new)}
    text = json.dumps(summary, indent=1)
    print(text)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text + "\n")
    if not args.out:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    import period_colors as pc
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 10.6), dpi=200)
    axA, axB, axC, axD = axes.ravel()
    ext = [a1[0] - st / 2, a1[-1] + st / 2, a2[0] - st / 2, a2[-1] + st / 2]
    P = np.where(lab == 1, per.astype(float), np.nan)
    cats = pc.categorize([P])

    def cells(ax):
        ax.imshow(np.where(lab == 0, 1.0, np.nan), origin="lower", extent=ext, aspect="auto",
                  cmap=ListedColormap(["0.85"]), interpolation="nearest")
        im = ax.imshow(cats.index(P), origin="lower", extent=ext, aspect="auto", cmap=cats.cmap, norm=cats.norm,
                       interpolation="nearest")
        ax.imshow(np.where(lab == 2, 1.0, np.nan), origin="lower", extent=ext, aspect="auto",
                  cmap=ListedColormap(["k"]), interpolation="nearest")
        ax.imshow(np.where(lab == 3, 1.0, np.nan), origin="lower", extent=ext, aspect="auto",
                  cmap=ListedColormap(["0.5"]), interpolation="nearest")
        return im

    def curves_on(ax, lw=1.4):
        for rec in old.values():
            if rec["points"]:
                Q = arr(rec["points"])
                ax.plot(Q[:, 0], Q[:, 1], "-" if rec["kind"] == "fold" else "--", color="0.55", lw=0.7, zorder=3)
        for n, rec in c.items():
            Q = arr(rec["points"])
            fold = rec["kind"] == "fold"
            ax.plot(Q[:, 0], Q[:, 1], "-" if fold else "--", color="k" if fold else "r", lw=lw, zorder=5)
        for rec in located.values():
            if rec["located"]:
                ax.plot(rec["final"]["alpha"], rec["final"]["alpha2"], "*", color="gold", ms=10, mec="k", mew=0.6, zorder=7)
        for n, ch in checks.items():
            for _, x, y in ch["cusp_candidates"]:
                if ch["kind"] == "fold":
                    ax.plot(x, y, "*", color="w", ms=11, mec="k", mew=0.9, zorder=7)

    cells(axA)
    curves_on(axA)
    M = np.array([[m["alpha"], m["alpha2"]] for m in march])
    axA.plot(M[:, 0], M[:, 1], "o-", color="deeppink", lw=0.9, ms=2.2, mec="w", mew=0.3, zorder=6,
             label="Newton march into the leg (68 cells)")
    axA.plot([], [], "k-", lw=1.4, label="fold curves of the leg (FM5L)")
    axA.plot([], [], "r--", lw=1.4, label="flip curves of the leg (FM5L)")
    axA.plot([], [], "-", color="0.55", lw=0.7, label="curves continued from the body")
    axA.plot([], [], "*", color="gold", mec="k", ms=9, ls="none", label="located cusps")
    axA.plot([], [], "*", color="w", mec="k", ms=9, ls="none", label="cusp flagged by the continuation")
    axA.set_xlim(-0.1305, -0.1180)
    axA.set_ylim(0.0790, 0.1030)
    axA.axhline(a2[0] - st / 2, color="0.4", lw=0.6, ls=":")
    axA.text(-0.1303, a2[0] - st / 2 - 0.0009, "below: not in the follow grid", fontsize=7, color="0.3")
    axA.legend(fontsize=6.5, loc="upper left")
    axA.set_title("(a) the lower leg and its own curves", fontsize=10)
    axA.set_xlabel(r"$\alpha_1$ (gratitude, block 4)")
    axA.set_ylabel(r"$\alpha_2$ (hostility, block 16)")
    cells(axB)
    curves_on(axB, lw=1.2)
    C5 = np.array([[a1[j], a2[i]] for i in rows for j in np.nonzero(m5[i])[0]])
    axB.plot(C5[:, 0], C5[:, 1], "o", color="w", mec="k", mew=0.5, ms=3.2, ls="none", zorder=8,
             label="centres of period-5 cells")
    axB.plot([], [], "r--", lw=1.2, label="flip curve (left edge of the strip)")
    axB.plot([], [], "k-", lw=1.2, label="fold curve (right edge)")
    axB.set_xlim(-0.12655, -0.12320)
    axB.set_ylim(0.0897, 0.0966)
    axB.set_xticks(np.arange(-0.1265, -0.1231, 0.0005))
    axB.legend(fontsize=7, loc="upper left")
    axB.set_title(f"(b) every period-5 cell of the leg lies in the strip ({leg5_in} of {leg5_in + len(leg5_out)})",
                  fontsize=10)
    axB.set_xlabel(r"$\alpha_1$")
    axB.set_ylabel(r"$\alpha_2$")
    axB.ticklabel_format(useOffset=False)
    y = np.linspace(lo_a2 + 1e-6, 0.0975, 800)
    w = (fd(y) - ff(y)) / st
    axC.semilogy(y, w, "k-", lw=1.4)
    axC.axhline(1.0, color="tab:blue", lw=0.8, ls="--")
    axC.text(0.0802, 1.12, "one grid step", color="tab:blue", fontsize=8)
    axC.axvline(lowest[1], color="tab:green", lw=0.8, ls=":")
    axC.text(lowest[1] + 0.0002, 0.012, "lowest leg cell\nreached by the follow run", color="tab:green", fontsize=8)
    axC.axvline(a2[0] - st / 2, color="0.4", lw=0.6, ls=":")
    axC.text(a2[0] - st / 2 - 0.0002, 2.2, "bottom of\nthe follow grid", color="0.3", fontsize=8, ha="right")
    axC.set_xlabel(r"$\alpha_2$")
    axC.set_ylabel(r"width of the strip, fold minus flip in $\alpha_1$ (grid steps)")
    axC.set_title("(c) the leg narrows below one grid step and continues", fontsize=10)
    for n, b in sorted(br.items()):
        lam = np.array([p["lambda"] for p in b["points"]])
        mu = np.array([max(p["multipliers_refined"] or p["multipliers"], key=lambda z: z[0] ** 2 + z[1] ** 2)[0]
                       for p in b["points"]])
        down = n.endswith("down")
        if down:
            axD.plot(lam, mu, "-", color="tab:purple", lw=1.4, label="branch toward smaller $\\alpha_1$")
        else:
            k = int(np.argmax(np.diff(lam) < 0))
            axD.plot(lam[:k + 1], mu[:k + 1], "-", color="tab:orange", lw=1.4, label="branch toward larger $\\alpha_1$")
            axD.plot(lam[k:], mu[k:], ":", color="tab:orange", lw=1.2, label="past the fold: the saddle")
    axD.axhline(1.0, color="k", lw=0.6)
    axD.axhline(-1.0, color="r", lw=0.6, ls="--")
    axD.axvspan(ff(0.094), fd(0.094), color="gold", alpha=0.45, lw=0)
    axD.axvline(march[-1]["alpha"], color="tab:blue", lw=0.6, ls=":")
    axD.set_xlim(-0.12580, -0.12430)
    axD.set_ylim(-15, 4)
    axD.set_xlabel(r"$\alpha_1$ at $\alpha_2 = 0.094$")
    axD.set_ylabel("leading multiplier of the period-5 orbit")
    axD.legend(fontsize=7, loc="lower left")
    axD.set_title("(d) across the leg: from +1 at the fold to -1 at the flip", fontsize=10)
    axD.ticklabel_format(useOffset=False)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")


if __name__ == "__main__":
    main()
