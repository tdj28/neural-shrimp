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
TB_WIDE = "llm-soft-loop/pod64/tb/zg-grat4-host16-t-0.1/tb-t-0.100.npz"
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


def load(run, extra=()):
    """Branches and curves of the run, curves of further runs (prefixed with the run's folder name), and located
    points (locate-*/locate.json, from tools/cusp_locate.py) of all runs."""
    branches, curves, located = {}, {}, {}
    for k, r in enumerate((run, *extra)):
        base = DATA / "explore" / r
        pre = "" if k == 0 else base.name + "/"
        if k == 0:
            for d in sorted(base.glob("branch-*")):
                if (d / "branch.json").exists():
                    branches[d.name] = json.loads((d / "branch.json").read_text())
        for d in sorted(base.glob("curve-*")):
            if (d / "curve.json").exists():
                curves[pre + d.name] = json.loads((d / "curve.json").read_text())
        for d in sorted(base.glob("locate-*")):
            if (d / "locate.json").exists():
                located[pre + d.name] = json.loads((d / "locate.json").read_text())
    return branches, curves, located


def locate_summary(rec):
    fin = rec["final"]
    return {"curve": rec["curve"], "left": rec["left"], "located": rec["located"], "checks": rec["checks"],
            "alpha": fin.get("alpha"), "alpha2": fin.get("alpha2"), "f": fin["f"], "initial_f_max": rec["initial_f_max"],
            "b": fin.get("b"), "c": fin.get("c"), "c_plateau_relative_spread": fin.get("c_plateau_relative_spread"),
            "det_J": fin.get("det_J"), "condition": fin.get("condition"), "multiplier": fin.get("multiplier"),
            "second_modulus": fin.get("second_modulus"), "normal_form_parameter": rec.get("normal_form_parameter")}


def self_crossings(P):
    """Crossings of a polyline with itself (non-adjacent segments), as (i, j, point)."""
    out = []
    for i in range(len(P) - 1):
        for j in range(i + 2, len(P) - 1):
            p1, p2, q1, q2 = P[i], P[i + 1], P[j], P[j + 1]
            d = (p2[0] - p1[0]) * (q2[1] - q1[1]) - (p2[1] - p1[1]) * (q2[0] - q1[0])
            if abs(d) < 1e-30:
                continue
            t = ((q1[0] - p1[0]) * (q2[1] - q1[1]) - (q1[1] - p1[1]) * (q2[0] - q1[0])) / d
            u = ((q1[0] - p1[0]) * (p2[1] - p1[1]) - (q1[1] - p1[1]) * (p2[0] - p1[0])) / d
            if 0 <= t <= 1 and 0 <= u <= 1:
                out.append((i, j, (p1 + t * (p2 - p1)).tolist()))
    return out


def figure(branches, curves, located, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    sys.path.insert(0, str(ROOT / "tools"))
    import dss_window as dw
    import period_colors as pc

    tb = np.load(DATA / "explore" / TB, allow_pickle=True)
    wide = np.load(DATA / "explore" / TB_WIDE, allow_pickle=True)
    z = np.load(DATA / "explore" / DSS / "grid.npz", allow_pickle=True)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), dpi=170)
    axes = axes.ravel()
    loc = [(r["final"]["alpha"], r["final"]["alpha2"], r["located"]) for r in located.values()]

    def is_located(a1, a2):
        return any(abs(a1 - x) < 1e-4 and abs(a2 - y) < 1e-4 and ok for x, y, ok in loc)   # within a curve step

    def stars(ax, ms):
        for x, y, ok in loc:
            ax.plot(x, y, "*", color="gold" if ok else "w", ms=ms, mec="k", mew=0.6, zorder=6)
        for rec in curves.values():                # reversals not (yet) located: white stars
            for q in rec["points"]:
                if q["cusp_candidate"] and not is_located(q["alpha"], q["alpha2"]):
                    ax.plot(q["alpha"], q["alpha2"], "*", color="w", ms=ms, mec="k", mew=0.6, zorder=6)

    def overlay(ax, ms=9, branches_too=True):
        for name, rec in sorted(curves.items()):
            pts = rec["points"]
            if not pts:
                continue
            fold = rec["kind"] == "fold"
            ax.plot([q["alpha"] for q in pts], [q["alpha2"] for q in pts], "-" if fold else "--",
                    color="k" if fold else "r", lw=1.3 if fold else 1.1)
        stars(ax, ms)
        if branches_too:
            for name, rec in sorted(branches.items()):
                ta = rec["args"]
                lam = [q["lambda"] for q in rec["points"]]
                if rec["free"] == "alpha":
                    ax.plot(lam, [ta["alpha2"]] * len(lam), ":", color="w", lw=0.8)
                else:
                    ax.plot([ta["alpha"]] * len(lam), lam, ":", color="w", lw=0.8)
            if branches:
                rec = next(iter(branches.values()))
                ax.plot(rec["args"]["alpha"], rec["args"]["alpha2"], "o", color="w", mec="k", ms=5, zorder=6)

    def periods(m):
        P = np.where(m["chaotic"], np.nan, m["period"].astype(float))
        P[P <= 0] = np.nan
        return P

    cats = pc.categorize([periods(tb), periods(wide)], max_blocks=9)      # one colour scheme for both maps

    def attractor_map(ax, m, label):
        P = periods(m)
        b1, b2 = m["alpha1"], m["alpha2"]
        e = [b1[0], b1[-1], b2[0], b2[-1]]
        ax.imshow(np.where(m["chaotic"], 1.0, np.nan), origin="lower", extent=e, aspect="auto",
                  cmap=ListedColormap(["0.15"]), interpolation="nearest")
        cmap = cats.cmap.copy()
        cmap.set_bad((0, 0, 0, 0))
        im = ax.imshow(cats.index(P), origin="lower", extent=e, aspect="auto", cmap=cmap, norm=cats.norm,
                       interpolation="nearest")
        if label:
            cats.colorbar(im, ax=ax, label="period (black: chaotic)")
        return e

    def box(ax, xs, ys, **kw):
        ax.plot([xs[0], xs[1], xs[1], xs[0], xs[0]], [ys[0], ys[0], ys[1], ys[1], ys[0]], "-", **kw)

    a1, a2 = z["alpha"], z["log10_temperature"]
    ax = axes[0]                                   # (a) all curves over the wide zoom
    ax.set_facecolor("0.93")
    attractor_map(ax, wide, True)
    overlay(ax, ms=8, branches_too=False)
    for rec in curves.values():                    # self-crossings of every fold curve (the dovetails)
        if rec["kind"] == "fold" and rec["points"]:
            for i, j, x in self_crossings(np.array([[q["alpha"], q["alpha2"]] for q in rec["points"]])):
                ax.plot(*x, "x", color="m", ms=7, mew=1.5, zorder=7)
    box(ax, (tb["alpha1"][0], tb["alpha1"][-1]), (tb["alpha2"][0], tb["alpha2"][-1]), color="c", lw=0.9)
    allx = [q["alpha"] for r in curves.values() for q in r["points"]]
    ally = [q["alpha2"] for r in curves.values() for q in r["points"]]
    ax.set_xlim(min(allx) - 0.001, max(max(allx), wide["alpha1"][-1]) + 0.0005)
    ax.set_ylim(min(min(ally), wide["alpha2"][0]) - 0.0005, max(ally) + 0.001)
    ax.set_title("(a) fold (black) and flip (red) curves of the period-5 orbit; x: self-crossings; cyan: zoom of (b)",
                 fontsize=7.5)

    ax = axes[1]                                   # (b) the body over the fine zoom
    e2 = attractor_map(ax, tb, True)
    overlay(ax)
    box(ax, (a1[0], a1[-1]), (a2[0], a2[-1]), color="w", lw=0.8)
    ax.set_xlim(e2[0], e2[1])
    ax.set_ylim(e2[2], e2[3])
    ax.set_title("(b) the body; dotted: branches from a common start; white box: grid of (c)", fontsize=7.5)

    ax = axes[2]                                   # (c) over the continued orbit's trace
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
    fig.colorbar(im, ax=ax, fraction=0.046, label=r"trace $\mu_1+\mu_2$ of the continued orbit")
    ax.contour(a1, a2, np.where(ok, t, np.nan), levels=[0], colors="0.3", linewidths=0.6)
    overlay(ax)
    pad1, pad2 = 4 * da1, 4 * da2
    ax.set_xlim(ext[0] - pad1, ext[1] + pad1)
    ax.set_ylim(ext[2] - pad2, ext[3] + pad2)
    ax.set_title("(c) the curves over the continued orbit's trace (gray: repelling)", fontsize=7.5)

    ax = axes[3]                                   # (d) the junction of the upper leg with the body
    styles = {"curve-a1-up-fold-minus": ("k", "-", 1.6), "curve-a2-up-fold-plus": ("tab:orange", "--", 1.0)}
    for name, rec in sorted(curves.items()):
        pts = rec["points"]
        if not pts:
            continue
        col, ls, lw = styles.get(name, ("k" if rec["kind"] == "fold" else "r", "-" if rec["kind"] == "fold" else "--", 0.9))
        ax.plot([q["alpha"] for q in pts], [q["alpha2"] for q in pts], ls, color=col, lw=lw, marker=".", ms=2.5)
    fold = [r for n, r in curves.items() if n == "curve-a1-up-fold-minus"]
    if fold:
        P = np.array([[q["alpha"], q["alpha2"]] for q in fold[0]["points"]])
        for i, j, x in self_crossings(P):
            ax.plot(*x, "x", color="m", ms=8, mew=1.6, zorder=7)
    stars(ax, 12)
    census = ROOT / "docs" / "oc" / "oc-summary.json"          # run OC: period-5 orbits found (attracting) per point
    if census.exists():
        for q in json.loads(census.read_text())["points"]:
            ax.plot(q["alpha"], q["alpha2"], "o", ms=4.5, color="tab:green" if q["match"] else "tab:orange", mec="k",
                    mew=0.4, zorder=8)
            ax.annotate(f"{q['found']} ({q['attracting']})", (q["alpha"], q["alpha2"]), xytext=(3, 2),
                        textcoords="offset points", fontsize=6.5, zorder=9)
    ax.set_xlim(-0.11935, -0.11880)
    ax.set_ylim(0.10255, 0.10425)
    ax.set_title("(d) junction of the upper leg: cusps (stars), crossing (x), orbits found (attracting)", fontsize=7.5)
    ax.ticklabel_format(useOffset=False)
    for ax in axes:
        ax.set_xlabel(r"$\alpha_1$ (gratitude, block 4)", fontsize=8)
        ax.set_ylabel(r"$\alpha_2$ (hostility, block 16)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", default="llm-soft-loop/pod64/fm5")
    ap.add_argument("--extra-run", nargs="*", default=[], help="further runs whose curves and located points are added")
    ap.add_argument("--json")
    ap.add_argument("--out")
    args = ap.parse_args()
    branches, curves, located = load(args.run, args.extra_run)
    if not branches and not curves:
        raise SystemExit(f"no branch or curve results under {args.run}")
    summary = {"run": args.run, "extra_runs": args.extra_run,
               "branches": {k: branch_summary(v) for k, v in branches.items()},
               "curves": {k: curve_summary(v) for k, v in curves.items()},
               "located": {k: locate_summary(v) for k, v in located.items()}}
    summary["fold_self_crossings"] = {                 # every fold curve (two traces of one curve repeat a crossing)
        name: [{"segments": [i, j], "at": x} for i, j, x in
               self_crossings(np.array([[p["alpha"], p["alpha2"]] for p in rec["points"]]))]
        for name, rec in curves.items() if rec["kind"] == "fold" and rec["points"]}
    text = json.dumps(summary, indent=1)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text + "\n")
    print(text)
    if args.out:
        figure(branches, curves, located, args.out)


if __name__ == "__main__":
    main()
