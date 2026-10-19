"""The anatomy of a shrimp's cell in the classical one-dimensional models, computed (figure fig:anatomy).

(a) Gallas's canonical period-1 cell: x -> (a - x^2)^2 - b, his eq. (5.3) rotated by 45 degrees as in his Fig. 7
(Gallas 1994, Physica A 202:196). Fold (multiplier +1), flip (-1) and superstable (0) curves are parametrized by the
fixed point x; the shading counts attracting fixed points on a grid. Labels: head H (the superstable curves of the two
critical points cross), cusp C of the fold curve, kite C-A-R-B where the two legs cross, tail T (superstable curves of
two different fixed points cross).
(b) The quartic model of Carcasses et al. (1991), x -> c + (1+a) x + b x^2 + x^4, in the plane of a and c at fixed
b = -1: its fixed-point fold curve has two cusps sharing a segment and a self-crossing, the "dovetail" of Mira et al.
(1996), around a pocket with four fixed points.

Both fixed-point equations are quartics x^4 + u x^2 + v x + w = 0, whose fold set in (u, v, w) is the swallowtail:
(a) is the section v = -1 (one cusp), (b) the section u = -1 (two cusps and a crossing).

Pair layout (--layout pair, the article's figure): (a) Gallas's cell with both axes reversed, a rotation by 180 degrees
that brings it to the orientation of the period-6 shrimp of the soft-token loop, drawn beside it in (b): the 640 x 640
float64 map HR10 with the continued fold and flip curves, the located cusp C and the pseudo-crossing R of the two flip
curves, gated as in plot_p6_shrimp.py (residual <= 1e-8, critical multiplier within 1e-3 of +-1).

Usage: python tools/plot_shrimp_anatomy.py [--out PNG] [--json OUT_JSON]
       python tools/plot_shrimp_anatomy.py --layout pair --out paper/llm/figures/fig-shrimp-anatomy-pair.png \
         --json paper/llm/figures/fig-shrimp-anatomy-pair.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.colors import ListedColormap                         # noqa: E402


def gallas(x, a, b):
    return (a - x * x) ** 2 - b


def gallas_mult(x, a):
    return -4.0 * x * (a - x * x)


def attracting_counts(coeffs_fn, mult_fn, A, B):
    """Number of real fixed points with |multiplier| < 1 at each grid point; also the number of real fixed points."""
    na = np.zeros(A.shape, dtype=int)
    nr = np.zeros(A.shape, dtype=int)
    for idx in np.ndindex(A.shape):
        r = np.roots(coeffs_fn(A[idx], B[idx]))
        r = r[np.abs(r.imag) < 1e-9].real
        nr[idx] = len(r)
        na[idx] = int(np.sum(np.abs(mult_fn(r, A[idx], B[idx])) < 1.0))
    return na, nr


def curve_gallas(target, xs):
    """Points (a, b) where a fixed point x of Gallas's map has multiplier `target` (+1 fold, -1 flip)."""
    # -4x(a - x^2) = target  ->  a = x^2 - target / (4x);  fixed point: b = (a - x^2)^2 - x
    a = xs * xs - target / (4.0 * xs)
    b = (a - xs * xs) ** 2 - xs
    return a, b


def curve_quartic(target, xs, bq):
    """Carcasses et al.'s map at fixed b = bq: points (a, c) where a fixed point x has multiplier `target`."""
    a = target - 1.0 - 2.0 * bq * xs - 4.0 * xs ** 3
    c = -xs ** 4 - bq * xs ** 2 - a * xs
    return a, c


def intersect(p, q):
    """Intersections of two polylines (arrays of shape (n, 2))."""
    out = []
    for i in range(len(p) - 1):
        for j in range(len(q) - 1):
            p0, p1, q0, q1 = p[i], p[i + 1], q[j], q[j + 1]
            d = (p1[0] - p0[0]) * (q1[1] - q0[1]) - (p1[1] - p0[1]) * (q1[0] - q0[0])
            if abs(d) < 1e-15:
                continue
            t = ((q0[0] - p0[0]) * (q1[1] - q0[1]) - (q0[1] - p0[1]) * (q1[0] - q0[0])) / d
            s = ((q0[0] - p0[0]) * (p1[1] - p0[1]) - (q0[1] - p0[1]) * (p1[0] - p0[0])) / d
            if 0 <= t <= 1 and 0 <= s <= 1:
                out.append(p0 + t * (p1 - p0))
    return out


def draw_gallas(ax, rec, cmap, pair=False):
    """Panel (a): Gallas's canonical cell. pair: smaller type for the article's two-panel figure, and both axes
    reversed, a rotation by 180 degrees that brings the cell to the orientation of the period-6 shrimp beside it."""
    ext = (-1.2, 2.2, -1.2, 2.2)
    av, bv = np.linspace(ext[0], ext[1], 341), np.linspace(ext[2], ext[3], 341)
    A, Bm = np.meshgrid(av, bv)
    na, _ = attracting_counts(lambda a, b: [1.0, 0.0, -2.0 * a, -1.0, a * a - b],
                              lambda r, a, b: gallas_mult(r, a), A, Bm)
    ax.imshow(np.minimum(na, 2), origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=2, interpolation="nearest",
              aspect="auto")
    xneg1 = -np.geomspace(1e-3, 3.0, 4000)[::-1]          # x < 0 (from -3 to -1e-3)
    xpos = np.geomspace(1e-3, 3.0, 4000)
    lines = {}
    for name, target, col, ls in (("fold", 1.0, "k", "-"), ("flip", -1.0, "r", "--")):
        for side, xs in (("neg", xneg1), ("pos", xpos)):
            a, b = curve_gallas(target, xs)
            ok = (a > ext[0] - 1) & (a < ext[1] + 1) & (b > ext[2] - 1) & (b < ext[3] + 1)
            ax.plot(a[ok], b[ok], ls, color=col, lw=(1.0 if pair else 1.3) if name == "fold" else (0.9 if pair else 1.1))
            lines[(name, side)] = np.c_[a, b]
    s = np.linspace(-1.5, 1.5, 600)
    ax.plot(s, s * s, ":", color="#1f4e9a", lw=1.1 if pair else 1.4)    # x = 0 superstable: b = a^2
    ax.plot(s * s, s, ":", color="#1f4e9a", lw=1.1 if pair else 1.4)    # x = -+sqrt(a): a = b^2
    # noble points
    xc = -0.5
    C = np.array([xc * xc - 1 / (4 * xc), 1 / (16 * xc * xc) - xc])
    H, T = np.array([0.0, 0.0]), np.array([1.0, 1.0])
    foldc = lines[("fold", "neg")]
    steep = foldc[(xneg1 > -0.5)]
    shallow = foldc[(xneg1 < -0.5)]
    flip_u = lines[("flip", "neg")]
    flip_c = lines[("flip", "pos")]
    Apt = intersect(steep, flip_u)
    Bpt = intersect(shallow, flip_c)
    Rpt = [p for p in intersect(flip_u, flip_c) if p[0] > 0.9]
    pts = {"H": H, "C": C, "T": T, "A": Apt[0] if Apt else None, "B": Bpt[0] if Bpt else None,
           "R": Rpt[0] if Rpt else None}
    for k, p in pts.items():
        if p is None:
            raise SystemExit(f"noble point {k} not found")
        ax.plot(*p, "o", ms=3.5 if pair else 4.5, color="#c00000" if k in "HT" else "k", zorder=5)
        if pair:                 # centred labels, offsets in data units: they turn with the reversed axes
            off = PAIR_OFFSETS[k]
            ax.text(p[0] + off[0], p[1] + off[1], k, fontsize=8, weight="bold", ha="center", va="center")
            continue
        off = {"H": (-0.22, -0.18), "C": (-0.24, 0.02), "T": (0.05, -0.17), "A": (-0.2, 0.06), "B": (0.05, -0.15),
               "R": (0.06, 0.02)}[k]
        ax.text(p[0] + off[0], p[1] + off[1], k, fontsize=10, weight="bold")
    rec["gallas"] = {k: [round(float(v), 6) for v in p] for k, p in pts.items()}
    if pair:
        for x, y, word in PAIR_WORDS:
            ax.text(x, y, word, fontsize=7, ha="center", va="center")
        ax.annotate("kite", xy=(0.97, 1.08), xytext=PAIR_KITE, fontsize=7, ha="center", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.5, shrinkA=1, shrinkB=0))
        ax.set_xlim(ext[1], ext[0]); ax.set_ylim(ext[3], ext[2])            # both reversed: turned by 180 degrees
        ax.set_xlabel("$a$"); ax.set_ylabel("$b$")
        ax.set_title("(a) Gallas's period-1 cell", fontsize=8)
        return
    ax.annotate("leg", xy=(1.55, 2.05), fontsize=9, ha="center")
    ax.annotate("leg", xy=(2.0, 1.36), fontsize=9, ha="center")
    ax.text(-0.95, -0.35, "main cell", fontsize=9)
    ax.annotate("kite", xy=(1.02, 1.03), xytext=(1.45, 0.62), fontsize=9, arrowprops=dict(arrowstyle="-", lw=0.6))
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("$a$"); ax.set_ylabel("$b$")
    ax.set_title("(a) Gallas's period-1 cell: $x\\mapsto(a-x^2)^2-b$", fontsize=10)



def draw_quartic(ax, rec, cmap):
    """Panel (b) of the classic layout: Carcasses et al.'s quartic at fixed b = -1, the dovetail."""
    bq = -1.0
    ext = (-1.1, 1.1, -0.45, 0.6)
    av, cv = np.linspace(ext[0], ext[1], 341), np.linspace(ext[2], ext[3], 341)
    A, Cg = np.meshgrid(av, cv)
    na, nr = attracting_counts(lambda a, c: [1.0, 0.0, bq, a, c],
                               lambda r, a, c: 1.0 + a + 2.0 * bq * r + 4.0 * r ** 3, A, Cg)
    ax.imshow(np.minimum(na, 2), origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=2, interpolation="nearest",
              aspect="auto")
    xs = np.linspace(-1.6, 1.6, 6000)
    for target, col, ls in ((1.0, "k", "-"), (-1.0, "r", "--")):
        a, c = curve_quartic(target, xs, bq)
        ok = (a > ext[0] - 0.5) & (a < ext[1] + 0.5) & (c > ext[2] - 0.5) & (c < ext[3] + 0.5)
        ax.plot(a[ok], c[ok], ls, color=col, lw=1.3 if target > 0 else 1.1)
    a0, c0 = curve_quartic(0.0, xs, bq)
    ax.plot(a0, c0, ":", color="#1f4e9a", lw=1.4)
    t = 1 / np.sqrt(6)
    U, L = np.array([-4 * t ** 3 + 2 * t, 3 * t ** 4 - t * t]), np.array([4 * t ** 3 - 2 * t, 3 * t ** 4 - t * t])
    X = np.array([0.0, 0.25])
    for k, p, off in (("$C_1$", U, (0.03, -0.05)), ("$C_2$", L, (-0.1, -0.05)), ("X", X, (0.03, 0.02))):
        ax.plot(*p, "*" if k != "X" else "x", ms=11 if k != "X" else 9, color="gold" if k != "X" else "m",
                mec="k", mew=0.6, zorder=5)
        ax.text(p[0] + off[0], p[1] + off[1], k, fontsize=10, weight="bold")
    # fixed-point counts in the regions
    for (a, c) in ((0.0, 0.03), (0.0, 0.45), (0.0, -0.35), (0.85, 0.15), (-0.85, 0.15)):
        i, j = np.argmin(np.abs(cv - c)), np.argmin(np.abs(av - a))
        ax.text(a, c, f"{nr[i, j]}", fontsize=11, ha="center", va="center", weight="bold", color="#333333")
        rec.setdefault("quartic_counts", []).append({"a": a, "c": c, "fixed_points": int(nr[i, j]),
                                                     "attracting": int(na[i, j])})
    rec["quartic"] = {"C1": U.round(6).tolist(), "C2": L.round(6).tolist(), "X": X.tolist()}
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("$a$"); ax.set_ylabel("$c$")
    ax.set_title("(b) $x\\mapsto c+(1+a)x-x^2+x^4$: the dovetail", fontsize=10)


# pair layout: label offsets (data units, centred labels), words and the kite's label, placed for the reversed axes
PAIR_OFFSETS = {"H": (-0.16, -0.14), "C": (-0.14, -0.13), "T": (0.08, -0.15), "A": (-0.13, 0.08), "B": (0.1, -0.1),
                "R": (0.1, 0.1)}
PAIR_WORDS = ((1.17, 2.02, "leg"), (2.02, 1.3, "leg"), (-0.2, 0.55, "main cell"))
PAIR_KITE = (0.5, 1.32)


def draw_shrimp(ax, args, rec):
    """Panel (b) of the pair layout: the fine float64 map of the period-6 shrimp with its continued fold and flip curves,
    the located cusps and the pseudo-crossing R, drawn and gated as by plot_p6_shrimp.py (the markers are named in the
    caption: gold star, the kite's cusp C; cyan diamond, R)."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import plot_p6_shrimp as p6

    img, ext, ly, chaos, _ = p6.map_image(args.scan)
    ax.imshow(img, origin="lower", aspect="auto", interpolation="nearest", extent=ext)
    curves, report = p6.load_curves(args.curves, 1e-8, 1e-3)          # the gate of plot_p6_shrimp.py's defaults
    located = sorted((p6.DATA / "explore" / args.curves).glob("locate-*/locate.json"))
    for extra in args.extra:
        c2, r2 = p6.load_curves(extra, 1e-8, 1e-3)
        curves += c2
        report += r2
        located += sorted((p6.DATA / "explore" / extra).glob("locate-*/locate.json"))
    p6.draw_curves(ax, curves, lw=0.9, ms=6)
    p6.draw_located(ax, located, ms=8)
    ax.plot(*args.pseudo, "D", color="cyan", mec="k", mew=0.6, ms=4.5, zorder=8)
    cusps = [json.loads(f.read_text())["final"] for f in located if json.loads(f.read_text()).get("located")]
    inside = [(c["alpha"], c["log10_temperature"]) for c in cusps
              if ext[0] <= c["alpha"] <= ext[1] and ext[2] <= c["log10_temperature"] <= ext[3]]
    if len(inside) != 1:
        raise SystemExit(f"expected one located cusp (the kite's C) in the window, found {len(inside)}")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel(r"steering strength $\alpha$"); ax.set_ylabel(r"$\log_{10}T$")
    ax.locator_params(axis="x", nbins=5)
    ax.set_title("(b) the period-6 shrimp", fontsize=8)
    rec["shrimp"] = {"scan": args.scan, "cells": int(ly.size), "chaotic": int(chaos.sum()), "window": [float(e) for e in ext],
                     "cusp_C": [float(v) for v in inside[0]], "pseudo_R": list(args.pseudo), "curves": report}


def pair_figure(args):
    """The article's figure: (a) Gallas's cell turned by 180 degrees beside (b) the period-6 shrimp."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rec = {}
    plt.rcParams.update({"font.size": 8, "xtick.labelsize": 7, "ytick.labelsize": 7})
    cmap = ListedColormap(["white", "#d9d9d9", "#8fb3d9"])            # 0, 1, 2 attracting fixed points
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.55))
    draw_gallas(axes[0], rec, cmap, pair=True)
    draw_shrimp(axes[1], args, rec)
    handles = [Line2D([], [], color="k", lw=1.0, label="fold ($\\mu=+1$)"),
               Line2D([], [], color="r", ls="--", lw=0.9, label="flip ($\\mu=-1$)"),
               Line2D([], [], color="#1f4e9a", ls=":", lw=1.1, label="superstable ($\\mu=0$)"),
               Patch(color="#d9d9d9", label="one attracting fixed point"),
               Patch(color="#8fb3d9", label="two attracting fixed points")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.5, frameon=False, handlelength=1.8,
               columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.06, 1, 1), w_pad=1.5)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=400)
    text = json.dumps(rec, indent=1)
    if args.json:
        Path(args.json).write_text(text + "\n")
    print(text[:1500])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="paper/llm/figures/fig-shrimp-anatomy.png")
    ap.add_argument("--json")
    ap.add_argument("--layout", choices=("classic", "pair"), default="classic",
                    help="classic: (a) Gallas's cell, (b) the dovetail; pair: (a) Gallas's cell turned by 180 degrees, (b) "
                    "the period-6 shrimp of the language model (the article's figure)")
    ap.add_argument("--scan", default="llm-soft-loop/pod64/hr10", help="pair: explore-relative folder of the map of (b)")
    ap.add_argument("--curves", default="llm-soft-loop/pod64/p6s", help="pair: the continued curves and located cusps")
    ap.add_argument("--extra", nargs="*", default=["llm-soft-loop/pod64/kite"], help="pair: further curves and cusps")
    ap.add_argument("--pseudo", type=float, nargs=2, default=(0.072912, -0.154104), metavar=("ALPHA", "LOG10T"),
                    help="pair: the verified pseudo-crossing R of the two flip curves (cyan diamond)")
    args = ap.parse_args()
    if args.layout == "pair":
        return pair_figure(args)
    rec = {}
    cmap = ListedColormap(["white", "#d9d9d9", "#8fb3d9"])            # 0, 1, 2 attracting fixed points
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), dpi=170)

    draw_gallas(axes[0], rec, cmap)
    draw_quartic(axes[1], rec, cmap)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="k", lw=1.3, label="fold ($\\mu=+1$)"),
               Line2D([], [], color="r", ls="--", lw=1.1, label="flip ($\\mu=-1$)"),
               Line2D([], [], color="#1f4e9a", ls=":", lw=1.4, label="superstable ($\\mu=0$)"),
               Patch(color="#d9d9d9", label="one attracting fixed point"),
               Patch(color="#8fb3d9", label="two attracting fixed points")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    text = json.dumps(rec, indent=1)
    if args.json:
        Path(args.json).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
