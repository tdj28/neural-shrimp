"""The three kinds of organization around a cusp of a fold curve (Mira 1987; Carcasses et al. 1991), computed from
the quartic map x -> c + (1+a)x + bx^2 + x^4 in planes of b and c at fixed a, as Carcasses et al. drew them
(figure fig:cuspkinds).

The fixed points solve x^4 + bx^2 + ax + c = 0; the multiplier is m = 1 + a + 2bx + 4x^3. Fold (m = +1), flip
(m = -1) and superstable (m = 0) curves are parametrized by the fixed point x; the shading counts attracting fixed
points (|m| < 1) on a grid. The cusp lies at x = t with a = 8t^3; its cubic coefficient is c3 = 4t.
- a = -1.1 (t < 0, c3 < 0): crossroad area; passing through the cusp splits a node into a saddle and two nodes, and
  the flip curves of the cusp's two branches cross.
- a = -2.05 (c3 < 0): spring area; a flip curve forms a loop around the cusp.
- a = +1.1 (t > 0, c3 > 0): saddle-area kind; passing through the cusp splits a saddle into a node and two saddles, and
  the attracting fixed point exists only inside the cusp's wedge.
Axis limits come from the geometry of each panel (cusp, wedge, the flip structure that defines the kind) with a
margin, so that no defining feature is clipped; the script refuses to draw a panel whose feature leaves the frame.

Usage: python tools/plot_cusp_kinds.py [--out PNG] [--json OUT_JSON]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.colors import ListedColormap                         # noqa: E402

XS = np.concatenate([-np.geomspace(4.0, 1e-3, 40000), np.geomspace(1e-3, 4.0, 40000)])


def curve(a, target, xs=XS):
    b = (target - 1.0 - a - 4.0 * xs ** 3) / (2.0 * xs)
    c = -xs ** 4 - b * xs ** 2 - a * xs
    return b, c


def counts(a, B, C):
    na = np.zeros(B.shape, int)
    for idx in np.ndindex(B.shape):
        r = np.roots([1.0, 0.0, B[idx], a, C[idx]])
        r = r[np.abs(r.imag) < 1e-9].real
        na[idx] = int(np.sum(np.abs(1.0 + a + 2.0 * B[idx] * r + 4.0 * r ** 3) < 1.0))
    return na


def crossings(P, Q, same=False, with_index=False):
    out = []
    for i in range(len(P) - 1):
        p0, p1 = P[i], P[i + 1]
        js = range(i + 2, len(Q) - 1) if same else range(len(Q) - 1)
        for j in js:
            q0, q1 = Q[j], Q[j + 1]
            d = (p1[0] - p0[0]) * (q1[1] - q0[1]) - (p1[1] - p0[1]) * (q1[0] - q0[0])
            if abs(d) < 1e-15:
                continue
            t = ((q0[0] - p0[0]) * (q1[1] - q0[1]) - (q0[1] - p0[1]) * (q1[0] - q0[0])) / d
            u = ((q0[0] - p0[0]) * (p1[1] - p0[1]) - (q0[1] - p0[1]) * (p1[0] - p0[0])) / d
            if 0 <= t <= 1 and 0 <= u <= 1:
                out.append((i, j, p0 + t * (p1 - p0)) if with_index else p0 + t * (p1 - p0))
    return out


def thin(b, c, box, step=4):
    ok = (b > box[0]) & (b < box[1]) & (c > box[2]) & (c < box[3])
    return np.c_[b, c][ok][::step]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="paper/llm/figures/fig-cusp-kinds.png")
    ap.add_argument("--json")
    args = ap.parse_args()
    cmap = ListedColormap(["white", "#d9d9d9", "#8fb3d9"])
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.9), dpi=170)
    rec = {}
    for ax, (a, kind, split) in zip(axes, [(-1.1, "crossroad area", "node $\\to$ saddle + 2 nodes"),
                                           (-2.05, "spring area", "node $\\to$ saddle + 2 nodes"),
                                           (1.1, "saddle area", "saddle $\\to$ node + 2 saddles")]):
        t = np.cbrt(a / 8.0)
        cusp = np.array([-6.0 * t * t, -3.0 * t ** 4])
        neg, pos = XS < 0, XS > 0
        bf, cf = curve(a, -1.0)
        feature = [cusp]
        big = (-6.0, 3.0, -3.0, 3.0)
        if kind == "crossroad area":       # where the flip curves of the two branches cross, nearest the cusp
            xs = crossings(thin(bf[neg], cf[neg], big), thin(bf[pos], cf[pos], big))
            if not xs:
                raise SystemExit("crossroad: no crossing of the two flip branches found")
            X = min(xs, key=lambda p: np.hypot(*(p - cusp)))
            feature.append(X)
            rec["crossroad_flip_crossing"] = [round(float(v), 4) for v in X]
        if kind == "spring area":          # the loop of the x < 0 flip branch: from its self-crossing, all the way round
            P = thin(bf[neg], cf[neg], big, step=8)
            xs = crossings(P, P, same=True, with_index=True)
            if not xs:
                raise SystemExit("spring: the flip branch has no self-crossing (no loop)")
            i, j, X = min(xs, key=lambda r: np.hypot(*(r[2] - cusp)))
            loop = np.vstack([X, P[i + 1:j + 1], X])            # the closed loop between the two passes
            ang = np.unwrap(np.arctan2(loop[:, 1] - cusp[1], loop[:, 0] - cusp[0]))
            winding = (ang[-1] - ang[0]) / (2 * np.pi)
            if abs(round(winding)) < 1:
                raise SystemExit(f"spring: the flip loop does not surround the cusp (winding {winding:.2f})")
            feature += list(loop)
            rec["spring_loop"] = {"self_crossing": [round(float(v), 4) for v in X], "winding_around_cusp": round(float(winding), 3),
                                  "b_range": [round(float(loop[:, 0].min()), 4), round(float(loop[:, 0].max()), 4)],
                                  "c_range": [round(float(loop[:, 1].min()), 4), round(float(loop[:, 1].max()), 4)]}
        # the wedge: fold branches from the cusp out to a fixed distance
        bfo, cfo = curve(a, 1.0)
        near_cusp = (np.abs(XS - t) < 0.35) & (np.abs(XS) > 0.2)    # the wedge, away from x = 0 where b diverges
        feature += list(np.c_[bfo, cfo][near_cusp][::50])
        F = np.array(feature)
        lo, hi = F.min(axis=0), F.max(axis=0)
        span = np.maximum(hi - lo, [0.6, 0.6])
        mid = 0.5 * (lo + hi)
        ext = (mid[0] - 0.62 * span[0], mid[0] + 0.62 * span[0], mid[1] - 0.62 * span[1], mid[1] + 0.62 * span[1])
        for p in F:                         # every defining point must be inside the frame
            assert ext[0] < p[0] < ext[1] and ext[2] < p[1] < ext[3]
        bv, cv = np.linspace(ext[0], ext[1], 261), np.linspace(ext[2], ext[3], 261)
        B, C = np.meshgrid(bv, cv)
        na = counts(a, B, C)
        ax.imshow(np.minimum(na, 2), origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=2, aspect="auto",
                  interpolation="nearest")
        for target, col, ls, lw in ((1.0, "k", "-", 1.4), (-1.0, "r", "--", 1.2), (0.0, "#1f4e9a", ":", 1.2)):
            b, c = curve(a, target)
            for sel in (neg, pos):
                ax.plot(b[sel], c[sel], ls, color=col, lw=lw)
        ax.plot(*cusp, "*", ms=13, color="gold", mec="k", zorder=6)
        if kind == "crossroad area":
            ax.plot(*X, "o", ms=6, color="r", mec="k", zorder=6)
        ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
        ax.set_xlabel("$b$"); ax.set_ylabel("$c$")
        ax.set_title(f"{kind} ($a={a:g}$, $c_3={4 * t:+.2f}$)\nthrough the cusp: {split}", fontsize=9.5)
        rec[kind] = {"a": a, "cusp": [round(float(v), 4) for v in cusp], "c3": round(float(4 * t), 4),
                     "frame": [round(float(v), 4) for v in ext],
                     "attracting_counts": {int(k): int(v) for k, v in zip(*np.unique(na, return_counts=True))}}
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Line2D([], [], color="k", lw=1.4, label="fold ($\\mu=+1$)"),
               Line2D([], [], color="r", ls="--", lw=1.2, label="flip ($\\mu=-1$)"),
               Line2D([], [], color="#1f4e9a", ls=":", lw=1.2, label="superstable ($\\mu=0$)"),
               Patch(color="#d9d9d9", label="one attracting fixed point"),
               Patch(color="#8fb3d9", label="two attracting fixed points")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    text = json.dumps(rec, indent=1)
    if args.json:
        Path(args.json).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
