"""Video: the quartic map of Carcasses et al. (1991), x -> c + (1+a)x + bx^2 + x^4, in the plane of b and c while a
sweeps from +1 to -3 (10 s). Every frame is computed: fold (multiplier +1), flip (-1) and superstable (0) curves of
the fixed points, parametrized by the fixed point x, and the number of attracting fixed points on a grid (batched
companion-matrix roots). The labels are computed too:
- the cusp sits at x = t, a = 8t^3, with cubic normal-form coefficient c3 = 4t; c3 > 0: saddle kind (passing through
  the cusp splits a saddle into a node and two saddles), c3 < 0: node kind (a node into a saddle and two nodes);
- crossroad: the two flip branches cross near the cusp; spring: a self-crossing of a flip branch closes a loop that
  winds once around the cusp.
At a = 0 the cusp passes through the swallowtail point (a = b = c = 0) and c3 changes sign; at a = -2 the flip
curves reconnect into the loop (the codimension-three point a7 of Carcasses et al.). Period-2 curves are not drawn.
A first pass fixes one frame that contains every frame's cusp, flip crossing and flip loop; rendering refuses to
start if any of them would leave it.

Usage: python tools/render_cusp_sweep.py [--out MP4] [--json OUT_JSON] [--fps 30] [--seconds 10]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                       # noqa: E402
from matplotlib.colors import ListedColormap                         # noqa: E402

XS = np.concatenate([-np.geomspace(4.0, 1e-3, 5000), np.geomspace(1e-3, 4.0, 5000)])
BIG = (-9.0, 4.0, -7.0, 7.0)


def curve(a, target, xs=XS):
    b = (target - 1.0 - a - 4.0 * xs ** 3) / (2.0 * xs)
    return b, -xs ** 4 - b * xs ** 2 - a * xs


def inside(b, c, box):
    return (b > box[0]) & (b < box[1]) & (c > box[2]) & (c < box[3])


def counts(a, B, C):
    n = B.size
    M = np.zeros((n, 4, 4))
    M[:, 0, 1], M[:, 0, 2], M[:, 0, 3] = -B.ravel(), -a, -C.ravel()
    M[:, 1, 0] = M[:, 2, 1] = M[:, 3, 2] = 1.0
    r = np.linalg.eigvals(M)
    real = np.abs(r.imag) < 1e-6 * np.maximum(1.0, np.abs(r.real))
    m = 1.0 + a + 2.0 * B.ravel()[:, None] * r.real + 4.0 * r.real ** 3
    return (real & (np.abs(m) < 1.0)).sum(axis=1).reshape(B.shape)


def seg_cross(P, Q, same=False, chunk=400):
    """All crossings of polylines P and Q as (i, j, point); with same=True, of P with itself (non-adjacent)."""
    res = []
    q0 = Q[:-1]
    s = Q[1:] - q0
    n = len(P) - 1
    for i0 in range(0, n, chunk):
        i1 = min(n, i0 + chunk)
        p0 = P[i0:i1]
        r = P[i0 + 1:i1 + 1] - p0
        den = r[:, None, 0] * s[None, :, 1] - r[:, None, 1] * s[None, :, 0]
        qp = q0[None, :, :] - p0[:, None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (qp[..., 0] * s[None, :, 1] - qp[..., 1] * s[None, :, 0]) / den
            u = (qp[..., 0] * r[:, None, 1] - qp[..., 1] * r[:, None, 0]) / den
        ok = (den != 0) & (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)
        if same:
            ok &= np.arange(len(q0))[None, :] > (np.arange(i0, i1)[:, None] + 1)
        for a_, b_ in zip(*np.nonzero(ok)):
            res.append((i0 + int(a_), int(b_), p0[a_] + t[a_, b_] * r[a_]))
    return res


def features(a):
    t = np.cbrt(a / 8.0)
    cusp = np.array([-6.0 * t * t, -3.0 * t ** 4])
    bf, cf = curve(a, -1.0)
    neg, pos = XS < 0, XS > 0
    P = np.c_[bf, cf][neg & inside(bf, cf, BIG)][::3]            # thinned for crossing detection only
    Q = np.c_[bf, cf][pos & inside(bf, cf, BIG)][::3]
    out = {"a": a, "c3": 4.0 * t, "cusp": cusp, "crossroad": None, "loop": None}
    if t < 0:
        cr = seg_cross(P, Q)
        if cr:
            out["crossroad"] = min((r[2] for r in cr), key=lambda p: np.hypot(*(p - cusp)))
        sc = seg_cross(P, P, same=True)
        for i, j, X in sorted(sc, key=lambda r: np.hypot(*(r[2] - cusp))):
            L = np.vstack([X, P[i + 1:j + 1], X])
            ang = np.unwrap(np.arctan2(L[:, 1] - cusp[1], L[:, 0] - cusp[0]))
            if abs(ang[-1] - ang[0]) > 1.9 * np.pi:          # winds once around the cusp
                out["loop"] = L
                break
    return out


def schedule(n):
    """a(t): +1 to -3, slower through a = 0 (swallowtail point) and a = -2 (loop reconnection)."""
    knots_t = [0.0, 0.22, 0.36, 0.62, 0.78, 1.0]
    knots_a = [1.0, 0.15, -0.15, -1.85, -2.15, -3.0]
    return np.interp(np.linspace(0.0, 1.0, n), knots_t, knots_a)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="../neural-shrimp-data/videos/cusp-sweep-quartic.mp4")
    ap.add_argument("--json")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--still", type=float, default=None, help="render only the frame nearest this a, as a PNG")
    ap.add_argument("--still-out", default=None, help="PNG path for --still")
    args = ap.parse_args()
    avals = schedule(int(round(args.fps * args.seconds)))
    cache = Path(args.out).with_suffix(".features.npy")
    if cache.exists():
        feats = list(np.load(cache, allow_pickle=True))
        if len(feats) != len(avals) or any(abs(f["a"] - a) > 1e-12 for f, a in zip(feats, avals)):
            feats = None
    else:
        feats = None
    if feats is None:
        feats = [features(a) for a in avals]
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, np.array(feats, dtype=object), allow_pickle=True)
    frame = (-5.0, 0.6, -2.0, 2.2)      # fixed; loops and cusps must fit, a far flip crossing is marked by an arrow
    for f in feats:
        for p in [f["cusp"]] + (list(f["loop"]) if f["loop"] is not None else []):
            if not (frame[0] < p[0] < frame[1] and frame[2] < p[1] < frame[3]):
                raise SystemExit(f"a={f['a']:.3f}: a defining point {p} leaves the frame {frame}")
    summary = {"frame": [round(v, 4) for v in frame], "frames": len(avals),
               "first_spring_a": next((round(float(f["a"]), 4) for f in feats if f["loop"] is not None), None),
               "last_crossroad_a": next((round(float(f["a"]), 4) for f in feats[::-1] if f["crossroad"] is not None), None)}
    print(json.dumps(summary))
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1) + "\n")

    cmap = ListedColormap(["white", "#d9d9d9", "#8fb3d9"])
    bv, cv = np.linspace(frame[0], frame[1], 384), np.linspace(frame[2], frame[3], 216)
    Bg, Cg = np.meshgrid(bv, cv)
    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    ax = fig.add_axes([0.07, 0.23, 0.64, 0.68])
    side = fig.add_axes([0.74, 0.23, 0.24, 0.68]); side.axis("off")
    bar = fig.add_axes([0.07, 0.065, 0.64, 0.032])

    def draw(k):
        f = feats[k]
        a = f["a"]
        ax.clear(); side.clear(); side.axis("off"); bar.clear()
        ax.imshow(np.minimum(counts(a, Bg, Cg), 2), origin="lower", extent=frame, cmap=cmap, vmin=0, vmax=2,
                  aspect="auto", interpolation="nearest")
        for target, col, ls, lw in ((1.0, "k", "-", 1.6), (-1.0, "r", "--", 1.3), (0.0, "#1f4e9a", ":", 1.2)):
            b, c = curve(a, target)
            for sel in (XS < 0, XS > 0):
                ax.plot(b[sel], c[sel], ls, color=col, lw=lw)
        ax.plot(*f["cusp"], "*", ms=16, color="gold", mec="k", zorder=6)
        if f["crossroad"] is not None:
            X = f["crossroad"]
            if frame[2] < X[1] < frame[3] and frame[0] < X[0] < frame[1]:
                ax.plot(*X, "o", ms=8, color="r", mec="k", zorder=6)
            else:                                               # off the frame: say where, do not hide it
                xb = min(max(X[0], frame[0] + 0.3), frame[1] - 0.3)
                ax.annotate("", xy=(xb, frame[3] - 0.02), xytext=(xb, frame[3] - 0.45),
                            arrowprops=dict(arrowstyle="-|>", color="r", lw=2), zorder=7)
                ax.text(xb + 0.08, frame[3] - 0.35, "flip curves cross above\n($b=%.2f$, $c=%.1f$)" % (X[0], X[1]),
                        color="r", fontsize=10, va="center")
        if f["loop"] is not None:
            ax.plot(f["loop"][:, 0], f["loop"][:, 1], "-", color="r", lw=3.0, alpha=0.35, zorder=4)
        ax.set_xlim(frame[0], frame[1]); ax.set_ylim(frame[2], frame[3])
        ax.set_xlabel("$b$", fontsize=13); ax.set_ylabel("$c$", fontsize=13)
        ax.set_title(r"$x\mapsto c+(1+a)\,x+b\,x^2+x^4$:   $a=%+.3f$" % a, fontsize=15)
        c3 = f["c3"]
        if abs(a) < 0.02:
            head, body = "swallowtail point", "the cusp passes through\n$a=b=c=0$; its cubic\ncoefficient $c_3=4t$\nchanges sign"
        elif c3 > 0:
            head, body = "saddle kind ($c_3>0$)", "through the cusp:\nsaddle $\\to$ node + 2 saddles;\nthe attracting fixed point\nlives only in the wedge"
        elif f["loop"] is not None:
            head, body = "spring area ($c_3<0$)", "a flip loop surrounds\nthe cusp (shaded red);\ninside it the fixed point\nattracts again"
        elif f["crossroad"] is not None:
            head, body = "crossroad area ($c_3<0$)", "through the cusp:\nnode $\\to$ saddle + 2 nodes;\nthe flip curves of the two\nbranches cross (red dot)"
        else:
            head, body = "node kind ($c_3<0$)", "the flip curves\nreconnect"
        side.text(0.0, 0.95, head, fontsize=16, weight="bold", va="top")
        side.text(0.0, 0.80, body, fontsize=13, va="top", linespacing=1.4)
        side.text(0.0, 0.30, "black: fold ($\\mu=+1$)\nred dashed: flip ($\\mu=-1$)\nblue dotted: superstable ($\\mu=0$)\n"
                  "grey: one attracting fixed point\nblue: two attracting fixed points\nperiod-2 curves not drawn",
                  fontsize=10.5, va="top", color="0.25", linespacing=1.5)
        bar.set_xlim(1.0, -3.0); bar.set_ylim(0, 1); bar.set_yticks([])
        bar.axvspan(1.0, a, color="#8fb3d9")
        for v, lab in ((0.0, "swallowtail\npoint"), (-2.0, "loop forms\n($a_7=-2$)")):
            bar.axvline(v, color="k", lw=1.0)
            bar.text(v, 1.15, lab, ha="center", va="bottom", fontsize=9)
        bar.set_xlabel("$a$", fontsize=12)

    if args.still is not None:
        draw(int(np.argmin(np.abs(avals - args.still))))
        fig.savefig(args.still_out)
        return
    from matplotlib.animation import FFMpegWriter
    writer = FFMpegWriter(fps=args.fps, codec="libx264", bitrate=4000, extra_args=["-pix_fmt", "yuv420p"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, args.out, dpi=100):
        for k in range(len(avals)):
            draw(k)
            writer.grab_frame()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
