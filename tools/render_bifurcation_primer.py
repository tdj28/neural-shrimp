"""Video: a primer on the terms of the cusp-sweep video (human, 18 Oct 2026: "begin it by defining flip, fold, cusp, node,
saddle and so on, so that it is a fully educational video").

Four computed sections, each with its own title, on one-dimensional maps (filled dots: attracting fixed points or
orbits; open dots: repelling):
1. fixed points and multipliers: the logistic map x -> r x (1 - x) and the diagonal; a cobweb converges to the fixed
   point at r = 2.6 (multiplier -0.6) and leaves it at r = 3.3 (multiplier -1.3);
2. fold: x -> x + m - x^2 as m falls through 0; the attracting point (multiplier 1 - 2 sqrt m) and the repelling one
   (1 + 2 sqrt m) meet at multiplier +1 and vanish;
3. flip: the logistic map as r passes 3; the fixed point's multiplier 2 - r passes -1, the point turns repelling and
   the attracting period-2 orbit x = ((r + 1) +- sqrt((r + 1)(r - 3))) / (2 r) opens out of it;
4. cusp: x -> x + a + b x - x^3 in the plane of (a, b): fold curves a = 2 s^3, b = 3 s^2 meet at the cusp; a dot crosses
   the wedge at b = 0.6, and the graph of a + b x - x^3 shows one fixed point outside and three inside (two attracting,
   one repelling).

Usage: python tools/render_bifurcation_primer.py --out primer.mp4 [--json OUT]
"""
import argparse
import json
from pathlib import Path

import numpy as np

SECTIONS = (("fixed", 12.0), ("fold", 12.0), ("flip", 12.0), ("cusp", 14.0))


def ease(u):
    u = min(max(u, 0.0), 1.0)
    return u * u * (3 - 2 * u)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--json")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    fig = plt.figure(figsize=(12.8, 7.2), dpi=100)
    writer = FFMpegWriter(fps=args.fps, codec="libx264", bitrate=4000, extra_args=["-pix_fmt", "yuv420p"])
    starts, t = {}, 0.0
    for name, dur in SECTIONS:
        starts[name] = t
        t += dur
    with writer.saving(fig, args.out, dpi=100):
        for name, dur in SECTIONS:
            n = int(round(dur * args.fps))
            for i in range(n):
                u = i / max(n - 1, 1)
                fig.clf()
                ax = fig.add_axes([0.07, 0.12, 0.50, 0.72])
                side = fig.add_axes([0.62, 0.12, 0.36, 0.72])
                side.set_axis_off()
                x = np.linspace(-0.05, 1.05, 400)
                if name == "fixed":
                    r = 2.6 if u < 0.5 else 3.3
                    f = lambda z: r * z * (1 - z)                          # noqa: E731
                    xs = 1 - 1 / r
                    mu = 2 - r
                    ax.plot(x, f(x), "k", lw=2)
                    ax.plot(x, x, color="0.6", lw=1)
                    z = 0.15 if r < 3 else xs + 0.012
                    steps = int(1 + 14 * ease((u % 0.5) / 0.45))
                    for _ in range(steps):
                        z2 = f(z)
                        ax.plot([z, z, z2], [z, z2, z2], color="tab:blue", lw=1.2)
                        z = z2
                    ax.plot(xs, xs, "o", ms=11, color="k" if abs(mu) < 1 else "w", mec="k", mew=2, zorder=5)
                    ax.set_xlim(-0.02, 1.02)
                    ax.set_ylim(-0.02, 1.02)
                    ax.set_xlabel("$x_n$", fontsize=13)
                    ax.set_ylabel("$x_{n+1}$", fontsize=13)
                    ax.set_title(f"logistic map $x\\mapsto rx(1-x)$, $r={r}$", fontsize=13)
                    head = "1. Fixed points and multipliers"
                    body = (f"fixed point: $f(x^*)=x^*$, where the graph\ncrosses the diagonal\n\n"
                            f"multiplier: the slope there, $\\mu={mu:+.1f}$\n\n"
                            + ("$|\\mu|<1$: nearby points move in.\nThe point attracts (a node)." if abs(mu) < 1 else
                               "$|\\mu|>1$: nearby points move out.\nThe point repels (in more dimensions,\n"
                               "often a saddle: attracting along\nsome directions, repelling along others)."))
                elif name == "fold":
                    m = 0.20 - 0.28 * ease(u / 0.85)
                    xx = np.linspace(-0.8, 0.8, 400)
                    ax.plot(xx, xx + m - xx ** 2, "k", lw=2)
                    ax.plot(xx, xx, color="0.6", lw=1)
                    if m > 0:
                        for s, filled in ((np.sqrt(m), True), (-np.sqrt(m), False)):
                            ax.plot(s, s, "o", ms=11, color="k" if filled else "w", mec="k", mew=2, zorder=5)
                        mus = f"multipliers ${1 - 2 * np.sqrt(m):.2f}$ (filled) and ${1 + 2 * np.sqrt(m):.2f}$ (open)"
                    else:
                        mus = "no fixed point left"
                    ax.set_xlim(-0.8, 0.8)
                    ax.set_ylim(-0.8, 0.8)
                    ax.set_xlabel("$x_n$", fontsize=13)
                    ax.set_ylabel("$x_{n+1}$", fontsize=13)
                    ax.set_title(f"$x\\mapsto x+m-x^2$, $m={m:+.3f}$", fontsize=13)
                    head = "2. Fold (multiplier $+1$)"
                    body = ("an attracting point (filled) and a\nrepelling one (open) approach each other,\n"
                            "meet where the graph touches the\ndiagonal, and vanish together\n\n" + mus)
                elif name == "flip":
                    r = 2.8 + 0.6 * ease(u / 0.85)
                    f = lambda z: r * z * (1 - z)                          # noqa: E731
                    xs = 1 - 1 / r
                    mu = 2 - r
                    ax.plot(x, f(x), "k", lw=2)
                    ax.plot(x, f(f(x)), color="tab:purple", lw=1.2, alpha=0.7)
                    ax.plot(x, x, color="0.6", lw=1)
                    ax.plot(xs, xs, "o", ms=11, color="k" if abs(mu) < 1 else "w", mec="k", mew=2, zorder=5)
                    if r > 3:
                        d = np.sqrt((r + 1) * (r - 3))
                        p, q = ((r + 1) + d) / (2 * r), ((r + 1) - d) / (2 * r)
                        ax.plot([p, p, q, q, p], [p, q, q, p, p], color="tab:red", lw=1.4)
                        ax.plot([p, q], [p, q], "o", ms=10, color="tab:red", mec="k", mew=1, zorder=6)
                    ax.set_xlim(-0.02, 1.02)
                    ax.set_ylim(-0.02, 1.02)
                    ax.set_xlabel("$x_n$", fontsize=13)
                    ax.set_ylabel("$x_{n+1}$", fontsize=13)
                    ax.set_title(f"logistic map, $r={r:.3f}$ (purple: the map applied twice)", fontsize=13)
                    head = "3. Flip (multiplier $-1$)"
                    body = (f"multiplier of the fixed point: $\\mu={mu:+.2f}$\n\n"
                            "when $\\mu$ passes $-1$ the fixed point\nstops attracting, and an orbit of\n"
                            "period two (red) takes over:\nperiod doubling"
                            + ("" if r <= 3 else "\n\nthe red orbit hops between two points"))
                else:
                    a_dot = -0.55 + 1.1 * ease(u / 0.9)
                    b0 = 0.6
                    s = np.linspace(-0.62, 0.62, 400)
                    ax.plot(2 * s ** 3, 3 * s ** 2, "k", lw=2)
                    ax.fill_between(np.linspace(-0.3, 0.3, 200), 3 * (np.abs(np.linspace(-0.3, 0.3, 200)) / 2) ** (2 / 3),
                                    1.2, color="#8fb3d9", alpha=0.5, lw=0)
                    ax.plot(0, 0, "*", ms=18, color="gold", mec="k", zorder=6)
                    ax.axhline(b0, color="0.5", lw=0.8, ls=":")
                    ax.plot(a_dot, b0, "o", ms=10, color="red", mec="w", mew=1.5, zorder=7)
                    ax.set_xlim(-0.6, 0.6)
                    ax.set_ylim(-0.15, 1.15)
                    ax.set_xlabel("parameter $a$", fontsize=13)
                    ax.set_ylabel("parameter $b$", fontsize=13)
                    ax.set_title("$x\\mapsto x+a+bx-x^3$: fold curves (black) and the cusp (star)", fontsize=12)
                    ins = fig.add_axes([0.66, 0.14, 0.30, 0.30])
                    xx = np.linspace(-1.1, 1.1, 300)
                    g = a_dot + b0 * xx - xx ** 3
                    ins.plot(xx, g, "k", lw=1.5)
                    ins.axhline(0, color="0.6", lw=0.8)
                    roots = np.roots([-1.0, 0.0, b0, a_dot])
                    roots = np.sort(roots[np.abs(roots.imag) < 1e-9].real)
                    for xr in roots:
                        mu = 1 + b0 - 3 * xr ** 2
                        ins.plot(xr, 0, "o", ms=8, color="k" if abs(mu) < 1 else "w", mec="k", mew=1.5, zorder=5)
                    ins.set_xlim(-1.1, 1.1)
                    ins.set_ylim(-0.9, 0.9)
                    ins.set_xticks([])
                    ins.set_yticks([])
                    ins.set_title(f"fixed points at the red dot: {len(roots)}", fontsize=10)
                    head = "4. Cusp"
                    body = ("two fold curves meet at a sharp tip,\nthe cusp (star). Inside its wedge\n(blue) the map has "
                            "three fixed points\ninstead of one: two attract, one\nrepels. Crossing a fold curve adds\n"
                            "or removes a pair of them.")
                side.text(0.0, 1.0, head, fontsize=17, weight="bold", va="top")
                side.text(0.0, 0.86, body, fontsize=12.5, va="top", linespacing=1.45)
                fig.text(0.07, 0.93, "Words of bifurcation theory, on one-dimensional maps", fontsize=15, color="0.25")
                fig.text(0.07, 0.03, "filled dot: attracting; open dot: repelling", fontsize=10.5, color="0.35")
                writer.grab_frame()
    info = {"out": args.out, "sections": {k: {"start": v, "seconds": d} for (k, d), v in zip(SECTIONS, starts.values())}}
    print(json.dumps(info))
    if args.json:
        Path(args.json).write_text(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
