"""The dovetail in Mira's foliated representation: parameters plus one state coordinate (figure fig:foliation).

(a) Model: the fixed points x of Carcasses et al.'s quartic map x -> c + (1+a)x + bx^2 + x^4 at b = -1, drawn as the
surface (a, c(a, x), x) with c = -x^4 - b x^2 - a x; attracting fixed points (|multiplier| < 1) in blue, others in grey.
The fold curve (multiplier +1) is a smooth curve on the surface; its shadow on the parameter plane is the dovetail.
(b) Data: the period-5 window of Qwen2.5-0.5B-Instruct steered at two depths (runs FM5, FM5c, OC). The continued fold
curve (curve-a1-up-fold-minus, which carries cusps U and L and the crossing X) and every orbit found by the census, at
(alpha1, alpha2, z), with z the projection of the orbit state on the first principal direction of all these states.
Each census orbit is stored at an arbitrary phase; it is iterated through its five phases (float64, CPU) and the phase
nearest the fold curve's state at the nearest curve point is kept, so that all states lie on one consistent sheet.

Usage:
  python tools/plot_foliation.py align [--threads 12]       # writes explore/llm-soft-loop/pod64/oc/foliation-aligned.npz
  python tools/plot_foliation.py plot [--out PNG] [--json OUT_JSON]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
POD = DATA / "explore" / "llm-soft-loop" / "pod64"
CURVE = POD / "fm5" / "curve-a1-up-fold-minus"
OC = POD / "oc"
PARTS = ("census", "census-2", "census-3")
ALIGNED = OC / "foliation-aligned.npz"


def census_records():
    recs = []
    for part in PARTS:
        for line in open(OC / part / "census.jsonl"):
            r = json.loads(line)
            f = OC / part / f"orbits-{r['name']}.npy"
            states = np.load(f) if f.exists() else np.zeros((0, 896))
            if len(states) != len(r["orbits"]):
                raise SystemExit(f"{r['name']}: {len(states)} stored states but {len(r['orbits'])} orbits")
            recs.append((r, states))
    return recs


def align(args):
    import window_census as wc
    curve = json.loads((CURVE / "curve.json").read_text())["points"]
    S = np.load(CURVE / "states.npy")
    P = np.array([[q["alpha"], q["alpha2"]] for q in curve])
    loop, make_cell = wc.setup("cpu", args.threads)
    names, params, states, attracting, moduli = [], [], [], [], []
    for r, X0 in census_records():
        k = int(np.argmin(np.hypot(P[:, 0] - r["alpha"], (P[:, 1] - r["alpha2"]) * 0.1)))   # nearest curve point
        ref = S[k]
        cell = make_cell(r["alpha"], r["alpha2"])
        for o, x in zip(r["orbits"], X0):
            ph = np.array(wc.phases(cell, wc.to_state(loop, x)))
            best = ph[int(np.argmin(np.linalg.norm(ph - ref, axis=1)))]
            names.append(r["name"]); params.append([r["alpha"], r["alpha2"]]); states.append(best)
            attracting.append(bool(o["attracting"])); moduli.append(float(o["leading_modulus"]))
        print(r["name"], len(X0), "orbits aligned to curve point", k, flush=True)
    np.savez(ALIGNED, names=np.array(names), params=np.array(params), states=np.array(states),
             attracting=np.array(attracting), moduli=np.array(moduli))
    print("wrote", ALIGNED)


def model_surface(ax):
    b = -1.0
    a = np.linspace(-1.0, 1.0, 161)
    x = np.linspace(-1.15, 1.15, 161)
    A, X = np.meshgrid(a, x)
    Cs = -X ** 4 - b * X ** 2 - A * X
    m = 1 + A + 2 * b * X + 4 * X ** 3
    from matplotlib.colors import ListedColormap
    face = ListedColormap(["#8fb3d9", "#d4d4d4"])(np.where(np.abs(m) < 1, 0.0, 1.0))
    z0 = -1.6
    ax.plot_surface(A, Cs, X, facecolors=face, rstride=2, cstride=2, linewidth=0, antialiased=True, shade=True,
                    alpha=0.85, zorder=1)
    t = np.linspace(-1.15, 1.15, 3000)                     # fold curve on the surface: a = -2bt - 4t^3
    fa = -2 * b * t - 4 * t ** 3
    fc = -t ** 4 - b * t ** 2 - fa * t
    ok = (fa >= -1.0) & (fa <= 1.0)
    ax.plot(fa[ok], fc[ok], t[ok], "k-", lw=1.8, zorder=10)
    ax.plot(fa[ok], fc[ok], np.full(ok.sum(), z0), "-", color="0.3", lw=1.0, zorder=2)
    tc = 1 / np.sqrt(6)
    for s_ in (tc, -tc):
        ca, cc = -2 * b * s_ - 4 * s_ ** 3, None
        cc = -s_ ** 4 - b * s_ ** 2 - ca * s_
        ax.plot([ca], [cc], [s_], "*", ms=12, color="gold", mec="k", zorder=11)
        ax.plot([ca], [cc], [z0], "*", ms=8, color="gold", mec="k", zorder=3)
    for s_ in (1 / np.sqrt(2), -1 / np.sqrt(2)):           # the two lifts of the crossing X = (0, 1/4)
        ax.plot([0.0, 0.0], [0.25, 0.25], [z0, s_], ":", color="m", lw=1.0, zorder=9)
        ax.plot([0.0], [0.25], [s_], "x", color="m", ms=9, mew=1.8, zorder=11)
    ap, cp = 0.0, 0.03                                     # four sheets over a pocket point
    r = np.roots([1.0, 0.0, b, ap, cp]); r = np.sort(r[np.abs(r.imag) < 1e-9].real)
    ax.plot([ap, ap], [cp, cp], [z0, r.max()], "-", color="0.45", lw=0.8, zorder=9)
    for xr in r:
        att = abs(1 + ap + 2 * b * xr + 4 * xr ** 3) < 1
        ax.plot([ap], [cp], [xr], "o", ms=5, mfc="#1f4e9a" if att else "white", mec="#1f4e9a", mew=1.1, zorder=12)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1]); ax.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1]); ax.set_zticks([-1.5, -1, -0.5, 0, 0.5, 1])
    ax.tick_params(labelsize=7, pad=0)
    ax.set_xlabel("$a$", labelpad=8); ax.set_ylabel("$c$", labelpad=8); ax.set_zlabel("fixed point $x$", labelpad=8)
    ax.set_zlim(z0, 1.2)
    ax.set_title("(a) model ($b=-1$): fixed points over the plane", fontsize=9)


def data_curve(ax, rec):
    curve = json.loads((CURVE / "curve.json").read_text())["points"]
    S = np.load(CURVE / "states.npy")
    P = np.array([[q["alpha"], q["alpha2"]] for q in curve])
    Z = np.load(ALIGNED)
    allS = np.vstack([S, Z["states"]])
    mu = allS.mean(axis=0)
    U, sv, Vt = np.linalg.svd(allS - mu, full_matrices=False)
    pc = Vt[0]
    zc = (S - mu) @ pc
    zo = (Z["states"] - mu) @ pc
    rec["pc1_variance_fraction"] = float(sv[0] ** 2 / np.sum(sv ** 2))
    sel0 = (P[:, 0] > -0.11935) & (P[:, 0] < -0.11885) & (P[:, 1] > 0.10255) & (P[:, 1] < 0.10425)
    zj = zc[sel0]                                                    # the curve near the junction
    fam = np.abs(zo - np.median(zj)) < 4 * (zj.max() - zj.min())    # the junction's family; others are annotated
    rec["off_family"] = [[str(n), round(float(z), 5)] for n, z, f in zip(Z["names"], zo, fam) if not f]
    lo_, hi_ = min(zj.min(), zo[fam].min()), max(zj.max(), zo[fam].max())
    z0 = lo_ - 0.25 * (hi_ - lo_)
    sel = (P[:, 0] > -0.11935) & (P[:, 0] < -0.11885) & (P[:, 1] > 0.10255) & (P[:, 1] < 0.10425)
    idx = np.where(sel)[0]
    runs = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
    for r_ in runs:
        ax.plot(P[r_, 0], P[r_, 1], zc[r_], "k-", lw=1.6)
        ax.plot(P[r_, 0], P[r_, 1], np.full(len(r_), z0), "-", color="0.35", lw=1.0)
    cand = [i for i, q in enumerate(curve) if q["cusp_candidate"]]
    for i in cand:
        ax.plot([P[i, 0]], [P[i, 1]], [zc[i]], "*", ms=11, color="gold", mec="k", zorder=10)
        ax.plot([P[i, 0]], [P[i, 1]], [z0], "*", ms=8, color="gold", mec="k")
    rec["cusp_points"] = [[float(P[i, 0]), float(P[i, 1]), float(zc[i])] for i in cand]
    import plot_fm5 as pf
    xs = pf.self_crossings(P)
    rec["crossing_lifts"] = []
    for i, j, x in xs:
        for k in (i, j):
            ax.plot([x[0], x[0]], [x[1], x[1]], [z0, zc[k]], ":", color="m", lw=1.0)
            ax.plot([x[0]], [x[1]], [zc[k]], "x", color="m", ms=8, mew=1.6)
            rec["crossing_lifts"].append(float(zc[k]))
    for (a1, a2), z, att, f in zip(Z["params"], zo, Z["attracting"], fam):
        if not f:
            ax.plot([a1, a1], [a2, a2], [z0, hi_], "-", color="0.75", lw=0.5)
            ax.text(a1, a2, hi_, " another orbit\n above (z=%.3f)" % z, fontsize=6, color="0.3")
            continue
        ax.plot([a1, a1], [a2, a2], [z0, z], "-", color="0.75", lw=0.5)
        ax.plot([a1], [a2], [z], "o", ms=4.5, mfc="#1f4e9a" if att else "white", mec="#1f4e9a", mew=1.0)
    rec["orbit_z"] = {str(n): [] for n in np.unique(Z["names"])}
    for n, z, att in zip(Z["names"], zo, Z["attracting"]):
        rec["orbit_z"][str(n)].append([round(float(z), 5), bool(att)])
    ax.set_xlabel(r"$\alpha_1$", labelpad=16); ax.set_ylabel(r"$\alpha_2$", labelpad=16)
    ax.set_zlabel("state (PC 1)", labelpad=14)
    from matplotlib.ticker import MaxNLocator
    ax.zaxis.set_major_locator(MaxNLocator(5)); ax.xaxis.set_major_locator(MaxNLocator(4)); ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.set_zlim(z0, hi_ + 0.05 * (hi_ - lo_))
    ax.ticklabel_format(useOffset=False)
    ax.tick_params(labelsize=6.5, pad=2)
    ax.set_title("(b) data: the period-5 window's fold curve and census orbits", fontsize=9)


def plot(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rec = {}
    fig = plt.figure(figsize=(11.5, 5.4), dpi=170)
    ax = fig.add_axes([0.0, 0.03, 0.47, 0.9], projection="3d")     # explicit positions: 3D labels need room
    model_surface(ax)
    ax.view_init(elev=args.elev, azim=args.azim)
    ax = fig.add_axes([0.48, 0.03, 0.44, 0.9], projection="3d")
    data_curve(ax, rec)
    ax.view_init(elev=args.elev, azim=args.azim2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    text = json.dumps(rec, indent=1)
    if args.json:
        Path(args.json).write_text(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    a = sub.add_parser("align")
    a.add_argument("--threads", type=int, default=12)
    p = sub.add_parser("plot")
    p.add_argument("--out", default="paper/llm/figures/fig-foliation.png")
    p.add_argument("--json")
    p.add_argument("--elev", type=float, default=24.0)
    p.add_argument("--azim", type=float, default=-62.0)
    p.add_argument("--azim2", type=float, default=-62.0)
    args = ap.parse_args()
    {"align": align, "plot": plot}[args.command](args)


if __name__ == "__main__":
    main()
