"""Figure fig-superstable-lines: the superstable structure of the hostile period-7 body (Qwen2.5-0.5B-Instruct,
hostility-vs-kindness at layer 12, float64).

(a) Largest Lyapunov exponent of the cells whose attractor is the period-7 loop, stitched from the float64 maps
    scan64/hub-arm (run M3b, partial) and scan64/hub-arm-rest, -top1, -top2 (run MB, same grid spacing): dark curves are
    superstable lines. Overlaid: float64 fold (black) and flip (red, dashed) curves of the period-7 orbit (hcurves/hub7-*),
    the two located cusps (codim2/hub7-*-cusp-*), and the box of the continuation grid of panel (b).
(b) The continuation grid of one period-7 orbit (dss/p7-blob, tools/doubly_superstable.py): log10 |mu| of its leading
    multiplier, and the zero curves of the per-step factors c_i (superstable curves of single steps), read from clean
    cells with the seed's token loop; crossings of two different steps' curves are doubly superstable points.

Usage: python tools/plot_superstable_lines.py [--out PNG] [--dss RUN]
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data")) / "explore/llm-soft-loop/pod64"
LOOP = 22                                       # family index of '! Yes heels \n\n defiance \n\n I' in scan64/hub-arm


def load_scan(name):
    path = DATA / "scan64" / name
    f = path / "scan.npz" if (path / "scan.npz").exists() else path / "scan-partial.npz"
    z = np.load(f, allow_pickle=True)
    lt, al = z["log10_temperature"], z["alpha"]
    P, L, F = z["period"].astype(float), z["lyapunov"].astype(float), z["family"].astype(float)
    keys = None
    if "family_keys" in z:
        k = z["family_keys"]
        keys = list(k.item() if k.ndim == 0 else k)            # stored as a pickled list (0-d object array) or an array
    if P.ndim == 1:
        done = int(z["done"])
        for A in (P, L, F):
            A[done:] = np.nan
        P, L, F = (A.reshape(len(lt), len(al)) for A in (P, L, F))
    return lt, al, P, L, F, keys


def stitched():
    """Rows of hub-arm (done rows only) and of the MB scans, on one grid (same alpha columns, same row spacing)."""
    lt, al, P, L, F, keys = load_scan("hub-arm")
    loop_tokens = keys[LOOP] if keys is not None else None
    rows = [(lt[i], P[i], L[i], F[i]) for i in range(len(lt)) if np.isfinite(P[i]).all()]
    for name in ("hub-arm-rest", "hub-arm-top1", "hub-arm-top2"):
        if not (DATA / "scan64" / name).exists():
            continue
        lt2, al2, P2, L2, F2, keys2 = load_scan(name)
        assert np.allclose(al2, al), name
        remap = {}
        if keys2 is not None and loop_tokens is not None:     # family indices differ between scans: match by tokens
            remap = {k: (LOOP if list(t) == list(loop_tokens) else -2) for k, t in enumerate(keys2)}
        for i in range(len(lt2)):
            if np.isfinite(P2[i]).all() and all(abs(lt2[i] - r[0]) > 1e-9 for r in rows):
                fam = np.array([remap.get(int(x), -2) if x >= 0 else -1 for x in F2[i]], dtype=float)
                rows.append((lt2[i], P2[i], L2[i], fam))
    rows.sort(key=lambda r: r[0])
    T = np.array([r[0] for r in rows])
    return T, al, np.array([r[1] for r in rows]), np.array([r[2] for r in rows]), np.array([r[3] for r in rows])


def curves():
    out = []
    for c in sorted((DATA / "hcurves").glob("hub7-*")):
        f = c / "curve.json"
        if f.exists():
            pts = json.loads(f.read_text())["points"]
            out.append((c.name, np.array([[p["alpha"], p["log10_temperature"]] for p in pts])))
    cusps = []
    for c in sorted((DATA / "codim2").glob("hub7-*-cusp-*")):
        d = json.loads((c / "locate.json").read_text())
        if d.get("located"):
            cusps.append((d["final"]["alpha"], d["final"]["log10_temperature"]))
    return out, cusps


def zero_segments(field, ok, xs, ys):
    """Line segments of the zero set of a grid field by marching squares over cells whose four corners are ok."""
    segs = []
    for i in range(len(ys) - 1):
        for j in range(len(xs) - 1):
            if not ok[i:i + 2, j:j + 2].all():
                continue
            v = [field[i, j], field[i, j + 1], field[i + 1, j + 1], field[i + 1, j]]
            p = [(xs[j], ys[i]), (xs[j + 1], ys[i]), (xs[j + 1], ys[i + 1]), (xs[j], ys[i + 1])]
            cut = []
            for k in range(4):
                a, b = v[k], v[(k + 1) % 4]
                if a == 0 or a * b < 0:
                    t = a / (a - b) if a != b else 0.0
                    cut.append((p[k][0] + t * (p[(k + 1) % 4][0] - p[k][0]), p[k][1] + t * (p[(k + 1) % 4][1] - p[k][1])))
            if len(cut) == 2:
                segs.append(cut)
            elif len(cut) == 4:                           # saddle cell: pair the cuts by the centre value
                segs += [[cut[0], cut[1]], [cut[2], cut[3]]] if np.mean(v) * v[0] > 0 else [[cut[0], cut[3]], [cut[1], cut[2]]]
    return segs


def crossings(segs_a, segs_b):
    """Intersection points of two sets of segments."""
    pts = []
    for (p1, p2) in segs_a:
        for (q1, q2) in segs_b:
            d = np.array(p2) - np.array(p1)
            e = np.array(q2) - np.array(q1)
            den = d[0] * e[1] - d[1] * e[0]
            if abs(den) < 1e-30:
                continue
            w = np.array(q1) - np.array(p1)
            t = (w[0] * e[1] - w[1] * e[0]) / den
            u = (w[0] * d[1] - w[1] * d[0]) / den
            if 0 <= t <= 1 and 0 <= u <= 1:
                pts.append(tuple(np.array(p1) + t * d))
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=str(ROOT / "paper/llm/figures/fig-superstable-lines.png"))
    ap.add_argument("--dss", default="p7-blob")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T, al, P, L, F = stitched()
    cv, cusps = curves()
    dss = DATA / "dss" / args.dss / "grid.npz"
    fig, axes = plt.subplots(1, 2 if dss.exists() else 1, figsize=(12.5 if dss.exists() else 6.4, 5.4), dpi=200,
                             squeeze=False)
    ax = axes[0, 0]
    M = np.where(P == 7, L, np.nan)                     # every period-7 cell: the loop's token variants ('defiance',
    other = np.where(np.isfinite(P) & (P != 7), 1.0, np.nan)   # 'pee', 'No') are one continued orbit family
    ext = [al[0], al[-1], T[0], T[-1]]
    ax.imshow(other, origin="lower", extent=ext, aspect="auto", cmap="Greys", vmin=0, vmax=6, interpolation="nearest")
    im = ax.imshow(M, origin="lower", extent=ext, aspect="auto", cmap="viridis", vmin=-1.0, vmax=0.0,
                   interpolation="nearest")
    fig.colorbar(im, ax=ax, label="largest Lyapunov exponent (period-7 cells)")
    for name, xy in cv:
        ax.plot(xy[:, 0], xy[:, 1], color="red" if "flip" in name else "black", ls="--" if "flip" in name else "-",
                lw=0.9)
    for a, t in cusps:
        ax.plot(a, t, "o", ms=6, mfc="gold", mec="black", mew=0.8)
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel("steering along 'hostility-vs-kindness'")
    ax.set_ylabel("log10 temperature")
    ax.set_title("(a) period-7 body: superstable lines are dark", fontsize=9)
    report = {"rows": int(len(T)), "log10T": [float(T[0]), float(T[-1])], "cusps": cusps}
    if dss.exists():
        g = np.load(dss)
        xs, ys = g["alpha"], g["log10_temperature"]
        ax.add_patch(plt.Rectangle((xs[0], ys[0]), xs[-1] - xs[0], ys[-1] - ys[0], fill=False, ec="white", lw=1.0))
        bx = axes[0, 1]
        mu, ok = g["mu"], g["ok"] & g["same_tokens"]
        good = ok & g["clean"]
        im2 = bx.imshow(np.log10(np.abs(np.where(ok, mu, np.nan))), origin="lower", aspect="auto", cmap="viridis",
                        extent=[xs[0], xs[-1], ys[0], ys[-1]], vmin=-4, vmax=0, interpolation="nearest")
        fig.colorbar(im2, ax=bx, label="log10 |leading multiplier|")
        C = g["factors"]
        p = C.shape[2]
        colours = plt.get_cmap("tab10")(np.arange(p))
        segs = {}
        for k in range(p):
            segs[k] = zero_segments(C[:, :, k], good, xs, ys)
            for s in segs[k]:
                bx.plot([s[0][0], s[1][0]], [s[0][1], s[1][1]], color=colours[k], lw=1.4)
            if segs[k]:
                bx.plot([], [], color=colours[k], lw=1.4, label=f"$c_{k}=0$")
        cross = []
        for a in range(p):
            for b in range(a + 1, p):
                for q in crossings(segs[a], segs[b]):
                    cross.append((a, b, q))
                    bx.plot(*q, "*", ms=11, mfc="white", mec="black")
        bx.set_xlabel("steering along 'hostility-vs-kindness'")
        bx.set_ylabel("log10 temperature")
        bx.set_title("(b) one continued period-7 orbit: zero curves of the step factors", fontsize=9)
        if any(segs.values()):
            bx.legend(fontsize=7, loc="upper left")
        report.update({"dss_cells": int(mu.size), "ok": int(ok.sum()), "clean": int(good.sum()),
                       "zero_segments": {k: len(v) for k, v in segs.items()},
                       "crossings": [{"steps": [a, b], "alpha": float(q[0]), "log10T": float(q[1])} for a, b, q in cross]})
    fig.tight_layout()
    fig.savefig(args.out)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
