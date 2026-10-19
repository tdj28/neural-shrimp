"""Analysis of the tracked edges of the hook (runs HOOKEDGE, HOOKTIP): which edge points lie on the kite's fold curve
and which are basin crossings of the fixed start, and how the basin-crossing edge meets the fold curve near the tip.

Theory being tested (saddle-node normal form u -> u + nu - a u^2 near the fold, the saddle at u = -sqrt(nu/a), the fixed
start arriving with a smooth coordinate u0(alpha, T)): the basin-crossing edge u0 = -sqrt(nu/a) meets the fold curve
tangentially at the tip, its distance from the fold growing like the square of the distance along it (exponent 2).

Distances are in grid steps of the HR12 map in each axis. A point closer than --on-fold steps to the continued fold
curve counts as on the fold.

Usage: python tools/hook_edge_analysis.py --edges llm-soft-loop/pod64/kite/hook-edge [--edges ...tip] \
    --out paper/llm/figures/fig-hook-edge.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import plot_p6_kite as pk  # noqa: E402


def seg_dist(q, P):
    """Distance from points q (m, 2) to the polyline P (n, 2), and the arclength position of the nearest point."""
    A, B = P[:-1], P[1:]
    d = B - A
    L = np.sqrt((d ** 2).sum(1))
    cum = np.concatenate([[0.0], np.cumsum(L)])
    out_d, out_s = [], []
    for x in q:
        t = np.clip(((x - A) * d).sum(1) / np.maximum((d ** 2).sum(1), 1e-30), 0, 1)
        proj = A + t[:, None] * d
        dist = np.sqrt(((proj - x) ** 2).sum(1))
        k = int(np.argmin(dist))
        out_d.append(float(dist[k]))
        out_s.append(float(cum[k] + t[k] * L[k]))
    return np.array(out_d), np.array(out_s)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--edges", nargs="+", required=True)
    ap.add_argument("--map", default="llm-soft-loop/pod64/hr12")
    ap.add_argument("--on-fold", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(pk.DATA / "explore" / args.map / "scan.npz")
    A, T, L = z["alpha"], z["log10_temperature"], z["lyapunov"]
    step = np.array([A[1] - A[0], T[1] - T[0]])
    E = []
    for e in args.edges:
        E += json.loads((pk.DATA / "explore" / e / "hook-edge.json").read_text())["edges"]
    pts = np.array([[e["alpha"], e["log10T_edge"]] for e in E])
    _, kp, _, _ = pk.curve(pk.POD / "kite/kite-la-curve-plus/curve.json")
    _, km, _, _ = pk.curve(pk.POD / "kite/kite-la-curve-minus/curve.json")
    fold = np.vstack([kp[::-1], km])
    d, s = seg_dist(pts / step, fold / step)
    on = d < args.on_fold
    rec = {"edges": len(E), "on_fold": int(on.sum()), "crossings": int((~on).sum())}
    # the tip: the crossing point nearest the fold; fit distance-from-fold against distance along it near the tip
    cr = np.nonzero(~on)[0]
    if len(cr) >= 5:
        k0 = cr[int(np.argmin(d[cr]))]
        ds = np.abs(s[cr] - s[k0])
        near = (ds > 0) & (ds < np.percentile(ds[ds > 0], 30))
        if near.sum() >= 4:
            p, logK = np.polyfit(np.log(ds[near]), np.log(np.maximum(d[cr][near], 1e-12)), 1)
            rec["tip"] = {"alpha": float(pts[k0, 0]), "log10T": float(pts[k0, 1]), "distance_steps": float(d[k0])}
            rec["fit_near_tip"] = {"points": int(near.sum()), "exponent": float(p), "prefactor": float(np.exp(logK)),
                                   "prediction": 2.0}
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))
    ext = [A[0], A[-1], T[0], T[-1]]
    ax.imshow(np.clip(np.where(L <= 1e-3, L, np.nan), -0.6, 0), origin="lower", extent=ext, aspect="auto", cmap="gray",
              vmin=-0.6, vmax=0, interpolation="nearest")
    ax.plot(fold[:, 0], fold[:, 1], "-", color="gold", lw=1.0, label="fold curve (continued)")
    ax.plot(pts[on, 0], pts[on, 1], ".", color="tab:red", ms=2.5, label="edge on the fold curve")
    ax.plot(pts[~on, 0], pts[~on, 1], ".", color="tab:cyan", ms=2.5, label="edge off the fold: basin crossing")
    ax.set_xlim(A[0], A[-1])
    ax.set_ylim(T[0], T[-1])
    ax.ticklabel_format(useOffset=False)
    ax.legend(fontsize=7, loc="lower right")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\log_{10}T$")
    if "fit_near_tip" in rec:
        ax2.loglog(np.abs(s[cr] - s[k0]), np.maximum(d[cr], 1e-12), ".", ms=3)
        xx = np.logspace(np.log10(max(1e-6, np.abs(s[cr] - s[k0]).min() + 1e-9)), np.log10(np.abs(s[cr] - s[k0]).max()), 50)
        ax2.loglog(xx, rec["fit_near_tip"]["prefactor"] * xx ** rec["fit_near_tip"]["exponent"], "k--", lw=0.8,
                   label=f"fit near the tip: exponent {rec['fit_near_tip']['exponent']:.2f}")
        ax2.set_xlabel("distance along the fold from the tip (grid steps)")
        ax2.set_ylabel("distance from the fold (grid steps)")
        ax2.legend(fontsize=7)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    Path(args.out).with_suffix(".json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
