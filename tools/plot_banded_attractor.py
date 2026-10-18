"""Figure: banded chaos and the unstable manifold that draws it, plus a three-dimensional view of a strange attractor.

(a) The strong-chaos attractor on its first three principal coordinates (a static 3-D view). (b)-(d) The banded attractor
just past a doubling cascade: its states fall into a few small bands, found by k-means on the first two principal
coordinates; each panel zooms into one band, with the unstable manifold of the embedded period-p saddle drawn over it
(tools/unstable_manifold.py), whose images trace the band.

Usage: python tools/plot_banded_attractor.py --strong llm-soft-loop/portraits/f64-strong \
         --banded llm-soft-loop/portraits/f64-cascade --manifold llm-soft-loop/portraits/f64-cascade-wu \
         --bands 3 --out FIG.png
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))


def kmeans(X, k, rng, iters=50, restarts=10):
    """Best of several seeded k-means runs (smallest within-cluster sum of squares); one random start can merge two
    bands and split another."""
    best = None
    for _ in range(restarts):
        C = X[rng.choice(len(X), k, replace=False)]
        for _ in range(iters):
            lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
            C = np.array([X[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
        sse = float(((X - C[lab]) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, lab, C)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--strong", required=True)
    ap.add_argument("--banded", required=True)
    ap.add_argument("--manifold", default=None)
    ap.add_argument("--bands", type=int, default=3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zs = np.load(DATA / "explore" / args.strong / "portrait.npz", allow_pickle=True)
    Ps = zs["coords"][zs["periods"] == 0].reshape(-1, zs["coords"].shape[-1])
    zb = np.load(DATA / "explore" / args.banded / "portrait.npz", allow_pickle=True)
    Pb = zb["coords"][zb["periods"] == 0].reshape(-1, zb["coords"].shape[-1])   # the chaotic attractor's trajectories
    rng = np.random.default_rng(0)
    lab, C = kmeans(Pb[:, :2], args.bands, rng)
    order = np.argsort(C[:, 0])
    fig = plt.figure(figsize=(18, 5.2), dpi=170)
    ax = fig.add_subplot(1, args.bands + 1, 1, projection="3d")
    sub = rng.choice(len(Ps), size=min(len(Ps), 120000), replace=False)
    ax.scatter(Ps[sub, 0], Ps[sub, 1], Ps[sub, 2], s=0.4, c=Ps[sub, 2], cmap="viridis", alpha=0.7, linewidths=0,
               rasterized=True)
    ax.view_init(elev=22, azim=-58)
    ax.set_xlabel("PC1", fontsize=7)
    ax.set_ylabel("PC2", fontsize=7)
    ax.set_zlabel("PC3", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title("(a) strange attractor, first three principal coordinates", fontsize=9)
    M = None
    if args.manifold:
        mz = np.load(DATA / "explore" / args.manifold / "manifold.npz")
        M = mz["proj"]
    summary = {"bands": []}
    for k, j in enumerate(order):
        axb = fig.add_subplot(1, args.bands + 1, k + 2)
        pts = Pb[lab == j][:, :2]
        lo, hi = np.percentile(pts, 0.2, axis=0), np.percentile(pts, 99.8, axis=0)   # stray states of other attractors
        pad = 0.08 * (hi - lo).max()
        axb.scatter(pts[:, 0], pts[:, 1], s=0.3, c="tab:purple", alpha=0.6, linewidths=0, rasterized=True,
                    label="attractor states")
        if M is not None:
            inb = np.all((M[:, :2] >= lo - pad) & (M[:, :2] <= hi + pad), axis=1)
            axb.scatter(M[inb, 0], M[inb, 1], s=0.15, c="k", alpha=0.7, linewidths=0, rasterized=True,
                        label="unstable manifold")
        axb.set_xlim(lo[0] - pad, hi[0] + pad)
        axb.set_ylim(lo[1] - pad, hi[1] + pad)
        axb.ticklabel_format(useOffset=False, style="sci", scilimits=(-3, 3))
        axb.set_title(f"({chr(ord('b') + k)}) band {k + 1}: {int(len(pts)):,} states, width {float((hi - lo).max()):.2g}",
                      fontsize=9)
        if k == 0:
            axb.legend(fontsize=6, markerscale=6)
        summary["bands"].append({"states": int(len(pts)), "extent": (hi - lo).tolist()})
    fig.tight_layout()
    fig.savefig(args.out)
    text = json.dumps(summary, indent=1)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")


if __name__ == "__main__":
    main()
