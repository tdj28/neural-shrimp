"""Figure: the strange attractor of the soft-token loop (portrait runs of tools/attractor_portrait.py).

(a) The attractor at one chaotic cell on its first two principal axes: every recorded state of all trajectories, colored
by the most probable next token (the text the loop writes). (b), (c) Two successive zooms into a dense part, showing
its layered fine structure. (d) The return map of the first principal coordinate, s_(n+1) against s_n: stretching and
folding. (e) Correlation sum C(r) on the leading principal coordinates, with the slope (correlation dimension) over the
scaling range. (f) A second cell: banded chaos just past a doubling cascade. Optionally an unstable-manifold run
(tools/unstable_manifold.py) is drawn over (f).

Usage: python tools/plot_attractor_portrait.py --main llm-soft-loop/portraits/f32-strong \
         --second llm-soft-loop/portraits/f32-cascade [--manifold llm-soft-loop/portraits/cascade-wu] --out FIG.png
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))


def load(run):
    z = np.load(DATA / "explore" / run / "portrait.npz", allow_pickle=True)
    rec = json.loads((DATA / "explore" / run / "portrait.json").read_text())
    return z, rec


def zoom_box(P, frac, center=None):
    """A square box of side frac x the data range around the densest grid cell (or a given center)."""
    lo, hi = P.min(0), P.max(0)
    side = frac * (hi - lo).max()
    if center is None:
        H, xe, ye = np.histogram2d(P[:, 0], P[:, 1], bins=60)
        i, j = np.unravel_index(np.argmax(H), H.shape)
        center = np.array([(xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2])
    return center - side / 2, center + side / 2, center


def corr_dim(X, rng, n=6000):
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    Y = X[idx]
    D = np.sqrt(((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1))
    d = D[np.triu_indices(len(Y), 1)]
    r = np.logspace(np.log10(np.percentile(d, 0.05)), np.log10(np.percentile(d, 50)), 30)
    C = np.array([(d < x).mean() for x in r])
    lr, lc = np.log(r), np.log(C)
    k = len(r)
    best = None
    for a in range(0, k - 8):
        b = a + 8
        sl = np.polyfit(lr[a:b], lc[a:b], 1)[0]
        resid = np.std(np.diff(lc[a:b]) / np.diff(lr[a:b]))
        if best is None or resid < best[0]:
            best = (resid, sl, a, b)
    return r, C, best


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--main", required=True)
    ap.add_argument("--second", default=None)
    ap.add_argument("--manifold", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", revision="7ae557604adf67be50417f59c2c2f167def9a775",
                                        cache_dir=str(DATA / "models/hf/hub"))
    z, rec = load(args.main)
    C = z["coords"]                                  # (starts, record, pcs)
    T = z["tokens"]
    P = C.reshape(-1, C.shape[-1])
    toks = T.reshape(-1)
    uniq, counts = np.unique(toks, return_counts=True)
    top = uniq[np.argsort(-counts)][:8]
    cmap = plt.get_cmap("tab10")
    colors = np.full(len(toks), 9)
    for k, t in enumerate(top):
        colors[toks == t] = k
    fig = plt.figure(figsize=(17, 10.5), dpi=170)
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(P[:, 0], P[:, 1], s=0.15, c=[cmap(c) for c in colors], alpha=0.6, linewidths=0, rasterized=True)
    for k, t in enumerate(top):
        ax.plot([], [], "o", color=cmap(k), label=repr(tok.decode([int(t)])))
    ax.legend(fontsize=6, markerscale=1.5, loc="best", title="next token", title_fontsize=6)
    lo1, hi1, c1 = zoom_box(P[:, :2], 0.08)
    ax.add_patch(plt.Rectangle(lo1, *(hi1 - lo1), fill=False, ec="k", lw=0.8))
    a = rec["args"]
    ax.set_title(f"(a) the attractor at $\\alpha={a['alpha']}$, $\\log_{{10}}T={a['log_t']}$ ({len(P):,} states)", fontsize=9)
    ax.set_xlabel("first principal coordinate")
    ax.set_ylabel("second principal coordinate")
    for col, (frac, label) in enumerate(((0.08, "(b) zoom"), (0.008, "(c) zoom, ten times closer"))):
        axz = fig.add_subplot(gs[0, col + 1])
        lo, hi, cc = zoom_box(P[:, :2], frac, c1)
        m = np.all((P[:, :2] >= lo) & (P[:, :2] <= hi), axis=1)
        axz.scatter(P[m, 0], P[m, 1], s=0.6, c=[cmap(c) for c in colors[m]], alpha=0.8, linewidths=0, rasterized=True)
        axz.set_xlim(lo[0], hi[0])
        axz.set_ylim(lo[1], hi[1])
        axz.set_title(f"{label} ({int(m.sum()):,} states)", fontsize=9)
        axz.ticklabel_format(useOffset=False, style="sci", scilimits=(-3, 3))
        if col == 0:
            lo2, hi2, _ = zoom_box(P[:, :2], 0.008, c1)
            axz.add_patch(plt.Rectangle(lo2, *(hi2 - lo2), fill=False, ec="k", lw=0.8))
    ax = fig.add_subplot(gs[1, 0])
    s = C[:, :, 0]
    ax.scatter(s[:, :-1].ravel(), s[:, 1:].ravel(), s=0.15, c="k", alpha=0.4, linewidths=0, rasterized=True)
    ax.set_xlabel("$s_n$ (first principal coordinate)")
    ax.set_ylabel("$s_{n+1}$")
    ax.set_title("(d) return map: stretching and folding", fontsize=9)
    ax = fig.add_subplot(gs[1, 1])
    rng = np.random.default_rng(0)
    r, Cr, best = corr_dim(P[:, :8], rng)
    ax.loglog(r, Cr, "o-", ms=3)
    _, sl, i0, i1 = best
    ax.loglog(r[i0:i1], Cr[i0:i1], "r-", lw=2, label=f"slope {sl:.2f} (correlation dimension)")
    ax.set_xlabel("$r$ (distance in the first 8 principal coordinates)")
    ax.set_ylabel("$C(r)$, share of pairs closer than $r$")
    ax.legend(fontsize=7)
    ax.set_title("(e) correlation sum", fontsize=9)
    summary = {"main": args.main, "states": int(len(P)), "census_main": rec["census"],
               "variance_top": rec["variance_explained_top"][:4], "correlation_dimension_8pc": float(sl),
               "top_tokens": [tok.decode([int(t)]) for t in top]}
    if args.second:
        z2, rec2 = load(args.second)
        P2 = z2["coords"].reshape(-1, z2["coords"].shape[-1])
        ax = fig.add_subplot(gs[1, 2])
        ax.scatter(P2[:, 0], P2[:, 1], s=0.15, c="tab:purple", alpha=0.5, linewidths=0, rasterized=True)
        if args.manifold:
            mz = np.load(DATA / "explore" / args.manifold / "manifold.npz")
            ax.scatter(mz["proj"][:, 0], mz["proj"][:, 1], s=0.1, c="k", alpha=0.6, linewidths=0, rasterized=True)
        a2 = rec2["args"]
        ax.set_title(f"(f) banded chaos past a cascade: $\\alpha={a2['alpha']}$", fontsize=9)
        ax.set_xlabel("first principal coordinate")
        ax.set_ylabel("second principal coordinate")
        summary["census_second"] = rec2["census"]
    fig.tight_layout()
    fig.savefig(args.out)
    text = json.dumps(summary, indent=1)
    print(text)
    if args.json:
        Path(args.json).write_text(text + "\n")


if __name__ == "__main__":
    main()
