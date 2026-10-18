"""The real shrimp (human, 18 Oct 2026: the Chaos article shows a theoretical shrimp, not ours; draw ours with the
verifications on it): the period-5 window of the plane steered with gratitude at block 4 and hostility at block 16
(Qwen2.5-0.5B-Instruct, log10 T = -0.1, float64), drawn as the domain of its attractor found by following it (run FA,
tools/follow_attractor.py), with the verified skeleton on top: the fold and flip curves of the period-5 orbit (FM5),
the located cusps (FM5c, FM5d), the self-crossing of the fold curve, the near-nilpotent point (the head), the orbit
census at 13 points (OC) and the trace-zero curve of the continued orbit (DSS5).

Boundary gate (run before the figure): where the followed domain ends, a computed curve must be near. For every pair
of 8-neighbouring cells with (i) period 5 next to period 10, the flip curve, and (ii) period 5 next to a cell where the
attractor was lost, a fold curve (or, where the lost cell lies beyond the shells, the edge of the cascade), the distance
from the midpoint of the pair to the nearest point of the relevant curves is measured in grid steps; the shares within
1 and 2 steps are reported. Rows are not dropped.

Usage: python tools/plot_real_shrimp.py [--fa llm-soft-loop/pod64/fa/t-0.100] [--json docs/fa/fa-100-summary.json]
         [--out paper/llm/figures/fig-real-shrimp.png]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
sys.path.insert(0, str(ROOT / "tools"))
import plot_fm5 as pf  # noqa: E402

HEAD = (-0.1184836, 0.1017217)      # near-nilpotent point (tools/nilpotent_search.py; paper, "The center")
DSS = "llm-soft-loop/pod64/dss/p5-grat4-host16"


def curve_points(curves, kinds):
    segs = []
    for rec in curves.values():
        if rec["kind"] in kinds and rec["points"]:
            segs.append(np.array([[q["alpha"], q["alpha2"]] for q in rec["points"]]))
    return segs


def dist_to_polylines(p, segs, scale):
    """Distance from point p to the nearest polyline segment, with coordinates divided by scale (grid steps)."""
    best = np.inf
    q = np.asarray(p) / scale
    for P in segs:
        A, B = P[:-1] / scale, P[1:] / scale
        d = B - A
        L2 = np.maximum((d ** 2).sum(1), 1e-30)
        t = np.clip(((q - A) * d).sum(1) / L2, 0, 1)
        proj = A + t[:, None] * d
        best = min(best, float(np.sqrt(((proj - q) ** 2).sum(1)).min()))
    return best


def boundary_gate(fa, curves):
    a1, a2, lab, per = fa["alpha1"], fa["alpha2"], fa["label"], fa["period"]
    step = float(a1[1] - a1[0])
    folds, flips = curve_points(curves, ("fold",)), curve_points(curves, ("flip",))
    n2, n1 = lab.shape
    pairs = {"5|10 (flip)": [], "5|lost (fold)": []}
    for i in range(n2):
        for j in range(n1):
            if not (lab[i, j] == 1 and per[i, j] == 5):
                continue
            for di, dj in pf_neighbours():
                ii, jj = i + di, j + dj
                if not (0 <= ii < n2 and 0 <= jj < n1) or lab[ii, jj] < 0:
                    continue
                mid = ((a1[j] + a1[jj]) / 2, (a2[i] + a2[ii]) / 2)
                if lab[ii, jj] == 1 and per[ii, jj] == 10:
                    pairs["5|10 (flip)"].append(dist_to_polylines(mid, flips, step))
                elif lab[ii, jj] == 0:
                    pairs["5|lost (fold)"].append(dist_to_polylines(mid, folds, step))
    out = {}
    for k, d in pairs.items():
        d = np.array(d)
        out[k] = {"pairs": int(len(d)), "within_1_step": int((d <= 1).sum()), "within_2_steps": int((d <= 2).sum()),
                  "median_steps": float(np.median(d)) if len(d) else None,
                  "max_steps": float(d.max()) if len(d) else None}
    return out


def pf_neighbours():
    return [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def figure(fa, curves, located, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    import dss_window as dw

    a1, a2, lab, lam, per = fa["alpha1"], fa["alpha2"], fa["label"], fa["lyapunov"], fa["period"]
    st = float(a1[1] - a1[0])
    ext = [a1[0] - st / 2, a1[-1] + st / 2, a2[0] - st / 2, a2[-1] + st / 2]
    fig = plt.figure(figsize=(12.5, 6.6), dpi=200)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], wspace=0.22, hspace=0.3)
    axA = fig.add_subplot(gs[:, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 1])
    loc = [(r["final"]["alpha"], r["final"]["alpha2"], r["located"], r["final"].get("c")) for r in located.values()]

    def skeleton(ax, ms=9, lw=1.2):
        for rec in curves.values():
            pts = rec["points"]
            if not pts:
                continue
            fold = rec["kind"] == "fold"
            ax.plot([q["alpha"] for q in pts], [q["alpha2"] for q in pts], "-" if fold else "--",
                    color="k" if fold else "r", lw=lw if fold else lw * 0.9, zorder=4)
        seen = []
        for x, y, ok, c in loc:
            if ok and not any(abs(x - s[0]) < 1e-6 and abs(y - s[1]) < 1e-6 for s in seen):
                seen.append((x, y))
                ax.plot(x, y, "*", color="gold", ms=ms, mec="k", mew=0.6, zorder=6)
        for rec in curves.values():
            if rec["kind"] == "fold" and rec["points"]:
                for _, _, x in pf.self_crossings(np.array([[q["alpha"], q["alpha2"]] for q in rec["points"]])):
                    ax.plot(*x, "x", color="m", ms=ms * 0.8, mew=1.6, zorder=7)
        ax.plot(*HEAD, "o", color="w", mec="k", mew=1.0, ms=ms * 0.6, zorder=8)

    # (a) the followed domain
    member_p = np.where(lab == 1, lam, np.nan)
    member_c = np.where(lab == 2, 1.0, np.nan)        # chaos behind the repeating period-5 text
    member_u = np.where(lab == 3, 1.0, np.nan)        # no period found, no positive exponent (slow convergence)
    lost = np.where(lab == 0, 1.0, np.nan)
    axA.set_facecolor("w")
    axA.imshow(lost, origin="lower", extent=ext, aspect="auto", cmap=ListedColormap(["0.86"]), interpolation="nearest")
    im = axA.imshow(member_p, origin="lower", extent=ext, aspect="auto", cmap="viridis", vmin=-1.2, vmax=0.0,
                    interpolation="nearest")
    axA.imshow(member_c, origin="lower", extent=ext, aspect="auto", cmap=ListedColormap(["k"]), interpolation="nearest")
    axA.imshow(member_u, origin="lower", extent=ext, aspect="auto", cmap=ListedColormap(["0.45"]),
               interpolation="nearest")
    cb = fig.colorbar(im, ax=axA, fraction=0.04, pad=0.02)
    cb.set_label("Lyapunov exponent of the followed attractor (periodic)", fontsize=8)
    skeleton(axA, ms=10, lw=1.1)
    m = lab >= 1
    ii, jj = np.where(m)
    pad = 6 * st
    axA.set_xlim(a1[jj.min()] - pad, a1[jj.max()] + pad)
    axA.set_ylim(a2[ii.min()] - pad, a2[ii.max()] + pad)
    axA.set_title("(a) the period-5 shrimp: domain of its attractor, followed from the body", fontsize=9)

    # (b) the body over the continued orbit's trace
    z = np.load(DATA / "explore" / DSS / "grid.npz", allow_pickle=True)
    b1, b2 = z["alpha"], z["log10_temperature"]
    mu1, mu2 = dw.leading_pair(z)
    ok = z["ok"]
    stable = ok & (np.abs(mu1) < 1)
    t = np.real(mu1 + mu2)
    d1, d2 = b1[1] - b1[0], b2[1] - b2[0]
    e2 = [b1[0] - d1 / 2, b1[-1] + d1 / 2, b2[0] - d2 / 2, b2[-1] + d2 / 2]
    im2 = axB.imshow(np.where(stable, t, np.nan), origin="lower", extent=e2, aspect="auto", cmap="RdBu_r", vmin=-0.6,
                     vmax=0.6, interpolation="nearest")
    axB.imshow(np.where(ok & ~stable, 1.0, np.nan), origin="lower", extent=e2, aspect="auto",
               cmap=ListedColormap(["0.8"]), interpolation="nearest")
    axB.contour(b1, b2, np.where(ok, t, np.nan), levels=[0], colors="k", linewidths=0.9)
    cb2 = fig.colorbar(im2, ax=axB, fraction=0.05, pad=0.02)
    cb2.set_label(r"trace $\mu_1+\mu_2$", fontsize=8)
    skeleton(axB, ms=9)
    axB.set_xlim(e2[0], e2[1])
    axB.set_ylim(e2[2], e2[3])
    axB.set_title("(b) the body: closed trace-zero curve around the head (white circle)", fontsize=9)

    # (c) the junction of the upper leg: two cusps, the crossing, the census
    skeleton(axC, ms=12)
    census = json.loads((ROOT / "docs" / "oc" / "oc-summary.json").read_text())["points"]
    for q in census:
        axC.plot(q["alpha"], q["alpha2"], "o", ms=4.5, color="tab:green" if q["match"] else "tab:orange", mec="k",
                 mew=0.4, zorder=8)
        axC.annotate(f"{q['found']} ({q['attracting']})", (q["alpha"], q["alpha2"]), xytext=(3, 2),
                     textcoords="offset points", fontsize=6.5, zorder=9)
    axC.set_xlim(-0.11935, -0.11880)
    axC.set_ylim(0.10255, 0.10425)
    axC.ticklabel_format(useOffset=False)
    axC.set_title("(c) the leg junction: cusps (stars), crossing (x); orbits found (attracting)", fontsize=9)
    for ax in (axA, axB, axC):
        ax.set_xlabel(r"$\alpha_1$ (gratitude, block 4)", fontsize=8)
        ax.set_ylabel(r"$\alpha_2$ (hostility, block 16)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.savefig(out, bbox_inches="tight")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fa", default="llm-soft-loop/pod64/fa/t-0.100")
    ap.add_argument("--json")
    ap.add_argument("--out")
    args = ap.parse_args()
    fa = np.load(DATA / "explore" / args.fa / "follow.npz", allow_pickle=True)
    _, curves, located = pf.load("llm-soft-loop/pod64/fm5", ["llm-soft-loop/pod64/fm5c", "llm-soft-loop/pod64/fm5d"])
    lab, per, lam = fa["label"], fa["period"], fa["lyapunov"]
    periods = {int(q): int(((lab == 1) & (per == q)).sum()) for q in np.unique(per[lab == 1])}
    summary = {"fa": args.fa, "computed": int((lab >= 0).sum()), "members": int((lab >= 1).sum()),
               "members_by_period": periods,
               "members_chaotic_repeating_text": int((lab == 2).sum()),
               "members_unresolved_slow": int((lab == 3).sum()),
               "not_members": int((lab == 0).sum()), "boundary_gate": boundary_gate(fa, curves),
               "head": HEAD}
    text = json.dumps(summary, indent=1)
    print(text)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(text + "\n")
    if args.out:
        figure(fa, curves, located, args.out)


if __name__ == "__main__":
    main()
