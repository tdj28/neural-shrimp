"""Figure of the kite of the period-6 shrimp (runs KITE, HR10; human, 19 Oct 2026: "Where the flip curves cross",
"Please figure out this shrimp").

(a) The 640 x 640 fixed-start map HR10 zoomed on the lower-left end of the body, with the two flip curves of the body
    (p6s-ad and p6s-td, red and dashed), the fold curve continued from the S-shaped branch (kite-la-curve-plus/minus,
    black), the corners of Gallas's kite (C: the fold curve's tangent reversal, gold star, filled once located by
    cusp_locate; A and B: where the fold curve meets the flip curves; R: where the flip curves cross), and the line of
    the branch (log10 T fixed) with the interval where the two attracting orbits coexist.
(b) The branch itself (kite-la-branch): the orbit's position along the leading principal direction of the branch's
    states against alpha, solid where the orbit is attracting and dashed where not; the two folds turn the branch into
    an S, and the two attracting sheets overlap in alpha.
Every curve point drawn passed the continuation gate; the figure reports the largest |mu -+ 1| of each curve.

Usage: python tools/plot_p6_kite.py --out paper/llm/figures/fig-p6-kite.png
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
POD = DATA / "explore" / "llm-soft-loop" / "pod64"


def curve(path):
    c = json.loads(path.read_text())
    pts = [q for q in c["points"] if not q.get("rejected")]
    target = 1.0 if c["kind"] == "fold" else -1.0
    gap = max(abs(q["critical_multiplier"][0] - target) for q in pts)
    return c["kind"], np.array([[q["alpha"], q["log10_temperature"]] for q in pts]), pts, gap


def crossings(A, B):
    out = []
    for i in range(len(A) - 1):
        p1, p2 = A[i], A[i + 1]
        for j in range(len(B) - 1):
            q1, q2 = B[j], B[j + 1]
            d1, d2 = p2 - p1, q2 - q1
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-30:
                continue
            t = ((q1[0] - p1[0]) * d2[1] - (q1[1] - p1[1]) * d2[0]) / den
            u = ((q1[0] - p1[0]) * d1[1] - (q1[1] - p1[1]) * d1[0]) / den
            if 0 <= t <= 1 and 0 <= u <= 1:
                out.append(p1 + t * d1)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scan", default="llm-soft-loop/pod64/hr10")
    ap.add_argument("--window", type=float, nargs=4, default=[0.07280, 0.07310, -0.15440, -0.15300],
                    metavar=("A0", "A1", "T0", "T1"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    d = np.load(DATA / "explore" / args.scan / "scan.npz", allow_pickle=True)
    lt, al, per, ly = d["log10_temperature"], d["alpha"], d["period"], d["lyapunov"]
    a0, a1, t0, t1 = args.window
    rows, cols = (lt >= t0) & (lt <= t1), (al >= a0) & (al <= a1)
    lt, al, per, ly = lt[rows], al[cols], per[np.ix_(rows, cols)], ly[np.ix_(rows, cols)]
    chaos = (ly > 1e-3) & (per <= 0)
    neg = np.clip(-ly, 0, None)
    cap = max(float(np.percentile(neg[~chaos], 99)) if (~chaos).any() else 1.0, 1e-12)
    grey = 1.0 - 0.9 * np.clip(neg / cap, 0, 1) ** 0.5
    img = np.stack([grey, grey, grey, np.ones_like(grey)], axis=-1)
    hot = Normalize(0.0, float(np.max(ly[chaos])) if chaos.any() else 1.0)
    img[chaos] = plt.get_cmap("plasma")(hot(ly[chaos]))
    da, dt = (al[1] - al[0]) / 2, (lt[1] - lt[0]) / 2
    ext = [al[0] - da, al[-1] + da, lt[0] - dt, lt[-1] + dt]

    _, adp, _, ad_gap = curve(POD / "p6s/p6s-ad-curve-plus/curve.json")
    _, adm, _, ad_gap2 = curve(POD / "p6s/p6s-ad-curve-minus/curve.json")
    _, tdp, _, td_gap = curve(POD / "p6s/p6s-td-curve-plus/curve.json")
    _, tdm, _, td_gap2 = curve(POD / "p6s/p6s-td-curve-minus/curve.json")
    _, kp, kpp, kp_gap = curve(POD / "kite/kite-la-curve-plus/curve.json")
    _, km, _, km_gap = curve(POD / "kite/kite-la-curve-minus/curve.json")
    ad, td, kite = np.vstack([adp[::-1], adm]), np.vstack([tdp[::-1], tdm]), np.vstack([kp[::-1], km])
    inwin = lambda p: a0 <= p[0] <= a1 and t0 <= p[1] <= t1          # noqa: E731
    R = [p for p in crossings(ad, td) if inwin(p)]
    A = [p for p in crossings(kite, td) if inwin(p)]
    B = [p for p in crossings(kite, ad) if inwin(p)]
    cands = [(q["index"], q["alpha"], q["log10_temperature"]) for q in kpp if q.get("cusp_candidate")
             and inwin((q["alpha"], q["log10_temperature"]))]
    located = []
    for f in sorted((POD / "kite").glob("locate-*/locate.json")):
        rec = json.loads(f.read_text())
        if rec.get("located") and inwin((rec["final"]["alpha"], rec["final"]["log10_temperature"])):
            located.append((rec["final"]["alpha"], rec["final"]["log10_temperature"], f.parent.name))
    br = json.loads((POD / "kite/kite-la-branch/branch.json").read_text())
    bt = float(br["args"]["log_temperature"])

    plt.rcParams.update({"font.size": 8})
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={"width_ratios": [1.0, 1.15]})
    axA.imshow(img, origin="lower", aspect="auto", interpolation="nearest", extent=ext)
    for xy, ls, col in ((ad, "--", "r"), (td, "--", "r"), (kite, "-", "k")):
        axA.plot(xy[:, 0], xy[:, 1], ls, color=col, lw=1.1, zorder=4)
    axA.axhline(bt, color="tab:green", lw=0.6, ls=":", zorder=3)
    pts = br["points"]
    lam = np.array([r["lambda"] for r in pts])
    rho = np.array([r["spectral_radius"] for r in pts])
    turns = [i for i, r in enumerate(pts) if r.get("turned_since_previous")]
    # the coexistence interval on the branch line: the left end of the first attracting sheet to the first fold
    first_fold = lam[turns[0]] if turns else None
    stable0 = np.nonzero(rho[: turns[0]] < 1)[0] if turns else []
    if len(stable0) and first_fold is not None:
        lo = float(lam[stable0[0]])
        axA.plot([lo, first_fold], [bt, bt], color="tab:green", lw=3.0, solid_capstyle="butt", zorder=5)
    if located:
        for x, y, _ in located:
            axA.plot(x, y, "*", color="gold", ms=12, mec="k", mew=0.6, zorder=7)
        Cxy = located[0][:2]
    elif cands:
        for _, x, y in cands:
            axA.plot(x, y, "*", mfc="none", mec="goldenrod", mew=1.2, ms=12, zorder=7)
        Cxy = cands[0][1:]
    else:
        Cxy = None
    labels = [("R", R), ("A", A), ("B", B)] + ([("C", [np.array(Cxy)])] if Cxy is not None else [])
    for name, P in labels:
        for p in P[:1]:
            if name != "C":
                axA.plot(p[0], p[1], "o", ms=3.5, color="w", mec="k", mew=0.6, zorder=8)
            axA.annotate(name, (p[0], p[1]), xytext=(5, 3), textcoords="offset points", fontsize=10, weight="bold",
                         color="k", zorder=9, bbox=dict(boxstyle="round,pad=0.1", fc="w", ec="none", alpha=0.75))
    axA.set_xlim(ext[0], ext[1]); axA.set_ylim(ext[2], ext[3])
    axA.set_xlabel(r"steering strength $\alpha$"); axA.set_ylabel(r"$\log_{10}T$")
    axA.ticklabel_format(useOffset=False)
    axA.tick_params(labelsize=6.5)
    axA.set_title("(a) the kite C-A-R-B on the map", fontsize=8)

    S = np.load(POD / "kite/kite-la-branch/states.npy").reshape(len(pts), -1)
    Sc = S - S.mean(0)
    u = np.linalg.svd(Sc, full_matrices=False)[2][0]
    xproj = Sc @ u
    st = rho < 1
    for k in range(len(pts) - 1):
        axB.plot(lam[k:k + 2], xproj[k:k + 2], "-" if (st[k] and st[k + 1]) else "--",
                 color="tab:blue" if (st[k] and st[k + 1]) else "0.45", lw=1.6 if (st[k] and st[k + 1]) else 1.0)
    for i in turns:
        axB.plot(lam[i], xproj[i], "o", ms=5, mfc="w", mec="k", zorder=5)
    if len(stable0) and first_fold is not None:
        axB.axvspan(lo, first_fold, color="tab:green", alpha=0.15, lw=0)
    axB.set_xlabel(r"steering strength $\alpha$ (at $\log_{10}T=%.7f$)" % bt)
    axB.set_ylabel("orbit point along the branch's leading direction")
    axB.ticklabel_format(useOffset=False)
    axB.tick_params(labelsize=6.5)
    axB.set_title("(b) the branch: an S between two folds (open circles)", fontsize=8)
    axB.plot([], [], "-", color="tab:blue", lw=1.6, label="attracting")
    axB.plot([], [], "--", color="0.45", lw=1.0, label="not attracting")
    axB.fill_between([], [], color="tab:green", alpha=0.15, label="two attracting orbits")
    axB.legend(fontsize=6.5, loc="best")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=400, bbox_inches="tight", pad_inches=0.03)
    summary = {"window": args.window, "R": [list(map(float, p)) for p in R], "A": [list(map(float, p)) for p in A],
               "B": [list(map(float, p)) for p in B], "cusp_candidates": cands, "located": located,
               "branch_log10_temperature": bt, "folds_alpha": [float(lam[i]) for i in turns],
               "coexistence_alpha": [lo, float(first_fold)] if len(stable0) and first_fold is not None else None,
               "gate_max_gap": {"ad": max(ad_gap, ad_gap2), "td": max(td_gap, td_gap2), "kite_fold": max(kp_gap, km_gap)}}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
