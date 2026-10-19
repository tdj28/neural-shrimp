"""Figure: why the hook inside the kite of the period-6 shrimp has its lower edge (runs HOOKEDGE, HOOKSADDLE, HOOK-EXIT).

(a) The fixed-start map of the hook (HR12; gray: contraction rate) with the continued fold curve of the kite and the
    tracked edges of the fixed start's outcome (bisected in log10 T to 2.6e-13; brackets whose end exponents differ by
    at least --min-sep). The star marks the steering strength of panels (b)-(d).
(b) Distance to the saddle orbit (a third period-6 orbit, one unstable multiplier mu_u) along trajectories from the
    fixed start at log10 T = edge - delta: the start comes in along the saddle's stable manifold, lingers, and leaves.
(c) Exit time from the saddle's neighborhood against ln(1/delta), on both sides of the edge, with the slope 6/ln mu_u
    that one unstable multiplier predicts.
(d) Closest approach to the saddle against delta (predicted proportional to delta).

Usage: python tools/plot_hook_theory.py --edges llm-soft-loop/pod64/kite/hook-edge \
    --exit llm-soft-loop/kite/hook-saddle-exit-v2 --out paper/llm/figures/fig-p6-hook-theory.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import plot_p6_kite as pk  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--edges", required=True)
    ap.add_argument("--exit", required=True)
    ap.add_argument("--map", default="llm-soft-loop/pod64/hr12")
    ap.add_argument("--min-sep", type=float, default=0.1, help="smallest difference of the bracket ends' exponents")
    ap.add_argument("--radius", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z = np.load(pk.DATA / "explore" / args.map / "scan.npz")
    A, T, L = z["alpha"], z["log10_temperature"], z["lyapunov"]
    E = json.loads((pk.DATA / "explore" / args.edges / "hook-edge.json").read_text())["edges"]
    pts = np.array([[e["alpha"], e["log10T_edge"]] for e in E if e["separation"] >= args.min_sep])
    _, kp, _, _ = pk.curve(pk.POD / "kite/kite-la-curve-plus/curve.json")
    _, km, _, _ = pk.curve(pk.POD / "kite/kite-la-curve-minus/curve.json")
    X = json.loads((pk.DATA / "explore" / args.exit / "hook-saddle-exit.json").read_text())
    D = np.load(pk.DATA / "explore" / args.exit / "distance-to-saddle.npz")
    mu_u, pred = X["source"]["mu_u"], X["predicted_slope_steps_per_e_fold"]
    a0, t0 = X["source"]["alpha"], X["source"]["edge_log10T"]

    fig, ax = plt.subplots(2, 2, figsize=(7.0, 6.4))
    a = ax[0, 0]
    a.imshow(np.clip(np.where(L <= 1e-3, L, np.nan), -0.6, 0), origin="lower", extent=[A[0], A[-1], T[0], T[-1]],
             aspect="auto", cmap="gray", vmin=-0.6, vmax=0, interpolation="nearest")
    for k, lab in ((kp, "fold curve (continued)"), (km, None)):
        a.plot(k[:, 0], k[:, 1], "-", color="gold", lw=0.9, label=lab)
    a.plot(pts[:, 0], pts[:, 1], ".", color="tab:cyan", ms=1.6, label="tracked edges of the outcome")
    a.plot(a0, t0, "*", color="tab:red", mec="k", mew=0.4, ms=9, label="steering strength of (b)-(d)")
    a.set_xlim(0.07293, 0.07301)
    a.set_ylim(-0.15372, -0.15333)
    a.ticklabel_format(useOffset=False)
    a.set_xticks([0.07294, 0.07296, 0.07298, 0.07300])
    a.tick_params(labelsize=6)
    a.set_xlabel(r"steering strength $\alpha$", fontsize=7)
    a.set_ylabel(r"$\log_{10}T$", fontsize=7)
    a.legend(fontsize=5.5, loc="lower left")
    a.set_title("(a) the hook and its tracked edges", fontsize=8)

    dist, dl, side = D["distance"], D["delta"], D["side"]
    below = [i for i in np.nonzero(side < 0)[0] if np.isclose(np.log10(dl[i]) % 2, 0) or np.isclose(np.log10(dl[i]) % 2, 2)]
    b = ax[0, 1]
    cmap = plt.get_cmap("viridis")
    for j, i in enumerate(below):
        b.semilogy(np.arange(1, dist.shape[0] + 1), dist[:, i], lw=0.8, color=cmap(j / max(1, len(below) - 1)),
                   label=rf"$\delta=10^{{{np.log10(dl[i]):.0f}}}$")
    b.axhline(args.radius, color="k", lw=0.6, ls=":")
    b.set_xlim(0, 700)
    b.set_xlabel("step", fontsize=7)
    b.set_ylabel("distance to the saddle orbit (relative)", fontsize=7)
    b.tick_params(labelsize=6)
    b.legend(fontsize=5.5, ncol=2, loc="lower right")
    b.set_title(r"(b) lingering at the saddle, $\log_{10}T=$ edge $-\,\delta$", fontsize=8)

    c = ax[1, 0]
    key = f"exit_{args.radius:g}"
    xs, ys = [], []
    for sd, mk, lab in (("below", "o", r"below the edge (ends on $S$)"), ("above", "s", r"above (ends on $W$)")):
        R = [r for r in X["rows"] if r["side"] == sd and r[key] is not None]
        x = np.log(1 / np.array([r["delta"] for r in R]))
        y = np.array([r[key] for r in R])
        c.plot(x, y, mk, ms=3.5, mfc="none", label=lab)
        xs.append(x)
        ys.append(y)
    x, y = np.concatenate(xs), np.concatenate(ys)
    off = np.median(y - pred * x)
    xx = np.linspace(x.min(), x.max(), 10)
    c.plot(xx, off + pred * xx, "k--", lw=0.8, label=rf"slope $6/\ln\mu_u$ = {pred:.2f}, $\mu_u$ = {mu_u:.4f}")
    c.set_xlabel(r"$\ln(1/\delta)$", fontsize=7)
    c.set_ylabel(f"last step within {args.radius:g} of the saddle", fontsize=7)
    c.tick_params(labelsize=6)
    c.legend(fontsize=6)
    c.set_title("(c) time spent at the saddle", fontsize=8)

    d = ax[1, 1]
    for sd, mk in (("below", "o"), ("above", "s")):
        R = [r for r in X["rows"] if r["side"] == sd]
        d.loglog([r["delta"] for r in R], [r["closest"] for r in R], mk, ms=3.5, mfc="none")
    dd = np.logspace(-14, -6, 10)
    k1 = np.median([r["closest"] / r["delta"] for r in X["rows"]])
    d.loglog(dd, k1 * dd, "k--", lw=0.8, label="proportional to $\\delta$")
    d.set_xlabel(r"distance $\delta$ from the edge in $\log_{10}T$", fontsize=7)
    d.set_ylabel("closest approach to the saddle (relative)", fontsize=7)
    d.tick_params(labelsize=6)
    d.legend(fontsize=6)
    d.set_title("(d) how close the start comes", fontsize=8)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=250)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
