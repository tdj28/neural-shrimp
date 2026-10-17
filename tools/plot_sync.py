"""Figure of run SYNC (coupled soft-token loops): paper/llm/figures/fig-sync.png.

(a) mean relative distance |x_A - x_B| / |x_A| against the coupling c for the three chaotic cells, with the predicted
    threshold c* = (1 - exp(-lambda_1)) / 2 marked; (b) measured against predicted transverse exponent; (c) the
    distance time series of the pair near threshold with the most bursts (if any pair bursts).

Usage: python tools/plot_sync.py [--sweep sweep-near] [--ts-sweep NAME --ts-cell I --ts-c C]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

COLORS = ["#1f77b4", "#d62728", "#2ca02c"]


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import explore_llm_attractors as ela

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default=str(ROOT / "docs" / "sync" / "sync-summary.json"))
    ap.add_argument("--sweep", default="sweep-near", help="sweep shown in panel (a)")
    ap.add_argument("--refine", nargs="*", default=[], help="further sweeps overlaid in panel (a)")
    ap.add_argument("--ts-sweep", default=None)
    ap.add_argument("--ts-cell", type=int, default=None)
    ap.add_argument("--ts-c", type=float, default=None)
    ap.add_argument("--out", default=str(ROOT / "paper" / "llm" / "figures" / "fig-sync.png"))
    args = ap.parse_args()
    S = json.loads(Path(args.summary).read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ax = axes[0]
    for k, (ci, c) in enumerate(sorted(S["cells"].items())):
        col = COLORS[k % 3]
        label = f"cell {ci}: $\\lambda_1$ = {c['lambda1_jvp']:.3f}"
        pts = {}
        for name in [args.sweep] + list(args.refine):
            sw = c["sweeps"].get(name)
            if sw is None:
                continue
            for r in sw["pairs"]:
                pts[r["c"]] = max(r["mean_dist_second_half"], 1e-17)
        if pts:
            xs = sorted(pts)
            ax.semilogy(xs, [pts[x] for x in xs], "o-", ms=3, color=col, label=label)
        ax.axvline(c["c_star_jvp"], color=col, ls="--", lw=1)
    ax.axhline(S["sync_tol"], color="0.5", ls=":", lw=1)
    ax.set_xlabel("coupling c")
    ax.set_ylabel(r"mean $|x_A - x_B| / |x_A|$ (second half)")
    ax.set_title("(a) synchronization; dashed: predicted $c^*$")
    ax.legend(fontsize=8, loc="lower left")
    ax = axes[1]
    lo, hi = np.inf, -np.inf
    for k, (ci, c) in enumerate(sorted(S["cells"].items())):
        pr = np.array([r["predicted"] for r in c["transverse"]])
        me = np.array([r["measured"] for r in c["transverse"]])
        se = np.array([r["measured_se"] or 0.0 for r in c["transverse"]])
        ax.errorbar(pr, me, yerr=2 * se, fmt="o", ms=4, color=COLORS[k % 3], label=f"cell {ci}")
        lo, hi = min(lo, pr.min(), me.min()), max(hi, pr.max(), me.max())
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel(r"predicted $\lambda_1 + \ln|1-2c|$ (per step)")
    ax.set_ylabel(r"measured transverse exponent (twin)")
    ax.set_title("(b) transverse exponent; bars: 2 s.e.")
    ax.legend(fontsize=8)
    ax = axes[2]
    pick = None
    if args.ts_sweep is not None:
        pick = (args.ts_sweep, args.ts_cell, args.ts_c)
    else:
        best = 0
        for ci, c in S["cells"].items():
            for name, sw in c["sweeps"].items():
                for r in sw["pairs"]:
                    if r["bursts_1e-8_to_1e-3"] > best:
                        best, pick = r["bursts_1e-8_to_1e-3"], (name, int(ci), r["c"])
    if pick is None:
        ax.text(0.5, 0.5, "no pair bursts\n(no passage from $<10^{-8}$ to $>10^{-3}$)", ha="center",
                va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        name, ci, cc = pick
        base = ela.DATA / "explore" / "sync" / name
        sj = json.loads((base / "sweep.json").read_text())
        dist = np.load(base / "sweep.npz")["dist"]
        k = [i for i, m in enumerate(sj["pairs"]) if m["cell"] == ci and abs(m["c"] - cc) < 1e-12]
        if not k:
            raise SystemExit(f"no pair cell {ci} c {cc} in {name}")
        d = np.maximum(dist[k[0]], 1e-18)
        ax.semilogy(np.arange(len(d)), d, lw=0.5, color=COLORS[ci % 3])
        cst = S["cells"][str(ci)]["c_star_jvp"]
        ax.set_title(f"(c) cell {ci}, c = {cc:g} ($c^*$ = {cst:.3f})")
        ax.set_xlabel("step")
        ax.set_ylabel(r"$|x_A - x_B| / |x_A|$")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(args.out, "time series:", pick)


if __name__ == "__main__":
    main()
