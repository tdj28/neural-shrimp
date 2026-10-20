"""Figure of run SYNC (coupled soft-token loops): paper/llm/figures/fig-sync.png.

(a) mean relative distance |x_A - x_B| / |x_A| against the coupling c for the three chaotic cells, with the predicted
    threshold c* = (1 - exp(-lambda_1)) / 2 marked; (b) measured against predicted transverse exponent; (c) the
    distance time series of the pair near threshold with the most bursts (if any pair bursts).

Usage: python tools/plot_sync.py [--sweep sweep-near] [--refine NAME ...] [--ts sweep:cell:c ...]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

COLORS = ["#1f77b4", "#d62728", "#2ca02c"]
OKABE = ["#0072B2", "#D55E00", "#009E73"]          # --compact: colorblind-safe colors with distinct markers
MARKERS = ["o", "s", "^"]


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import explore_llm_attractors as ela

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", default=str(ROOT / "docs" / "sync" / "sync-summary.json"))
    ap.add_argument("--sweep", default="sweep-near", help="sweep shown in panel (a)")
    ap.add_argument("--refine", nargs="*", default=[], help="further sweeps overlaid in panel (a)")
    ap.add_argument("--ts", nargs="*", default=None, help="time-series panels, each sweep:cell:c (default: the pair "
                    "with the most bursts)")
    ap.add_argument("--out", default=str(ROOT / "paper" / "llm" / "figures" / "fig-sync.png"))
    ap.add_argument("--compact", action="store_true", help="panels (a) and (b) stacked for one column (the article)")
    args = ap.parse_args()
    S = json.loads(Path(args.summary).read_text())
    if args.compact:
        return compact(S, args, plt)
    picks = [(a, int(b), float(c)) for a, b, c in (t.split(":") for t in args.ts)] if args.ts else []
    npan = 2 + max(1, len(picks))
    fig, axes = plt.subplots(1, npan, figsize=(4.5 * npan, 4.0))
    ax = axes[0]
    for k, (ci, c) in enumerate(sorted(S["cells"].items())):
        col = COLORS[k % 3]
        label = f"cell {int(ci) + 1}: $\\lambda_1$ = {c['lambda1_jvp']:.3f}"
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
        ax.errorbar(pr, me, yerr=2 * se, fmt="o", ms=4, color=COLORS[k % 3], label=f"cell {int(ci) + 1}")
        lo, hi = min(lo, pr.min(), me.min()), max(hi, pr.max(), me.max())
    ax.plot([lo, hi], [lo, hi], "k-", lw=0.8)
    ax.axhline(0, color="0.6", lw=0.6)
    ax.axvline(0, color="0.6", lw=0.6)
    ax.set_xlabel(r"predicted $\lambda_1 + \ln|1-2c|$ (per step)")
    ax.set_ylabel(r"measured transverse exponent (twin)")
    ax.set_title("(b) transverse exponent; bars: 2 s.e.")
    ax.legend(fontsize=8)
    if not picks:
        best, pick = 0, None
        for ci, c in S["cells"].items():
            for name, sw in c["sweeps"].items():
                for r in sw["pairs"]:
                    if r["bursts_1e-8_to_1e-3"] > best:
                        best, pick = r["bursts_1e-8_to_1e-3"], (name, int(ci), r["c"])
        picks = [pick]
    for j, pick in enumerate(picks):
        ax = axes[2 + j]
        letter = "cdefg"[j]
        if pick is None:
            ax.text(0.5, 0.5, "no pair bursts\n(no passage from $<10^{-8}$ to $>10^{-3}$)", ha="center",
                    va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        name, ci, cc = pick
        base = ela.DATA / "explore" / "sync" / name
        sj = json.loads((base / "sweep.json").read_text())
        dist = np.load(base / "sweep.npz")["dist"]
        k = [i for i, m in enumerate(sj["pairs"]) if m["cell"] == ci and abs(m["c"] - cc) < 1e-12]
        if not k:
            raise SystemExit(f"no pair cell {ci} c {cc} in {name}")
        d = np.maximum(dist[k[0]], 1e-18)
        cells = sorted(S["cells"])
        ax.semilogy(np.arange(len(d)), d, lw=0.4, color=COLORS[cells.index(str(ci)) % 3])
        cst = S["cells"][str(ci)]["c_star_jvp"]
        ax.set_title(f"({letter}) cell {int(ci) + 1}, c = {cc:g} (predicted $c^*$ = {cst:.3f})")
        ax.set_xlabel("step")
        ax.set_ylabel(r"$|x_A - x_B| / |x_A|$")
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200)
    print(args.out, "time series:", picks)


def compact(S, args, plt):
    """Panels (a) and (b) stacked in one column, short titles; the caption names the curves' meaning."""
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(3.4, 4.6), gridspec_kw={"hspace": 0.42})
    for k, (ci, c) in enumerate(sorted(S["cells"].items())):
        col, mk = OKABE[k % 3], MARKERS[k % 3]
        pts = {}
        for name in [args.sweep] + list(args.refine):
            sw = c["sweeps"].get(name)
            if sw is None:
                continue
            for r in sw["pairs"]:
                pts[r["c"]] = max(r["mean_dist_second_half"], 1e-17)
        xs = sorted(pts)
        ax.semilogy(xs, [pts[x] for x in xs], "-", marker=mk, ms=2.6, lw=0.9, color=col,
                    label=f"$\\lambda_1={c['lambda1_jvp']:.3f}$")
        ax.axvline(c["c_star_jvp"], color=col, ls="--", lw=0.9)
        pr = np.array([r["predicted"] for r in c["transverse"]])
        me = np.array([r["measured"] for r in c["transverse"]])
        se = np.array([r["measured_se"] or 0.0 for r in c["transverse"]])
        bx.errorbar(pr, me, yerr=2 * se, fmt=mk, ms=3, color=col, lw=0.8, capsize=0)
    ax.axhline(S["sync_tol"], color="0.5", ls=":", lw=0.8)
    ax.set_xlabel("coupling $c$", fontsize=7.5)
    ax.set_ylabel(r"$|x_A-x_B|/|x_A|$", fontsize=7.5)
    ax.set_title("(a)", fontsize=8, loc="left")
    ax.legend(fontsize=6, loc="upper right", bbox_to_anchor=(1.0, 0.9), frameon=False, handlelength=1.6)
    lo, hi = -0.85, 0.47
    bx.plot([lo, hi], [lo, hi], "k-", lw=0.7)
    bx.axhline(0, color="0.6", lw=0.5)
    bx.axvline(0, color="0.6", lw=0.5)
    bx.set_xlabel(r"predicted $\lambda_1+\ln|1-2c|$ (per step)", fontsize=7.5)
    bx.set_ylabel("measured (per step)", fontsize=7.5)
    bx.set_title("(b)", fontsize=8, loc="left")
    for a in (ax, bx):
        a.tick_params(labelsize=6.5)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", pad_inches=0.03)
    print(args.out)


if __name__ == "__main__":
    main()
