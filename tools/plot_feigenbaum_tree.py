"""Figure: the Feigenbaum tree of the crescent cascade (run FT). Three float64 line scans along log10 T = -0.156
(Qwen2.5-0.5B-Instruct, philosophy direction at layer 12), each a bifurcation diagram (the 256 recorded attractor
states of every cell, projected on the unit steering direction; red where the cell is chaotic) over its largest
Lyapunov exponent. The flips located by Newton's method (cascade/crescent-t-0.156*, Table tab:cascade156) are drawn
as dashed lines, and the accumulation point estimated from the last two flips with Feigenbaum's delta as a dotted line.

Check (written into the JSON): along each line, the first cell whose detected period is 2p after the last cell of
period p, against the Newton flip p -> 2p; a period above the scan's maximum (48) is detected as none.

Usage: python tools/plot_feigenbaum_tree.py OUT.png OUT.json [SCAN_PREFIX]   (default llm-soft-loop/pod64/scan64/ft-)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
CASCADE = ("crescent-t-0.156", "crescent-t-0.156-b", "crescent-t-0.156-c", "crescent-t-0.156-d", "crescent-t-0.156-f")
DELTA = 4.669201609102990
# y windows of the zooms, chosen on one branch so that its splittings are visible (self-similar zooms); None = full
Y_WINDOWS = {"a": None, "b": (0.00120, 0.00170), "c": (0.001243, 0.001258)}


def flips():
    """Newton-located flips p -> 2p along log10 T = -0.156, by period p (the first value found per period)."""
    import explore_llm_attractors as ela
    out = {}
    for run in CASCADE:
        f = ela.DATA / "explore" / "llm-soft-loop" / "pod64" / "cascade" / run / "cascade.json"
        for r in json.loads(f.read_text())["flips"]:
            out.setdefault(int(r["period"]), float(r["flip"]["lambda"]))
    return dict(sorted(out.items()))


def first_doubling(al, per, p, max_period):
    """First alpha whose detected period is 2p (or none, when 2p exceeds max_period) after the last cell of period p."""
    idx = np.nonzero(per == p)[0]
    if idx.size == 0:
        return None
    after = np.arange(idx[-1] + 1, per.size)
    want = (per[after] == 2 * p) if 2 * p <= max_period else (per[after] <= 0)
    hit = after[want]
    return float(al[hit[0]]) if hit.size else None


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches
    import matplotlib.ticker

    import explore_llm_attractors as ela

    out_png, out_json = sys.argv[1], sys.argv[2]
    prefix = sys.argv[3] if len(sys.argv) > 3 else "llm-soft-loop/pod64/scan64/ft-"
    lines = [l for l in "abc" if (ela.DATA / "explore" / f"{prefix}{l}" / "scan.npz").exists()]
    if not lines:
        raise SystemExit(f"no finished line scans under {prefix}")
    fl = flips()
    pers = sorted(fl)
    a_inf = fl[pers[-1]] + (fl[pers[-1]] - fl[pers[-2]]) / (DELTA - 1.0)
    summary = {"flips": {str(p): fl[p] for p in pers}, "accumulation_estimate": a_inf, "lines": {}}
    fig = plt.figure(figsize=(7.2, 3.4 * len(lines)))
    outer = fig.add_gridspec(len(lines), 1, hspace=0.32)
    prev = None                                                 # the previous panel's axes, to mark this zoom on it
    for k, name in enumerate(lines):
        d = ela.DATA / "explore" / f"{prefix}{name}"
        z = np.load(d / "scan.npz")
        meta = json.loads((d / "scan.json").read_text())
        al, L, P = z["alpha"], z["lyapunov"][0], z["period"][0]
        proj = z["projection"][0, :, :, 0]                       # (cells, recorded states) on the steering direction
        chaotic = (L > 1e-3) & (P <= 0)
        maxp = int(meta["args"]["max_period"])
        inner = outer[k].subgridspec(2, 1, height_ratios=[3.2, 1.0], hspace=0.06)
        ax = fig.add_subplot(inner[0])
        lx = fig.add_subplot(inner[1], sharex=ax)
        xs = np.repeat(al[:, None], proj.shape[1], axis=1)
        # solid points at full opacity: periodic branches read as lines, chaotic bands as filled red regions
        ax.plot(xs[~chaotic].ravel(), proj[~chaotic].ravel(), "o", ms=1.1, mew=0, color="black", rasterized=True)
        ax.plot(xs[chaotic].ravel(), proj[chaotic].ravel(), "o", ms=0.8, mew=0, color="#d62728", alpha=0.8,
                rasterized=True)
        lo, hi = float(al[0]), float(al[-1])
        checks = {}
        for p in pers:
            if lo < fl[p] < hi:
                for a in (ax, lx):
                    a.axvline(fl[p], color="0.45", lw=0.6, ls=(0, (3, 2)), zorder=0)
                found = first_doubling(al, P, p, maxp) if p <= maxp else None
                step = float(al[1] - al[0])
                checks[f"{p}->{2 * p}"] = {"newton": fl[p], "scan": found,
                                          "cells_off": None if found is None else round((found - fl[p]) / step, 2)}
        if lo < a_inf < hi:
            for a in (ax, lx):
                a.axvline(a_inf, color="0.2", lw=0.7, ls=(0, (1, 1.5)), zorder=0)
        lx.plot(al, np.clip(L, -1.0, None), color="black", lw=0.6)
        lx.axhline(0.0, color="0.5", lw=0.5)
        lx.set_ylabel(r"$\lambda_{\max}$", fontsize=8)
        ax.set_ylabel("state on the\nsteering direction", fontsize=8)
        yw = Y_WINDOWS.get(name)
        if yw is not None:
            ax.set_ylim(*yw)
        ax.text(0.01, 0.97, f"({name})", transform=ax.transAxes, fontsize=9, va="top", ha="left",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.0))
        if prev is not None:                                    # this zoom's window, drawn on the previous panel
            y0, y1 = ax.get_ylim()
            prev.add_patch(matplotlib.patches.Rectangle((lo, y0), hi - lo, y1 - y0, fill=False, edgecolor="#1f77b4",
                                                        lw=0.9))
        prev = ax
        plt.setp(ax.get_xticklabels(), visible=False)
        for a in (ax, lx):
            a.tick_params(labelsize=7)
            a.set_xlim(lo, hi)
        lx.ticklabel_format(axis="x", useOffset=False, style="plain")
        lx.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
        if k == len(lines) - 1:
            lx.set_xlabel(r"steering strength $\alpha$ (philosophy, layer 12; $\log_{10}T=-0.156$)", fontsize=8)
        summary["lines"][name] = {"alpha": [lo, hi], "cells": int(al.size), "chaotic": int(chaotic.sum()),
                                  "periods": {str(int(q)): int(c) for q, c in zip(*np.unique(P, return_counts=True))},
                                  "transient": meta["args"]["transient"], "record": meta["args"]["record"],
                                  "doubling_checks": checks}
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.05)
    Path(out_json).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v["doubling_checks"] for k, v in summary["lines"].items()}, indent=1))


if __name__ == "__main__":
    main()
