"""Figure fig-p6-ditches: where the period-6 orbit's leading multiplier changes sign over the body of the period-6 shrimp
(the measurements behind the dark ditches of fig-shrimp-anatomy-pair; human request of 20 Oct 2026: the dots in a figure
of their own in the full report and the blog, not in the article).

(a) The fine float64 map of the body (llm-soft-loop/pod64/hr10, drawn as plot_p6_shrimp.py draws it) with the sign
    changes of three measurements: the 31 x 31 grid over the body (dss/p6-body; blue dots: midpoints of neighbouring
    solved cells whose real leading multipliers have opposite signs, and cells where it is a complex pair), the finer
    patch across the top (dss/ditch2d-b0..b3; orange dots, same rule) and the two lines (p6s/ditch-top-a across the dim
    top ditch, p6s/ditch-main-b across the main ditch; black ticks), with the minimum of line A (open circle).
(b) Along line A: the leading multiplier (real, signed) and the next modulus against log10 T. The leading multiplier dips
    but keeps its sign, so the top ditch is a same-sign valley at this resolution.
(c) Along line B (control): the leading multiplier against alpha; it changes sign through a complex pair (open markers:
    the modulus of a complex leading pair).

Usage: python tools/plot_p6_ditches.py [--out paper/llm/figures/fig-p6-ditches.png]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import plot_p6_shrimp as p6  # noqa: E402
import plot_shrimp_anatomy as anat  # noqa: E402
from explore_llm_soft_loop import DATA  # noqa: E402

GRID = "llm-soft-loop/pod64/dss/p6-body"
PATCH = [f"llm-soft-loop/pod64/dss/ditch2d-b{k}" for k in range(4)]
LINE_A, LINE_B = "llm-soft-loop/p6s/ditch-top-a", "llm-soft-loop/p6s/ditch-main-b"
BLUE, ORANGE = "#4f8fd8", "#E69F00"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scan", default="llm-soft-loop/pod64/hr10")
    ap.add_argument("--out", default=str(ROOT / "paper/llm/figures/fig-p6-ditches.png"))
    ap.add_argument("--json", default=str(ROOT / "paper/llm/figures/fig-p6-ditches.json"))
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.0, 3.3))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], hspace=0.55, wspace=0.32)
    ax = fig.add_subplot(gs[:, 0])
    img, ext, _, _, _ = p6.map_image(args.scan)
    ax.imshow(img, origin="lower", aspect="auto", interpolation="nearest", extent=ext)
    rec = {}
    g = anat.sign_changes(DATA / "explore" / GRID / "grid.npz")
    rec["grid_31x31"] = {k: v for k, v in g.items() if k != "points"}
    if g["points"]:
        x, y = zip(*g["points"])
        ax.plot(x, y, "o", ms=2.6, mfc=BLUE, mec="white", mew=0.3, ls="none")
    pts, rec["patch"] = [], []
    for band in PATCH:
        s = anat.sign_changes(DATA / "explore" / band / "grid.npz")
        pts += s["points"]
        rec["patch"].append({k: v for k, v in s.items() if k != "points"})
    if pts:
        x, y = zip(*pts)
        ax.plot(x, y, "o", ms=1.9, mfc=ORANGE, mec="black", mew=0.2, ls="none")
    za = np.load(DATA / "explore" / LINE_A / "grid.npz", allow_pickle=True)
    zb = np.load(DATA / "explore" / LINE_B / "grid.npz", allow_pickle=True)
    ax.plot(np.full(za["log10_temperature"].size, za["alpha"][0]), za["log10_temperature"], "|", color="black", ms=3, mew=0.6)
    ax.plot(zb["alpha"], np.full(zb["alpha"].size, zb["log10_temperature"][0]), "_", color="black", ms=3, mew=0.6)
    ia = int(np.nanargmin(za["mu"][:, 0]))
    ax.plot(za["alpha"][0], za["log10_temperature"][ia], "o", ms=6, mfc="none", mec="black", mew=0.8)
    ax.set_xlim(0.07240, 0.07380)
    ax.set_ylim(-0.15450, -0.15000)
    ax.set_xlabel(r"steering strength $\alpha$", fontsize=7.5)
    ax.set_ylabel(r"$\log_{10}T$", fontsize=7.5)
    ax.set_title("(a)", fontsize=8, loc="left")
    ax.tick_params(labelsize=6.5)
    # (b) line A across the dim top ditch
    bx = fig.add_subplot(gs[0, 1])
    lt, mu, nx = za["log10_temperature"], za["mu"][:, 0], za["next_modulus"][:, 0]
    bx.plot(lt, mu, "-o", color="black", ms=2.2, lw=0.9)
    bx.plot(lt, nx, "--", color="0.45", lw=0.9)
    bx.axhline(0, color="0.6", lw=0.5)
    bx.set_xlabel(r"$\log_{10}T$ at $\alpha=0.07318$", fontsize=7)
    bx.set_ylabel("multiplier", fontsize=7)
    bx.set_title("(b) across the top ditch", fontsize=8, loc="left")
    bx.tick_params(labelsize=6)
    # (c) line B across the main ditch (control)
    cx = fig.add_subplot(gs[1, 1])
    al, mb, mib = zb["alpha"], zb["mu"][0], np.nan_to_num(zb["mu_imag"][0])
    real = np.abs(mib) == 0
    cx.plot(al[real], mb[real], "o", color="black", ms=2.2)
    cx.plot(al[~real], np.hypot(mb, mib)[~real], "o", mfc="none", mec="black", ms=3.2, mew=0.7)
    cx.plot(al, mb, "-", color="black", lw=0.6)
    cx.axhline(0, color="0.6", lw=0.5)
    cx.set_xlabel(r"$\alpha$ at $\log_{10}T=-0.153$", fontsize=7)
    cx.set_xticks([0.07292, 0.07296, 0.07300, 0.07304])
    cx.set_ylabel("multiplier", fontsize=7)
    cx.set_title("(c) across the main ditch", fontsize=8, loc="left")
    cx.tick_params(labelsize=6)
    fig.savefig(args.out, dpi=300, bbox_inches="tight", pad_inches=0.03)
    rec["line_a"] = {"alpha": float(za["alpha"][0]), "rows": int(lt.size), "min_leading": float(mu[ia]),
                     "min_at_log10T": float(lt[ia]), "max_next_modulus": float(np.nanmax(nx)),
                     "leading_positive_everywhere": bool(np.all(mu > 0))}
    rec["line_b"] = {"log10T": float(zb["log10_temperature"][0]), "cells": int(al.size), "complex_cells": int((~real).sum()),
                     "leading_min": float(np.nanmin(mb)), "leading_max": float(np.nanmax(mb))}
    Path(args.json).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps(rec, indent=1)[:1500])


if __name__ == "__main__":
    main()
