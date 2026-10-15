"""Statistics and figure for a doubly superstable grid of one periodic orbit (tools/doubly_superstable.py) in a
two-dimensional reading: the two leading multipliers mu1, mu2 (by modulus) of the continued orbit, their trace
t = mu1 + mu2 and product d = mu1 mu2 (real for a complex pair). Near a superstable curve of a one-dimensional map,
mu1 would pass through zero; in a map with a second, comparable multiplier, mu1 instead meets mu2, and the two form
a complex pair (d > 0) or exchange through real values (d < 0) where t changes sign. The zero set of t takes the
place of the superstable curves, and a crossing of two of its curves shows at cell resolution as a 2 x 2 block whose
trace signs alternate like a checkerboard.

The modulus of the leading multiplier is read from the stored multipliers, not from grid.npz's "mu", which keeps only
the real part (a complex pair with small real part would look superstable).

Usage:
  python tools/dss_window.py RUN [--tb TB_NPZ] [--json OUT_JSON] [--out PNG] [--partial]
    --partial: a grid cut by its time limit has no grid.json; read the seed from the last saved grid.npz (written every
    25 cells) and mark the statistics as partial, with the number of cells it holds
    RUN: folder with grid.npz and grid.json (for example explore/llm-soft-loop/pod64/dss/p5-grat4-host16)
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))


def leading_pair(z):
    """(mu1, mu2) as complex arrays of shape (rows, cols) from the stored three leading multipliers."""
    m = z["multipliers"]
    lam = m[..., 0] + 1j * m[..., 1]
    return lam[..., 0], lam[..., 1]


def checkerboards(sign, valid):
    """2 x 2 blocks of valid cells whose signs alternate (s00 = s11 != s01 = s10): the cell-resolution signature of a
    crossing of two zero curves. Returns a list of (row, col) of each block's lower-left cell."""
    out = []
    for i in range(sign.shape[0] - 1):
        for j in range(sign.shape[1] - 1):
            if not valid[i:i + 2, j:j + 2].all():
                continue
            s = sign[i:i + 2, j:j + 2]
            if s[0, 0] == s[1, 1] and s[0, 1] == s[1, 0] and s[0, 0] != s[0, 1] and 0 not in s:
                out.append((i, j))
    return out


def sign_changes(sign, valid):
    """Number of edges between neighbouring valid cells across which the sign changes."""
    n = 0
    for di, dj in ((0, 1), (1, 0)):
        a = sign[:sign.shape[0] - di, :sign.shape[1] - dj]
        b = sign[di:, dj:]
        v = valid[:valid.shape[0] - di, :valid.shape[1] - dj] & valid[di:, dj:]
        n += int((v & (a * b < 0)).sum())
    return n


def stats(z, period):
    ok, clean, same = z["ok"], z["clean"], z["same_tokens"]
    mu1, mu2 = leading_pair(z)
    mod1, mod2 = np.abs(mu1), np.abs(mu2)
    cplx = ok & (np.abs(mu1.imag) > 1e-8)
    stable = ok & (mod1 < 1)
    t = np.real(mu1 + mu2)
    d = np.real(mu1 * mu2)
    q = [0, 0.1, 0.5, 0.9, 1]
    i_min = np.unravel_index(np.argmin(np.where(ok, mod1, np.inf)), mod1.shape)
    sgn = np.sign(np.where(stable, t, 0.0))
    boards = checkerboards(sgn, stable)
    return {
        "cells": int(ok.size), "solved": int(ok.sum()), "same_tokens": int((ok & same).sum()),
        "attracting": int(stable.sum()), "repelling": int((ok & ~stable).sum()),
        "attracting_complex_leading": int((stable & cplx).sum()),
        "attracting_clean": int((stable & clean).sum()),
        "attracting_real_unclean": int((stable & ~cplx & ~clean).sum()),
        "min_leading_modulus": float(mod1[i_min]), "min_leading_at_row_col": [int(i_min[0]), int(i_min[1])],
        "min_leading_value": [float(mu1[i_min].real), float(mu1[i_min].imag)],
        "ratio_mu2_mu1_quantiles_attracting": np.quantile((mod2 / mod1)[stable], q).tolist(),
        "pair_area_per_step_quantiles_attracting": np.quantile(np.abs(d[stable]) ** (1.0 / period), q).tolist(),
        "mu1_per_step_quantiles_attracting": np.quantile(mod1[stable] ** (1.0 / period), q).tolist(),
        "mu2_per_step_quantiles_attracting": np.quantile(mod2[stable] ** (1.0 / period), q).tolist(),
        "pair_product_negative_attracting": int((stable & (d < 0)).sum()),
        "trace_sign_changes_attracting": sign_changes(sgn, stable),
        "trace_checkerboards_attracting": [list(b) for b in boards],
        "stretching_min_per_step": np.nanmin(np.where(ok[..., None], z["stretchings"], np.nan), axis=(0, 1)).tolist(),
    }


def figure(z, tb, out, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    a1, a2 = z["alpha"], z["log10_temperature"]
    ok = z["ok"]
    mu1, mu2 = leading_pair(z)
    mod1 = np.abs(mu1)
    stable = ok & (mod1 < 1)
    cplx = ok & (np.abs(mu1.imag) > 1e-8)
    t = np.real(mu1 + mu2)
    da1, da2 = a1[1] - a1[0], a2[1] - a2[0]
    ext = [a1[0] - da1 / 2, a1[-1] + da1 / 2, a2[0] - da2 / 2, a2[-1] + da2 / 2]
    n = 3 if tb is not None else 2
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.4), dpi=170)
    k = 0
    if tb is not None:
        import period_colors as pc
        ax = axes[0]
        P = np.where(tb["chaotic"], np.nan, tb["period"].astype(float))
        P[P <= 0] = np.nan
        b1, b2 = tb["alpha1"], tb["alpha2"]
        e2 = [b1[0], b1[-1], b2[0], b2[-1]]
        ax.imshow(np.where(tb["chaotic"], 1.0, np.nan), origin="lower", extent=e2, aspect="auto",
                  cmap=ListedColormap(["0.15"]), interpolation="nearest")
        cats = pc.categorize([P], max_blocks=9)
        cmap = cats.cmap.copy()
        cmap.set_bad((0, 0, 0, 0))
        im = ax.imshow(cats.index(P), origin="lower", extent=e2, aspect="auto", cmap=cmap, norm=cats.norm,
                       interpolation="nearest")
        cats.colorbar(im, ax=ax, label="period (black: chaotic)")
        # the map must cover the grid (up to its half-cell padding): the axes stop at the map, never in blank space
        tx, ty = 0.05 * (ext[1] - ext[0]), 0.05 * (ext[3] - ext[2])
        if not (e2[0] <= ext[0] + tx and ext[1] <= e2[1] + tx and e2[2] <= ext[2] + ty and ext[3] <= e2[3] + ty):
            raise SystemExit(f"the map {e2} does not cover the grid {ext}; pass a wider --tb")
        ax.plot([ext[0], ext[1], ext[1], ext[0], ext[0]], [ext[2], ext[2], ext[3], ext[3], ext[2]], "-", color="w",
                lw=1.0)
        ax.set_xlim(e2[0], e2[1])
        ax.set_ylim(e2[2], e2[3])
        ax.set_title("(a) attractor from a fixed start, " + title, fontsize=7.5)
        k = 1
    ax = axes[k]
    lim = 0.6
    im = ax.imshow(np.where(stable, t, np.nan), origin="lower", extent=ext, aspect="auto", cmap="RdBu_r", vmin=-lim,
                   vmax=lim, interpolation="nearest")
    ax.imshow(np.where(ok & ~stable, 1.0, np.nan), origin="lower", extent=ext, aspect="auto",
              cmap=ListedColormap(["0.75"]), interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, label=r"trace $\mu_1+\mu_2$ of the leading pair")
    ax.contour(a1, a2, np.where(ok, t, np.nan), levels=[0], colors="k", linewidths=0.8)
    ax.contour(a1, a2, np.where(ok, mod1, np.nan), levels=[1], colors="m", linewidths=1.0)
    ax.set_title(f"({'b' if k else 'a'}) continued period-{len(z['seed_tokens'])} orbit: trace (black: zero; "
                 "magenta: |mu1| = 1; gray: repelling)", fontsize=7.5)
    ax = axes[k + 1]
    im = ax.imshow(np.where(ok, np.log10(mod1), np.nan), origin="lower", extent=ext, aspect="auto", cmap="viridis",
                   vmin=-2, vmax=0.3, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, label=r"$\log_{10}|\mu_1|$")
    yy, xx = np.nonzero(cplx)
    ax.plot(a1[xx], a2[yy], "x", ms=2.2, color="w", mew=0.6)
    ax.set_title(f"({'c' if k else 'b'}) modulus of the leading multiplier (crosses: complex pair)", fontsize=7.5)
    for ax in axes:
        ax.set_xlabel(r"$\alpha_1$ (block 4)", fontsize=8)
        ax.set_ylabel(r"$\alpha_2$ (block 16)", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run")
    ap.add_argument("--tb", help="classifier map (tb-*.npz) of the same plane for panel (a)")
    ap.add_argument("--json")
    ap.add_argument("--out")
    ap.add_argument("--title", default="")
    ap.add_argument("--partial", action="store_true", help="read a grid cut by its time limit (no grid.json)")
    args = ap.parse_args()
    run = Path(args.run)
    if not run.is_absolute() and not run.exists():
        run = DATA / "explore" / run
    z = np.load(run / "grid.npz", allow_pickle=True)
    if str(z["row_axis"]) != "alpha2":
        raise SystemExit("grid rows are not alpha2: this reading is written for two-block planes")
    if (run / "grid.json").exists():
        meta = json.loads((run / "grid.json").read_text())
    elif args.partial:                                  # cut by its time limit: the seed is in the saved grid
        meta = {"seed_tokens": [int(t) for t in z["seed_tokens"]], "seed_text": None}
    else:
        raise SystemExit(f"{run}: no grid.json (unfinished run?); pass --partial to read the last saved grid.npz")
    s = stats(z, len(meta["seed_tokens"]))
    if args.partial and not (run / "grid.json").exists():
        s.update({"partial": True, "cells_converged_in_saved_grid": int(np.isfinite(z["mu"]).sum())})
    s.update({"run": str(run.relative_to(DATA)) if str(run).startswith(str(DATA)) else str(run),
              "seed_text": meta["seed_text"], "alpha1_range": [float(z["alpha"][0]), float(z["alpha"][-1])],
              "alpha2_range": [float(z["log10_temperature"][0]), float(z["log10_temperature"][-1])],
              "resolution": [int(len(z["alpha"])), int(len(z["log10_temperature"]))]})
    text = json.dumps(s, indent=1, ensure_ascii=False)
    if args.json:
        Path(args.json).write_text(text + "\n")
    print(text)
    if args.out:
        import sys
        sys.path.insert(0, str(ROOT / "tools"))
        tb = np.load(args.tb, allow_pickle=True) if args.tb else None
        figure(z, tb, args.out, args.title)


if __name__ == "__main__":
    main()
