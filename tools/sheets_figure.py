"""Run SH figure: what the fixed-start map shows, what exists underneath, and why the visible attractor switches.

Panels:
(a) the fixed-start map at the run's settings, colored by attractor family (the palette of the paper's cross-512
    figure for its main families; other periodic families gray, chaos black);
(b) the two groups' footprints over all maps of the run: only the 2-group exists (blue), only the 3-group (orange),
    both (purple), with the fixed start's 2|3 boundary drawn on top;
(c) the settle step of the fixed start (log scale): long transients mark basin boundaries and fold ghosts;
(d) cross-sections along one or two temperature rows: the blank-line probability of the attractor reached in each
    map against alpha, so that the two groups appear as two sheets, with the fixed start's choice in black.

Usage: python tools/sheets_figure.py --analysis sheets/analysis --maps <dirs as for sheets_analyze> --rows -0.148 -0.1415
       --out paper/llm/figures/fig-sheets.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sheets_analyze as sa  # noqa: E402
from explore_llm_soft_loop import DATA  # noqa: E402

PALETTE = {(2, " is it"): "#1f77b4", (8, ". This is there is a is a"): "#aec7e8",
           (3, "\n\nThe existence"): "#ff7f0e", (10, ". This is there is it is a is a"): "#ffbb78",
           (6, "\n\nThe question\n\nThe question"): "#2ca02c", (12, ". This is there is it is it is a is a"): "#98df8a",
           (6, "\n\nThe question\n\nThe existence"): "#d62728", (6, "\n\nThe existence\n\nThe existence"): "#ff9896",
           (6, ". This is there is a"): "#9467bd", (3, "\n\nThe question"): "#c5b0d5",
           (14, ". This is there is it is it is it is a is a"): "#8c564b"}


def rotations(s_tokens):
    return {"".join(s_tokens[i:] + s_tokens[:i]) for i in range(len(s_tokens))}


def family_colors(meta):
    """Map family index -> color, matching the palette by token loop up to rotation."""
    from matplotlib.colors import to_rgb
    pal = {}
    for f in meta["families"]:
        rots = rotations(f["tokens"])
        col = None
        for (q, text), c in PALETTE.items():
            if q == f["period"] and text in rots:
                col = c
        pal[f["family"]] = to_rgb(col) if col else to_rgb("#8c8c99")
    return pal


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--maps", nargs="+", required=True)
    ap.add_argument("--default", default="default")
    ap.add_argument("--rows", type=float, nargs="+", default=[-0.148, -0.1415])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    F = np.load(DATA / "explore" / args.analysis / "fields.npz")
    lt, al = F["log10_temperature"], F["alpha"]
    maps, _, _, keys = sa.load_maps(args.maps)
    bi = keys.index("\n\n")
    m0 = maps[args.default]
    meta = json.loads((Path(m0["source"]).parent / "sheets.json").read_text())
    pal = family_colors(meta)
    img = np.zeros(m0["family"].shape + (3,))
    for idx, col in pal.items():
        img[m0["family"] == idx] = col
    chaos = (m0["family"] < 0) & (m0["lyapunov"] > 1e-3)
    img[chaos] = (0, 0, 0)
    img[(m0["family"] < 0) & ~chaos] = (0.8, 0.8, 0.8)
    ext = [al[0], al[-1], lt[0], lt[-1]]
    nrow = len(args.rows)
    fig = plt.figure(figsize=(11.5, 8.6))
    gs = fig.add_gridspec(2 + (1 if nrow > 2 else 0), 3, height_ratios=[1, 0.62] + ([0.62] if nrow > 2 else []),
                          hspace=0.42, wspace=0.28)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img, origin="lower", extent=ext, aspect="auto", interpolation="nearest")
    ax.set_title("(a) what the map shows (fixed start)", fontsize=9, loc="left")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\log_{10}T$")
    ax = fig.add_subplot(gs[0, 1])
    f3, f2 = F["foot3"], F["foot2"]
    code = np.where(f3 & f2, 3, np.where(f3, 2, np.where(f2, 1, 0)))
    cmap = ListedColormap(["#ffffff", "#9ec9ea", "#ffbf80", "#b48ad6"])
    ax.imshow(code, origin="lower", extent=ext, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=3)
    g0 = F["groups"][list(F["names"]).index(args.default)]
    ax.contour(al, lt, (g0 == 1).astype(float), levels=[0.5], colors="k", linewidths=0.6)
    ax.set_title("(b) what exists: 2-group, 3-group, both", fontsize=9, loc="left")
    ax.set_xlabel(r"$\alpha$")
    for r in args.rows:
        ax.axhline(r, color="k", ls=":", lw=0.6)
    ax = fig.add_subplot(gs[0, 2])
    st = np.log10(np.maximum(F["settle_default"], 1))
    im = ax.imshow(st, origin="lower", extent=ext, aspect="auto", interpolation="nearest", cmap="magma")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label=r"$\log_{10}$ settle step")
    ax.contour(al, lt, (g0 == 1).astype(float), levels=[0.5], colors="c", linewidths=0.6)
    ax.set_title("(c) how long the fixed start takes to settle", fontsize=9, loc="left")
    ax.set_xlabel(r"$\alpha$")
    groups = F["groups"]
    gnames = list(F["names"])
    gcol = {1: "#ff7f0e", -1: "#1f77b4", 2: "#999999"}
    for k, r in enumerate(args.rows):
        ax = fig.add_subplot(gs[1 + k // 3, k % 3]) if nrow > 1 else fig.add_subplot(gs[1, :])
        i = int(np.argmin(np.abs(lt - r)))
        for name, m in maps.items():
            g = groups[gnames.index(name)][i]
            pb = m["keyprob"][i, :, bi]
            for gv, c in gcol.items():
                sel = g == gv
                ax.plot(al[sel], pb[sel], ".", ms=2.0, color=c, alpha=0.75)
        # fate ribbon: the group the fixed start reaches at each alpha
        g0r = groups[gnames.index(args.default)][i]
        ymin = -0.045
        for gv, c in gcol.items():
            sel = g0r == gv
            ax.scatter(al[sel], np.full(sel.sum(), ymin), marker="|", s=40, color=c, linewidths=1.2)
        ax.axhline(ymin + 0.018, color="0.85", lw=0.5)
        ax.set_ylim(ymin - 0.02, 0.27)
        ax.set_title(f"(d{k + 1}) cross-section at " + r"$\log_{10}T=$" + f"{lt[i]:.4f}", fontsize=9, loc="left")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"mean $p(\hookleftarrow\hookleftarrow)$ on the attractor")
        ax.text(0.01, 0.06, "fixed start", transform=ax.transAxes, fontsize=6, color="0.3")
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
        if k == 0:
            from matplotlib.lines import Line2D
            hs = [Line2D([], [], marker=".", ls="", color=gcol[1], label="crescent family (periodic)"),
                  Line2D([], [], marker=".", ls="", color=gcol[-1], label="2-cycle and stripes (periodic)"),
                  Line2D([], [], marker=".", ls="", color=gcol[2], label="chaotic")]
            ax.legend(handles=hs, fontsize=6, frameon=False, loc="upper right")
    for ax in fig.axes[:3]:
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=220, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
