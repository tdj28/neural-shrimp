"""Run SH extra figures (exploratory): the crescent's hidden strip, and the two families as sheets in 3D.

fig-hidden-strip: along each temperature row of the crescent zone (alpha < 0.0676), the smallest alpha at which the
crescent family's periodic attractor exists in any map of the run (its fold edge), and the smallest alpha at which the
fixed start reaches it (its visible edge); with the width of the hidden strip between them, and the fold located by
continuation at log10 T -0.148 (explore/e2e/crescent-cut-t-0.148).

fig-sheets-3d: the attractor's mean probability of the blank-line token (height) over (alpha, log10 T), for every
cell of every map of the run, colored by family. The crescent family forms a sheet near 0.2, and the 2-cycle with its
stripes forms a sheet near 0. The cells chosen by the fixed start are shown in a second panel.

Usage: python tools/sheets_extra_figures.py --analysis sheets/analysis --maps <dirs> --out-dir paper/llm/figures
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sheets_analyze as sa  # noqa: E402
from explore_llm_soft_loop import DATA  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analysis", required=True)
    ap.add_argument("--maps", nargs="+", required=True)
    ap.add_argument("--default", default="default")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    F = np.load(DATA / "explore" / args.analysis / "fields.npz")
    lt, al, G, names = F["log10_temperature"], F["alpha"], F["groups"], list(F["names"])
    d = G[names.index(args.default)]
    P3 = (G == 1).any(0)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # hidden strip
    rows = []
    zone = al < 0.0676
    for i in range(len(lt)):
        e = np.nonzero(P3[i] & zone)[0]
        v = np.nonzero((d[i] == 1) & zone)[0]
        if len(e) and len(v):
            rows.append((lt[i], al[e.min()], al[v.min()]))
    R = np.array(rows)
    R = R[R[:, 0] <= -0.142]                   # above, the leftmost crescent-family cells are band and period-3 patches
    trunc = R[:, 1] <= al[0] + 1e-12            # the fold edge has left the window: the strip width is a lower bound
    fold = json.loads((DATA / "explore" / "e2e" / "crescent-cut-t-0.148" / "cut.json").read_text())
    fold_a = fold["directions"]["down"]["boundary"]["lambda"]
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    ax = axs[0]
    ax.plot(R[:, 1], R[:, 0], "-", color="#ff7f0e", lw=1.3, label="crescent exists from (fold edge)")
    ax.plot(R[:, 2], R[:, 0], "-", color="k", lw=1.0, label="fixed start reaches it from (visible edge)")
    ax.fill_betweenx(R[:, 0], R[:, 1], R[:, 2], color="#b48ad6", alpha=0.45, label="hidden strip: both exist,\nfixed start shows the 2-cycle")
    ax.plot([fold_a], [-0.148], "*", color="#d62728", ms=10, label="fold by continuation")
    ax.set_xlabel(r"steering strength $\alpha$")
    ax.set_ylabel(r"$\log_{10}T$")
    ax.set_title("(a) the crescent's left edge", fontsize=9, loc="left")
    ax.legend(fontsize=6.5, frameon=False, loc="lower right")
    ax = axs[1]
    w = R[:, 2] - R[:, 1]
    ax.plot(R[~trunc, 0], w[~trunc] * 1e4, ".-", color="#7b4fa0", ms=3, lw=0.8, label="fold edge inside the window")
    ax.plot(R[trunc, 0], w[trunc] * 1e4, ".", color="#c8b3dd", ms=3, label="fold edge beyond the window\n(width is a lower bound)")
    ax.legend(fontsize=6.5, frameon=False)
    ax.axhline((al[1] - al[0]) * 1e4, color="0.6", ls=":", lw=0.8)
    ax.text(R[0, 0], (al[1] - al[0]) * 1e4 * 1.15, "grid spacing", fontsize=6.5, color="0.4")
    ax.set_xlabel(r"$\log_{10}T$")
    ax.set_ylabel(r"width of the hidden strip ($10^{-4}$ in $\alpha$)")
    ax.set_title("(b) the strip narrows toward the top", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(out / "fig-hidden-strip.png", dpi=220)
    plt.close(fig)
    np.savez_compressed(out / "fig-hidden-strip-data.npz", log10_temperature=R[:, 0], fold_edge=R[:, 1],
                        visible_edge=R[:, 2])
    # 3D sheets
    maps, _, _, keys = sa.load_maps(args.maps)
    bi = keys.index("\n\n")
    A, T = np.meshgrid(al, lt)
    colors = {1: "#ff7f0e", -1: "#1f77b4", 2: "#9a9a9a"}
    fig = plt.figure(figsize=(11.5, 5.2))
    rng = np.random.default_rng(0)
    for p, which in enumerate(("all maps", "hidden")):
        ax = fig.add_subplot(1, 2, p + 1, projection="3d")
        g0 = G[names.index(args.default)]
        for name, m in maps.items():
            g = G[names.index(name)]
            z = m["keyprob"][..., bi]
            for gv, c in colors.items():
                sel = (g == gv) & np.isfinite(z)
                if which != "all maps":                      # hidden: found from another start, not by the fixed start
                    sel &= (g0 != gv) & (gv != 2)
                idx = np.flatnonzero(sel)
                if len(idx) == 0:
                    continue
                if len(idx) > 9000:
                    idx = rng.choice(idx, 9000, replace=False)
                ax.scatter(A.flat[idx], T.flat[idx], z.flat[idx], s=0.8 if which == "all maps" else 1.6, c=c, alpha=0.4 if which == "all maps" else 0.7, linewidths=0)
        ax.set_xlabel(r"$\alpha$", fontsize=8)
        ax.set_ylabel(r"$\log_{10}T$", fontsize=8)
        ax.set_zlabel(r"mean $p(\hookleftarrow\hookleftarrow)$", fontsize=8)
        ax.set_xlim(al[0], al[-1])
        ax.set_ylim(lt[0], lt[-1])
        ax.set_zlim(-0.01, 0.27)
        ax.view_init(elev=24, azim=-58)
        ax.tick_params(labelsize=6)
        ax.set_title("(a) every attractor found (four starts)" if p == 0 else
                     "(b) what the map hides: found from another start,\nnot reached by the fixed start", fontsize=9)
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], marker="o", ls="", color=colors[1], label="crescent family (periodic)"),
          Line2D([], [], marker="o", ls="", color=colors[-1], label="2-cycle and stripes (periodic)"),
          Line2D([], [], marker="o", ls="", color=colors[2], label="chaotic")]
    fig.legend(handles=hs, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.08, top=0.93, wspace=0.02)
    fig.savefig(out / "fig-sheets-3d.png", dpi=200)
    print(json.dumps({"hidden_strip_rows": len(R), "strip_width_range": [float(w.min()), float(w.max())],
                      "fold_by_continuation_at_-0.148": fold_a}))


if __name__ == "__main__":
    main()
