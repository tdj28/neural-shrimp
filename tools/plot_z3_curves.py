"""Figure: the period-6 window of the Qwen2.5-0.5B-Instruct soft-token map (philosophy direction, layer 12) with its
float64 fold curves, flip curve and the two normal-form-tested cusps (runs in p6z3), over a background map merged
from horizontal bands of one grid (run F6; float32 scans, labelled as such). Chaotic cells are coloured by the
largest exponent, periodic cells in grey by how strongly they contract.

Usage: python tools/plot_z3_curves.py OUT.png BAND_PREFIX NBANDS
       (bands BAND_PREFIX0 .. BAND_PREFIX{NBANDS-1}, explore-relative, each a scan.npz of consecutive rows)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
CURVES = "llm-soft-loop/pod64/p6z3"
CUSPS = (("curve-foldA-plus", "cusp-check-foldA-plus.jsonl"), ("curve-foldD-minus", "cusp-check-foldD-minus.jsonl"))


def merge(prefix, n):
    import explore_llm_attractors as ela

    parts = [np.load(ela.DATA / "explore" / f"{prefix}{b}" / "scan.npz") for b in range(n)]
    al = parts[0]["alpha"]
    for p in parts[1:]:
        if not np.allclose(p["alpha"], al, rtol=0, atol=1e-12):
            raise SystemExit("bands have different alpha grids")
    lt = np.concatenate([p["log10_temperature"] for p in parts])
    step = np.diff(lt)
    if not np.allclose(step, step[0], rtol=1e-6):
        raise SystemExit("bands do not join into one temperature grid")
    L = np.concatenate([p["lyapunov"] for p in parts])
    P = np.concatenate([p["period"] for p in parts])
    return L, P, al, lt


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    import explore_llm_attractors as ela

    out, prefix, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    L, P, al, lt = merge(prefix, n)
    chaotic = (L > 1e-3) & (P <= 0)
    print("cells", L.size, "chaotic share", round(float(chaotic.mean()), 4))
    ext = [al[0], al[-1], lt[0], lt[-1]]
    fig, ax = plt.subplots(figsize=(8.8, 7.6))
    grey = np.where(~chaotic, np.clip(L, -1.0, 0.0), np.nan)
    ax.imshow(grey, origin="lower", extent=ext, aspect="auto", cmap="Greys_r", vmin=-1.0, vmax=0.0,
              interpolation="nearest")
    hot = np.where(chaotic, np.clip(L, 0.0, 0.5), np.nan)
    im = ax.imshow(hot, origin="lower", extent=ext, aspect="auto", cmap=ListedColormap(plt.get_cmap("plasma")(
        np.linspace(0.15, 0.85, 256))), vmin=0.0, vmax=0.5, interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("largest Lyapunov exponent (chaotic cells)", fontsize=9)
    base = ela.DATA / "explore" / CURVES
    for d in sorted(base.glob("curve-*")):
        c = json.loads((d / "curve.json").read_text())
        x = [p["alpha"] for p in c["points"]]
        y = [p["log10_temperature"] for p in c["points"]]
        if c.get("kind") == "flip":
            ax.plot(x, y, color="magenta", ls=(0, (1.5, 1.5)), lw=1.8)
        else:                       # solid: some branches were continued from two starts and overlap
            ax.plot(x, y, color="cyan", lw=1.6)
    for curve, check in CUSPS:
        rows = [json.loads(line) for line in (base / check).read_text().splitlines() if line.strip()]
        c = json.loads((base / curve / "curve.json").read_text())
        flags = [i for i, p in enumerate(c["points"]) if p.get("cusp_candidate")]
        if not flags:
            raise SystemExit(f"{curve}: no flagged cusp point")
        p = c["points"][flags[0]]
        ax.plot(p["alpha"], p["log10_temperature"], "o", ms=9, mfc="none", mec="yellow", mew=2)
        print("cusp", curve, round(p["alpha"], 5), round(p["log10_temperature"], 5), "checked rows", len(rows))
    ax.set_xlim(al[0], al[-1])
    ax.set_ylim(lt[0], lt[-1])
    ax.set_xlabel("steering strength alpha (philosophy, layer 12)")
    ax.set_ylabel("log10 temperature")
    ax.set_title("Qwen2.5-0.5B-Instruct: the period-6 window, its fold and flip curves and two cusps", fontsize=10)
    handles = [Patch(color=plt.get_cmap("plasma")(0.6), label="chaotic (colour: largest exponent)"),
               Patch(color="0.2", label="periodic, strongly contracting"),
               Patch(color="0.85", label="periodic, weakly contracting"),
               Line2D([], [], color="cyan", lw=1.6, label="fold curve (float64)"),
               Line2D([], [], color="magenta", ls=(0, (1.5, 1.5)), lw=1.8, label="flip curve (float64)"),
               Line2D([], [], marker="o", ls="", mfc="none", mec="yellow", mew=2, ms=9, label="cusp (normal form)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=300)


if __name__ == "__main__":
    main()
