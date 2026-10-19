"""Figure: the 64 x 64 zoom of Llama-3.3-70B-Instruct's chaotic ridge along its hostility concept vector (run Z70),
stitched from four bands of 16 temperature rows. The bands must join into one temperature grid, share the steering
grid, and carry concept vectors of identical norm (built on each pod); otherwise the script stops.

Left: largest Lyapunov exponent on one colour bar: grey from -1 (black: strongly contracting, or below) to 0 (white)
for the cells that are not chaotic, colour from 0 to 0.5 for the chaotic cells. Right: the period of the attractor reached from the fixed start; chaotic cells black, cells with neither a
detected period nor a positive exponent light grey.

Usage: python tools/plot_zoom70.py OUT.png [BAND_PREFIX]   (default llm-soft-loop/pod64/controls70/hostility-vs-kindness-zoom-b)
  The bands are the folders BAND_PREFIX0, BAND_PREFIX1, ... (or zero-padded: BAND_PREFIX00, ...), numbered from 0
  without gaps. Run Z70 (pre-fix, 64 x 64) has four bands (zoom-b); run Z128 (post-fix, 128 x 128) sixteen (z128-b).
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def band_dirs(prefix):
    """The band folders of PREFIX in band order; stops unless they are numbered 0..N-1 without gaps."""
    import explore_llm_attractors as ela
    root = ela.DATA / "explore" / prefix
    found = {}
    for d in root.parent.glob(root.name + "*"):
        tail = d.name[len(root.name):]
        if tail.isdigit():
            found.setdefault(int(tail), []).append(d)
    if not found or sorted(found) != list(range(len(found))) or any(len(v) > 1 for v in found.values()):
        raise SystemExit(f"bands of {prefix}: found {sorted(found)}; need one folder per band 0..N-1")
    return [found[k][0] for k in range(len(found))]


def stitch(dirs, concept="hostility-vs-kindness"):
    """Join band folders (each with scan/scan.npz and vectors/vectors.npz) into one map; stops on any mismatch."""
    parts, norms = [], []
    for d in dirs:
        parts.append(np.load(Path(d) / "scan" / "scan.npz"))
        norms.append(float(np.linalg.norm(np.load(Path(d) / "vectors" / "vectors.npz")[concept])))
    if max(norms) - min(norms) > 1e-9 * max(norms):
        raise SystemExit(f"concept vectors differ between bands: norms {norms}")
    al = parts[0]["alpha"]
    for p in parts[1:]:
        if p["alpha"].shape != al.shape or not np.allclose(p["alpha"], al, rtol=0, atol=1e-12):
            raise SystemExit("bands have different steering grids")
    lt = np.concatenate([p["log10_temperature"] for p in parts])
    step = np.diff(lt)
    if step.size == 0 or step[0] <= 0 or not np.allclose(step, step[0], rtol=1e-6):
        raise SystemExit("bands do not join into one temperature grid")
    cat = lambda k: np.concatenate([p[k] for p in parts])
    return {"log10_temperature": lt, "alpha": al, "lyapunov": cat("lyapunov"), "period": cat("period"),
            "family": cat("family"), "norms": norms}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, TwoSlopeNorm

    import explore_llm_attractors as ela
    import period_colors as pc

    out = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "llm-soft-loop/pod64/controls70/hostility-vs-kindness-zoom-b"
    dirs = band_dirs(prefix)
    m = stitch(dirs)
    lt, al, L, P, norms = m["log10_temperature"], m["alpha"], m["lyapunov"], m["period"], m["norms"]
    chaotic = (L > 1e-3) & (P <= 0)
    print(json.dumps({"cells": int(L.size), "chaotic": int(chaotic.sum()), "share": round(float(chaotic.mean()), 4),
                      "periodic": int((P > 0).sum()), "unresolved": int(((P <= 0) & ~chaotic).sum()),
                      "max_exponent": round(float(L.max()), 4), "norms": norms,
                      "log10T": [float(lt[0]), float(lt[-1])], "alpha": [float(al[0]), float(al[-1])]}))
    da, dt = al[1] - al[0], lt[1] - lt[0]
    ext = [al[0] - da / 2, al[-1] + da / 2, lt[0] - dt / 2, lt[-1] + dt / 2]     # cell edges
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.6))
    # one colour bar for both kinds of cell: grey from -1 (black) to 0 (white) where the cell is not chaotic, plasma
    # from 0 to 0.5 where it is; TwoSlopeNorm puts each range on its own half of the bar
    lo, hi = -1.0, 0.5
    cmap = ListedColormap(np.vstack([plt.get_cmap("Greys_r")(np.linspace(0.0, 1.0, 256)),
                                     plt.get_cmap("plasma")(np.linspace(0.15, 0.85, 256))]))
    val = np.where(chaotic, np.clip(L, 1e-9, hi), np.clip(L, lo, 0.0))
    im = ax.imshow(val, origin="lower", extent=ext, aspect="auto", interpolation="nearest", cmap=cmap,
                   norm=TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi))
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=[-1.0, -0.75, -0.5, -0.25, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    cb.ax.set_yticklabels(["$\\leq -1$", "$-0.75$", "$-0.5$", "$-0.25$", "0", "0.1", "0.2", "0.3", "0.4", "$\\geq 0.5$"],
                          fontsize=7)
    cb.set_label("largest Lyapunov exponent per step\n(gray: not chaotic; color: chaotic)", fontsize=8)
    ax.set_title(f"largest exponent; chaotic {100 * chaotic.mean():.0f} % of {L.size} cells", fontsize=9)
    periodic = P > 0
    unresolved = ~periodic & ~chaotic
    # one colour and colour-bar block per period, from the palette safe under colour-vision deficiencies
    cats = pc.categorize(P, max_blocks=pc.blocks_that_fit(bx), kind="cvd")
    img = cats.rgba(P, fill=(1.0, 1.0, 1.0, 1.0))
    img[chaotic] = (0.0, 0.0, 0.0, 1.0)
    img[unresolved] = (0.82, 0.82, 0.82, 1.0)
    bx.imshow(img, origin="lower", extent=ext, aspect="auto", interpolation="nearest")
    cb2 = cats.colorbar(ax=bx, fraction=0.04, pad=0.02)
    cb2.set_label("period (periodic cells)", fontsize=8)
    bx.set_title(f"period (black: chaotic; light gray: no period up to 32 and no positive exponent, "
                 f"{100 * unresolved.mean():.0f} %)", fontsize=9)
    for a in (ax, bx):
        a.set_xlabel("steering strength alpha (hostility concept vector, layer 50)")
    ax.set_ylabel("log10 temperature")
    fig.suptitle(f"Llama-3.3-70B-Instruct: zoom on the chaotic ridge ({len(lt)} x {len(al)} float64 cells, "
                 f"{len(dirs)} stitched bands)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=250)


if __name__ == "__main__":
    main()
