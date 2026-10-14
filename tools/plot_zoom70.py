"""Figure: the 64 x 64 zoom of Llama-3.3-70B-Instruct's chaotic ridge along its hostility concept vector (run Z70),
stitched from four bands of 16 temperature rows. The bands must join into one temperature grid, share the steering
grid, and carry concept vectors of identical norm (built on each pod); otherwise the script stops.

Left: largest Lyapunov exponent (chaotic cells coloured by the exponent, periodic cells in grey by how strongly they
contract). Right: the attractor families (leading-token loops) of periodic cells, the most frequent in colour.

Usage: python tools/plot_zoom70.py OUT.png [BAND_PREFIX]   (default llm-soft-loop/pod64/controls70/hostility-vs-kindness-zoom-b)
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


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
    from matplotlib.colors import ListedColormap

    import explore_llm_attractors as ela

    out = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else "llm-soft-loop/pod64/controls70/hostility-vs-kindness-zoom-b"
    m = stitch([ela.DATA / "explore" / f"{prefix}{b}" for b in range(4)])
    lt, al, L, P, norms = m["log10_temperature"], m["alpha"], m["lyapunov"], m["period"], m["norms"]
    chaotic = (L > 1e-3) & (P <= 0)
    print(json.dumps({"cells": int(L.size), "chaotic": int(chaotic.sum()), "share": round(float(chaotic.mean()), 4),
                      "max_exponent": round(float(L.max()), 4), "norms": norms,
                      "log10T": [float(lt[0]), float(lt[-1])], "alpha": [float(al[0]), float(al[-1])]}))
    ext = [al[0], al[-1], lt[0], lt[-1]]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 5.6))
    grey = np.where(~chaotic, np.clip(L, -1.0, 0.0), np.nan)
    ax.imshow(grey, origin="lower", extent=ext, aspect="auto", cmap="Greys_r", vmin=-1.0, vmax=0.0,
              interpolation="nearest")
    hot = np.where(chaotic, np.clip(L, 0.0, 0.5), np.nan)
    im = ax.imshow(hot, origin="lower", extent=ext, aspect="auto", interpolation="nearest", vmin=0.0, vmax=0.5,
                   cmap=ListedColormap(plt.get_cmap("plasma")(np.linspace(0.15, 0.85, 256))))
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_label("largest Lyapunov exponent (chaotic cells)", fontsize=8)
    ax.set_title(f"largest exponent; chaotic {100 * chaotic.mean():.0f} % of {L.size} cells", fontsize=9)
    fam = m["family"]
    img = np.ones(fam.shape + (4,))
    img[chaotic] = (0.0, 0.0, 0.0, 1.0)
    vals, counts = np.unique(fam[(fam >= 0) & ~chaotic], return_counts=True)
    order = vals[np.argsort(-counts)][:10]
    palette = plt.get_cmap("tab10")
    for k, f in enumerate(order):
        img[(fam == f) & ~chaotic] = palette(k)
    bx.imshow(img, origin="lower", extent=ext, aspect="auto", interpolation="nearest")
    bx.set_title("attractor families (black: chaotic; white: other or no period)", fontsize=9)
    for a in (ax, bx):
        a.set_xlabel("steering strength alpha (hostility concept vector, layer 50)")
    ax.set_ylabel("log10 temperature")
    fig.suptitle("Llama-3.3-70B-Instruct: zoom on the chaotic ridge (64 x 64 float64 cells, four stitched bands)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=250)


if __name__ == "__main__":
    main()
