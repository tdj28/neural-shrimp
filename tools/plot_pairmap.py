"""Same architecture, other weights (run S10): Qwen2.5-0.5B-Instruct against Qwen2-0.5B-Instruct (identical Qwen2
architecture and size). Figure: 128 x 128 float64 maps along gratitude and philosophy for each model (period of the
attractor reached from the fixed start, chaotic cells black). Printed summary: chaotic share, period spectrum and
largest loop families per map; the overlap of the two models' loop families (identical token loops); and the
gratitude share of sampled self-dialogue against the dose (lexicon of tools/p1_controller.gratitude_hit) with the
first turn of a gratitude reply.

Usage: python tools/plot_pairmap.py OUT.png
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

MODELS = {"qwen0.5b": "Qwen2.5-0.5B-Instruct", "qwen2-0.5b": "Qwen2-0.5B-Instruct"}
CONCEPTS = ("gratitude", "philosophy")
DOSES = ("000", "010", "020", "040")


def load_map(key, concept):
    import explore_llm_attractors as ela

    d = ela.DATA / "explore" / "llm-soft-loop" / "pod64" / f"pairmap-{key}" / f"map128-{concept}"
    z = np.load(d / "scan.npz")
    meta = json.loads((d / "scan.json").read_text())
    return z, meta


def dialogue(key, dose):
    import explore_llm_attractors as ela
    import p1_controller as pc

    d = ela.DATA / "explore" / "llm-soft-loop" / "pod64" / "msm" / f"pair-{key}-grat{dose}"
    texts = json.loads((d / "sample.json").read_text())["texts"]
    g = np.array([[pc.gratitude_hit(t) for t in row] for row in texts])
    first = [int(np.argmax(r)) if r.any() else None for r in g]
    entered = [f for f in first if f is not None]
    return {"share_5_39": float(g[:, 5:40].mean()), "share_35_39": float(g[:, 35:40].mean()),
            "never": int(sum(f is None for f in first)),
            "entry_quartiles": [float(q) for q in np.percentile(entered, [25, 50, 75])] if entered else None}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import period_colors as pc

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6))
    # one colour per period of the non-chaotic cells, the same in all four maps; rare periods share an "other"
    # block if the labels would not fit
    shown = []
    for key in MODELS:
        for concept in CONCEPTS:
            z, _ = load_map(key, concept)
            shown.append(np.where(z["lyapunov"] > 1e-3, 0, z["period"]))
    cats = pc.categorize(shown, max_blocks=pc.blocks_that_fit(axes[0, 0]))
    loops = {}
    for col, key in enumerate(MODELS):
        for row, concept in enumerate(CONCEPTS):
            z, meta = load_map(key, concept)
            lt, al, P, L = z["log10_temperature"], z["alpha"], z["period"], z["lyapunov"]
            chaotic = L > 1e-3
            fams = meta["families"]
            loops[(key, concept)] = {"".join(f["tokens"]): f["cells"] for f in fams}
            print(json.dumps({"model": key, "concept": concept, "chaotic_share": round(float(chaotic.mean()), 4),
                              "period_counts": meta["period_counts"],
                              "largest_families": [(f["period"], f["cells"], "".join(f["tokens"])[:50])
                                                   for f in fams[:8]]}, ensure_ascii=False))
            ax = axes[row, col]
            ext = [al[0], al[-1], lt[0], lt[-1]]
            im = ax.imshow(cats.index(P, valid=~chaotic), origin="lower", extent=ext, aspect="auto",
                           cmap=cats.cmap, norm=cats.norm, interpolation="nearest")
            cats.colorbar(im, ax=ax, label="period (black: chaotic)")
            ax.imshow(np.where(chaotic, 1.0, np.nan), origin="lower", extent=ext, aspect="auto", cmap="gray_r",
                      vmin=0, vmax=1, interpolation="nearest")
            ax.set_title(f"{MODELS[key]}, {concept} ({100 * chaotic.mean():.0f} % chaotic)", fontsize=9)
            ax.set_xlabel(f"steering along '{concept}'")
            ax.set_ylabel("log10 temperature")
    for concept in CONCEPTS:
        a, b = loops[("qwen0.5b", concept)], loops[("qwen2-0.5b", concept)]
        shared = sorted(set(a) & set(b), key=lambda k: -(a[k] + b[k]))
        print(json.dumps({"concept": concept, "families": [len(a), len(b)], "shared_loops": len(shared),
                          "shared_cells": [sum(a[k] for k in shared), sum(b[k] for k in shared)],
                          "top_shared": [(k[:40], a[k], b[k]) for k in shared[:6]]}, ensure_ascii=False))
    for key in MODELS:
        try:
            print(json.dumps({"model": key, "gratitude_dialogue": {d: dialogue(key, d) for d in DOSES}}))
        except FileNotFoundError as err:
            print(json.dumps({"model": key, "gratitude_dialogue": f"missing: {err}"}))
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=200)


if __name__ == "__main__":
    main()
