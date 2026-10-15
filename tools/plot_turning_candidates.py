"""Figure for inspecting run TP's flagged cells: return maps of the one-dimensional phases with two or more turning
points (tools/turning_scan.py), recomputed locally from the cells' trajectories is not possible (the scan keeps no
trajectories), so this script lists the flagged cells per configuration and writes a cells file for a follow-up run
that records them (tools/return_map.py style) and plots their per-phase return maps.

Usage: python tools/plot_turning_candidates.py [--min-turning 2] [--out-cells docs/tp/flagged.json]
"""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data")) / "explore/llm-soft-loop/pod64/tp"


def plot_followups(out_png, min_turning=2, min_explained=0.99):
    """Return maps (s_(n+1) against s_n) of every flagged one-dimensional phase in the follow-up runs tp/f-* (run TPF,
    which saved the projections), one panel per phase, at most 24 panels."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = []
    for run in sorted(p.name for p in DATA.iterdir() if p.name.startswith("f-") and (p / "projections.npz").exists()):
        d = json.loads((DATA / run / "turning.json").read_text())
        z = np.load(DATA / run / "projections.npz")
        for ci, c in enumerate(d["cells"]):
            for ph in c["phases"]:
                if ph["explained"] >= min_explained and ph["turning"] >= min_turning and f"c{ci}_p{ph['phase']}" in z:
                    panels.append((run, c, ph, z[f"c{ci}_p{ph['phase']}"]))
    panels = panels[:24]
    if not panels:
        return 0
    cols = 6
    rows = (len(panels) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.6 * cols, 2.6 * rows), dpi=150, squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (run, c, ph, sv) in zip(axes.ravel(), panels):
        ax.axis("on")
        ax.plot(sv[:-1], sv[1:], ".", ms=1, color="k")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{run} {[round(x, 4) for x in c['cell']]}\nq={c['q']} phase {ph['phase']}: {ph['turning']} TP, "
                     f"expl {ph['explained']:.3f}", fontsize=5.5)
    fig.tight_layout()
    fig.savefig(out_png)
    return len(panels)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--min-turning", type=int, default=2)
    ap.add_argument("--out-cells", default=str(ROOT / "docs/tp/flagged.json"))
    ap.add_argument("--plot", help="PNG of the follow-up runs' flagged return maps (run TPF)")
    args = ap.parse_args()
    if args.plot:
        print(plot_followups(args.plot), "panels")
        return
    summary, flagged = {}, {}
    for run in sorted(p.name for p in DATA.iterdir() if (p / "turning.json").exists()):
        d = json.loads((DATA / run / "turning.json").read_text())
        cells = d["cells"]
        hits = []
        for c in cells:
            ph = [p for p in c["phases"] if p["explained"] >= d["args"]["min_explained"] and p["turning"] >= args.min_turning]
            if ph:
                hits.append({"cell": c["cell"], "q": c["q"], "phases": [(p["phase"], p["turning"], round(p["explained"], 3))
                                                                       for p in ph]})
        summary[run] = {"cells": len(cells), "flagged": len(hits),
                        "q_distribution": {str(q): sum(1 for c in cells if c["q"] == q) for q in sorted({c["q"] for c in cells})},
                        "one_d_cells": sum(1 for c in cells if c["phases_1d"] > 0)}
        flagged[run] = {"args": {k: d["args"][k] for k in ("model", "layer", "vectors", "concept", "layer2", "vectors2",
                                                            "concept2")}, "hits": hits}
    Path(args.out_cells).write_text(json.dumps({"note": "TP cells with >= %d turning points in a one-dimensional phase"
                                                        % args.min_turning, "flagged": flagged}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
