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


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--min-turning", type=int, default=2)
    ap.add_argument("--out-cells", default=str(ROOT / "docs/tp/flagged.json"))
    args = ap.parse_args()
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
