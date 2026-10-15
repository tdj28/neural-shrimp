"""Which orbits meet at a self-crossing of a continued fold curve (runs FM5, FM5c): for each crossing recorded in
docs/fm5/fm5-summary.json, the four curve points that bracket it (i, i+1 on one arm, j, j+1 on the other) are iterated
through the five phases of their period-5 orbit at their own parameters (float64, CPU), and the arms are compared at
the best phase alignment: RMS over the five points of the state difference, relative to the RMS state norm. The
result is set against the state change between neighbouring curve points on each arm. Writes
docs/fm5/crossing-orbits.json.

Usage: python tools/fold_crossing_orbits.py [--threads 12] [--json docs/fm5/crossing-orbits.json]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
DATA = Path(os.environ.get("NEURAL_SHRIMP_DATA", ROOT.parent / "neural-shrimp-data"))
POD = DATA / "explore" / "llm-soft-loop" / "pod64"


def rel_distance(a, b):
    """Relative RMS distance between two five-point orbits (arrays (5, n)) at the best cyclic phase."""
    best = min(np.sqrt(np.mean(np.sum((np.roll(b, s, axis=0) - a) ** 2, axis=1))) for s in range(len(b)))
    return float(best / np.sqrt(np.mean(np.sum(a ** 2, axis=1))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--summary", default=str(ROOT / "docs" / "fm5" / "fm5-summary.json"))
    ap.add_argument("--json", default=str(ROOT / "docs" / "fm5" / "crossing-orbits.json"))
    args = ap.parse_args()
    import window_census as wc
    crossings = json.loads(Path(args.summary).read_text())["fold_self_crossings"]
    loop, make_cell = wc.setup("cpu", args.threads)
    out = []
    for name, lst in crossings.items():
        run = POD / ("fm5/" + name if "/" not in name else name)
        if not lst:
            continue
        pts = json.loads((run / "curve.json").read_text())["points"]
        S = np.load(run / "states.npy")
        for c in lst:
            i, j = c["segments"]
            orb, toks = {}, {}
            for k in (i, i + 1, j, j + 1):
                cell = make_cell(pts[k]["alpha"], pts[k]["alpha2"])
                X = wc.to_state(loop, S[k])
                orb[k] = np.array(wc.phases(cell, X))
                toks[k] = wc.token_loop(cell, X)
            pairs = {f"{a}-{b}": rel_distance(orb[a], orb[b]) for a in (i, i + 1) for b in (j, j + 1)}
            steps = {f"{i}-{i + 1}": rel_distance(orb[i], orb[i + 1]), f"{j}-{j + 1}": rel_distance(orb[j], orb[j + 1])}
            d = list(pairs.values())
            s = list(steps.values())
            rec = {"curve": name, "segments": [i, j], "at": c["at"], "cross_arm_distance": pairs,
                   "per_step_change": steps, "tokens": {str(k): v for k, v in toks.items()},
                   "distance_range": [min(d), max(d)], "ratio_range": [min(d) / max(s), max(d) / min(s)]}
            out.append(rec)
            print(json.dumps(rec), flush=True)
    Path(args.json).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
