"""Analysis of SAE readouts of soft-token-map attractors (run L13, tools/llm_sae_readout.py read).

For each SAE: the reconstruction quality on the map's states (FVE) next to the hook-point check on ordinary text;
the attractors' top features and their overlap (Jaccard index of the top-k sets) within and across steering
concepts; features shared by every attractor of one map (candidates for the steering vector's own footprint, since
alpha v is added upstream) separated from features specific to one attractor; and, when a feature-profile file is
given (mean activation of each feature on balanced labelled text categories), each specific feature's category
profile, which is evidence about what the feature responds to and not a label.

Usage: python tools/analyze_sae_readout.py READOUT_DIR [READOUT_DIR ...] [--profiles PROFILE.json] [--top 15]
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def load(folder):
    import explore_llm_attractors as ela

    d = json.loads((ela.DATA / "explore" / folder / "readout.json").read_text())
    concept = d["args"]["concept"]
    return concept, d


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("readouts", nargs="+")
    parser.add_argument("--profiles", default=None)
    parser.add_argument("--top", type=int, default=15)
    args = parser.parse_args()
    cells = []
    for folder in args.readouts:
        concept, d = load(folder)
        for c in d["cells"]:
            cells.append(dict(c, concept=concept))
    saes = sorted({name for c in cells for name in c["saes"]})
    profiles = json.loads(Path(args.profiles).read_text()) if args.profiles else None
    report = {"cells": len(cells), "saes": {}}
    for sae in saes:
        rows = []
        tops = []
        for c in cells:
            s = c["saes"][sae]
            top = [t["feature"] for t in s["top"][:args.top]]
            tops.append(set(top))
            loop = "".join(c["tokens"][-8:]).replace("\n", "|")
            rows.append({"concept": c["concept"], "kind": c["kind"], "log10_T": round(c["log10_T"], 4),
                         "alpha": round(c["alpha"], 4), "loop_tail": loop[:40], "fve": s["fve_on_orbit_states"],
                         "l0": s["l0_mean"], "top": top[:8]})
        n = len(cells)
        jac = np.zeros((n, n))
        for i, j in itertools.product(range(n), range(n)):
            u = tops[i] | tops[j]
            jac[i, j] = len(tops[i] & tops[j]) / len(u) if u else np.nan
        same, cross = [], []
        for i, j in itertools.combinations(range(n), 2):
            (same if cells[i]["concept"] == cells[j]["concept"] else cross).append(jac[i, j])
        by_concept = {}
        for concept in sorted({c["concept"] for c in cells}):
            idx = [i for i, c in enumerate(cells) if c["concept"] == concept]
            common = set.intersection(*[tops[i] for i in idx]) if idx else set()
            specific = {}
            for i in idx:
                others = set().union(*[tops[j] for j in range(n) if j != i])
                specific[rows[i]["loop_tail"]] = sorted(tops[i] - others)
            by_concept[concept] = {"cells": len(idx), "shared_by_all_cells": sorted(common),
                                   "specific_to_one_attractor": specific}
            if profiles and sae in profiles:
                prof = profiles[sae]
                by_concept[concept]["profiles_of_shared"] = {f: prof.get(str(f)) for f in sorted(common)}
        report["saes"][sae] = {
            "fve_on_orbit_states": {"median": float(np.nanmedian([r["fve"] for r in rows if r["fve"] is not None])),
                                    "min": float(np.nanmin([r["fve"] for r in rows if r["fve"] is not None]))},
            "jaccard_top_same_concept_mean": float(np.nanmean(same)) if same else None,
            "jaccard_top_cross_concept_mean": float(np.nanmean(cross)) if cross else None,
            "by_concept": by_concept, "cells": rows}
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
