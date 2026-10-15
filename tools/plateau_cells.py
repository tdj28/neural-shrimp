"""Draw the cells of a plateau test from a stitched zoom map (tools/plot_zoom70.py), by a rule fixed before the draw:
N_CHAOTIC chaotic cells (exponent above 1e-3, no period), N_PERIODIC periodic cells and N_UNRESOLVED unresolved cells
(no period, exponent at most 1e-3), each drawn uniformly without replacement with numpy default_rng(SEED) in that
order. A class with too few cells gives all of them, and the shortfall is recorded, not filled from another class.
The output lists [log10T, alpha] pairs (the format of hidden_chaos.py --cells-file) with each cell's map label.

Usage: python tools/plateau_cells.py BAND_PREFIX OUT_JSON [--seed S] [--chaotic N] [--periodic N] [--unresolved N]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def draw(L, P, lt, al, counts, seed):
    """counts: (chaotic, periodic, unresolved). Returns the cells and the shortfall per class."""
    chaotic = (L > 1e-3) & (P <= 0)
    classes = {"chaotic": chaotic, "periodic": P > 0, "unresolved": (P <= 0) & ~chaotic}
    rng = np.random.default_rng(seed)
    cells, short = [], {}
    for (name, mask), n in zip(classes.items(), counts):
        idx = np.flatnonzero(mask)                          # row-major over (temperature, alpha)
        take = rng.choice(idx, size=min(n, idx.size), replace=False) if idx.size else np.array([], int)
        short[name] = int(n - take.size)
        for k in np.sort(take):
            r, c = np.unravel_index(k, L.shape)
            cells.append({"log10T": float(lt[r]), "alpha": float(al[c]), "map_label": name,
                          "map_exponent": float(L[r, c]), "map_period": int(P[r, c])})
    return cells, short


def main():
    import plot_zoom70 as pz
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("prefix")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--chaotic", type=int, default=12)
    ap.add_argument("--periodic", type=int, default=6)
    ap.add_argument("--unresolved", type=int, default=6)
    args = ap.parse_args()
    m = pz.stitch(pz.band_dirs(args.prefix))
    cells, short = draw(m["lyapunov"], m["period"], m["log10_temperature"], m["alpha"],
                        (args.chaotic, args.periodic, args.unresolved), args.seed)
    rec = {"note": (f"Plateau-test cells drawn from {args.prefix} by tools/plateau_cells.py: {args.chaotic} chaotic, "
                    f"{args.periodic} periodic, {args.unresolved} unresolved, numpy default_rng({args.seed}), "
                    "each class uniformly without replacement; shortfalls recorded, not filled."),
           "shortfall": short, "cells": [[c["log10T"], c["alpha"]] for c in cells], "meta": cells}
    Path(args.out).write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"cells": len(cells), "shortfall": short}))


if __name__ == "__main__":
    main()
