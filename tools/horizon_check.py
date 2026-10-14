"""Horizon check of finite-horizon map labels (run H2).

Compares the labels of the paper's maps (short horizon, one twin offset) with tools/hidden_chaos.py results at a long
horizon with two twin offsets, cell by cell:
- line: the hostile cascade line scan (hcascade/tline-a0.0655) against the same grid at the long horizon;
- cells: the stratified random cells of docs/h2-horizon-cells.json against their long-horizon results.
A map cell is chaotic when its exponent exceeds 1e-3 and no period up to the map's maximum is found, periodic when a
period is found. Long-horizon cells are chaotic when the exponent is positive (above 1e-3) at both offsets and no
period up to the map's own maximum is found (like for like), periodic when a period up to that maximum is found,
and otherwise "neither" (no period, exponent not positive at both offsets).

Usage: python tools/horizon_check.py OUT.json
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
LINE_SHORT = "llm-soft-loop/pod64/hcascade/tline-a0.0655"
LINE_LONG = "llm-soft-loop/pod64/hidden/h2-line-exact"
CELLS_LONG = "llm-soft-loop/pod64/hidden/h2-cells"
CELLS_FILE = ROOT / "docs" / "h2-horizon-cells.json"


def long_label(r, max_period):
    if 0 < r["period"] <= max_period:
        return "periodic"
    if all(l > 1e-3 for l in r["lyapunov"]):
        return "chaotic"
    return "neither"


def short_label(lyap, period):
    if period > 0:
        return "periodic"
    return "chaotic" if lyap > 1e-3 else "neither"


def table(pairs):
    out = {}
    for s in ("chaotic", "periodic", "neither"):
        got = Counter(l for a, l in pairs if a == s)
        n = sum(got.values())
        out[s] = {"n": n, **{k: got.get(k, 0) for k in ("chaotic", "periodic", "neither")},
                  "same_label": (got.get(s, 0) / n) if n else None}
    return out


def main():
    import explore_llm_attractors as ela

    data = ela.DATA / "explore"
    result = {}
    z = np.load(data / LINE_SHORT / "scan.npz")
    sargs = json.loads((data / LINE_SHORT / "scan.json").read_text())["args"]
    long_line = json.loads((data / LINE_LONG / "hidden-chaos.json").read_text())["cells"]
    lt_s, L, P = z["log10_temperature"], z["lyapunov"].ravel(), z["period"].ravel()
    lt_l = np.array([r["log10_temperature"] for r in long_line])
    if len(lt_s) != len(lt_l) or not np.allclose(lt_s, lt_l, rtol=0, atol=1e-12):
        raise SystemExit("line grids differ")
    mp = int(sargs["max_period"])
    pairs = [(short_label(L[i], P[i]), long_label(long_line[i], mp)) for i in range(len(lt_s))]
    result["line"] = {"short": {k: sargs[k] for k in ("transient", "samples", "eps", "tol", "max_period")},
                      "cells": len(pairs), "table": table(pairs)}
    spec = json.loads(CELLS_FILE.read_text())
    long_cells = json.loads((data / CELLS_LONG / "hidden-chaos.json").read_text())["cells"]
    if len(long_cells) != len(spec["cells"]):
        raise SystemExit("cell lists differ")
    for r, (lt, a) in zip(long_cells, spec["cells"]):
        if abs(r["log10_temperature"] - lt) > 1e-12 or abs(r["alpha"] - a) > 1e-12:
            raise SystemExit("cell order differs")
    result["cells"] = {}
    for src in sorted(set(spec["source"])):
        idx = [i for i, s in enumerate(spec["source"]) if s == src]
        pairs = [(spec["label"][i], long_label(long_cells[i], 32)) for i in idx]
        result["cells"][src] = {"cells": len(idx), "map_settings": "float64, 300 transient + 300 measured steps, "
                                "twin offset 1e-6, period up to 32", "table": table(pairs)}
    Path(sys.argv[1]).write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
