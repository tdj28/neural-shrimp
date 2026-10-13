"""Frozen candidate rule for the cusp-unfolding pilot (docs/qwen2-cusp-unfolding-pilot.md; committed before the
Qwen2-0.5B maps of run S10b were retrieved or inspected beyond the run log's period counts).

Candidates are windows embedded in chaos in a finished float64 scan: connected components (4-neighbour) of cells
that share one attractor family (rotation-invariant leading-token loop) with period 3 <= p <= 12, of at least 6
cells, not touching the map's border, and with at least a quarter of the component's outer-neighbour cells chaotic
(largest exponent > 1e-3). They are ranked by size; the first three are returned, each with its most interior cell
(largest 4-neighbour distance to the component's edge; ties: smallest row, then column).

Usage: python tools/select_window_candidates.py SCAN_DIR [--top 3]
"""
import argparse
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

PERIODS = (3, 12)
MIN_CELLS = 6
CHAOS_SHARE = 0.25
CHAOS_LEVEL = 1e-3


def components(mask):
    labels = -np.ones(mask.shape, dtype=int)
    comps = []
    for start in zip(*np.nonzero(mask)):
        if labels[start] >= 0:
            continue
        k = len(comps)
        cells, queue = [], deque([start])
        labels[start] = k
        while queue:
            i, j = queue.popleft()
            cells.append((i, j))
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a, b = i + di, j + dj
                if 0 <= a < mask.shape[0] and 0 <= b < mask.shape[1] and mask[a, b] and labels[a, b] < 0:
                    labels[a, b] = k
                    queue.append((a, b))
        comps.append(cells)
    return comps


def interior_cell(cells):
    s = set(cells)
    dist = {c: 0 for c in cells if any((c[0] + d[0], c[1] + d[1]) not in s
                                       for d in ((1, 0), (-1, 0), (0, 1), (0, -1)))}
    queue = deque(dist)
    while queue:
        c = queue.popleft()
        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c[0] + d[0], c[1] + d[1])
            if n in s and n not in dist:
                dist[n] = dist[c] + 1
                queue.append(n)
    best = max(dist.values())
    return min(c for c, v in dist.items() if v == best), best


def select(folder, top=3):
    import explore_llm_attractors as ela

    z = np.load(ela.DATA / "explore" / folder / "scan.npz")
    lt, al, F, P, L = z["log10_temperature"], z["alpha"], z["family"], z["period"], z["lyapunov"]
    chaotic = L > CHAOS_LEVEL
    picks = []
    for f in np.unique(F[F >= 0]):
        cells_f = F == f
        p = int(np.median(P[cells_f]))
        if not PERIODS[0] <= p <= PERIODS[1]:
            continue
        for comp in components(cells_f):
            if len(comp) < MIN_CELLS:
                continue
            if any(i in (0, F.shape[0] - 1) or j in (0, F.shape[1] - 1) for i, j in comp):
                continue
            s = set(comp)
            ring = {(i + di, j + dj) for i, j in comp for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1))} - s
            share = float(np.mean([chaotic[c] for c in ring]))
            if share < CHAOS_SHARE:
                continue
            (i, j), depth = interior_cell(comp)
            picks.append({"family": int(f), "period": p, "cells": len(comp), "chaotic_ring_share": share,
                          "interior_cell": {"log10_T": float(lt[i]), "alpha": float(al[j]), "row": int(i),
                                            "col": int(j), "depth": int(depth)}})
    picks.sort(key=lambda c: (-c["cells"], c["interior_cell"]["row"], c["interior_cell"]["col"]))
    return picks[:top]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scan")
    parser.add_argument("--top", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(select(args.scan, args.top), indent=1))


if __name__ == "__main__":
    main()
