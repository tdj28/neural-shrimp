"""Return maps of chaotic attractors of the soft-token map (human, 15 Oct 2026: "You should be able to find the return
map"): does the one-dimensional reduction stay unimodal near the hostile period-7 body, or does a second turning point
appear?

For each cell the loop (window 1, float64) runs transient + record steps from the fixed start; the recorded states are
projected on their own leading principal components. Near the period-7 body the chaos sits behind a 7-periodic text,
so the relevant one-dimensional map is the 7-step return within each phase: s_{n+7} against s_n for n = phase mod 7,
with s the projection on the leading principal component of that phase's points. Writes return-map.npz (projections
per cell and phase) and a figure; counts the turning points of a smoothed (binned-median) return curve per phase.

Usage: python tools/return_map.py --cells 0.07136:0.20621 0.0655:0.18 --device cuda --out llm-soft-loop/pod64/retmap/x
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def turning_points(x, y, bins=40, min_count=5):
    """Turning points of the binned-median curve y(x): sign changes of its slope over bins with enough points."""
    edges = np.linspace(np.min(x), np.max(x), bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, bins - 1)
    cx, cy = [], []
    for b in range(bins):
        m = idx == b
        if m.sum() >= min_count:
            cx.append(0.5 * (edges[b] + edges[b + 1]))
            cy.append(float(np.median(y[m])))
    cy = np.array(cy)
    if len(cy) < 3:
        return 0, np.array(cx), cy
    slope = np.sign(np.diff(cy))
    slope = slope[slope != 0]
    return int(np.sum(slope[1:] != slope[:-1])), np.array(cx), cy


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cells", nargs="+", required=True, metavar="ALPHA:LOG10T")
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vectors", default="llm-phase-map/vectors-l12-p1b")
    ap.add_argument("--concept", default="hostility-vs-kindness")
    ap.add_argument("--period", type=int, default=7, help="phase period of the return map")
    ap.add_argument("--transient", type=int, default=1000)
    ap.add_argument("--record", type=int, default=7000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    cells = [tuple(float(x) for x in c.split(":")) for c in args.cells]
    loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    v = np.asarray(elpm.load_vectors(args.vectors)[args.concept], dtype=np.float64)
    tic = time.perf_counter()
    X = loop.prepare(np.array([a * v for a, _ in cells]))
    temps = np.array([10.0 ** t for _, t in cells])
    for _ in range(args.transient):
        X, _ = loop.step(X, temps)
    states = np.zeros((args.record, len(cells), X.shape[-1]), dtype=np.float64)
    for n in range(args.record):
        X, _ = loop.step(X, temps)
        states[n] = X[:, -1, :].double().cpu().numpy()
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    q = args.period
    result, proj = [], {}
    for c, (a, t) in enumerate(cells):
        S = states[:, c, :]
        rec = {"alpha": a, "log10_temperature": t, "phases": []}
        for ph in range(q):
            P = S[ph::q]
            Pc = P - P.mean(0)
            _, sv, vt = np.linalg.svd(Pc, full_matrices=False)
            s = Pc @ vt[0]
            x, y = s[:-1], s[1:]
            tp, cx, cy = turning_points(x, y)
            proj[f"c{c}_p{ph}"] = s
            rec["phases"].append({"phase": ph, "turning_points": tp, "explained_1": float(sv[0] ** 2 / np.sum(sv ** 2)),
                                  "explained_2": float(sv[1] ** 2 / np.sum(sv ** 2)), "spread": float(np.ptp(s))})
        result.append(rec)
        print(json.dumps({"cell": [a, t], "turning_points": [p["turning_points"] for p in rec["phases"]],
                          "explained_1": [round(p["explained_1"], 4) for p in rec["phases"]]}), flush=True)
    np.savez_compressed(out / "return-map.npz", cells=np.array(cells), period=q, **proj)
    (out / "return-map.json").write_text(json.dumps({"args": vars(args), "cells": result,
                                                     "seconds": round(time.perf_counter() - tic, 1)}, indent=1))


if __name__ == "__main__":
    main()
