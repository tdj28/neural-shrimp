"""Where does the loop's reduced map have two turning points? A scan of return maps over chaotic cells (run TP).

A shrimp centre needs an orbit that passes two critical points at once, so the effectively one-dimensional return map
must have two turning points. For each cell the loop (window 1, float64; steering alpha v after block L, optionally
plus alpha2 w after block L2) runs a transient and records the states. The leading-token sequence gives a symbolic
period q (the smallest q <= qmax over which the recorded tokens repeat exactly; q = 1 if none). For each phase of q,
the states of that phase are projected on their leading principal component s, and the return map s_(n+1) against
s_n (consecutive visits of the phase) is reduced to a curve by medians over equal-count bins of s_n, lightly smoothed.
Turning points are sign changes of the curve's slope whose rise and fall both exceed `prominence` times the curve's
range. A phase counts only if its leading component carries at least `min_explained` of the variance (an effectively
one-dimensional phase).

Output per cell: q, per-phase turning points and explained variance, the largest count over one-dimensional phases.

Usage: python tools/turning_scan.py --cells-file F --layer 12 --vectors V --concept C [--layer2 L2 --vectors2 V2
       --concept2 C2] --device cuda --out llm-soft-loop/pod64/tp/x
       cells-file: JSON with 'cells', a list of [log10T, alpha] or [log10T, alpha, alpha2] entries
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def symbolic_period(tokens, qmax):
    n = len(tokens)
    for q in range(1, qmax + 1):
        if all(tokens[i] == tokens[i + q] for i in range(n - q)):
            return q
    return 1


def return_curve(s, bins):
    """Equal-count binned medians of s_(n+1) against s_n, smoothed by a three-point running mean."""
    x, y = s[:-1], s[1:]
    order = np.argsort(x)
    x, y = x[order], y[order]
    edges = np.linspace(0, len(x), bins + 1).astype(int)
    cx = np.array([np.median(x[a:b]) for a, b in zip(edges[:-1], edges[1:]) if b > a])
    cy = np.array([np.median(y[a:b]) for a, b in zip(edges[:-1], edges[1:]) if b > a])
    if len(cy) >= 3:
        cy = np.convolve(np.pad(cy, 1, mode="edge"), np.ones(3) / 3, mode="valid")
    return cx, cy


def count_turning(cy, prominence):
    """Turning points of a curve: interior pivots of its zigzag after pruning every swing smaller than
    prominence * range (the smallest swing first). A single hump gives 1, an N-shape 2, a monotone curve 0."""
    cy = np.asarray(cy, dtype=float)
    if len(cy) < 5 or np.ptp(cy) == 0:
        return 0
    thr = prominence * float(np.ptp(cy))
    d = np.sign(np.diff(cy))
    d = d[d != 0] if np.any(d != 0) else d
    idx = [0]
    sd = np.sign(np.diff(cy))
    last = 0
    for i in range(1, len(cy) - 1):
        if sd[i - 1] != 0:
            last = sd[i - 1]
        if sd[i] != 0 and last != 0 and sd[i] != last:
            idx.append(i)
    idx.append(len(cy) - 1)
    piv = [cy[i] for i in idx]
    while len(piv) > 2:
        sw = np.abs(np.diff(piv))
        j = int(np.argmin(sw))
        if sw[j] >= thr:
            break
        if j == 0:
            del piv[1]
        elif j == len(piv) - 2:
            del piv[-2]
        else:
            del piv[j:j + 2]
        # merge pivots that no longer alternate (monotone runs)
        k = 1
        while k < len(piv) - 1:
            if (piv[k] - piv[k - 1]) * (piv[k + 1] - piv[k]) >= 0:
                del piv[k]
            else:
                k += 1
    return len(piv) - 2


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    ap.add_argument("--concept", default="philosophy")
    ap.add_argument("--layer2", type=int, default=None)
    ap.add_argument("--vectors2", default=None)
    ap.add_argument("--concept2", default=None)
    ap.add_argument("--cells-file", required=True)
    ap.add_argument("--transient", type=int, default=1000)
    ap.add_argument("--record", type=int, default=6000)
    ap.add_argument("--qmax", type=int, default=24)
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--prominence", type=float, default=0.05)
    ap.add_argument("--min-explained", type=float, default=0.9)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save", action="store_true", help="also save every phase's projection s (projections.npz)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    cells = json.loads(Path(args.cells_file).read_text())["cells"]
    loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    torch = loop.torch
    v = np.asarray(elpm.load_vectors(args.vectors)[args.concept], dtype=np.float64)
    w = None
    if args.layer2 is not None:
        loop.sm.add_steer_layer(args.layer2)
        w = np.asarray(elpm.load_vectors(args.vectors2)[args.concept2], dtype=np.float64)
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    results = []
    saved = {}
    for s0 in range(0, len(cells), args.chunk):
        part = cells[s0:s0 + args.chunk]
        steer = np.array([c[1] * v for c in part])
        if w is not None:
            loop.extra = (args.layer2, np.array([c[2] * w for c in part]))
        temps = torch.as_tensor(10.0 ** np.array([c[0] for c in part]), dtype=loop.state_dtype, device=loop.device)
        X = loop.prepare(steer)
        for _ in range(args.transient):
            X, _ = loop.step(X, temps)
        states = np.zeros((args.record, len(part), X.shape[-1]), dtype=np.float32)
        toks = np.zeros((args.record, len(part)), dtype=np.int64)
        for i in range(args.record):
            X, p = loop.step(X, temps)
            states[i] = X[:, -1].float().cpu().numpy()
            toks[i] = p.argmax(-1).cpu().numpy()
        loop.extra = None
        for k, c in enumerate(part):
            q = symbolic_period(toks[:, k].tolist(), args.qmax)
            phases = []
            for ph in range(q):
                P = states[ph::q, k].astype(np.float64)
                if len(P) < 4 * args.bins:
                    continue
                Pc = P - P.mean(0)
                _, sv, vt = np.linalg.svd(Pc, full_matrices=False)
                expl = float(sv[0] ** 2 / max(np.sum(sv ** 2), 1e-300))
                sproj = Pc @ vt[0]
                if args.save:
                    saved[f"c{s0 + k}_p{ph}"] = sproj.astype(np.float32)
                _, cy = return_curve(sproj, args.bins)
                phases.append({"phase": ph, "explained": expl, "turning": count_turning(cy, args.prominence),
                               "spread": float(np.ptp(sproj))})
            one_d = [ph for ph in phases if ph["explained"] >= args.min_explained]
            results.append({"cell": c, "q": q, "phases": phases,
                            "max_turning_1d": max((ph["turning"] for ph in one_d), default=None),
                            "phases_1d": len(one_d)})
        line = json.dumps({"done": s0 + len(part), "of": len(cells), "seconds": round(time.perf_counter() - tic, 1),
                           "two_or_more_1d": int(sum(1 for r in results if (r["max_turning_1d"] or 0) >= 2))})
        print(line, flush=True)
        with open(out / "progress.jsonl", "a") as f:
            f.write(line + "\n")
        (out / "turning.json").write_text(json.dumps({"args": vars(args), "cells": results}, indent=1))
        if args.save:
            np.savez_compressed(out / "projections.npz", **saved)


if __name__ == "__main__":
    main()
