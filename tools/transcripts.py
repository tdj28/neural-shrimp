"""What the soft-token map writes: transcripts of single cells (paper table of periodic cycles against nearby chaotic
cells). From the fixed start, run TRANSIENT steps, then record RECORD steps: the leading token of the next-token
distribution at each step, its probability, the distribution's entropy (how far the soft token is from a hard token)
and the top three tokens. The period is the smallest q <= MAX_PERIOD with state return distance below TOL times the
state norm (as in the scans); a chaotic cell has none.

Usage: python tools/transcripts.py --vectors VEC --concept C --cell LOG10T,ALPHA [--cell ...] --out OUT
       [--model qwen0.5b --layer 12 --transient 600 --record 64 --dtype float64 --device cpu]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def run_cell(loop, v, log_t, alpha, transient, record, max_period, tol):
    torch = loop.torch
    X = loop.prepare((alpha * v)[None])
    temps = np.array([10.0 ** log_t])
    for _ in range(transient):
        X, _ = loop.step(X, temps)
    tok = loop.sm.tok
    steps, states = [], []
    for _ in range(record):
        X, p = loop.step(X, temps)
        p = p[0].double()
        top = torch.topk(p, 3)
        ent = float(-(p * torch.log(torch.clamp(p, min=1e-300))).sum())
        steps.append({"token": tok.decode([int(top.indices[0])]), "p": float(top.values[0]), "entropy": ent,
                      "top3": [[tok.decode([int(i)]), float(q)] for i, q in zip(top.indices, top.values)]})
        states.append(X[0, -1].double().cpu().numpy())
    S = np.array(states)
    scale = max(1.0, float(np.linalg.norm(S[-1])))
    period = 0
    for q in range(1, max_period + 1):
        if np.linalg.norm(S[-1 - q] - S[-1]) < tol * scale:
            period = q
            break
    return {"log10_temperature": log_t, "alpha": alpha, "period": period, "steps": steps}


def main():
    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--cell", action="append", required=True, help="LOG10T,ALPHA (write --cell=-0.115,0.07)")
    ap.add_argument("--transient", type=int, default=600)
    ap.add_argument("--record", type=int, default=64)
    ap.add_argument("--max-period", type=int, default=32)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, args.dtype, args.device)
    v = elpm.load_vectors(args.vectors)[args.concept]
    cells = []
    for c in args.cell:
        lt, a = (float(x) for x in c.split(","))
        r = run_cell(loop, v, lt, a, args.transient, args.record, args.max_period, args.tol)
        cells.append(r)
        txt = "".join(s["token"] for s in r["steps"][:32])
        print(f"log10T {lt:+.4f} alpha {a:.6f} period {r['period']:2d} mean entropy "
              f"{np.mean([s['entropy'] for s in r['steps']]):.2f}  {txt!r}", flush=True)
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "transcripts.json").write_text(json.dumps({"args": vars(args), "cells": cells}, indent=1))


if __name__ == "__main__":
    main()
