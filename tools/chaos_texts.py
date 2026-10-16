"""What the soft-token loop writes in a chaotic zone, and how a 1e-10 difference splits the text (run CT).

At each cell (alpha, log10 T) the loop runs in float64 from the fixed start (the opener state, as in every map) for
a transient. The cell is labeled from twin exponents at relative offsets 1e-6 and 1e-8 (renormalized every step,
--lyapunov-steps steps) and the period of the base orbit (relative return distance below --tol, up to 64): chaotic
when both exponents exceed 1e-3 and no period is found, as in the maps. The tool then records --record steps of:
- **Base text:** the most probable next token at every step, its probability, and the three most probable tokens with
  their probabilities.
- **Twin text:** a free twin started at relative distance --eps-text from the base state and never renormalized; its
  most probable token at every step and its relative distance from the base.
- **Split:** the first step at which the two most probable tokens differ.
So the output shows both what a chaotic loop writes and the sensitivity that makes it chaotic, in words.

Usage: python tools/chaos_texts.py --model qwen0.5b --layer 12 --vectors llm-phase-map/vectors-l12 \\
           --concept philosophy --cells 0.0652,-0.156 0.07,-0.1 --json docs/chaos-texts/q05.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def twin_exponent(cell, X, steps, rel, seed=0):
    torch = cell.torch
    rng = np.random.default_rng(seed)
    d = torch.as_tensor(rng.normal(size=X.shape), dtype=X.dtype, device=X.device)
    d0 = rel * float(X.norm())
    Y = X + d0 * d / d.norm()
    total = 0.0
    for _ in range(steps):
        X, Y = cell.f(X), cell.f(Y)
        dist = float((Y - X).norm())
        total += np.log(dist / d0)
        Y = X + (Y - X) * (d0 / dist)
    return total / steps


def step_tokens(cell, X, tok):
    loop = cell.loop
    cell._use(1)
    Xn, p = loop.step(X, np.array([cell.T]))
    top = np.argsort(-p[0].cpu().numpy())[:3] if hasattr(p, "cpu") else np.argsort(-p[0])[:3]
    pv = p[0].cpu().numpy() if hasattr(p, "cpu") else p[0]
    return Xn, [{"token": int(k), "text": tok.decode([int(k)]), "p": float(pv[k])} for k in top]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vectors", default="llm-phase-map/vectors-l12", help="explore-relative; '' for unsteered")
    ap.add_argument("--concept", default="philosophy")
    ap.add_argument("--cells", nargs="+", required=True, help="alpha,log10T pairs")
    ap.add_argument("--transient", type=int, default=1000)
    ap.add_argument("--record", type=int, default=60)
    ap.add_argument("--lyapunov-steps", type=int, default=1000)
    ap.add_argument("--eps-text", type=float, default=1e-10)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--json", required=True)
    args = ap.parse_args()

    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, "float64", "cpu")
    tok = loop.sm.tok
    torch = loop.torch
    v = elpm.load_vectors(args.vectors)[args.concept] if args.vectors else np.zeros(loop.E.shape[1])
    out = {"args": vars(args), "cells": []}
    for spec in args.cells:
        a, lt = (float(s) for s in spec.split(","))
        cell = esl.SoftOrbit(loop, a * v, 10.0 ** lt)
        X = cell.iterate(cell.start, args.transient)
        per = cell.detect_period(X, 64, args.tol)
        lams = [twin_exponent(cell, X.clone(), args.lyapunov_steps, rel) for rel in (1e-6, 1e-8)]
        chaotic = bool(per == 0 and all(x > 1e-3 for x in lams))
        rng = np.random.default_rng(1)
        d = torch.as_tensor(rng.normal(size=X.shape), dtype=X.dtype, device=X.device)
        Y = X + args.eps_text * float(X.norm()) * d / d.norm()
        steps, split = [], None
        for t in range(args.record):
            X, top = step_tokens(cell, X, tok)
            Y, ytop = step_tokens(cell, Y, tok)
            dist = float((Y - X).norm() / X.norm())
            if split is None and ytop[0]["token"] != top[0]["token"]:
                split = t + 1
            steps.append({"top": top, "twin": ytop[0]["text"], "log10_distance": round(np.log10(max(dist, 1e-300)), 2)})
        rec = {"alpha": a, "log10T": lt, "period": int(per), "lyapunov": lams, "chaotic": chaotic,
               "split_step": split, "text": "".join(s["top"][0]["text"] for s in steps),
               "twin_text": "".join(s["twin"] for s in steps), "steps": steps}
        out["cells"].append(rec)
        print(json.dumps({"alpha": a, "log10T": lt, "period": per, "lyapunov": [round(x, 4) for x in lams],
                          "chaotic": chaotic, "split": split, "text": rec["text"][:160]}, ensure_ascii=False),
              flush=True)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
