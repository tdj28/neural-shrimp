"""False chaos and real chaos: what noise does to a twin test of the soft-token loop (run FC).

A twin test estimates the largest Lyapunov exponent from a second row started at a relative offset eps from the base
row and renormalized back to eps after every step. Noise of relative size sigma in both rows (rounding, or the
float32 final normalization of the early two-GPU runs) adds a fresh difference of size about sqrt(2) sigma at every
step. Suppose the change survives along one leading direction while the others contract within a step. Then the twin
offset settles at a stationary angle to the leading direction, and the estimate is
    lambda_measured = max(lambda, log(sqrt(2) sigma / eps)):
the true exponent while eps > sqrt(2) sigma e^(-lambda), and the noise term below that. Noise can therefore raise an
estimate (a contracting orbit measured with an offset below sqrt(2) sigma looks chaotic), but in this model it never
lowers one. The real map has no single leading direction with all others contracting at once, so the run tests the
prediction quantitatively.

The tool runs Qwen2.5-0.5B-Instruct in float64 on the paper's prompt at three cells of the philosophy-steered map at
block 12 (chaotic, a weakly contracting period-3 orbit, a strongly contracting orbit), each with twins at relative
offsets 1e-12 to 1e-4 and one free twin (offset 1e-12, never renormalized). It runs under five conditions:
- **clean float64;**
- **Gaussian noise** of relative size 1e-10, 1e-8 or 1e-6 added to every row's next soft token, independently per row;
- **the float32 final normalization:** the final RMS normalization computed as Hugging Face code computes it,
  rounding the normalized state to float32, as in the two-GPU runs before 14 October.
A twin whose difference becomes exactly zero has merged with its base row (possible with deterministic rounding).
It is counted, its step is skipped, and it is restarted along a fresh random direction.

Usage: python tools/false_chaos_demo.py --threads 8 --out false-chaos/fc1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

CELLS = [(0.0652, -0.156, "chaotic"), (0.0642, -0.156, "period 3, weakly contracting"),
         (0.0, 0.18, "strongly contracting")]
EPS = [10.0 ** k for k in range(-12, -3)]
CONDITIONS = ["clean", "noise-1e-10", "noise-1e-08", "noise-1e-06", "float32-final-norm"]


def float32_norm_forward(self, hidden_states):
    """Hugging Face's RMSNorm: computes in float32 whatever the input dtype, then multiplies by the weight."""
    import torch

    input_dtype = hidden_states.dtype
    h = hidden_states.to(torch.float32)
    variance = h.pow(2).mean(-1, keepdim=True)
    h = h * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * h.to(input_dtype)


def main():
    import types

    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transient", type=int, default=500)
    ap.add_argument("--record", type=int, default=1000)
    ap.add_argument("--free-steps", type=int, default=300, help="steps of the free twin's distance kept")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    loop = esl.SoftLoop("qwen0.5b", 12, args.threads, "open", ela.OPENER, 1, "float64", "cpu")
    torch, E = loop.torch, loop.E
    v = elpm.load_vectors("llm-phase-map/vectors-l12")["philosophy"]
    norm = loop.model.model.norm
    G = 2 + len(EPS)                                  # rows per cell: base, free twin, renormalized twins
    nC = len(CELLS)
    steer = np.concatenate([[c[0] * v] * G for c in CELLS])
    temps = np.repeat([10.0 ** c[1] for c in CELLS], G)
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    results = {"args": vars(args), "cells": [{"alpha": c[0], "log10_temperature": c[1], "label": c[2]}
                                             for c in CELLS], "eps": EPS, "conditions": {}}
    nrm = lambda a: a.norm(dim=-1)                                                # noqa: E731
    for cond in args.conditions:
        tic = time.perf_counter()
        rng = np.random.default_rng(args.seed)
        sigma = float(cond.split("-")[1]) if cond.startswith("noise") else 0.0
        if cond == "float32-final-norm":
            norm.forward = types.MethodType(float32_norm_forward, norm)
        X = loop.prepare(steer)
        for _ in range(args.transient):                # every row of a cell follows the base orbit (same inputs)
            X, _ = loop.step(X, temps)
            if sigma:
                F = X[:, -1]
                xi = torch.as_tensor(rng.normal(size=tuple(F.shape)), dtype=F.dtype) / np.sqrt(F.shape[1])
                X = (F + sigma * nrm(F)[:, None] * xi)[:, None, :]
            for c in range(nC):                        # keep the rows of a cell identical during the transient
                X[c * G:(c + 1) * G] = X[c * G]
        F = X[:, -1].clone()
        for c in range(nC):
            b = F[c * G]
            for j, e in enumerate([EPS[0]] + EPS):
                g = torch.as_tensor(rng.normal(size=b.shape[0]), dtype=F.dtype)
                F[c * G + 1 + j] = b + e * nrm(b) * g / g.norm()
        X = F[:, None, :]
        growth = np.zeros((nC, len(EPS)))
        counted = np.zeros((nC, len(EPS)))
        merges = np.zeros((nC, len(EPS)), dtype=int)
        free = np.full((nC, args.free_steps), np.nan)
        tail = []
        for i in range(args.record):
            Xn, _ = loop.step(X, temps)
            Fn = Xn[:, -1]
            if sigma:
                xi = torch.as_tensor(rng.normal(size=tuple(Fn.shape)), dtype=Fn.dtype) / np.sqrt(Fn.shape[1])
                Fn = Fn + sigma * nrm(Fn)[:, None] * xi
            Fn = Fn.clone()
            for c in range(nC):
                b0, bn = X[c * G, -1], Fn[c * G]
                fd = (nrm(Fn[c * G + 1] - bn) / nrm(bn)).item()
                if i < args.free_steps:
                    free[c, i] = np.log10(fd) if fd > 0 else -np.inf
                for j, e in enumerate(EPS):
                    r = c * G + 2 + j
                    dx = nrm(X[r, -1] - b0).item()
                    df = nrm(Fn[r] - bn).item()
                    if df == 0.0:                       # merged: count, skip, restart along a fresh direction
                        merges[c, j] += 1
                    else:
                        growth[c, j] += np.log(df / dx)
                        counted[c, j] += 1
                    if df == 0.0:
                        g = torch.as_tensor(rng.normal(size=bn.shape[0]), dtype=Fn.dtype)
                        Fn[r] = bn + e * nrm(bn) * g / g.norm()
                    else:
                        Fn[r] = bn + (Fn[r] - bn) * (e * nrm(bn) / df)
            tail.append(Fn[[c * G for c in range(nC)]].numpy().copy())
            if len(tail) > 66:
                tail.pop(0)
            X = Fn[:, None, :]
        if cond == "float32-final-norm":
            del norm.forward                            # back to the class's (float64) forward
        S = np.array(tail)
        periods = []
        for c in range(nC):
            s = S[:, c]
            periods.append(next((q for q in range(1, min(65, len(s))) if np.linalg.norm(s[-1 - q] - s[-1])
                                 < 1e-8 * max(1.0, np.linalg.norm(s[-1]))), 0))
        lam = np.where(counted > 0, growth / np.maximum(counted, 1), np.nan)
        results["conditions"][cond] = {"sigma": sigma, "exponent": lam.tolist(), "merges": merges.tolist(),
                                       "counted": counted.tolist(), "period": periods,
                                       "free_log10_distance": [[None if not np.isfinite(x) else round(float(x), 3)
                                                                for x in row] for row in free],
                                       "seconds": round(time.perf_counter() - tic, 1)}
        print(json.dumps({"condition": cond, "seconds": results["conditions"][cond]["seconds"],
                          "exponent_at_1e-6": [round(lam[c, 6], 3) for c in range(nC)],
                          "exponent_at_1e-12": [round(lam[c, 0], 3) for c in range(nC)], "period": periods}),
              flush=True)
        (out / "false-chaos.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
