"""Sensitive dependence in soft-thinking decoding with a growing context (run ST).

The soft-token map of the paper feeds the expected embedding x = E^T softmax(z / T) back at a fixed position with a
one-token window. Soft Thinking (Zhang et al. 2025) instead appends each expected embedding to the context and keeps
all of them. Here a base trajectory and twins whose first soft token is displaced by eps0 (random unit directions)
run that growing-context decoding in float64; the distance d_t between the newest soft tokens of twin and base,
without renormalization, measures whether an early perturbation grows. Per twin: the finite-time growth rate, the slope
of log d_t against t over the steps where 10 eps0 <= d_t <= DMAX (the linear regime), or over all steps when d_t never
leaves the start (decay or no growth); and the first step at which d_t exceeds DMAX.

Usage: python tools/soft_thinking_twins.py --model qwen0.5b --log-temperature LO HI --n N --steps S --out OUT
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
DMAX = 1e-3


def growth_rate(d, eps0, dmax=DMAX):
    """Slope of log d_t (t = 1, 2, ...) over the linear regime 10 eps0 <= d <= dmax, the first step above dmax (or
    None), and the number of points used; if d never reaches 10 eps0, the slope over all steps."""
    d = np.maximum(np.asarray(d, dtype=float), 1e-300)
    t = np.arange(1, len(d) + 1)
    above = np.nonzero(d > dmax)[0]
    first = int(above[0]) + 1 if len(above) else None
    stop = above[0] if len(above) else len(d)
    sel = np.nonzero((d[:stop] >= 10 * eps0) & (d[:stop] <= dmax))[0]
    if len(sel) < 5:
        sel = np.arange(stop)
    if len(sel) < 2:
        return float("nan"), first, int(len(sel))
    slope = np.polyfit(t[sel], np.log(d[sel]), 1)[0]
    return float(slope), first, int(len(sel))


def run(loop, log_ts, steps, eps0, n_twins, seed=0):
    torch = loop.torch
    n = len(log_ts)
    B = n * (1 + n_twins)
    X = loop.prepare(np.zeros((B, loop.E.shape[1])))          # (B, 1, d): start state at position L0, unsteered
    rng = np.random.default_rng(seed)
    base = X[:n]
    parts = [base]
    for _ in range(n_twins):
        u = torch.as_tensor(rng.normal(size=tuple(base.shape)), dtype=loop.state_dtype, device=loop.device)
        parts.append(base + u / u.flatten(1).norm(dim=1)[:, None, None] * eps0)
    X = torch.cat(parts)
    T = torch.as_tensor(np.tile(10.0 ** np.asarray(log_ts), 1 + n_twins), dtype=loop.state_dtype,
                        device=loop.device)[:, None]
    dist = np.zeros((steps, n_twins, n))
    ent = np.zeros((steps, n))
    loop.sm.steer = loop.steer_rows
    try:
        with torch.no_grad():
            for t in range(steps):
                z = loop.model(inputs_embeds=X.to(loop.dtype), past_key_values=loop.cache, use_cache=True,
                               logits_to_keep=1).logits[:, -1, :].to(loop.device, loop.state_dtype)
                p = torch.softmax(z / T, dim=-1)
                X = (p @ loop.E)[:, None, :]                     # appended: the cache keeps every soft token
                for k in range(n_twins):
                    dist[t, k] = (X[n * (k + 1):n * (k + 2), 0] - X[:n, 0]).norm(dim=-1).double().cpu().numpy()
                p0 = p[:n].double()
                ent[t] = (-(p0 * torch.log(torch.clamp(p0, min=1e-300))).sum(-1)).cpu().numpy()
    finally:
        loop.sm.steer = None
    out = []
    for c in range(n):
        rates = [growth_rate(dist[:, k, c], eps0) for k in range(n_twins)]
        out.append({"log10_temperature": float(log_ts[c]), "growth_rates": [r[0] for r in rates],
                    "first_above_dmax": [r[1] for r in rates], "points": [r[2] for r in rates],
                    "growth_rate_median": float(np.nanmedian([r[0] for r in rates])),
                    "entropy_mean": float(ent[:, c].mean()), "final_distance": dist[-1, :, c].tolist()})
    return out, dist, ent


def main():
    import explore_llm_attractors as ela
    import explore_llm_soft_loop as esl

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--log-temperature", type=float, nargs=2, required=True)
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--eps0", type=float, default=1e-10)
    ap.add_argument("--twins", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=24, help="temperatures per batch")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--weights", default="bf16", choices=("native", "bf16"))
    ap.add_argument("--prefix-file", default=None)
    ap.add_argument("--prefix-index", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    prompt, opener = "open", ela.OPENER
    if args.prefix_file is not None:
        pref = json.loads(Path(args.prefix_file).read_text())["prefixes"][args.prefix_index]
        ela.PROMPTS["custom"] = pref["system"]
        prompt, opener = "custom", pref["opener"]
    loop = esl.SoftLoop(args.model, 0, 16, prompt, opener, 1, "float64", args.device, args.weights)
    log_ts = np.linspace(args.log_temperature[0], args.log_temperature[1], args.n)
    cells, dists, ents = [], [], []
    for s in range(0, args.n, args.chunk):
        got, dist, ent = run(loop, log_ts[s:s + args.chunk], args.steps, args.eps0, args.twins, args.seed + s)
        cells += got
        dists.append(dist)
        ents.append(ent)
        print(json.dumps({"done": len(cells), "of": args.n}), flush=True)
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "soft-thinking-twins.json").write_text(json.dumps({"args": vars(args), "dmax": DMAX, "cells": cells}, indent=1))
    np.savez_compressed(out / "distances.npz", distance=np.concatenate(dists, axis=2), entropy=np.concatenate(ents, axis=1),
                        log10_temperature=log_ts)
    for c in cells:
        print(json.dumps({k: c[k] for k in ("log10_temperature", "growth_rate_median", "entropy_mean")}))


if __name__ == "__main__":
    main()
