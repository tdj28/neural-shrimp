"""Superstable curves of one periodic orbit of the soft-token map in the (alpha, log10 T) plane, and where two of them
cross: a doubly superstable point, the centre of a shrimp (Gallas 1993; Jones 2012, arXiv:1201.4343).

For a period-p orbit x_0, ..., x_{p-1} with leading multiplier mu and its eigenvector v, the chain
v_{i+1} = DF(x_i) v_i splits mu into one factor per map step. With fixed reference directions r_i (unit vectors: the
chain directions at the seed point), the signed factors

    c_i = <DF(x_i) v_i, r_{i+1}> / <v_i, r_i>,   i = 0..p-1 (r_p = r_0),

are scale-invariant in v and multiply to mu exactly (the product telescopes to <M v, r_0> / <v, r_0>). In an
effectively one-dimensional map, c_i is the derivative of the reduced map at x_i, and a superstable curve is where
one factor vanishes. Where the zero curves of two different factors cross, the orbit is doubly superstable. A single
factor's zero curve does not cross itself generically, and a crossing seen on a map could also be two coexisting
orbits; here one orbit is followed by continuation and its token loop is checked at every point.

Near a zero of mu the leading multiplier falls below the transverse ones and its eigenvector is no longer defined;
such points are flagged (clean: |mu_1| > 10 |mu_2|), and zero curves are read from clean points only.

grid: continuation over a box, breadth-first from the seed cell, each cell warm-started from an already solved
neighbour (Newton on F^p(x) = x with the low-rank solver of tools/llm_feigenbaum.py, Ritz multipliers, the factors
and the token loop). Writes grid.npz and grid.json under explore/OUT.

Usage:
  python tools/doubly_superstable.py grid --period 7 --seed 0.0700 0.2040 --alpha 0.0686 0.0714 \
      --log-temperature 0.199 0.209 --resolution 29 29 --vectors llm-phase-map/vectors-l12-p1b \
      --concept hostility-vs-kindness --device cuda --out llm-soft-loop/pod64/dss/p7-blob
"""
import argparse
import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import llm_feigenbaum as lfb  # noqa: E402


def factors(cell, X, period, v, refs):
    """Signed per-step factors c_i of the leading multiplier along the orbit from X, chain directions (unit), and
    the argmax token of every step."""
    torch = cell.torch
    vi = v / np.linalg.norm(v)
    cs, dirs, toks = [], [], []
    for i in range(period):
        toks.append(int(cell.probabilities(X).argmax()))
        w = lfb._cols(cell.jvp_rows(X, lfb._rows(vi[:, None], X)))[:, 0]
        r_in, r_out = refs[i], refs[(i + 1) % period]
        cs.append(float(np.dot(w, r_out) / np.dot(vi, r_in)))
        dirs.append(vi)
        vi = w / np.linalg.norm(w)
        X = cell.f(X)
    return np.array(cs), dirs, toks


def leading(vals, vecs):
    """Leading multiplier (largest modulus) as a real number when it is real, its real eigenvector, and whether it
    dominates the transverse spectrum."""
    mu = vals[0]
    v = vecs[:, 0]
    v = np.real(v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))])))
    clean = bool(len(vals) < 2 or abs(mu) > 10.0 * abs(vals[1]))
    return mu, v, clean


def grid(args):
    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    opener = args.opener if args.opener is not None else ela.OPENER
    loop = esl.SoftLoop(args.model, args.layer, args.threads, args.prompt, opener, args.window, "float64", args.device)
    v = elpm.load_vectors(args.vectors)[args.concept]
    p = args.period

    def make_cell(a, lt):
        return esl.SoftOrbit(loop, a * v, 10.0 ** lt, chunk=args.chunk)

    lts = np.linspace(*args.log_temperature, args.resolution[0])
    als = np.linspace(*args.alpha, args.resolution[1])
    nt, na = len(lts), len(als)
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "progress.jsonl", "a")

    def emit(rec):
        line = json.dumps(rec)
        print(line, flush=True)
        log.write(line + "\n")
        log.flush()

    # seed: the attractor reached from the fixed start must have period p
    tic = time.perf_counter()
    a0, lt0 = args.seed
    cell = make_cell(a0, lt0)
    X = cell.iterate(cell.start, args.transient)
    found = cell.detect_period(X, max(p, 32), 1e-6)
    if found != p:
        raise SystemExit(f"the attractor at the seed ({a0}, {lt0}) has period {found}, not {p}: refusing to start")
    Q = np.linalg.qr(np.random.default_rng(0).normal(size=(X.numel(), args.k)))[0]
    X, Q, hist = lfb.newton(cell, X, p, Q, lfb.tol_for(p, args.tol), 12)
    if not hist[-1] < lfb.tol_for(p, args.tol):
        raise SystemExit(f"Newton did not converge at the seed: {hist}")
    vals, vecs, res, rh, Q = lfb.ritz(cell, X, p, Q)
    mu, vlead, clean = leading(vals, vecs)
    if not clean or abs(mu.imag) > 1e-8:
        raise SystemExit(f"leading multiplier at the seed is {mu!r} (next {abs(vals[1]):.3g}): choose a seed away from "
                         "the superstable curves")
    refs = [np.ones(1)] * p
    _, dirs, seed_toks = factors(cell, X, p, vlead, [np.ones(X.numel())] * p)
    refs = [d / np.linalg.norm(d) for d in dirs]
    c_seed, _, _ = factors(cell, X, p, vlead, refs)
    emit({"seed": [a0, lt0], "mu": float(mu.real), "next_modulus": float(abs(vals[1])), "factors": c_seed.tolist(),
          "product": float(np.prod(c_seed)), "tokens": seed_toks, "seconds": round(time.perf_counter() - tic, 1)})

    # breadth-first continuation from the grid cell nearest the seed
    i0, j0 = int(np.argmin(abs(lts - lt0))), int(np.argmin(abs(als - a0)))
    MU = np.full((nt, na), np.nan)
    MU_IM = np.full((nt, na), np.nan)
    NEXT = np.full((nt, na), np.nan)
    CLEAN = np.zeros((nt, na), dtype=bool)
    OK = np.zeros((nt, na), dtype=bool)
    SAME = np.zeros((nt, na), dtype=bool)
    C = np.full((nt, na, p), np.nan)
    PROD = np.full((nt, na), np.nan)
    RES = np.full((nt, na), np.nan)
    COS = np.full((nt, na), np.nan)
    states = {}
    queue = collections.deque([((i0, j0), None)])
    seen = {(i0, j0)}
    states[None] = (X, Q)
    n_done = 0
    while queue:
        (i, j), parent = queue.popleft()
        Xs, Qs = states[parent]
        t = time.perf_counter()
        cell = make_cell(als[j], lts[i])
        try:
            Xn, Qn, hist = lfb.newton(cell, Xs, p, Qs, lfb.tol_for(p, args.tol), 12)
            ok = bool(hist[-1] < lfb.tol_for(p, args.tol))
        except Exception as err:                       # noqa: BLE001 - recorded, the cell stays unsolved
            ok, hist = False, [str(err)]
        rec = {"i": i, "j": j, "alpha": float(als[j]), "log10_temperature": float(lts[i]), "converged": ok}
        if ok:
            split = min((lfb.return_distance(cell, Xn, q) for q in range(1, p) if p % q == 0), default=1.0)
            vals, vecs, res, rh, Qn = lfb.ritz(cell, Xn, p, Qn)
            mu, vlead, clean = leading(vals, vecs)
            c, dirs, toks = factors(cell, Xn, p, vlead, refs)
            OK[i, j] = split > 1e-7
            MU[i, j], MU_IM[i, j], NEXT[i, j], CLEAN[i, j] = mu.real, mu.imag, abs(vals[1]), clean
            C[i, j], PROD[i, j], RES[i, j] = c, np.prod(c), hist[-1]
            COS[i, j] = min(abs(float(np.dot(d, r))) for d, r in zip(dirs, refs))
            SAME[i, j] = toks == seed_toks
            states[(i, j)] = (Xn, Qn)
            rec.update({"mu": float(mu.real), "mu_imag": float(mu.imag), "next_modulus": float(abs(vals[1])),
                        "clean": clean, "factors": c.tolist(), "product": float(np.prod(c)), "same_tokens": SAME[i, j].item(),
                        "min_reference_cosine": float(COS[i, j]), "newton": float(hist[-1]), "split": float(split)})
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (i + di, j + dj)
                if 0 <= nb[0] < nt and 0 <= nb[1] < na and nb not in seen:
                    seen.add(nb)
                    queue.append((nb, (i, j)))
        rec["seconds"] = round(time.perf_counter() - t, 2)
        emit(rec)
        n_done += 1
        if n_done % 25 == 0 or not queue:
            np.savez_compressed(out / "grid.npz", alpha=als, log10_temperature=lts, mu=MU, mu_imag=MU_IM, next_modulus=NEXT,
                                clean=CLEAN, ok=OK, same_tokens=SAME, factors=C, product=PROD, newton=RES,
                                min_reference_cosine=COS, seed=np.array(args.seed), seed_tokens=np.array(seed_toks))
        # free states that no queued cell still needs
        live = {par for _, par in queue}
        for key in [k for k in states if k is not None and k not in live]:
            del states[key]
    tok = loop.sm.tok
    (out / "grid.json").write_text(json.dumps({
        "args": vars(args), "seed_tokens": seed_toks, "seed_text": [tok.decode([t]) for t in seed_toks],
        "cells": int(nt * na), "solved": int(OK.sum()), "clean": int((OK & CLEAN).sum()),
        "same_tokens": int((OK & SAME).sum()), "seconds": round(time.perf_counter() - tic, 1)}, indent=1,
        ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)
    g = sub.add_parser("grid")
    g.add_argument("--period", type=int, required=True)
    g.add_argument("--seed", type=float, nargs=2, required=True, metavar=("ALPHA", "LOG10T"))
    g.add_argument("--alpha", type=float, nargs=2, required=True)
    g.add_argument("--log-temperature", type=float, nargs=2, required=True)
    g.add_argument("--resolution", type=int, nargs=2, required=True, help="rows (temperature) and columns (alpha)")
    g.add_argument("--model", default="qwen0.5b")
    g.add_argument("--layer", type=int, default=12)
    g.add_argument("--threads", type=int, default=8)
    g.add_argument("--device", default="cpu")
    g.add_argument("--prompt", default="open")
    g.add_argument("--opener", default=None)
    g.add_argument("--window", type=int, default=1)
    g.add_argument("--vectors", default="llm-phase-map/vectors-l12-p1b")
    g.add_argument("--concept", default="hostility-vs-kindness")
    g.add_argument("--transient", type=int, default=900)
    g.add_argument("--k", type=int, default=6)
    g.add_argument("--tol", type=float, default=1e-12)
    g.add_argument("--chunk", type=int, default=512)
    g.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.command == "grid":
        grid(args)


if __name__ == "__main__":
    main()
