"""Attractor portraits of the soft-token map (human, 18 Oct 2026: "where is my strange attractor? can't we reconstruct
one from the data?").

At one cell (alpha, log10 T) of a steered plane, many trajectories start from the fixed start displaced by --spread
|x| along seeded random directions. After --transient steps every trajectory is recorded for --record steps, in
float64. Their states sample the attractors the starts reach.
- Pass 1 records --pca-starts of the trajectories and fixes the principal axes of their states (the attractor's own
  coordinates).
- Pass 2 runs all --starts and stores each recorded state's coordinates on the leading --pcs axes, the most probable
  next token, and the full states of the first --keep-full trajectories (for dimension estimates).
- Each start's final state is classified against the periodic orbits found among the trajectories (period up to
  --max-period by full-state recurrence, tolerance --tol), so coexisting attractors are told apart and counted.

Output: DATA/explore/OUT/portrait.npz (coords, tokens, periods, orbit ids, basis, mean, kept states) and
portrait.json (cell, settings, attractor census).

Usage: python tools/attractor_portrait.py --alpha 0.0653 --log-t -0.148 --device cuda --out llm-soft-loop/portraits/x
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import explore_llm_attractors as ela  # noqa: E402
import explore_llm_phase_map as elpm  # noqa: E402
import explore_llm_soft_loop as esl  # noqa: E402


def run_batch(loop, steer_row, T, n, spread, seed, transient, record, keep_states, torch, project=None, pcs=0):
    """n trajectories from the displaced fixed start; returns recorded coordinates (record, n, pcs) or states, tokens
    (record, n) and the last max_period+1 states for the period test."""
    X = loop.prepare(np.repeat(steer_row[None], n, axis=0))
    g = torch.Generator(device=X.device).manual_seed(seed)
    U = torch.randn(X[:, -1].shape, generator=g, dtype=X.dtype, device=X.device)
    U = U / U.norm(dim=1, keepdim=True)
    X = X.clone()
    X[:, -1] = X[:, -1] + spread * X[:, -1].norm(dim=1, keepdim=True) * U
    temps = torch.full((n,), T, dtype=loop.state_dtype, device=loop.device)
    with torch.no_grad():
        for _ in range(transient):
            X, _ = loop.step(X, temps)
        coords, toks, states = [], [], []
        for _ in range(record):
            X, p = loop.step(X, temps)
            x = X[:, -1].double()
            toks.append(p.argmax(-1).cpu().numpy())
            if project is not None:
                mean, basis = project
                coords.append(((x - mean) @ basis[:, :pcs]).cpu().numpy())
            if keep_states:
                states.append(x.cpu().numpy())
    return coords, toks, states, X


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    ap.add_argument("--concept", default="philosophy")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--log-t", type=float, required=True)
    ap.add_argument("--starts", type=int, default=512)
    ap.add_argument("--pca-starts", type=int, default=64)
    ap.add_argument("--spread", type=float, default=0.05)
    ap.add_argument("--transient", type=int, default=400)
    ap.add_argument("--record", type=int, default=300)
    ap.add_argument("--pcs", type=int, default=16)
    ap.add_argument("--keep-full", type=int, default=16)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-period", type=int, default=64)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--dtype", default="float64", choices=("float64", "float32"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    tic = time.perf_counter()
    loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, args.dtype, args.device)
    torch = loop.torch
    v = np.asarray(elpm.load_vectors(args.vectors)[args.concept], dtype=np.float64)
    steer = args.alpha * v
    T = 10.0 ** args.log_t
    # pass 1: principal axes of the attractor states
    _, _, st, _ = run_batch(loop, steer, T, args.pca_starts, args.spread, 1, args.transient, args.record, True, torch)
    S = np.concatenate(st)                                       # (record * pca_starts, d)
    mean = S.mean(0)
    C = np.cov((S - mean).T)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    meant = torch.as_tensor(mean, dtype=torch.float64, device=loop.device)
    basist = torch.as_tensor(V[:, :args.pcs].copy(), dtype=torch.float64, device=loop.device)
    # pass 2: all starts, projected
    coords, toks, kept, finals = [], [], [], []
    for b0 in range(0, args.starts, args.batch):
        n = min(args.batch, args.starts - b0)
        keep = max(0, min(n, args.keep_full - b0))
        c, t, s, X = run_batch(loop, steer, T, n, args.spread, 100 + b0, args.transient, args.record, keep > 0, torch,
                               (meant, basist), args.pcs)
        coords.append(np.stack(c, axis=1))                     # (n, record, pcs)
        toks.append(np.stack(t, axis=1))
        if keep:
            kept.append(np.stack(s, axis=1)[:keep])
        # period of each final state by full-state recurrence over max_period further steps
        with torch.no_grad():
            Y = [X[:, -1].double().clone()]
            temps = torch.full((n,), T, dtype=loop.state_dtype, device=loop.device)
            Z = X
            for _ in range(args.max_period):
                Z, _ = loop.step(Z, temps)
                Y.append(Z[:, -1].double().clone())
            Y = torch.stack(Y)                                  # (max_period+1, n, d)
            per = np.zeros(n, dtype=int)
            for q in range(1, args.max_period + 1):
                dq = ((Y[q] - Y[0]).norm(dim=1) / Y[0].norm(dim=1)).cpu().numpy()
                per[(per == 0) & (dq < args.tol)] = q
            finals.append((per, Y[0].cpu().numpy()))
        print(json.dumps({"done": b0 + n, "of": args.starts, "seconds": round(time.perf_counter() - tic, 1)}),
              flush=True)
    coords = np.concatenate(coords)
    toks = np.concatenate(toks)
    periods = np.concatenate([f[0] for f in finals])
    ends = np.concatenate([f[1] for f in finals])
    # census: periodic attractors told apart by the closest pair of points of their orbits (same orbit, any phase)
    orbit_id = np.full(len(periods), -1)
    reps = []
    for i in np.nonzero(periods > 0)[0]:
        for k, (q, x) in enumerate(reps):
            if q == periods[i] and np.min(np.linalg.norm(coords[i, -q:, :] - coords[x, -q:, :][:, None], axis=-1)) \
                    < 1e-6 * max(1.0, np.linalg.norm(coords[x, -1])):
                orbit_id[i] = k
                break
        else:
            reps.append((periods[i], i))
            orbit_id[i] = len(reps) - 1
    census = {"aperiodic": int(np.sum(periods == 0)),
              "orbits": [{"period": int(q), "starts": int(np.sum(orbit_id == k)), "example_start": int(x)}
                         for k, (q, x) in enumerate(reps)]}
    np.savez_compressed(out / "portrait.npz", coords=coords, tokens=toks, periods=periods, orbit_id=orbit_id,
                        basis=V[:, :args.pcs], mean=mean, eigenvalues=w[:64],
                        kept_states=np.concatenate(kept) if kept else np.zeros((0,)), final_states=ends)
    rec = {"args": vars(args), "census": census, "variance_explained_top": (w[:10] / w.sum()).tolist(),
           "seconds": round(time.perf_counter() - tic, 1)}
    (out / "portrait.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
