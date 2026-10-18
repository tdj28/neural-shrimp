"""Run FA (exploratory; human, 18 Oct 2026: show the real shrimp): the domain of one attractor family in a two-block
plane, found by following the attractor outward from a seed cell, cell by cell.

The fixed-start maps (tools/two_block_map.py) start every cell from the same text, so where two attractors coexist
they show whichever the fixed start reaches: in the legs of the period-5 window of the gratitude@4 / hostility@16
plane the fixed start settles on a period-4 attractor, and the legs do not show. Here every cell starts instead from
the final state of an already computed neighbour that is still on the followed attractor (attractor continuation by
iteration, as in a hysteresis scan), so the attractor is kept wherever it exists and is reachable through the grid.

1. Seed: the fixed start at --seed, iterated --seed-transient steps, must reach a period --seed-period orbit (full-state
   recurrence); its leading tokens over one period give the reference word.
2. Waves: every unvisited cell next to (8-neighbourhood) a member cell is queued, with the state of its most stable
   member neighbour (smallest exponent). A wave iterates its cells --transient steps from those states, then
   --record steps with one twin (offset --eps, renormalized every step: the exponent) and records the leading tokens.
   Period: smallest q <= --max-period with |x_t - x_(t-q)| and |x_t - x_(t-2q)| below --tol * max(1, |x_t|) at the
   last step (0: none found).
3. Membership: the dominant token of each of the --seed-period phases over the last --token-steps steps forms the
   cell's word, with its agreement (the share of steps of that phase carrying the dominant token). A cell belongs to
   the family if its word matches the seed's reference word (best rotation) in at least --min-match phases, and
   - its period is a multiple of --seed-period (label 1, propagates), or
   - no period was found but its tokens repeat exactly (agreement 1 in every phase) and its exponent is below
     --max-slow-exponent: a slowly converging orbit next to a fold or flip (label 3, propagates), or
   - no period was found, its tokens repeat exactly and its exponent is larger: chaos behind a repeating text, the
     chaotic bands of the cascade (label 2, a member that does not propagate).
   Anything else is not a member (label 0). The reference is global, so the rule cannot drift through intermittent
   chaos (FA-100a, the first attempt, used the parent's word and a majority vote and leaked across the fold into the
   chaotic sea and into a period-4 attractor). A non-member is retried once (--max-attempts) if another propagating
   member neighbour appears later.

Labels: -1 not computed, 0 not a member, 1 periodic member, 2 chaotic member with an exactly repeating text,
3 member without a period found and without a positive exponent.

Output: DATA/explore/OUT/follow.npz (grids, labels, exponents, periods, words, agreement, parents, attempts, waves),
states.npz (final states of members, float32), summary.json, progress.jsonl.

Usage: python tools/follow_attractor.py --log-temperature -0.1 --seed -0.118730 0.100693 \
         --alpha1 -0.1320 -0.1040 --alpha2 0.0860 0.1225 --step 1.25e-4 --device cuda --out llm-soft-loop/pod64/fa/t-0.100
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

NEIGHBOURS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def dominant_word(toks, p):
    """Most frequent token of each phase k (positions k, k + p, ...) of toks (steps, n); and its frequency."""
    steps, n = toks.shape
    words = np.zeros((n, p), dtype=np.int64)
    agree = np.zeros((n, p))
    for k in range(p):
        col = toks[k::p]
        for c in range(n):
            vals, counts = np.unique(col[:, c], return_counts=True)
            m = int(np.argmax(counts))
            words[c, k], agree[c, k] = vals[m], counts[m] / len(col)
    return words, agree


def best_match(word, ref):
    """Largest number of equal phases between word and any rotation of ref."""
    p = len(ref)
    return max(int(np.sum(word == np.roll(ref, s))) for s in range(p))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--block1", type=int, default=4)
    ap.add_argument("--vectors1", default="llm-phase-map/vectors-l4")
    ap.add_argument("--concept1", default="gratitude")
    ap.add_argument("--block2", type=int, default=16)
    ap.add_argument("--vectors2", default="llm-phase-map/vectors-l16")
    ap.add_argument("--concept2", default="hostility")
    ap.add_argument("--log-temperature", type=float, required=True)
    ap.add_argument("--alpha1", type=float, nargs=2, required=True)
    ap.add_argument("--alpha2", type=float, nargs=2, required=True)
    ap.add_argument("--step", type=float, required=True, help="grid spacing in both alphas")
    ap.add_argument("--seed", type=float, nargs=2, required=True, metavar=("A1", "A2"))
    ap.add_argument("--seed-period", type=int, default=5)
    ap.add_argument("--seed-transient", type=int, default=900)
    ap.add_argument("--transient", type=int, default=250)
    ap.add_argument("--record", type=int, default=300)
    ap.add_argument("--token-steps", type=int, default=200)
    ap.add_argument("--max-slow-exponent", type=float, default=0.005)
    ap.add_argument("--max-period", type=int, default=64)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--min-match", type=int, default=4)
    ap.add_argument("--max-attempts", type=int, default=2)
    ap.add_argument("--max-batch", type=int, default=1536)
    ap.add_argument("--max-minutes", type=float, default=1e9)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ring = 2 * args.max_period + 1
    if args.record < max(ring, args.token_steps):
        raise SystemExit("--record must cover 2 * max_period + 1 steps and --token-steps")

    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "progress.jsonl", "a")

    def emit(rec):
        print(json.dumps(rec), flush=True)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

    tic = time.perf_counter()
    loop = esl.SoftLoop(args.model, args.block1, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    loop.sm.add_steer_layer(args.block2)
    torch, dev = loop.torch, loop.device
    u = np.asarray(elpm.load_vectors(args.vectors1)[args.concept1], dtype=np.float64)
    w = np.asarray(elpm.load_vectors(args.vectors2)[args.concept2], dtype=np.float64)
    T = 10.0 ** args.log_temperature
    a1 = np.arange(args.alpha1[0], args.alpha1[1] + args.step / 2, args.step)
    a2 = np.arange(args.alpha2[0], args.alpha2[1] + args.step / 2, args.step)
    n2, n1 = len(a2), len(a1)
    rng = np.random.default_rng(0)

    def run(cells, X0, transient, record):
        """Iterate cells [(i, j)] from states X0 (n, K, d); returns final states, exponents, periods, tokens."""
        n = len(cells)
        steer = np.array([a1[j] * u for i, j in cells])
        loop.extra = (args.block2, np.array([a2[i] * w for i, j in cells]))
        loop.prepare(np.concatenate([steer, steer]))
        temps = np.full(2 * n, T)
        with torch.no_grad():
            X = torch.cat([X0, X0])
            for _ in range(transient):
                X, _ = loop.step(X, temps)
            base = X[:n]
            d = torch.as_tensor(rng.normal(size=tuple(base.shape)), dtype=loop.state_dtype, device=dev)
            X = torch.cat([base, base + d / d.flatten(1).norm(dim=1)[:, None, None] * args.eps])
            growth = np.zeros(n)
            buf = torch.zeros((ring, n, X.shape[2]), dtype=loop.state_dtype, device=dev)
            toks = np.zeros((record, n), dtype=np.int64)
            for t in range(record):
                X, p = loop.step(X, temps)
                base, tw = X[:n], X[n:]
                diff = tw - base
                dist = np.maximum(diff.flatten(1).norm(dim=1).double().cpu().numpy(), args.eps * 1e-30)
                growth += np.log(dist / args.eps)
                tw = base + diff * torch.as_tensor(args.eps / dist, dtype=loop.state_dtype, device=dev)[:, None, None]
                X = torch.cat([base, tw])
                buf[t % ring] = base[:, -1, :]
                toks[t] = p[:n].argmax(-1).cpu().numpy()
            t = record - 1
            cur = buf[t % ring]
            scale = torch.clamp(cur.norm(dim=1), min=1.0) * args.tol
            found = torch.zeros(n, dtype=torch.long, device=dev)
            for q in range(args.max_period, 0, -1):              # descending, so the smallest period wins
                ok = ((buf[(t - q) % ring] - cur).norm(dim=1) < scale) & ((buf[(t - 2 * q) % ring] - cur).norm(dim=1)
                                                                           < scale)
                found = torch.where(ok, torch.full_like(found, q), found)
        loop.extra = None
        return base.clone(), growth / record, found.cpu().numpy(), toks

    label = np.full((n2, n1), -1, dtype=np.int8)
    lam = np.full((n2, n1), np.nan)
    period = np.zeros((n2, n1), dtype=np.int32)
    words = np.zeros((n2, n1, args.seed_period), dtype=np.int64)
    agree = np.zeros((n2, n1, args.seed_period))
    parent = np.full((n2, n1, 2), -1, dtype=np.int32)
    attempts = np.zeros((n2, n1), dtype=np.int8)
    wave_of = np.full((n2, n1), -1, dtype=np.int32)
    states = {}

    # 1. seed from the fixed start
    i0, j0 = int(np.argmin(np.abs(a2 - args.seed[1]))), int(np.argmin(np.abs(a1 - args.seed[0])))
    loop.extra = (args.block2, np.array([a2[i0] * w]))
    X0 = loop.prepare(np.array([a1[j0] * u]))
    loop.extra = None
    Xs, ls, ps, tk = run([(i0, j0)], X0, args.seed_transient, args.record)
    if ps[0] != args.seed_period:
        raise SystemExit(f"the fixed start at the seed cell ({a1[j0]}, {a2[i0]}) has period {ps[0]}, "
                         f"not {args.seed_period}")
    wd, ag = dominant_word(tk[-args.token_steps:], args.seed_period)
    ref = wd[0]
    decode = loop.sm.tok.decode
    label[i0, j0], lam[i0, j0], period[i0, j0], words[i0, j0], agree[i0, j0] = 1, ls[0], ps[0], ref, ag[0]
    wave_of[i0, j0] = 0
    states[(i0, j0)] = Xs[0:1]
    emit({"seed": [float(a1[j0]), float(a2[i0])], "period": int(ps[0]), "exponent": float(ls[0]),
          "word": ref.tolist(), "text": [decode([int(t)]) for t in ref], "grid": [n2, n1],
          "seconds": round(time.perf_counter() - tic, 1)})

    def frontier():
        """Unvisited (or retryable) cells next to members, each with its most stable member neighbour."""
        cand = {}
        for (i, j) in states:
            for di, dj in NEIGHBOURS:
                ii, jj = i + di, j + dj
                if not (0 <= ii < n2 and 0 <= jj < n1) or label[ii, jj] >= 1:
                    continue
                if label[ii, jj] == 0 and (attempts[ii, jj] >= args.max_attempts      # a lost cell is retried
                                           or tuple(parent[ii, jj]) == (i, j)):    # only from another member
                    continue
                best = cand.get((ii, jj))
                if best is None or lam[i, j] < lam[best]:
                    cand[(ii, jj)] = (i, j)
        return cand

    wave = 0
    while True:
        cand = frontier()
        if not cand:
            emit({"stop": "no frontier", "wave": wave})
            break
        if (time.perf_counter() - tic) / 60 > args.max_minutes:
            emit({"stop": "time limit", "wave": wave, "frontier": len(cand)})
            break
        wave += 1
        items = sorted(cand.items())[:args.max_batch]
        cells = [c for c, _ in items]
        X0 = torch.cat([states[par] for _, par in items])
        Xf, lw, pw, tk = run(cells, X0, args.transient, args.record)
        wd, ag = dominant_word(tk[-args.token_steps:], args.seed_period)
        new = 0
        for k, ((i, j), (pi, pj)) in enumerate(items):
            attempts[i, j] += 1
            parent[i, j] = (pi, pj)
            wave_of[i, j] = wave
            lam[i, j], period[i, j], words[i, j], agree[i, j] = lw[k], pw[k], wd[k], ag[k]
            exact = bool(np.all(ag[k] >= 1.0))
            if best_match(wd[k], ref) < args.min_match:
                label[i, j] = 0
            elif pw[k] > 0 and pw[k] % args.seed_period == 0:
                label[i, j] = 1
            elif pw[k] == 0 and exact and lw[k] < args.max_slow_exponent:
                label[i, j] = 3
            elif pw[k] == 0 and exact:
                label[i, j] = 2
            else:
                label[i, j] = 0
            if label[i, j] in (1, 3):
                states[(i, j)] = Xf[k:k + 1]
                new += 1
        emit({"wave": wave, "cells": len(cells), "propagating": new, "total_propagating": len(states),
              "computed": int((label >= 0).sum()), "seconds": round(time.perf_counter() - tic, 1)})
        if wave % 20 == 0:
            np.savez_compressed(out / "follow.npz", alpha1=a1, alpha2=a2, log10_temperature=args.log_temperature,
                                label=label, lyapunov=lam, period=period, words=words, agreement=agree,
                                parent=parent, attempts=attempts, wave=wave_of, reference_word=ref)

    np.savez_compressed(out / "follow.npz", alpha1=a1, alpha2=a2, log10_temperature=args.log_temperature, label=label,
                        lyapunov=lam, period=period, words=words, agreement=agree, parent=parent, attempts=attempts,
                        wave=wave_of, reference_word=ref)
    keys = sorted(states)
    np.savez_compressed(out / "states.npz", index=np.array(keys, dtype=np.int32),
                        states=torch.cat([states[k] for k in keys]).float().cpu().numpy())
    per = {int(q): int(((label == 1) & (period == q)).sum()) for q in np.unique(period[label == 1])}
    summary = {"args": vars(args), "grid": [n2, n1], "waves": wave, "computed": int((label >= 0).sum()),
               "members": int((label >= 1).sum()), "members_periodic": int((label == 1).sum()),
               "members_chaotic_repeating_text": int((label == 2).sum()),
               "members_unresolved_slow": int((label == 3).sum()), "not_members": int((label == 0).sum()),
               "member_periods": per, "reference_word": ref.tolist(),
               "reference_text": [decode([int(t)]) for t in ref], "seconds": round(time.perf_counter() - tic, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    emit({k: summary[k] for k in ("waves", "computed", "members", "not_members", "member_periods", "seconds")})


if __name__ == "__main__":
    main()
