"""Hidden chaos: chaotic soft-token dynamics behind a fixed or periodic leading token.

The leading token (argmax of the next-token distribution) partitions the state space. When a chaotic attractor lies
inside one cell of that partition, the greedy readout repeats one token while the blend moves chaotically: positive
largest Lyapunov exponent, zero symbolic information. This tool measures, per cell:
- the largest exponent from renormalised twins at two offsets (to check it is not a rounding effect),
- the state period (state return distance below TOL times the state norm) and the attractor's diameter relative to
  the state norm, with the band count for 2^k bands (period of the coarse return pattern),
- the leading-token sequence: number of distinct tokens and its smallest period (symbolic period),
- the probability of the leading token and the entropy of the distribution (mean and spread).
A cell is "hidden chaos" when the exponent is positive at both offsets, there is no state period, and the leading-token
sequence is periodic (symbolic period >= 1).

cell   one cell with a long record (--trace also saves its per-step leading tokens, probabilities, entropies and
       states, trace.npz).
line   every cell of a 1D line (log10 T range at fixed alpha), batched.
cells  a list of cells from a JSON file {"cells": [[log10T, alpha], ...]} (horizon checks of map labels), batched.

Usage: python tools/hidden_chaos.py cell --vectors VEC --concept C --log-temperature LT --alpha A --out OUT [--trace]
       python tools/hidden_chaos.py line --vectors VEC --concept C --alpha A --log-temperature LO HI --n N --out OUT
       python tools/hidden_chaos.py cells --vectors VEC --concept C --cells-file FILE --out OUT
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def symbolic_period(seq, max_q=64):
    seq = list(seq)
    for q in range(1, max_q + 1):
        if len(seq) > 2 * q and all(seq[k] == seq[k - q] for k in range(q, len(seq))):
            return q
    return 0


KEEP = 400          # states kept per cell: the diameter uses the last 400, the period check the last 65


def run_batch(loop, v, log_ts, alphas, transient, record, eps_list, tol, seed=0, trace=False):
    """Trajectories for a batch of cells with one twin per offset; returns per-cell statistics (and, with trace,
    the per-step arrays of the first cell)."""
    torch = loop.torch
    n = len(log_ts)
    m = len(eps_list)
    steer = np.array([a * v for a in alphas])
    rows = np.concatenate([steer] * (m + 1))
    temps = np.tile(10.0 ** np.asarray(log_ts), m + 1)
    X = loop.prepare(rows)
    for _ in range(transient):
        X, _ = loop.step(X, temps)
    rng = np.random.default_rng(seed)
    base = X[:n]
    twins = []
    for e in eps_list:
        d = torch.as_tensor(rng.normal(size=tuple(base.shape)), dtype=loop.state_dtype, device=loop.device)
        twins.append(base + d / d.flatten(1).norm(dim=1)[:, None, None] * e)
    X = torch.cat([base] + twins)
    growth = np.zeros((m, n))
    lead, pmax, ent, states, first = [], [], [], [], []
    for _ in range(record):
        X, p = loop.step(X, temps)
        base = X[:n]
        new = [base]
        for k, e in enumerate(eps_list):
            Y = X[n * (k + 1):n * (k + 2)]
            diff = Y - base
            dist = np.maximum(diff.flatten(1).norm(dim=1).double().cpu().numpy(), e * 1e-30)
            growth[k] += np.log(dist / e)
            new.append(base + diff * torch.as_tensor(e / dist, dtype=loop.state_dtype, device=loop.device)[:, None, None])
        X = torch.cat(new)
        p0 = p[:n].double()
        top = p0.max(dim=-1)
        lead.append(top.indices.cpu().numpy())
        pmax.append(top.values.cpu().numpy())
        ent.append((-(p0 * torch.log(torch.clamp(p0, min=1e-300))).sum(-1)).cpu().numpy())
        states.append(base[:, -1].double().cpu().numpy())
        if len(states) > KEEP:
            states.pop(0)
        if trace:
            first.append(states[-1][0].copy())
    lead, pmax, ent, S = np.array(lead), np.array(pmax), np.array(ent), np.array(states)   # (record|KEEP, n[, d])
    out = []
    for c in range(n):
        s = S[:, c]
        scale = max(1.0, float(np.linalg.norm(s[-1])))
        q = 0
        for qq in range(1, 65):
            if np.linalg.norm(s[-1 - qq] - s[-1]) < tol * scale:
                q = qq
                break
        tail = s[-min(record, KEEP):]
        diam = float(np.max(np.linalg.norm(tail - tail.mean(0), axis=1)) * 2 / scale)
        seq = lead[:, c].tolist()
        out.append({"log10_temperature": float(log_ts[c]), "alpha": float(alphas[c]),
                    "lyapunov": [float(g[c] / record) for g in growth], "eps": list(eps_list), "period": q,
                    "diameter_rel": diam, "lead_distinct": len(set(seq)), "lead_period": symbolic_period(seq),
                    "lead_tokens": [int(t) for t in sorted(set(seq), key=seq.index)[:8]],
                    "pmax_mean": float(pmax[:, c].mean()), "pmax_sd": float(pmax[:, c].std()),
                    "entropy_mean": float(ent[:, c].mean()), "entropy_sd": float(ent[:, c].std())})
    if trace:
        return out, {"lead": lead[:, 0], "pmax": pmax[:, 0], "entropy": ent[:, 0], "states": np.array(first)}
    return out


def classify(r):
    pos = all(l > 1e-3 for l in r["lyapunov"])
    if r["period"] > 0:
        return "periodic"
    if pos and r["lead_period"] >= 1:
        return "hidden chaos"
    if pos:
        return "visible chaos"
    return "irregular, exponent not positive at both offsets"


def main():
    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("cell", "line", "cells"))
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vectors", required=True)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--log-temperature", type=float, nargs="+")
    ap.add_argument("--cells-file", help="cells mode: JSON file with a list 'cells' of [log10T, alpha] pairs")
    ap.add_argument("--trace", action="store_true", help="cell mode: save the per-step arrays (trace.npz)")
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--transient", type=int, default=1000)
    ap.add_argument("--record", type=int, default=2000)
    ap.add_argument("--eps", type=float, nargs="+", default=[1e-6, 1e-9])
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, args.dtype, args.device)
    v = elpm.load_vectors(args.vectors)[args.concept]
    if args.mode == "cells":
        if not args.cells_file:
            raise SystemExit("cells mode needs --cells-file")
        pairs = np.array(json.loads(Path(args.cells_file).read_text())["cells"], dtype=float)
        log_ts, alphas = pairs[:, 0], pairs[:, 1]
    else:
        if args.alpha is None or not args.log_temperature:
            raise SystemExit(f"{args.mode} mode needs --alpha and --log-temperature")
        if args.mode == "cell":
            log_ts = np.array([args.log_temperature[0]])
        else:
            log_ts = np.linspace(args.log_temperature[0], args.log_temperature[1], args.n)
        alphas = np.full(len(log_ts), args.alpha)
    results, tr = [], None
    for s in range(0, len(log_ts), args.chunk):
        got = run_batch(loop, v, log_ts[s:s + args.chunk], alphas[s:s + args.chunk], args.transient, args.record,
                        args.eps, args.tol, trace=args.trace and args.mode == "cell")
        if isinstance(got, tuple):
            got, tr = got
        results += got
        print(json.dumps({"done": len(results), "of": len(log_ts)}), flush=True)
    for r in results:
        r["class"] = classify(r)
    tok = loop.sm.tok
    for r in results:
        r["lead_text"] = [tok.decode([t]) for t in r["lead_tokens"]]
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "hidden-chaos.json").write_text(json.dumps({"args": vars(args), "cells": results}, indent=1))
    if tr is not None:
        np.savez_compressed(out / "trace.npz", **tr)
    counts = {}
    for r in results:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    print(json.dumps(counts))
    if args.mode == "cell":
        print(json.dumps(results[0], indent=1)[:1500])


if __name__ == "__main__":
    main()
