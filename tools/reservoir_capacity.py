"""Memory capacity of the soft-token loop used as a reservoir, across its regimes (human, 15 Oct 2026: "since the
broader edge-of-chaos question remains untested <- how do we test?"; the classical test of Bertschinger and
Natschlaeger 2004 and Jaeger 2001, applied to the language model's own map).

A weak scalar input u_n ~ U(-1, 1), independent per cell and step, enters the map before the network:
x_(n+1) = F(x_n + eps |x| u_n r), with r a fixed unit direction (seeded) and |x| the cell's state norm after an
input-free settling run. Linear readouts of the recorded state x_(n+1) (centred; ridge regularisation chosen on a
validation block, one value for all delays) reconstruct u_(n-k), k = 0..K. The memory capacity MC is the sum over k of
the held-out R^2 (negative values counted as 0) on a final test block. Readout blocks are contiguous: train 60 %,
validation 20 %, test 20 % of the record.

Cells: a cells file (list 'cells' of [log10T, alpha], optional 'label' and 'map_exponent' lists), or a line
(--line LOG10T ALPHA_LO ALPHA_HI N). With --exponents, each cell's input-free twin exponents and label are also
computed (tools/hidden_chaos.run_batch and classify; transient 1000, record 2000, offsets 1e-6 and 1e-9).

Usage:
  python tools/reservoir_capacity.py --line -0.14798 0.0640 0.0670 151 --exponents --out llm-soft-loop/pod64/rc/feig
  python tools/reservoir_capacity.py --cells-file docs/h2-horizon-cells.json --out llm-soft-loop/pod64/rc/h2
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))


def inject_run(loop, v, log_ts, alphas, eps_rel, settle, record, seed):
    """States (record, n, d) in float32 of the loop driven by weak inputs, and the inputs (record, n)."""
    torch = loop.torch
    n = len(log_ts)
    temps = 10.0 ** np.asarray(log_ts, dtype=float)
    X = loop.prepare(np.array([a * v for a in alphas]))
    for _ in range(settle):
        X, _ = loop.step(X, temps)
    norm = X[:, -1].norm(dim=1)
    g = np.random.default_rng(seed)
    r = torch.as_tensor(g.normal(size=X.shape[-1]), dtype=X.dtype, device=X.device)
    r = r / r.norm()
    U = g.uniform(-1.0, 1.0, size=(record, n))
    S = np.zeros((record, n, X.shape[-1]), dtype=np.float32)
    for t in range(record):
        kick = (eps_rel * norm * torch.as_tensor(U[t], dtype=X.dtype, device=X.device))[:, None] * r[None, :]
        Xin = X.clone()
        Xin[:, -1] = Xin[:, -1] + kick
        X, _ = loop.step(Xin, temps)
        S[t] = X[:, -1].float().cpu().numpy()
    return S, U


def r2(y, yhat):
    ss = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum((y - yhat) ** 2)) / ss if ss > 0 else float("nan")


def memory_capacity(S, u, K, betas=(1e-10, 1e-8, 1e-6, 1e-4, 1e-2)):
    """MC and the held-out R^2 per delay k = 0..K from states S (N, d) and inputs u (N,), state t driven by u_t."""
    N = len(u)
    rows = np.arange(K, N)
    X = S[rows].astype(np.float64)
    Y = np.stack([u[rows - k] for k in range(K + 1)], axis=1)
    n_tr, n_va = int(0.6 * len(rows)), int(0.2 * len(rows))
    tr, va, te = slice(0, n_tr), slice(n_tr, n_tr + n_va), slice(n_tr + n_va, len(rows))
    mx, my = X[tr].mean(0), Y[tr].mean(0)
    Xc, Yc = X - mx, Y - my
    G = Xc[tr].T @ Xc[tr]
    B = Xc[tr].T @ Yc[tr]
    scale = np.trace(G) / G.shape[0]
    best = None
    for b in betas:
        W = np.linalg.solve(G + b * scale * np.eye(G.shape[0]), B)
        score = np.nanmean([r2(Yc[va, k], Xc[va] @ W[:, k]) for k in range(K + 1)])
        if best is None or score > best[0]:
            best = (score, b, W)
    _, b, W = best
    test = [r2(Yc[te, k], Xc[te] @ W[:, k]) for k in range(K + 1)]
    return float(np.nansum(np.clip(test, 0, None))), test, b


def main():
    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl
    import hidden_chaos as hc

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    ap.add_argument("--concept", default="philosophy")
    ap.add_argument("--cells-file", default=None)
    ap.add_argument("--line", type=float, nargs=4, default=None, metavar=("LOG10T", "ALPHA_LO", "ALPHA_HI", "N"))
    ap.add_argument("--eps", type=float, nargs="+", default=[1e-4, 1e-3], help="input sizes relative to |x|")
    ap.add_argument("--settle", type=int, default=1000)
    ap.add_argument("--record", type=int, default=3000)
    ap.add_argument("--delays", type=int, default=40)
    ap.add_argument("--exponents", action="store_true")
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float64")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    meta = {}
    if args.cells_file:
        spec = json.loads(Path(args.cells_file).read_text())
        cells = np.array(spec["cells"], dtype=float)
        meta = {k: spec[k] for k in ("label", "map_exponent", "map_period", "source") if k in spec}
    elif args.line:
        lt, lo, hi, n = args.line
        cells = np.array([[lt, a] for a in np.linspace(lo, hi, int(n))])
    else:
        raise SystemExit("give --cells-file or --line")
    loop = esl.SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, args.dtype, args.device)
    v = np.asarray(elpm.load_vectors(args.vectors)[args.concept], dtype=np.float64)
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = [{"log10_temperature": float(c[0]), "alpha": float(c[1]),
             **{k: meta[k][i] for k in meta}} for i, c in enumerate(cells)]
    tic = time.perf_counter()
    if args.exponents:
        for s in range(0, len(cells), args.chunk):
            c = cells[s:s + args.chunk]
            got = hc.run_batch(loop, v, c[:, 0], c[:, 1], 1000, 2000, [1e-6, 1e-9], 1e-8)
            for r, g in zip(rows[s:s + args.chunk], got):
                r["lyapunov"], r["period"], r["class"] = g["lyapunov"], g["period"], hc.classify(g)
        print(json.dumps({"exponents": len(rows), "seconds": round(time.perf_counter() - tic, 1)}), flush=True)
    for eps in args.eps:
        for s in range(0, len(cells), args.chunk):
            c = cells[s:s + args.chunk]
            S, U = inject_run(loop, v, c[:, 0], c[:, 1], eps, args.settle, args.record, args.seed + s)
            for j, r in enumerate(rows[s:s + args.chunk]):
                mc, curve, beta = memory_capacity(S[:, j], U[:, j], args.delays)
                r.setdefault("capacity", {})[str(eps)] = {"mc": mc, "r2": curve, "ridge": beta,
                                                          "state_spread": float(S[:, j].std(0).sum())}
            print(json.dumps({"eps": eps, "done": min(s + args.chunk, len(cells)), "of": len(cells),
                              "seconds": round(time.perf_counter() - tic, 1)}), flush=True)
            (out / "capacity.json").write_text(json.dumps({"args": vars(args), "cells": rows}, indent=1))
    (out / "capacity.json").write_text(json.dumps({"args": vars(args), "cells": rows}, indent=1))


if __name__ == "__main__":
    main()
