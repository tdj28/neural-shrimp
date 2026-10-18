"""Run BH (exploratory; human 18 Oct 2026: "i want us to have a theory of bounds and show that they apply to what we
already have"): the hierarchy of upper bounds on the largest Lyapunov exponent of the soft-token map, with exact norms.

Along an orbit x_0, x_1, ... of F_{T,alpha}, let P_{i,k} = DF(x_{i+k-1}) ... DF(x_i). Submultiplicativity of the
operator norm gives, for the N recorded steps and block lengths k that divide N,

    Lambda_N = (1/N) log ||P_{0,N}||  <=  B_k = (1/N) sum_j log ||P_{jk,k}||  <=  B_{k/2}  <= ... <=  B_1,

    B_1 = (1/N) sum_i log ||DF(x_i)||  <=  (1/N) sum_i [ log(||Sigma_p(x_i)|| / T) + log ||J_h(x_i)|| ],

with dyadic k (each block of length 2k is the product of two blocks of length k). As N grows, Lambda_N tends to the
largest exponent lambda along almost every orbit of an ergodic invariant measure, and by Kingman's subadditive ergodic
theorem the block bounds close onto lambda as k grows. Every norm here is exact up to float64 rounding: DF(x_i) is
formed densely by forward-mode AD (896 Jacobian-vector products per step), Sigma_p as E^T diag(p) E - F F^T, J_h (the
Jacobian of the final normalized hidden state) densely at every --j-stride-th step, and all norms by SVD. The dense
J_h also checks the factorization DF = Sigma_p J_h / T (tied embeddings) at those steps.

Per cell (a cells file as docs/bh/bh-cells.json: [log10 T, alpha] at the philosophy direction, layer 12):
--transient steps from the fixed start, as the maps; --record N steps with dense DF; then --tangent further steps of
one renormalized tangent vector, the long-run reference exponent lambda_long. Reported per cell: B_k for the dyadic k
up to N, the overlapping-window averages (1/k) mean_i log ||P_{i,k}|| for every k <= N, Lambda_N, the readout and
network terms, the per-step gap log(||Sigma_p|| ||J_h|| / T) - log ||DF|| (never negative), the factorization error,
lambda_long and the map's label and exponent. Cells run in the file's order until --budget-minutes is spent; a cell
not started is listed as such.

Output: DATA/explore/OUT/bh.json and progress.jsonl.

Usage: python tools/bound_hierarchy.py --cells-file docs/bh/bh-cells.json --device cuda --out llm-soft-loop/pod64/bh/a
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import explore_llm_attractors as ela  # noqa: E402
import explore_llm_phase_map as elpm  # noqa: E402
from explore_llm_soft_loop import DATA, SoftLoop, SoftOrbit  # noqa: E402


def h_jacobian(orb, X, chunk):
    """Dense Jacobian of the final normalized hidden state h(x) at X (1, 1, d), by forward-mode AD on a shallow copy
    of the repeated prefix cache (as SoftOrbit.jvp_rows does for the whole step)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch, loop = orb.torch, orb.loop
    n = orb.n
    eye = torch.eye(n, dtype=X.dtype, device=X.device).reshape((n,) + orb.shape)
    cols = []
    for s in range(0, n, chunk):
        V = eye[s:s + chunk]
        m = V.shape[0]
        orb._use(m)
        base = orb.caches[m]
        cache = copy.copy(base)
        cache.layers = [copy.copy(layer) for layer in base.layers]
        steer = orb.steer1.repeat(m, 1, 1)

        def h_of(inp):
            loop.sm.steer = steer
            try:
                out = loop.model.model(inputs_embeds=inp.to(loop.dtype), past_key_values=cache, use_cache=True)
                return out.last_hidden_state[:, -1, :].to(loop.state_dtype)
            finally:
                loop.sm.steer = None

        with sdpa_kernel(SDPBackend.MATH):
            _, jv = torch.func.jvp(h_of, (X.repeat(m, 1, 1),), (V,))
        cols.append(jv)
    return torch.cat(cols).T                                   # J[:, i] = d h / d x_i


def log_norm_product(mats, torch):
    """log ||M_k ... M_1|| for mats = [M_1, ..., M_k] (torch, float64), renormalizing by the Frobenius norm at every
    product so that strongly expanding or contracting blocks neither overflow nor underflow."""
    P = mats[0]
    scale = 0.0
    for M in mats[1:]:
        P = M @ P
        f = float(torch.linalg.matrix_norm(P))
        P = P / f
        scale += np.log(f)
    return scale + float(np.log(float(torch.linalg.matrix_norm(P, ord=2))))


def cell_bounds(orb, args, torch, E):
    X = orb.iterate(orb.start, args.transient)
    d = orb.n
    T = orb.T
    DF, log_df, log_read, log_net, gaps, fact = [], [], [], [], [], []
    for i in range(args.record):
        D = torch.as_tensor(orb.jacobian(X, "ad"), device=X.device)            # (d, d), float64
        p = orb.probabilities(X).to(E.dtype)
        F = p @ E
        S = (E * p[:, None]).T @ E - torch.outer(F, F)                      # Sigma_p, exactly
        sD = float(torch.linalg.matrix_norm(D, ord=2))
        sS = float(torch.linalg.eigvalsh(S)[-1])
        DF.append(D)
        log_df.append(np.log(sD))
        log_read.append(np.log(sS / T))
        if i % args.j_stride == 0:
            J = h_jacobian(orb, X, args.chunk)
            sJ = float(torch.linalg.matrix_norm(J, ord=2))
            log_net.append(np.log(sJ))
            gaps.append(np.log(sS) + np.log(sJ) - np.log(T) - np.log(sD))
            R = S @ J / T
            fact.append(float(torch.linalg.matrix_norm(D - R) / torch.linalg.matrix_norm(D)))
        X = orb.f(X)
    N = args.record
    dyadic = {}
    k = 1
    while k <= N:
        if N % k == 0:
            dyadic[k] = sum(log_norm_product(DF[j * k:(j + 1) * k], torch) for j in range(N // k)) / N
        k *= 2
    overlap = {}
    for k in range(1, N + 1):
        vals = [log_norm_product(DF[i:i + k], torch) / k for i in range(0, N - k + 1)]
        overlap[k] = float(np.mean(vals))
    g = torch.Generator(device=X.device).manual_seed(0)
    w = torch.randn(d, generator=g, dtype=DF[0].dtype, device=X.device)
    w = w / w.norm()
    tan = []
    for D in DF:                                                           # tangent exponent along the window
        w = D @ w
        nw = float(w.norm())
        tan.append(np.log(nw))
        w = w / nw
    long = []
    v = w.reshape((1,) + orb.shape).to(X.dtype)
    for _ in range(args.tangent):                                          # long-run reference exponent
        v = orb.jvp_rows(X, v)
        nv = float(v.norm())
        long.append(np.log(nv))
        v = v / nv
        X = orb.f(X)
    chain_dyadic = [dyadic[k] for k in sorted(dyadic)]
    return {"B_dyadic": {str(k): float(v) for k, v in sorted(dyadic.items())},
            "B_overlap": {str(k): v for k, v in overlap.items()},
            "Lambda_N": float(dyadic[N]) if N in dyadic else None,
            "chain_monotone": bool(all(a >= b - 1e-12 for a, b in zip(chain_dyadic, chain_dyadic[1:]))),
            "one_step_mean": float(np.mean(log_df)), "readout_mean": float(np.mean(log_read)),
            "network_mean_strided": float(np.mean(log_net)), "readout_mean_strided": float(np.mean(log_read[::args.j_stride])),
            "one_step_mean_strided": float(np.mean(log_df[::args.j_stride])),
            "gap_min": float(np.min(gaps)), "gap_median": float(np.median(gaps)),
            "factorization_rel_err_max": float(np.max(fact)),
            "tangent_window": float(np.mean(tan)), "lambda_long": float(np.mean(long)),
            "lambda_long_steps": args.tangent}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cells-file", required=True)
    ap.add_argument("--model", default="qwen0.5b")
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    ap.add_argument("--concept", default="philosophy")
    ap.add_argument("--transient", type=int, default=1000)
    ap.add_argument("--record", type=int, default=32)
    ap.add_argument("--j-stride", type=int, default=4)
    ap.add_argument("--tangent", type=int, default=2000)
    ap.add_argument("--chunk", type=int, default=448)
    ap.add_argument("--first", type=int, default=0, help="index of the first cell to run (cells run in file order)")
    ap.add_argument("--stride", type=int, default=1, help="run every stride-th cell from --first (split over processes)")
    ap.add_argument("--budget-minutes", type=float, default=1e9, help="start no new cell after this many minutes")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    path = Path(args.cells_file)
    if not path.is_absolute() and not path.exists():
        path = ROOT / args.cells_file
    spec = json.loads(path.read_text())
    out = DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    prog = (out / "progress.jsonl").open("a")
    loop = SoftLoop(args.model, args.layer, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    torch = loop.torch
    E = loop.E.to(torch.float64)
    v = np.asarray(elpm.load_vectors(args.vectors)[args.concept], dtype=np.float64)
    tic = time.perf_counter()
    rows = []
    idx = list(range(args.first, len(spec["cells"]), args.stride))
    for c in idx:
        lt, al = spec["cells"][c]
        base = {"index": c, "log10_temperature": lt, "alpha": al, "source": spec["source"][c],
                "map_label": spec["label"][c], "map_exponent": spec["map_exponent"][c]}
        if (time.perf_counter() - tic) / 60 > args.budget_minutes:
            rows.append({**base, "status": "not started (budget)"})
            continue
        t0 = time.perf_counter()
        orb = SoftOrbit(loop, al * v, 10.0 ** lt, chunk=args.chunk)
        try:
            res = cell_bounds(orb, args, torch, E)
            row = {**base, "status": "done", "seconds": round(time.perf_counter() - t0, 1), **res}
        except Exception as e:                       # noqa: BLE001 - a failed cell is recorded, never substituted
            import traceback
            row = {**base, "status": f"failed: {type(e).__name__}: {e}"[:400],
                   "traceback": traceback.format_exc()[-1500:]}
        rows.append(row)
        brief = {k: row.get(k) for k in ("index", "map_label", "status", "seconds", "Lambda_N", "lambda_long",
                                         "one_step_mean", "factorization_rel_err_max", "gap_min", "chain_monotone")}
        prog.write(json.dumps(brief) + "\n")
        prog.flush()
        print(json.dumps(brief), flush=True)
        (out / "bh.json").write_text(json.dumps({"args": vars(args), "cells": rows}, indent=0))
    (out / "bh.json").write_text(json.dumps({"args": vars(args), "cells": rows,
                                             "seconds": round(time.perf_counter() - tic, 1)}, indent=0))


if __name__ == "__main__":
    main()
