"""Near-nilpotent points of a periodic orbit's monodromy in the two-block plane (generalizes tools/p5_nilpotent.py, the
referee's period-5 computation): does the leading part of the spectrum vanish (T = mu1 + mu2 = 0 and D = mu1 mu2 = 0)
somewhere near a target? Same map and settings as tools/doubly_superstable.py grid in the two-block plane:
Qwen2.5-0.5B-Instruct, gratitude at block 4 (alpha1), hostility at block 16 (alpha2), log10 T = -0.1, Newton
tolerance 1e-12 (tol_for), k = 6 Ritz vectors.

From the seed (whose fixed-start attractor must have period p) the orbit is marched to the target in steps no larger
than --step, then a damped two-parameter Newton iteration solves c1 = c2 = 0, with c1 the sum of the Ritz values and
c2 the sum of their pairwise products (smooth surrogates of T and D that stay defined where the leading pair is not
isolated). Every solve records the multipliers, the token loop, the Newton residual and the return distance at the
divisor periods (a solve onto a divisor-period orbit stops the search). Writes OUT/log.jsonl.

Usage: python tools/nilpotent_search.py --period 4 --seed A1 A2 --target A1 A2 --step S1 S2 --out RUN [--threads 16]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--period", type=int, required=True)
    ap.add_argument("--seed", type=float, nargs=2, required=True)
    ap.add_argument("--target", type=float, nargs=2, required=True)
    ap.add_argument("--step", type=float, nargs=2, required=True, help="largest march step in alpha1, alpha2")
    ap.add_argument("--iterations", type=int, default=6)
    ap.add_argument("--stop-modulus", type=float, default=3e-3)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl
    import llm_feigenbaum as lfb

    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "log.jsonl", "a")

    def emit(rec):
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    P = args.period
    loop = esl.SoftLoop("qwen0.5b", 4, args.threads, "open", ela.OPENER, 1, "float64", "cpu")
    v = elpm.load_vectors("llm-phase-map/vectors-l4")["gratitude"]
    loop.sm.add_steer_layer(16)
    w = elpm.load_vectors("llm-phase-map/vectors-l16")["hostility"]

    def make_cell(a1, a2):
        loop.extra = (16, np.array([a2 * w]))
        return esl.SoftOrbit(loop, a1 * v, 10.0 ** -0.1, chunk=512)

    def solve(a1, a2, X, Q):
        t = time.perf_counter()
        cell = make_cell(a1, a2)
        X, Q, hist = lfb.newton(cell, X, P, Q, lfb.tol_for(P, 1e-12), 12)
        ok = bool(hist[-1] < lfb.tol_for(P, 1e-12))
        split = min((lfb.return_distance(cell, X, q) for q in range(1, P) if P % q == 0), default=1.0)
        vals, vecs, res, rh, Q = lfb.ritz(cell, X, P, Q)
        toks, Y = [], X
        for _ in range(P):
            toks.append(int(cell.probabilities(Y).argmax()))
            Y = cell.f(Y)
        c1 = float(np.sum(vals).real)
        c2 = float(sum(vals[i] * vals[j] for i in range(len(vals)) for j in range(i + 1, len(vals))).real)
        rec = {"a1": a1, "a2": a2, "newton": float(hist[-1]), "converged": ok, "split": float(split), "tokens": toks,
               "mu": [[float(z.real), float(z.imag)] for z in vals], "moduli": [float(abs(z)) for z in vals],
               "ritz_res": [float(r) for r in res[:3]], "T2": float((vals[0] + vals[1]).real),
               "D2": float((vals[0] * vals[1]).real), "c1": c1, "c2": c2, "seconds": round(time.perf_counter() - t, 1)}
        return rec, X, Q

    tic = time.perf_counter()
    a0 = tuple(args.seed)
    cell = make_cell(*a0)
    X = cell.iterate(cell.start, 900)
    per = cell.detect_period(X, 32, 1e-6)
    emit({"seed": a0, "period_detected": int(per), "seconds": round(time.perf_counter() - tic, 1)})
    if per != P:
        raise SystemExit(f"the attractor at the seed has period {per}, not {P}")
    Q = np.linalg.qr(np.random.default_rng(0).normal(size=(X.numel(), 6)))[0]
    rec, X, Q = solve(*a0, X, Q)
    rec["tag"] = "seed"
    emit(rec)
    tgt = tuple(args.target)
    n = int(np.ceil(max(abs(tgt[0] - a0[0]) / args.step[0], abs(tgt[1] - a0[1]) / args.step[1]))) + 1
    for s in np.linspace(0, 1, n + 1)[1:]:
        a = (a0[0] + s * (tgt[0] - a0[0]), a0[1] + s * (tgt[1] - a0[1]))
        rec, X, Q = solve(*a, X, Q)
        rec["tag"] = "march"
        emit(rec)
        if not rec["converged"] or rec["split"] <= 1e-7:
            raise SystemExit("march failed (no convergence, or a divisor-period orbit)")
    a = np.array(tgt, dtype=float)
    h = np.array(args.step) / 20
    lim = np.array(args.step) / 4
    for it in range(args.iterations):
        r0, X, Q = solve(a[0], a[1], X, Q)
        r0["tag"] = f"newton2d-{it}"
        emit(r0)
        if max(r0["moduli"][:3]) < args.stop_modulus or not r0["converged"] or r0["split"] <= 1e-7:
            break
        F0 = np.array([r0["c1"], r0["c2"]])
        J = np.zeros((2, 2))
        for k in range(2):
            b = a.copy()
            b[k] += h[k]
            rk, _, _ = solve(b[0], b[1], X.clone(), Q.copy())
            J[:, k] = (np.array([rk["c1"], rk["c2"]]) - F0) / h[k]
        step = np.linalg.solve(J, -F0)
        scale = min(1.0, float(np.min(lim / np.maximum(np.abs(step), 1e-30))))
        emit({"tag": f"step-{it}", "J": J.tolist(), "step": step.tolist(), "damping": scale})
        a = a + scale * step
    emit({"done": True, "seconds": round(time.perf_counter() - tic, 1)})


if __name__ == "__main__":
    main()
