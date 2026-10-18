"""Run NT (exploratory; human: "use temperature as the third parameter"): continue the near-nilpotent point of the
period-5 window of the two-block plane in log10 T, to see whether its head becomes one-dimensional.

At the near-nilpotent point (tools/nilpotent_search.py) the sum of the leading Ritz values and the sum of their
pairwise products vanish; at log10 T = -0.1 the three leading multipliers there have nearly equal moduli, 3.4 to
3.6e-3, which is the scale of the directions transverse to the one-dimensional reduction. In a map whose transverse
directions contract more strongly, the point would approach a doubly superstable point of the reduction, the head of a
one-dimensional shrimp, and the avoided crossing of its superstable curves would shrink. This tool follows the point
through a list of temperatures:

1. At the first temperature, the fixed start at --seed must reach period --period; the orbit is marched to --start
   and a damped two-parameter Newton iteration solves c1 = c2 = 0 (c1: sum of the --k Ritz values, c2: sum of their
   pairwise products) in (alpha1, alpha2), as tools/nilpotent_search.py.
2. Each next temperature: predictor = the last point (or the linear extrapolation of the last two); the orbit is
   marched from the last solution to the predictor in --march-steps steps of (alpha1, alpha2, log10 T) together,
   then the same Newton iteration.
3. Every solve records the Ritz values (moduli and arguments), the token loop, the Newton residual of the orbit, the
   return distance at the divisor periods (a solve onto a divisor-period orbit stops the track), and the Ritz
   residuals. A temperature whose Newton iteration does not reach |c1|, |c2| < --tol-f is reported as not converged
   and ends the track (no fallback point).

Output: DATA/explore/OUT/progress.jsonl (every solve) and track.json (one row per temperature).

Usage: python tools/nilpotent_track.py --seed -0.118730 0.100693 --start -0.1184836 0.1017217 \
         --log-t -0.100 -0.105 -0.110 --device cuda --out llm-soft-loop/pod64/nt/down
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
    ap.add_argument("--period", type=int, default=5)
    ap.add_argument("--seed", type=float, nargs=2, required=True, help="fixed-start cell at the first temperature")
    ap.add_argument("--start", type=float, nargs=2, required=True, help="first guess of the point")
    ap.add_argument("--log-t", type=float, nargs="+", required=True)
    ap.add_argument("--march-steps", type=int, default=6)
    ap.add_argument("--seed-march", type=float, default=2e-4, help="largest march step from --seed to --start")
    ap.add_argument("--iterations", type=int, default=10)
    ap.add_argument("--tol-f", type=float, default=1e-7)
    ap.add_argument("--h", type=float, default=5e-6, help="finite-difference step in alpha")
    ap.add_argument("--cap", type=float, default=5e-4, help="largest Newton step in alpha")
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl
    import llm_feigenbaum as lfb

    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "progress.jsonl", "a")

    def emit(rec):
        print(json.dumps(rec), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

    P = args.period
    loop = esl.SoftLoop("qwen0.5b", 4, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    v = elpm.load_vectors("llm-phase-map/vectors-l4")["gratitude"]
    loop.sm.add_steer_layer(16)
    w = elpm.load_vectors("llm-phase-map/vectors-l16")["hostility"]
    tol = lfb.tol_for(P, 1e-12)

    def make_cell(a1, a2, lt):
        loop.extra = (16, np.array([a2 * w]))
        return esl.SoftOrbit(loop, a1 * v, 10.0 ** lt, chunk=512)

    def solve(a1, a2, lt, X, Q, full=True):
        t = time.perf_counter()
        cell = make_cell(a1, a2, lt)
        X, Q, hist = lfb.newton(cell, X, P, Q, tol, 12)
        ok = bool(hist[-1] < tol)
        rec = {"a1": float(a1), "a2": float(a2), "log10_T": float(lt), "newton": float(hist[-1]), "converged": ok}
        if not full:
            return rec, X, Q
        split = min((lfb.return_distance(cell, X, q) for q in range(1, P) if P % q == 0), default=1.0)
        vals, vecs, res, rh, Q = lfb.ritz(cell, X, P, Q)
        toks, Y = [], X
        for _ in range(P):
            toks.append(int(cell.probabilities(Y).argmax()))
            Y = cell.f(Y)
        c1 = float(np.sum(vals).real)
        c2 = float(sum(vals[i] * vals[j] for i in range(len(vals)) for j in range(i + 1, len(vals))).real)
        rec.update({"split": float(split), "tokens": toks, "mu": [[float(z.real), float(z.imag)] for z in vals],
                    "moduli": [float(abs(z)) for z in vals],
                    "arguments_deg": [float(np.degrees(np.angle(z))) for z in vals],
                    "ritz_res": [float(r) for r in res[:4]], "c1": c1, "c2": c2,
                    "seconds": round(time.perf_counter() - t, 1)})
        return rec, X, Q

    def march(p0, p1, X, Q, n):
        for s in np.linspace(0, 1, n + 1)[1:]:
            p = p0 + s * (p1 - p0)
            rec, X, Q = solve(p[0], p[1], p[2], X, Q, full=False)
            if not rec["converged"]:
                return False, rec, X, Q
        return True, rec, X, Q

    def newton2d(a, lt, X, Q):
        """Damped Newton on (c1, c2) = 0 in (alpha1, alpha2) at fixed lt; returns (converged, last record, a, X, Q)."""
        rec = None
        for it in range(args.iterations):
            rec, X, Q = solve(a[0], a[1], lt, X, Q)
            rec["tag"] = f"newton2d-{it}"
            emit(rec)
            if not rec["converged"] or rec["split"] <= 1e-7:
                return False, rec, a, X, Q
            F0 = np.array([rec["c1"], rec["c2"]])
            if np.max(np.abs(F0)) < args.tol_f:
                return True, rec, a, X, Q
            J = np.zeros((2, 2))
            for k in range(2):
                b = a.copy()
                b[k] += args.h
                rk, _, _ = solve(b[0], b[1], lt, X.clone(), Q.copy())
                J[:, k] = (np.array([rk["c1"], rk["c2"]]) - F0) / args.h
            try:
                step = np.linalg.solve(J, -F0)
            except np.linalg.LinAlgError:
                emit({"tag": f"step-{it}", "error": "singular Jacobian", "J": J.tolist()})
                return False, rec, a, X, Q
            scale = min(1.0, args.cap / max(float(np.max(np.abs(step))), 1e-30))
            emit({"tag": f"step-{it}", "J": J.tolist(), "step": step.tolist(), "damping": scale})
            a = a + scale * step
        rec, X, Q = solve(a[0], a[1], lt, X, Q)
        rec["tag"] = "newton2d-final"
        emit(rec)
        ok = rec["converged"] and rec["split"] > 1e-7 and max(abs(rec["c1"]), abs(rec["c2"])) < args.tol_f
        return ok, rec, a, X, Q

    tic = time.perf_counter()
    lt0 = args.log_t[0]
    cell = make_cell(args.seed[0], args.seed[1], lt0)
    X = cell.iterate(cell.start, 900)
    per = cell.detect_period(X, 32, 1e-6)
    emit({"seed": args.seed, "log10_T": lt0, "period_detected": int(per)})
    if per != P:
        raise SystemExit(f"the attractor at the seed has period {per}, not {P}")
    Q = np.linalg.qr(np.random.default_rng(0).normal(size=(X.numel(), args.k)))[0]
    p0 = np.array([args.seed[0], args.seed[1], lt0])
    p1 = np.array([args.start[0], args.start[1], lt0])
    n = int(np.ceil(np.max(np.abs(p1[:2] - p0[:2])) / args.seed_march)) + 1
    ok, rec, X, Q = march(p0, p1, X, Q, n)
    if not ok:
        raise SystemExit(f"march from the seed to the start failed: {rec}")
    rows, pts = [], []
    a = np.array(args.start, dtype=float)
    for m, lt in enumerate(args.log_t):
        if m > 0:
            prev = np.array([pts[-1][0], pts[-1][1], args.log_t[m - 1]])
            if len(pts) >= 2:
                dlt = args.log_t[m - 2] - args.log_t[m - 1]
                slope = (np.array(pts[-1]) - np.array(pts[-2])) / (args.log_t[m - 1] - args.log_t[m - 2])
                a = np.array(pts[-1]) + slope * (lt - args.log_t[m - 1]) if abs(dlt) > 0 else np.array(pts[-1])
            else:
                a = np.array(pts[-1])
            ok, rec, X, Q = march(prev, np.array([a[0], a[1], lt]), X, Q, args.march_steps)
            if not ok:
                emit({"stop": "march lost the orbit", "log10_T": lt, "last": rec})
                break
        ok, rec, a, X, Q = newton2d(a, lt, X, Q)
        row = {"log10_T": lt, "converged": ok, "a1": float(a[0]), "a2": float(a[1]),
               "moduli": rec.get("moduli"), "arguments_deg": rec.get("arguments_deg"), "tokens": rec.get("tokens"),
               "c1": rec.get("c1"), "c2": rec.get("c2"), "newton": rec.get("newton"), "split": rec.get("split"),
               "seconds": round(time.perf_counter() - tic, 1)}
        rows.append(row)
        emit({"row": row})
        (out / "track.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1))
        if not ok:
            emit({"stop": "Newton did not converge", "log10_T": lt})
            break
        pts.append([float(a[0]), float(a[1])])
    emit({"done": True, "temperatures": len(rows), "seconds": round(time.perf_counter() - tic, 1)})


if __name__ == "__main__":
    main()
