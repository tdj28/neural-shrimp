"""Run P5C (exploratory; human: "we must find a verifiable shrimp"): locate a doubly superstable point of the
one-dimensional reduction of the period-5 window, the centre of a shrimp in Gallas' sense.

In the effectively one-dimensional map, the leading multiplier of a period-p orbit splits into one signed factor per
step (tools/doubly_superstable.py): c_i = <DF(x_i) v_i, r_(i+1)> / <v_i, r_i>, with v_i the chain of the leading
eigenvector and r_i fixed reference directions. A superstable curve of the reduction is where one factor vanishes; a
doubly superstable point is where the factors of two different steps vanish together. In the full map no multiplier is
exactly zero (each step's Jacobian is invertible), so at such a point the leading multiplier collapses to the scale of
the transverse ones, and the trace-zero curves of the leading pair form an avoided crossing around it.

1. Start: iterate the fixed start at --march-from (period p required), then continue the orbit by Newton along a line
   to --start (as doubly_superstable --march-from).
2. References: the chain directions of the leading eigenvector at the start point (refreshed with --refresh-refs at
   every accepted Newton step, so they stay local; their cosines to the chain are recorded).
3. Newton on F = (c_a, c_b) for the steps --steps A B in the two --free parameters (alpha1, alpha2 or log10T; the
   third fixed), finite-difference Jacobian, step caps, the orbit warm-started at every evaluation.
4. At the solution: leading multipliers, stretchings |DF(x_i) d_i|, the token loop, the reference cosines, and a local
   check grid (--check-n^2 points within +-(--check-radius) of each free parameter) of signed factors and multipliers:
   a doubly superstable point shows all four sign combinations of (c_a, c_b) around it.

Output: DATA/explore/OUT/centre.json and progress.jsonl.

Usage: python tools/shrimp_centre.py --log-t -0.105 --march-from -0.117872 0.102936 --start -0.12125 0.10359 \
         --steps 0 1 --device cuda --out llm-soft-loop/pod64/shrimp/centre-t105
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
import llm_feigenbaum as lfb  # noqa: E402
from doubly_superstable import chain_dirs, factors, leading  # noqa: E402

P = 5


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-t", type=float, required=True, help="log10 T at the start (fixed unless freed)")
    ap.add_argument("--alpha2", type=float, default=None, help="fixed alpha2 when --free alpha1 logt")
    ap.add_argument("--march-from", type=float, nargs=2, required=True, metavar=("A1", "A2"))
    ap.add_argument("--march-steps", type=int, default=30)
    ap.add_argument("--march-path", default=None, help="JSON list of [alpha1, alpha2] waypoints to march through")
    ap.add_argument("--march-sub", type=int, default=2)
    ap.add_argument("--start", type=float, nargs=2, required=True, metavar=("A1", "A2"))
    ap.add_argument("--free", nargs=2, default=["alpha1", "alpha2"], choices=["alpha1", "alpha2", "logt"])
    ap.add_argument("--steps", type=int, nargs=2, required=True, metavar=("A", "B"))
    ap.add_argument("--newton", type=int, default=12)
    ap.add_argument("--h", type=float, nargs=2, default=[2e-6, 2e-6], help="finite-difference steps per free parameter")
    ap.add_argument("--cap", type=float, nargs=2, default=[1e-4, 1e-4], help="Newton step caps per free parameter")
    ap.add_argument("--tol-f", type=float, default=1e-7)
    ap.add_argument("--tol-x", type=float, default=1e-10)
    ap.add_argument("--refresh-refs", action="store_true")
    ap.add_argument("--check-n", type=int, default=7)
    ap.add_argument("--check-radius", type=float, nargs=2, default=[4e-6, 4e-6])
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--transient", type=int, default=900)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = ela.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "progress.jsonl", "a")

    def emit(rec):
        print(json.dumps(rec), flush=True)
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

    loop = esl.SoftLoop("qwen0.5b", 4, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    v = elpm.load_vectors("llm-phase-map/vectors-l4")["gratitude"]
    loop.sm.add_steer_layer(16)
    w = elpm.load_vectors("llm-phase-map/vectors-l16")["hostility"]

    def make_cell(a1, a2, lt):
        loop.extra = (16, np.array([a2 * w]))
        return esl.SoftOrbit(loop, a1 * v, 10.0 ** lt, chunk=512)

    tol = lfb.tol_for(P, 1e-12)

    def solve(par, X, Q):
        cell = make_cell(*par)
        X, Q, hist = lfb.newton(cell, X, P, Q, tol, 12)
        return cell, X, Q, bool(hist[-1] < tol), float(hist[-1])

    def evaluate(par, X, Q, refs):
        cell, X, Q, ok, res = solve(par, X, Q)
        vals, vecs, _, _, Q = lfb.ritz(cell, X, P, Q)
        mu, vlead, _ = leading(vals, vecs)
        c, dirs, toks = factors(cell, X, P, vlead, refs)
        _, s, _ = chain_dirs(cell, X, P, vlead)
        cos = [float(abs(np.dot(d, r) / (np.linalg.norm(d) * np.linalg.norm(r)))) for d, r in zip(dirs, refs)]
        rec = {"par": [float(x) for x in par], "converged": ok, "newton": res, "factors": c.tolist(),
               "stretchings": s.tolist(), "tokens": toks, "ref_cos_min": float(min(cos)),
               "multipliers": [[float(z.real), float(z.imag)] for z in vals[:3]]}
        return rec, X, Q, vlead, dirs

    def full(par2):
        """Map the two free parameters to (alpha1, alpha2, log10T)."""
        base = {"alpha1": args.start[0], "alpha2": args.start[1] if args.alpha2 is None else args.alpha2,
                "logt": args.log_t}
        for name, val in zip(args.free, par2):
            base[name] = val
        return (base["alpha1"], base["alpha2"], base["logt"])

    tic = time.perf_counter()
    cell = make_cell(args.march_from[0], args.march_from[1], args.log_t)
    X = cell.iterate(cell.start, args.transient)
    per = cell.detect_period(X, 32, 1e-6)
    if per != P:
        raise SystemExit(f"the fixed start at --march-from has period {per}, not {P}")
    Q = np.linalg.qr(np.random.default_rng(0).normal(size=(X.numel(), args.k)))[0]
    _, X, Q, ok, res = solve((args.march_from[0], args.march_from[1], args.log_t), X, Q)
    if not ok:
        raise SystemExit("Newton failed at --march-from")
    if args.march_path:
        way = [tuple(q) for q in json.loads(Path(args.march_path).read_text())] + [tuple(args.start)]
        pts, prev = [], tuple(args.march_from)
        for q in way:
            for kk in range(1, args.march_sub + 1):
                u = kk / args.march_sub
                pts.append((prev[0] + u * (q[0] - prev[0]), prev[1] + u * (q[1] - prev[1])))
            prev = q
    else:
        pts = [(args.march_from[0] + k / args.march_steps * (args.start[0] - args.march_from[0]),
                args.march_from[1] + k / args.march_steps * (args.start[1] - args.march_from[1]))
               for k in range(1, args.march_steps + 1)]
    for k, q in enumerate(pts, 1):
        par = (q[0], q[1], args.log_t)
        _, X, Q, ok, res = solve(par, X, Q)
        if not ok:
            raise SystemExit(f"march lost the orbit at step {k} {par}: residual {res}")
    emit({"marched_to": list(args.start), "seconds": round(time.perf_counter() - tic, 1)})
    # references at the start point
    cell = make_cell(args.start[0], args.start[1], args.log_t)
    vals, vecs, _, _, Q = lfb.ritz(cell, X, P, Q)
    _, vlead, _ = leading(vals, vecs)
    refs, _, _ = chain_dirs(cell, X, P, vlead)
    start_vals = {"alpha1": args.start[0], "alpha2": args.start[1] if args.alpha2 is None else args.alpha2,
                  "logt": args.log_t}
    x = np.array([start_vals[n] for n in args.free], dtype=float)
    a, b = args.steps
    hist = []
    rec = None
    for it in range(args.newton):
        rec, X, Q, vl, dirs = evaluate(full(x), X, Q, refs)
        F0 = np.array([rec["factors"][a], rec["factors"][b]])
        rec.update({"iteration": it, "F": F0.tolist()})
        emit(rec)
        hist.append(rec)
        if not rec["converged"]:
            break
        if np.max(np.abs(F0)) < args.tol_f:
            break
        J = np.zeros((2, 2))
        for kk in range(2):
            xp = x.copy()
            xp[kk] += args.h[kk]
            rk, _, _, _, _ = evaluate(full(xp), X.clone(), Q.copy(), refs)
            J[:, kk] = (np.array([rk["factors"][a], rk["factors"][b]]) - F0) / args.h[kk]
        try:
            step = np.linalg.solve(J, -F0)
        except np.linalg.LinAlgError:
            emit({"iteration": it, "error": "singular Jacobian", "J": J.tolist()})
            break
        scale = min(1.0, float(np.min(np.array(args.cap) / np.maximum(np.abs(step), 1e-30))))
        x = x + scale * step
        if args.refresh_refs:
            refs = [d / np.linalg.norm(d) for d in dirs]
        if np.max(np.abs(scale * step)) < args.tol_x:
            break
    centre = {"free": args.free, "x": x.tolist(), "par": [float(t) for t in full(x)], "last": rec,
              "converged_F": bool(rec and np.max(np.abs([rec["factors"][a], rec["factors"][b]])) < args.tol_f),
              "seconds": round(time.perf_counter() - tic, 1)}
    # local check grid of sign(c_a), sign(c_b) and the leading multiplier
    grid = []
    offs = np.linspace(-1, 1, args.check_n)
    for di in offs:
        for dj in offs:
            xp = x + np.array([di * args.check_radius[0], dj * args.check_radius[1]])
            rk, _, _, _, _ = evaluate(full(xp), X.clone(), Q.copy(), refs)
            grid.append({"d": [float(di), float(dj)], "ca": rk["factors"][a], "cb": rk["factors"][b],
                         "mu1": rk["multipliers"][0], "ref_cos_min": rk["ref_cos_min"], "converged": rk["converged"]})
    quads = {(int(np.sign(g["ca"])), int(np.sign(g["cb"]))) for g in grid if g["converged"]}
    centre.update({"check_grid": grid, "sign_quadrants": sorted([list(q) for q in quads]),
                   "all_four_quadrants": len({q for q in quads if 0 not in q}) == 4})
    (out / "centre.json").write_text(json.dumps(centre, indent=1))
    emit({k: centre[k] for k in ("par", "converged_F", "all_four_quadrants", "seconds")})


if __name__ == "__main__":
    main()
