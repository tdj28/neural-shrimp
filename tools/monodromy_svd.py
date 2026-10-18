"""Run RK (exploratory; consult 15, first recommendation): is the return map of a periodic orbit still nearly of rank
one at a shrimp's head?

Equal moduli of the leading multipliers at a near-nilpotent point are forced by the conditions that define it, and a
strongly nonnormal operator of rank nearly one can have several tiny eigenvalues of comparable modulus. Singular values
decide the rank question. For the monodromy M = J_(phi+p-1) ... J_phi of a period-p orbit started at phase phi, this tool
estimates the leading singular values by a randomized range finder with power iterations: Jacobian-vector products
(forward mode) and vector-Jacobian products (reverse mode) chained along the orbit, all in float64. Reported per point
and phase: sigma_1..sigma_q, d1 = sigma_2/sigma_1, d3 = sigma_4/sigma_1, the leading multipliers (subspace iteration),
the orbit's Newton residual and token loop. A convergence check repeats the estimate with one more power iteration and
records the relative change of every reported ratio; a ratio whose change exceeds --tol-change is flagged unresolved.

Controls (--controls): the same routine on synthetic operators with identical eigenvalues and different rank structure,
A1 = eps I + e1 e2^T and A2 = eps I + e1 e2^T + e2 e3^T (n = 896), whose singular structure it must separate.

Points: --points A1 A2 [A1 A2 ...] in the two-block plane at --log-t; the orbit is reached from the fixed start at --seed
(period --period required) by marching to the first point, then from point to point.

Usage: python tools/monodromy_svd.py --period 5 --seed -0.118730 0.100693 --points -0.1184836 0.1017217 ... \
         --phases 0 2 4 --device cuda --out llm-soft-loop/rk/p5-head
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))


def orth(rows):
    """Orthonormal rows spanning the given rows (q, n)."""
    q, _ = np.linalg.qr(rows.T)
    return q.T


def rsvd(fwd, bwd, n, q, power, rng):
    """Leading singular values of an operator given as fwd(V) = M V and bwd(U) = M^T U on rows (q, n)."""
    Y = fwd(rng.normal(size=(q, n)))
    for _ in range(power):
        Y = fwd(orth(bwd(orth(Y))))
    B = bwd(orth(Y))                       # rows (M^T q_i)^T: B = Q^T M
    return np.linalg.svd(B, compute_uv=False)


def ratios(s):
    return {"d1": float(s[1] / s[0]) if len(s) > 1 else None, "d3": float(s[3] / s[0]) if len(s) > 3 else None}


def controls(q, power, eps_list=(1e-6, 1e-4, 1e-2)):
    n = 896
    rng = np.random.default_rng(1)
    out = []
    for eps in eps_list:
        for name in ("A1", "A2"):
            def fwd(V, name=name, eps=eps):
                W = eps * V.copy()
                W[:, 0] += V[:, 1]                        # e1 e2^T
                if name == "A2":
                    W[:, 1] += V[:, 2]                    # e2 e3^T
                return W

            def bwd(U, name=name, eps=eps):
                W = eps * U.copy()
                W[:, 1] += U[:, 0]
                if name == "A2":
                    W[:, 2] += U[:, 1]
                return W
            s = rsvd(fwd, bwd, n, q, power, rng)
            out.append({"operator": name, "eps": eps, "sigma": s[:4].tolist(), **ratios(s),
                        "eigenvalues": "all equal to eps (nilpotent part plus eps I)"})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--period", type=int, required=True)
    ap.add_argument("--block1", type=int, default=4)
    ap.add_argument("--vectors1", default="llm-phase-map/vectors-l4")
    ap.add_argument("--concept1", default="gratitude")
    ap.add_argument("--block2", type=int, default=16)
    ap.add_argument("--vectors2", default="llm-phase-map/vectors-l16")
    ap.add_argument("--concept2", default="hostility")
    ap.add_argument("--log-t", type=float, default=-0.1)
    ap.add_argument("--seed", type=float, nargs=2, required=True)
    ap.add_argument("--points", type=float, nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--march-step", type=float, default=1e-4)
    ap.add_argument("--phases", type=int, nargs="+", default=[0])
    ap.add_argument("--q", type=int, default=8)
    ap.add_argument("--power", type=int, default=3)
    ap.add_argument("--tol-change", type=float, default=0.05)
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if len(args.points) % 2:
        raise SystemExit("--points needs pairs")
    pts = [(args.points[k], args.points[k + 1]) for k in range(0, len(args.points), 2)]
    labels = args.labels or [f"p{k}" for k in range(len(pts))]

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

    if args.controls:
        emit({"controls": controls(args.q, args.power)})
    P = args.period
    loop = esl.SoftLoop("qwen0.5b", args.block1, args.threads, "open", ela.OPENER, 1, "float64", args.device)
    torch = loop.torch
    v = elpm.load_vectors(args.vectors1)[args.concept1]
    loop.sm.add_steer_layer(args.block2)
    w = elpm.load_vectors(args.vectors2)[args.concept2]
    T = 10.0 ** args.log_t
    tol = lfb.tol_for(P, 1e-12)

    def make_cell(a1, a2):
        loop.extra = (args.block2, np.array([a2 * w]))
        return esl.SoftOrbit(loop, a1 * v, T, chunk=512)

    def vjp_rows(cell, X, U):
        """J(X)^T u for cotangent rows U (m, 1, d) at X (1, 1, d), by reverse mode through the model (window 1)."""
        m = U.shape[0]
        cell._use(m)
        base = cell.caches[m]
        cache = copy.copy(base)
        cache.layers = [copy.copy(layer) for layer in base.layers]
        with torch.enable_grad():
            Xr = X.repeat(m, 1, 1).detach().requires_grad_(True)
            loop.sm.steer = cell.steer1.repeat(m, 1, 1)
            if cell.extra1 is not None:
                loop.sm.steer_extra = {loop.extra[0]: cell.extra1.repeat(m, 1, 1)}
            try:
                z = loop.model(inputs_embeds=Xr.to(loop.dtype), past_key_values=cache, use_cache=True,
                               logits_to_keep=1).logits[:, -1, :].to(loop.state_dtype)
            finally:
                loop.sm.steer = None
                loop.sm.steer_extra = {}
            y = torch.softmax(z / cell.T, dim=-1) @ loop.E
            (g,) = torch.autograd.grad((y * U[:, -1, :]).sum(), Xr)
        return g

    def solve(a1, a2, X, Q):
        cell = make_cell(a1, a2)
        X, Q, hist = lfb.newton(cell, X, P, Q, tol, 12)
        return cell, X, Q, float(hist[-1])

    tic = time.perf_counter()
    cell = make_cell(*args.seed)
    if cell.start.shape[1] != 1:
        raise SystemExit("window 1 only")
    X = cell.iterate(cell.start, 900)
    per = cell.detect_period(X, 32, 1e-6)
    if per != P:
        raise SystemExit(f"the fixed start at --seed has period {per}, not {P}")
    Q = np.linalg.qr(np.random.default_rng(0).normal(size=(X.numel(), 6)))[0]
    cur = np.array(args.seed)
    rng = np.random.default_rng(2)
    n = X.numel()
    for lab, p in zip(labels, pts):
        p = np.array(p)
        steps = int(np.ceil(np.max(np.abs(p - cur)) / args.march_step)) + 1
        ok = True
        for s in np.linspace(0, 1, steps + 1)[1:]:
            q_ = cur + s * (p - cur)
            cell, X, Q, res = solve(q_[0], q_[1], X, Q)
            if res > tol:
                ok = False
                break
        if not ok:
            emit({"label": lab, "point": p.tolist(), "error": "march lost the orbit", "residual": res})
            break
        cur = p
        vals, _, rres, _, Q = lfb.ritz(cell, X, P, Q)
        orbit = [X]
        for _ in range(P - 1):
            orbit.append(cell.f(orbit[-1]))
        toks = [int(cell.probabilities(x).argmax()) for x in orbit]
        rec = {"label": lab, "point": p.tolist(), "newton": res, "tokens": toks,
               "multipliers": [[float(z.real), float(z.imag)] for z in vals[:4]],
               "moduli": [float(abs(z)) for z in vals[:4]], "phases": []}
        for phi in args.phases:
            seq = [orbit[(phi + k) % P] for k in range(P)]

            def fwd(V, seq=seq):
                Vt = torch.as_tensor(V.reshape(V.shape[0], 1, -1), dtype=X.dtype, device=X.device)
                for x in seq:
                    Vt = cell.jvp_rows(x, Vt)
                return Vt.reshape(V.shape[0], -1).double().cpu().numpy()

            def bwd(U, seq=seq):
                Ut = torch.as_tensor(U.reshape(U.shape[0], 1, -1), dtype=X.dtype, device=X.device)
                for x in reversed(seq):
                    Ut = vjp_rows(cell, x, Ut)
                return Ut.reshape(U.shape[0], -1).double().cpu().numpy()

            s1 = rsvd(fwd, bwd, n, args.q, args.power, rng)
            s2 = rsvd(fwd, bwd, n, args.q, args.power + 1, rng)
            r1, r2 = ratios(s1), ratios(s2)
            change = {k: (abs(r2[k] - r1[k]) / max(abs(r2[k]), 1e-300)) if r1[k] is not None else None for k in r1}
            rec["phases"].append({"phase": phi, "sigma": s2[:6].tolist(), **r2, "relative_change": change,
                                  "unresolved": [k for k, c in change.items() if c is not None and c > args.tol_change]})
        rec["seconds"] = round(time.perf_counter() - tic, 1)
        emit(rec)
    emit({"done": True, "seconds": round(time.perf_counter() - tic, 1)})


if __name__ == "__main__":
    main()
