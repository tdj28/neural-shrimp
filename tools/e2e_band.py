"""Run E2E, Q2: the boundaries of a window on one-parameter cuts, and a coexistence census.

cut     on a cut at fixed alpha (free log10 T) or fixed log10 T (free alpha), seed the window's orbit from the fixed
        start (or a token embedding or a saved state) at a parameter inside the window, verify its period, continue
        it both ways (tools/e2e_branches.continue_branch, central-difference multipliers) until it loses stability,
        then locate each boundary: a flip by secant iteration on mu + 1 (real crossing multiplier only), a fold by
        (1 - mu)^2 extrapolation on the stable side. The crossing is classified before anything is computed from it;
        the multipliers on both sides of a located flip are recomputed with forward-mode JVPs.
census  at given cells, iterate from the fixed start and from saved states of candidate attractors (each start
        also with a twin at relative offset 1e-7); after the transient, record the period (state recurrence),
        the largest Lyapunov exponent (twin renormalized every step) and the leading-token loop of every start.

Usage: python tools/e2e_band.py cut --fixed alpha --value 0.070 --center -0.1405 --period 6 --span 0.01 --out band-cut-a0.0700
       python tools/e2e_band.py census --cells 0.069 -0.1427 0.070 -0.1405 --starts NAME=path.npy ... --out census
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import e2e_branches as eb  # noqa: E402
import e2e_model as em  # noqa: E402
import e2e_numerics as en  # noqa: E402


def make_cell(model, fixed, value, lam):
    return model.cell(value, lam) if fixed == "alpha" else model.cell(lam, value)


def refine_flip(model, fixed, value, p, a, xa, b, xb, k=6, iters=5, tol=1e-6, log=print):
    """Secant (regula falsi) on mu + 1 between a stable record a (mu > -1) and an unstable one b (mu < -1), both
    with a real leading multiplier; states xa, xb as initial guesses. Returns the bracketing solves with AD
    multipliers and the interpolated flip parameter."""
    def lead(info):
        vals = np.array([complex(u, w) for u, w in info["multipliers"]])
        return vals[np.argsort(-np.abs(vals))][0]

    la, ma = a["lambda"], complex(*a["leading"])
    lb, mb = b["lambda"], complex(*b["leading"])
    for m in (ma, mb):
        if abs(m.imag) > 1e-9:
            raise ValueError(f"crossing multiplier {m} is not real: not a flip")
    ma, mb = ma.real, mb.real
    if not (ma > -1 > mb):
        raise ValueError(f"multipliers {ma}, {mb} do not bracket -1 from the stable side")
    hist = []
    for _ in range(iters):
        lam = en.flip_root(la, ma, lb, mb)
        x0 = xa + (xb - xa) * (lam - la) / (lb - la)
        cell = make_cell(model, fixed, value, lam)
        x, info, _ = cell.solve(x0, p, k=k, ad=False)
        if not info["converged"]:
            log(json.dumps({"flip_refine": lam, "failed": info["residuals"][-3:]}))
            break
        m = lead(info)
        if abs(m.imag) > 1e-9:
            raise ValueError(f"complex leading multiplier {m} inside the flip bracket")
        hist.append({"lambda": lam, "mu": m.real})
        log(json.dumps({"flip_refine": lam, "mu": m.real}))
        if m.real > -1:
            la, ma, xa = lam, m.real, x
        else:
            lb, mb, xb = lam, m.real, x
        if abs(m.real + 1) < tol:
            break
    sides = []
    for lam, x in ((la, xa), (lb, xb)):
        cell = make_cell(model, fixed, value, lam)
        Q, H, vals, conv, _ = cell.multipliers(x, p, k=k, iters=10, ad=True)
        res = float(np.linalg.norm(cell.Fp(x, p) - x) / np.linalg.norm(x))
        sides.append({"lambda": lam, "residual": res, "multipliers_ad": [[float(z.real), float(z.imag)] for z in vals],
                      "converged": bool(conv), "classified": en.classify(vals)})
    flip = en.flip_root(sides[0]["lambda"], sides[0]["classified"]["multiplier"][0],
                        sides[1]["lambda"], sides[1]["classified"]["multiplier"][0])
    return {"type": "flip", "lambda": flip, "bracket": [la, lb], "sides": sides, "secant": hist}


def boundary(model, fixed, value, p, records, states, stop, log):
    """Locate and classify the boundary at the end of one continuation direction."""
    if not records:
        return {"type": None, "reason": "no points"}
    last = records[-1]
    lead = complex(*last["leading"])
    if abs(lead) >= 1 and len(records) >= 2:
        cls = last["crossing_side"]
        if cls["type"] == "real -1":
            try:
                return refine_flip(model, fixed, value, p, records[-2], states[-2], last, states[-1], log=log)
            except ValueError as err:
                return {"type": "unresolved", "reason": str(err), "classified": cls}
        return {"type": "unresolved", "reason": f"lost stability through {cls['type']}", "classified": cls,
                "between": [records[-2]["lambda"], last["lambda"]]}
    fold = eb.fold_estimate(records)
    if fold["fold"] is None:
        return {"type": "unresolved", "reason": f"{stop}; {fold['reason']}", "last_leading": last["leading"]}
    return {"type": "fold", "lambda": fold["fold"], "fit": fold, "stop": stop,
            "classified_last": last["crossing_side"]}


def cut(args):
    model = em.Model(args.threads)
    out = em.OUT / args.out
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "progress.jsonl", "w")

    def log(line):
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    cell = make_cell(model, args.fixed, args.value, args.center)
    x = np.load(args.seed_state).reshape(-1) if args.seed_state else (
        model.E[em.TOKENS[args.seed_token]].copy() if args.seed_token else cell.start.copy())
    X, found, d = eb.verify_period(cell, x, args.period, args.transient)
    if found != args.period:
        raise SystemExit(f"seed reached period {found} (wanted {args.period}); return distances {d[:12]}")
    free = "log_t" if args.fixed == "alpha" else "alpha"
    res = {"args": vars(args), "free": free, "seed_text": cell.text(X, args.period), "directions": {}}
    for name, to in (("up", args.center + args.span), ("down", args.center - args.span)):
        recs, sts, stop = eb.continue_branch(model, args.value, free, args.center, X, args.period, to, args.step,
                                             args.step_min, args.step_max, out, k=args.k, mu_stop=args.mu_stop,
                                             log=log, ad=False, name=f"branch-{name}.json")
        b = boundary(model, args.fixed, args.value, args.period, recs, sts, stop, log)
        res["directions"][name] = {"stop": stop, "points": len(recs), "boundary": b,
                                   "first_text": [ph["top"][:2] for ph in recs[0]["phases"]] if recs else None,
                                   "last_text": [ph["top"][:2] for ph in recs[-1]["phases"]] if recs else None}
        log(json.dumps({"direction": name, "boundary": b}, ensure_ascii=False))
        (out / "cut.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    np.save(out / "seed-state.npy", X)
    print(json.dumps({d: res["directions"][d]["boundary"].get("lambda") for d in res["directions"]}))


def loop_label(model, P):
    """Rotation-invariant leading-token loop of a recorded sequence of distributions (rows), or None."""
    toks = [int(np.argmax(r)) for r in P]
    n = len(toks)
    for q in range(1, n // 3 + 1):
        if all(toks[i] == toks[i + q] for i in range(n - q)):
            cyc = toks[-q:]
            k = min(range(q), key=lambda j: cyc[j:] + cyc[:j])
            return [model.decode(t) for t in cyc[k:] + cyc[:k]]
    return None


def census(args):
    model = em.Model(args.threads)
    out = em.OUT / args.out
    out.mkdir(parents=True, exist_ok=True)
    cells = [(args.cells[i], args.cells[i + 1]) for i in range(0, len(args.cells), 2)]
    starts = {"fixed": None}
    for s in args.starts:
        name, path = s.split("=", 1)
        starts[name] = np.load(path).reshape(-1)
    rng = np.random.default_rng(args.seed)
    rows = []
    for alpha, log_t in cells:
        cell = model.cell(alpha, log_t)
        names = list(starts)
        X0 = np.array([cell.start if starts[n] is None else starts[n] for n in names])
        B = len(names)
        tic = time.perf_counter()
        X = X0.copy()
        for _ in range(args.transient):
            X = cell.F(X)
        delta = rng.normal(size=X.shape)
        delta *= (args.eps * np.linalg.norm(X, axis=1) / np.linalg.norm(delta, axis=1))[:, None]
        Y = X + delta
        logs = np.zeros(B)
        hist = []
        for _ in range(args.steps):
            both = cell.F(np.concatenate([X, Y]))
            X, Y = both[:B], both[B:]
            dist = np.linalg.norm(Y - X, axis=1)
            logs += np.log(dist / (args.eps * np.linalg.norm(X, axis=1)))
            Y = X + (Y - X) * (args.eps * np.linalg.norm(X, axis=1) / dist)[:, None]
            hist.append(X.copy())
        hist = np.array(hist)                                   # (steps, B, d)
        P = cell.probs(hist[-args.record:].reshape(-1, model.d)).reshape(args.record, B, -1)
        for b, name in enumerate(names):
            ref = hist[-1, b]
            dists = [float(np.linalg.norm(hist[-1 - q, b] - ref) / np.linalg.norm(ref)) for q in range(1, 49)]
            period = next((q for q, dq in enumerate(dists, 1) if dq < args.tol), 0)
            lam = float(logs[b] / args.steps)
            rec = {"alpha": alpha, "log10_temperature": log_t, "start": name, "period": period, "lyapunov": lam,
                   "chaotic": bool(lam > 1e-3 and period == 0), "loop": loop_label(model, P[:, b]),
                   "smallest_return_distance": min(dists), "final_state_norm": float(np.linalg.norm(ref))}
            rows.append(rec)
            print(json.dumps(rec, ensure_ascii=False), flush=True)
        np.save(out / f"final-a{alpha:.6f}-t{log_t:.6f}.npy", hist[-48:])
        print(json.dumps({"cell": [alpha, log_t], "seconds": round(time.perf_counter() - tic, 1)}), flush=True)
        (out / "census.json").write_text(json.dumps({"args": vars(args), "rows": rows}, indent=1, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cut")
    c.add_argument("--fixed", choices=("alpha", "log_t"), required=True)
    c.add_argument("--value", type=float, required=True)
    c.add_argument("--center", type=float, required=True)
    c.add_argument("--period", type=int, required=True)
    c.add_argument("--span", type=float, required=True)
    c.add_argument("--seed-token", default=None, choices=sorted(em.TOKENS))
    c.add_argument("--seed-state", default=None)
    c.add_argument("--transient", type=int, default=600)
    c.add_argument("--step", type=float, default=2e-4)
    c.add_argument("--step-min", type=float, default=1e-8)
    c.add_argument("--step-max", type=float, default=5e-4)
    c.add_argument("--mu-stop", type=float, default=0.995)
    c.add_argument("--k", type=int, default=6)
    c.add_argument("--threads", type=int, default=8)
    c.add_argument("--out", required=True)
    s = sub.add_parser("census")
    s.add_argument("--cells", type=float, nargs="+", required=True, help="alpha log10T pairs")
    s.add_argument("--starts", nargs="*", default=[], help="NAME=path.npy (one state each)")
    s.add_argument("--transient", type=int, default=2000)
    s.add_argument("--steps", type=int, default=1000)
    s.add_argument("--record", type=int, default=96)
    s.add_argument("--eps", type=float, default=1e-7)
    s.add_argument("--tol", type=float, default=1e-8)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--threads", type=int, default=8)
    s.add_argument("--out", required=True)
    args = ap.parse_args()
    {"cut": cut, "census": census}[args.cmd](args)


if __name__ == "__main__":
    main()
