"""Feigenbaum's delta inside a language model: the period-doubling cascade of the soft-token map along a
one-parameter cut, located in float64.

The orbits of a period-doubling cascade of this map have one multiplier of order one and the rest small (the map is
strongly dissipative; on the period-8 stripe the others are about 1e-6). Each orbit is computed by Newton's method on
the p-fold map F^p(x) - x, with the Newton matrix M - I replaced by its action on a k-dimensional dominant subspace of
the monodromy M. The subspace comes from subspace iteration with exact forward-mode Jacobian-vector products chained
along the orbit (k tangents per model call), so no dense Jacobian is formed. The neglected part of M has the size of
the (k+1)-th multiplier, which sets Newton's linear rate; the residual is always the exact float64 F^p(x) - x. The
leading multipliers are Rayleigh-Ritz values on the same subspace, reported with the residual of the critical pair.

A flip (period doubling) of the period-p orbit is where its critical multiplier is a real -1. Inside a bracket whose
ends lie on either side, it is located by the Illinois variant of regula falsi on mu(lambda) + 1, each orbit
warm-started from the nearest evaluated parameter. Every evaluation classifies the crossing first (AGENTS.md): the
critical multiplier must be real and negative, otherwise the search stops with an error; the next-largest modulus is
recorded. The period-2p orbit is then reached by forward iteration inside the next band, from the p-orbit displaced
along the flip eigenvector, and its minimal period is checked before its own flip is sought. Bracket and switching
points use the last measured interval ratio (4 before any is known) only to place the search; the flip parameters
themselves come from the roots.

Successive flips lambda_n give the interval ratios delta_n = (lambda_n - lambda_{n-1}) / (lambda_{n+1} - lambda_n),
which converge to Feigenbaum's delta = 4.6692016... when the cascade is governed by a one-dimensional reduction with a
quadratic extremum.

Usage (CPU, float64):
  python tools/llm_feigenbaum.py cascade --log-temperature -0.14798 --alpha 0.0650 --free alpha --period 3 \
      --next 0.0656 --flips 6 --threads 16 --out llm-soft-loop/feigenbaum-crescent
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

FEIGENBAUM_DELTA = 4.669201609102990


class SearchError(RuntimeError):
    """A precondition of the cascade search failed (no convergence, wrong crossing type, no bracket)."""


def _rows(Q, X):
    """Columns of Q (n, k) as tangent rows (k, *X.shape[1:]) in X's dtype and device."""
    torch = __import__("torch")
    k = Q.shape[1]
    return torch.as_tensor(np.ascontiguousarray(Q.T).reshape((k,) + tuple(X.shape[1:])), dtype=X.dtype,
                           device=X.device)


def _cols(V):
    """Tangent rows (k, ...) as columns (n, k) of a float64 numpy array."""
    return V.reshape(V.shape[0], -1).double().cpu().numpy().T


def _flat(X):
    return X.reshape(-1).double().cpu().numpy()


def chain(cell, X, Q, period):
    """M Q for the monodromy M of the p-fold map at X: Jacobian-vector products chained along the orbit."""
    V = _rows(Q, X)
    for _ in range(period):
        V = cell.jvp_rows(X, V)
        X = cell.f(X)
    return _cols(V)


def newton(cell, X, period, Q, tol=1e-12, max_iter=12):
    """Newton for F^p(X) = X with M - I acting on span(Q) only (M Q from chained JVPs, M = 0 on the complement).
    Returns (X, Q, relative residual history); Q is refreshed by one subspace-iteration step per Newton step."""
    torch = cell.torch
    k = Q.shape[1]
    history = []
    for it in range(max_iter + 1):
        r = _flat(cell.iterate(X, period)) - _flat(X)
        rel = float(np.linalg.norm(r) / max(np.linalg.norm(_flat(X)), 1e-300))
        history.append(rel)
        if rel < tol or it == max_iter or not np.isfinite(rel):
            break
        W = chain(cell, X, Q, period)
        a = np.linalg.solve(Q.T @ W - np.eye(k), -Q.T @ r)
        w = W @ a + r
        dx = Q @ a + w - Q @ (Q.T @ w)
        X = X + torch.as_tensor(dx.reshape(tuple(X.shape)), dtype=X.dtype, device=X.device)
        Q = np.linalg.qr(W)[0]
    return X, Q, history


def ritz(cell, X, period, Q, passes=3, max_passes=12, res_tol=1e-10, significant=0.05):
    """Leading multipliers of the p-fold map at X by subspace iteration (chained products) and Rayleigh-Ritz on the
    last subspace: at least `passes` passes, continued until the residuals |M v - mu v| / |v| of the critical pair (the
    smallest real part, see Evaluation) and of every pair with modulus above `significant` are below res_tol, or
    max_passes is reached. Returns (multipliers sorted by modulus, Ritz vectors as columns in the same order, residuals
    of the pairs, history of the three leading moduli, refreshed Q)."""
    history = []
    for it in range(max_passes):
        W = chain(cell, X, Q, period)
        vals, Y = np.linalg.eig(Q.T @ W)
        order = np.argsort(-np.abs(vals))
        vals, Y = vals[order], Y[:, order]
        V = Q @ Y
        residuals = np.linalg.norm(W @ Y - V * vals, axis=0) / np.linalg.norm(V, axis=0)
        history.append([float(abs(z)) for z in vals[:3]])
        Q = np.linalg.qr(W)[0]
        watched = (np.abs(vals) > significant) | (np.arange(len(vals)) == int(np.argmin(vals.real)))
        if it + 1 >= passes and float(residuals[watched].max()) < res_tol:
            break
    return vals, V, residuals, history, Q


class Evaluation:
    """The period-p orbit at one parameter value: state, subspace, multipliers. The critical multiplier is the one with
    the smallest real part, the one that can cross -1: it varies continuously along the band (from +1 after the flip
    that created the orbit, through 0, to -1), whereas the largest-modulus multiplier switches identity where the
    critical one passes through 0. g = Re(critical) + 1 changes sign at a flip."""

    def __init__(self, lam, X, Q, vals, vecs, residuals, newton_history, ritz_history, seconds):
        self.lam, self.X, self.Q = float(lam), X, Q
        self.vals, self.vecs, self.residuals = vals, vecs, residuals
        self.newton_history, self.ritz_history, self.seconds = newton_history, ritz_history, seconds
        self.ci = int(np.argmin(vals.real))

    @property
    def critical(self):
        return self.vals[self.ci]

    @property
    def g(self):
        return float(self.critical.real) + 1.0

    @property
    def residual(self):
        return float(self.residuals[self.ci])

    @property
    def next_modulus(self):
        others = np.delete(np.abs(self.vals), self.ci)
        return float(others.max()) if others.size else None

    @property
    def vec(self):
        v = self.vecs[:, self.ci]
        v = np.real(v * np.exp(-1j * np.angle(v[np.argmax(np.abs(v))])))
        return v / np.linalg.norm(v)

    def record(self):
        mu = self.critical
        return {"lambda": self.lam, "critical_multiplier": [float(mu.real), float(mu.imag), float(abs(mu)),
                                                            float(np.angle(mu) / (2 * np.pi))],
                "next_modulus": self.next_modulus,
                "multipliers": [[float(z.real), float(z.imag)] for z in self.vals],
                "ritz_residual": self.residual, "ritz_leading_moduli": self.ritz_history,
                "newton_residuals": self.newton_history, "seconds": round(self.seconds, 1)}


def evaluate(make_cell, lam, X0, period, Q0, tol=1e-12, max_newton=12, passes=3, real_tol=1e-8):
    """Converged period-p orbit at lam from the warm start X0, with its multipliers. Refuses a complex critical
    multiplier near -1 (Re + 1 < 0.5): what approaches the circle there is not a real multiplier, so not a flip."""
    tic = time.perf_counter()
    cell = make_cell(lam)
    X, Q, hist = newton(cell, X0, period, Q0, tol, max_newton)
    if not hist[-1] < tol:
        raise SearchError(f"period-{period} Newton did not converge at lambda = {lam!r}: residuals {hist}")
    vals, vecs, residuals, rhist, Q = ritz(cell, X, period, Q, passes)
    ev = Evaluation(lam, X, Q, vals, vecs, residuals, hist, rhist, time.perf_counter() - tic)
    mu = ev.critical
    if abs(mu.imag) > real_tol and ev.g < 0.5:
        raise SearchError(f"period-{period} critical multiplier at lambda = {lam!r} is {mu!r} (next modulus "
                          f"{ev.next_modulus:.3g}): a complex pair near -1, so no flip can be located here")
    return ev


def locate_flip(make_cell, period, a, b, lam_tol, g_tol=1e-11, max_eval=40, log=print, **kw):
    """Flip of the period-p orbit between evaluations a and b (g = mu + 1 of opposite signs) by the Illinois method.
    Returns (root evaluation, list of all evaluation records)."""
    ga, gb = a.g, b.g
    if not ga * gb < 0:
        raise SearchError(f"no flip bracketed for period {period}: mu + 1 = {ga:.3g} at {a.lam!r} and {gb:.3g} at "
                          f"{b.lam!r}")
    evals = []
    best = a if abs(ga) < abs(gb) else b
    for _ in range(max_eval):
        if abs(b.lam - a.lam) <= lam_tol or abs(best.g) <= g_tol:
            break
        lam = (a.lam * gb - b.lam * ga) / (gb - ga)
        start = a if abs(lam - a.lam) < abs(lam - b.lam) else b
        c = evaluate(make_cell, lam, start.X, period, start.Q, **kw)
        evals.append(c.record())
        log(json.dumps({"period": period, "lambda": c.lam, "mu": float(c.critical.real), "g": c.g,
                        "next_modulus": evals[-1]["next_modulus"], "newton": c.newton_history[-1],
                        "ritz_residual": c.residual, "seconds": round(c.seconds, 1)}))
        gc = c.g
        if gc * gb < 0:
            a, ga = b, gb
        else:
            ga = ga / 2.0
        b, gb = c, gc
        if abs(gc) < abs(best.g):
            best = c
    else:
        raise SearchError(f"flip of period {period} not located within {max_eval} evaluations")
    mu = best.critical
    if abs(mu.imag) > 1e-8 or abs(best.g) > 1e-6 or best.residual > 1e-6 or not best.next_modulus < 1:
        raise SearchError(f"the located root of period {period} at lambda = {best.lam!r} is not a clean flip: critical "
                          f"multiplier {mu!r}, Ritz residual {best.residual:.3g}, next modulus {best.next_modulus}")
    return best, evals


def return_distance(cell, X, q):
    return float(np.linalg.norm(_flat(cell.iterate(X, q)) - _flat(X)) / np.linalg.norm(_flat(X)))


def switch(make_cell, flip, period, lam, displacement, steps, k, tol=1e-12, max_newton=12, min_split=1e-7):
    """The period-2p orbit at lam (inside the doubled band): forward iteration from the p-orbit at the flip
    displaced along the flip eigenvector, then Newton for period 2p. Refuses a result that is still p-periodic."""
    torch = __import__("torch")
    cell = make_cell(lam)
    X = flip.X + torch.as_tensor((displacement * np.linalg.norm(_flat(flip.X)) * flip.vec).reshape(
        tuple(flip.X.shape)), dtype=flip.X.dtype, device=flip.X.device)
    X = cell.iterate(X, steps)
    Q = np.linalg.qr(np.column_stack([flip.vec, flip.Q[:, : k - 1]]))[0]
    X, Q, hist = newton(cell, X, 2 * period, Q, tol, max_newton)
    if not hist[-1] < tol:
        raise SearchError(f"period-{2 * period} Newton after switching at lambda = {lam!r} did not converge: {hist}")
    split = return_distance(cell, X, period)
    if split < min_split:
        raise SearchError(f"switch at lambda = {lam!r} returned to the period-{period} orbit (|F^p x - x| / |x| = "
                          f"{split:.3g}); move lambda closer to the flip or enlarge the displacement")
    return X, Q, hist, split


def interval_ratios(lams):
    d = np.diff(np.asarray(lams, dtype=np.float64))
    return [float(d[i] / d[i + 1]) for i in range(len(d) - 1)]


def cascade(make_cell, X0, Q0, period, lam0, lam_next, flips, direction, steps_per_period=24, k=6,
            displacement=1e-3, tol=1e-12, lam_rel_tol=1e-13, log=print, tokens=None, on_flip=None):
    """Successive flips along the cut, starting from the period-`period` orbit near X0 at lam0; lam_next lies inside
    the first doubled band (beyond the first flip). Returns the list of flip records."""
    s = 1.0 if direction > 0 else -1.0
    a = evaluate(make_cell, lam0, X0, period, Q0, tol)
    log(json.dumps({"start": {"period": period, "lambda": lam0, "mu": float(a.critical.real)}}))
    b = evaluate(make_cell, lam_next, a.X, period, a.Q, tol)
    results, ratio, split = [], 4.0, None
    for n in range(flips):
        width = abs(b.lam - a.lam)
        root, evals = locate_flip(make_cell, period, a, b, lam_tol=max(lam_rel_tol * abs(root_scale(a, b)),
                                                                         1e-9 * width * 1e-3), log=log, tol=tol)
        rec = {"period": period, "flip": root.record(), "evaluations": evals}
        if tokens is not None:
            rec["tokens"] = tokens(make_cell(root.lam), root.X, period)
        results.append(rec)
        lams = [r["flip"]["lambda"] for r in results]
        if len(lams) >= 2:
            if len(lams) >= 3:
                ratio = interval_ratios(lams)[-1]
            width = abs(lams[-1] - lams[-2]) / ratio
        log(json.dumps({"flip": n + 1, "period": period, "lambda": root.lam, "mu": float(root.critical.real),
                        "next_modulus": rec["flip"]["next_modulus"], "ratios": interval_ratios(lams)}))
        if on_flip is not None:
            on_flip(results)
        if n + 1 == flips:
            break
        # the doubled orbit, inside its band, and a bracket for its flip
        if n == 0:
            lam_s = lam_next
            band = abs(lam_next - root.lam) * 2.0
        else:
            band = width
            lam_s = root.lam + s * 0.5 * band                # the middle of the doubled band (estimated)
        X2, Q2, hist, split = switch(make_cell, root, period, lam_s, displacement, steps_per_period * 2 * period,
                                     k, tol)
        displacement = max(split / 2.5, 1e-6)
        period *= 2
        a = evaluate(make_cell, lam_s, X2, period, Q2, tol)
        if a.g <= 0:
            raise SearchError(f"period-{period} orbit at the switching point {lam_s!r} is already beyond its flip "
                              f"(mu = {a.critical.real:.4g}); move the switching point closer to the last flip")
        far = lam_s + s * band                            # beyond the estimated next flip
        for _ in range(4):
            b = evaluate(make_cell, far, a.X, period, a.Q, tol)
            if b.g < 0:
                break
            far = lam_s + s * 2.0 * abs(far - lam_s)
        else:
            raise SearchError(f"no flip of period {period} found up to lambda = {far!r}")
    return results


def root_scale(a, b):
    return max(abs(a.lam), abs(b.lam))


def orbits(args, loop, v, tok, esl):
    """Periodic orbits at given (log10 T, alpha, period) points: forward iteration from the opener state, a check that
    the attractor has the requested period, low-rank Newton and Ritz multipliers; the orbit's leading tokens."""
    rows = []
    for spec in args.points:
        lt_s, a_s, p_s = spec.split(":")
        log_t, alpha, period = float(lt_s), float(a_s), int(p_s)
        tic = time.perf_counter()
        cell = esl.SoftOrbit(loop, alpha * v, 10.0 ** log_t, chunk=args.chunk)
        X = cell.iterate(cell.start, args.transient)
        found = cell.detect_period(X, max(period, 32), 1e-6)
        rec = {"log10_temperature": log_t, "alpha": alpha, "period": period, "detected_period": found}
        if found != period:
            rec["error"] = f"attractor period {found}, not {period}"
            rows.append(rec)
            print(json.dumps(rec), flush=True)
            continue
        Q0 = np.linalg.qr(np.random.default_rng(0).normal(size=(X.numel(), args.k)))[0]
        X, Q, hist = newton(cell, X, period, Q0, args.tol, 12)
        vals, vecs, residuals, rhist, Q = ritz(cell, X, period, Q)
        toks, Y = [], X
        for _ in range(period):
            pr = cell.probabilities(Y).double().cpu().numpy()
            toks.append(tok.decode([int(pr.argmax())]))
            Y = cell.f(Y)
        seps = [return_distance(cell, X, q) for q in range(1, period) if period % q == 0]
        rec.update({"newton_residuals": hist, "converged": hist[-1] < args.tol,
                    "multipliers": [[float(z.real), float(z.imag), float(abs(z))] for z in vals],
                    "ritz_residuals": [float(r) for r in residuals], "divisor_return_distances": seps,
                    "tokens": toks, "seconds": round(time.perf_counter() - tic, 1)})
        rows.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k not in ("ritz_residuals",)}, ensure_ascii=False), flush=True)
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "orbits.json").write_text(json.dumps({"args": vars(args), "orbits": rows}, indent=1, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    o = sub.add_parser("orbits", help="periodic orbits at given points (low-rank Newton, Ritz multipliers, tokens)")
    o.add_argument("--points", nargs="+", required=True, metavar="LOG10T:ALPHA:PERIOD")
    o.add_argument("--model", default="qwen0.5b")
    o.add_argument("--layer", type=int, default=12)
    o.add_argument("--threads", type=int, default=8)
    o.add_argument("--device", default="cpu")
    o.add_argument("--prompt", default="open")
    o.add_argument("--opener", default=None)
    o.add_argument("--window", type=int, default=1)
    o.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    o.add_argument("--concept", default="philosophy")
    o.add_argument("--transient", type=int, default=900)
    o.add_argument("--k", type=int, default=6)
    o.add_argument("--tol", type=float, default=1e-12)
    o.add_argument("--chunk", type=int, default=512)
    o.add_argument("--out", required=True)
    c = sub.add_parser("cascade")
    c.add_argument("--model", default="qwen0.5b")
    c.add_argument("--layer", type=int, default=12)
    c.add_argument("--threads", type=int, default=16)
    c.add_argument("--device", default="cpu")
    c.add_argument("--prompt", default="open")
    c.add_argument("--opener", default=None)
    c.add_argument("--window", type=int, default=1)
    c.add_argument("--vectors", default="llm-phase-map/vectors-l12")
    c.add_argument("--concept", default="philosophy")
    c.add_argument("--log-temperature", type=float, required=True)
    c.add_argument("--alpha", type=float, required=True)
    c.add_argument("--free", choices=("alpha", "temperature"), default="alpha")
    c.add_argument("--period", type=int, required=True, help="period of the orbit at the start")
    c.add_argument("--next", type=float, required=True,
                   help="free-parameter value inside the first doubled band (beyond the first flip)")
    c.add_argument("--flips", type=int, default=6)
    c.add_argument("--transient", type=int, default=900)
    c.add_argument("--k", type=int, default=6, help="dimension of the dominant subspace")
    c.add_argument("--steps-per-period", type=int, default=24,
                   help="forward iterations after a switch, in units of the doubled period")
    c.add_argument("--tol", type=float, default=1e-12)
    c.add_argument("--chunk", type=int, default=512)
    c.add_argument("--out", required=True)
    args = parser.parse_args()

    import explore_llm_attractors as ela
    import explore_llm_phase_map as elpm
    import explore_llm_soft_loop as esl

    opener = args.opener if args.opener is not None else ela.OPENER
    loop = esl.SoftLoop(args.model, args.layer, args.threads, args.prompt, opener, args.window, "float64", args.device)
    v = elpm.load_vectors(args.vectors)[args.concept]
    tok = loop.sm.tok
    if args.command == "orbits":
        return orbits(args, loop, v, tok, esl)

    def make_cell(lam):
        if args.free == "alpha":
            return esl.SoftOrbit(loop, lam * v, 10.0 ** args.log_temperature, chunk=args.chunk)
        return esl.SoftOrbit(loop, args.alpha * v, 10.0 ** lam, chunk=args.chunk)

    def tokens(cell, X, period):
        out = []
        for _ in range(period):
            p = cell.probabilities(X).double().cpu().numpy()
            out.append(tok.decode([int(p.argmax())]))
            X = cell.f(X)
        return out

    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    progress = open(out / "progress.jsonl", "a")

    def log(line):
        print(line, flush=True)
        progress.write(line + "\n")
        progress.flush()

    lam0 = args.alpha if args.free == "alpha" else args.log_temperature
    tic = time.perf_counter()
    cell = make_cell(lam0)
    X = cell.iterate(cell.start, args.transient)
    found = cell.detect_period(X, max(args.period, 32), 1e-6)
    if found != args.period:
        raise SystemExit(f"the attractor at the start has period {found}, not {args.period}: refusing to start")
    Q0 = np.linalg.qr(np.random.default_rng(0).normal(size=(X.numel(), args.k)))[0]

    def save(results):
        lams = [r["flip"]["lambda"] for r in results]
        rec = {"args": vars(args), "feigenbaum_delta": FEIGENBAUM_DELTA, "flips": results,
               "flip_parameters": lams, "intervals": np.diff(lams).tolist(), "ratios": interval_ratios(lams),
               "seconds": round(time.perf_counter() - tic, 1)}
        (out / "cascade.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))

    results = cascade(make_cell, X, Q0, args.period, lam0, args.next, args.flips, np.sign(args.next - lam0),
                      args.steps_per_period, args.k, tol=args.tol, log=log, tokens=tokens, on_flip=save)
    save(results)
    log(json.dumps({"done": True, "ratios": interval_ratios([r["flip"]["lambda"] for r in results])}))


if __name__ == "__main__":
    main()
