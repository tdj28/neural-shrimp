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


def tol_for(period, tol):
    """Newton tolerance for the p-fold map: float64 rounding accumulates along the orbit, and the relative residual of
    the 96-step crescent orbit stagnated at 1.3e-12 to 6.7e-12 (run C2), above a fixed 1e-12; so the tolerance is
    max(tol, 1e-13 * p), which stays above the measured floor (AGENTS.md: tolerances above the measured error)."""
    return max(float(tol), 1e-13 * int(period))


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


class Collapsed(SearchError):
    """Newton for period p converged to the period-p/2 orbit (also p-periodic), not to the doubled orbit."""


def evaluate(make_cell, lam, X0, period, Q0, tol=1e-12, max_newton=12, passes=3, real_tol=1e-8, half=None,
             min_split=1e-7):
    """Converged period-p orbit at lam from the warm start X0, with its multipliers. Refuses a complex critical
    multiplier near -1 (Re + 1 < 0.5): what approaches the circle there is not a real multiplier, so not a flip. With
    `half` (= p/2 for a doubled orbit), refuses a solution with |F^half x - x| / |x| < min_split: just beyond the flip
    that created the doubled orbit, the half-period orbit is a period-p solution too (multiplier mu_half^2 > 1), and
    Newton can slide onto it (run C3, period 96)."""
    tic = time.perf_counter()
    cell = make_cell(lam)
    tol = tol_for(period, tol)
    X, Q, hist = newton(cell, X0, period, Q0, tol, max_newton)
    if not hist[-1] < tol:
        raise SearchError(f"period-{period} Newton did not converge at lambda = {lam!r}: residuals {hist}")
    if half:
        split = return_distance(cell, X, half)
        if split < min_split:
            raise Collapsed(f"period-{period} solution at lambda = {lam!r} is the period-{half} orbit "
                            f"(|F^{half} x - x| / |x| = {split:.3g})")
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
        first, second = (a, b) if abs(lam - a.lam) < abs(lam - b.lam) else (b, a)
        try:
            c = evaluate(make_cell, lam, first.X, period, first.Q, **kw)
        except Collapsed:
            try:                                           # warm start from the other bracket end
                c = evaluate(make_cell, lam, second.X, period, second.Q, **kw)
            except Collapsed:                              # bisect instead, from the nearer end
                lam = 0.5 * (a.lam + b.lam)
                c = evaluate(make_cell, lam, first.X, period, first.Q, **kw)
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
    tol = tol_for(2 * period, tol)
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
    half = None                                           # the start orbit is not checked against a half period
    a = evaluate(make_cell, lam0, X0, period, Q0, tol)
    log(json.dumps({"start": {"period": period, "lambda": lam0, "mu": float(a.critical.real)}}))
    b = evaluate(make_cell, lam_next, a.X, period, a.Q, tol)
    results, ratio, split = [], 4.0, None
    for n in range(flips):
        width = abs(b.lam - a.lam)
        root, evals = locate_flip(make_cell, period, a, b, lam_tol=max(lam_rel_tol * abs(root_scale(a, b)),
                                                                         1e-9 * width * 1e-3), log=log, tol=tol,
                                  half=half)
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
        half = period // 2
        a = evaluate(make_cell, lam_s, X2, period, Q2, tol, half=half)
        if a.g <= 0:
            raise SearchError(f"period-{period} orbit at the switching point {lam_s!r} is already beyond its flip "
                              f"(mu = {a.critical.real:.4g}); move the switching point closer to the last flip")
        # walk from the switching point toward and beyond the estimated next flip by continuation in small
        # substeps (a single long jump lets Newton slide onto the half-period orbit); stop at the first point past
        # the flip (g < 0), which brackets it together with the previous point
        sub = 0.25 * band
        prev, b = a, None
        for _ in range(16):
            cur = evaluate(make_cell, prev.lam + s * sub, prev.X, period, prev.Q, tol, half=half)
            if cur.g < 0:
                a, b = prev, cur
                break
            prev = cur
        if b is None:
            raise SearchError(f"no flip of period {period} found up to lambda = {prev.lam!r}")
    return results


def root_scale(a, b):
    return max(abs(a.lam), abs(b.lam))


def edges(make_cell, X0, Q0, period, lam0, step, tol=1e-12, min_step=1e-10, fit_points=4, max_steps=200, log=print,
          directions=(+1.0, -1.0)):
    """Edges of the window of the period-p orbit found near X0 at lam0, walking in both directions.

    In each direction the parameter advances by `step` (halved whenever Newton fails, i.e. beyond a fold) while the
    critical multipliers are recorded. A flip edge (min Re mu crossing -1) is located by the Illinois method. A fold edge
    is where the orbit ceases to exist: the multiplier nearest +1 tends to 1 and (1 - mu)^2 is linear in the parameter
    near the fold (normal form), so the fold is estimated by a linear fit of (1 - mu)^2 over the last `fit_points`
    converged points, extrapolated to zero; the last converged parameter is reported with it. For an even period the
    walk stops where the orbit collapses onto its half-period orbit (the flip that created it; Newton would otherwise
    continue on the half-period orbit, which is also p-periodic). Returns one record per direction."""
    out = []
    half = period // 2 if period % 2 == 0 else None
    for direction in directions:
        a = evaluate(make_cell, lam0, X0, period, Q0, tol, half=half)
        h, pts, rec = float(step), [], {"direction": direction}
        pts.append((a.lam, float(np.max(a.vals.real))))
        for _ in range(max_steps):
            lam = a.lam + direction * h
            try:
                b = evaluate(make_cell, lam, a.X, period, a.Q, tol, half=half)
            except Collapsed:
                rec.update({"kind": "collapse onto the half-period orbit (birth flip)", "between": [a.lam, lam],
                            "last_mu_max": float(np.max(a.vals.real))})
                break
            except SearchError:
                if h / 2 < min_step:
                    lams = np.array([q[0] for q in pts[-fit_points:]])
                    y = (1.0 - np.array([q[1] for q in pts[-fit_points:]])) ** 2
                    slope, icpt = np.polyfit(lams, y, 1)
                    rec.update({"kind": "fold", "lambda": float(-icpt / slope), "last_converged": a.lam,
                                "last_mu_max": float(np.max(a.vals.real)), "fit_points": [list(q) for q in pts[-fit_points:]]})
                    break
                h /= 2
                continue
            if b.g < 0:                                       # a real multiplier passed -1 between a and b: a flip
                root, evals = locate_flip(make_cell, period, a, b, lam_tol=1e-13 * abs(lam), log=lambda s: None,
                                          tol=tol)
                rec.update({"kind": "flip", "lambda": root.lam, "record": root.record()})
                break
            mu_max = float(np.max(b.vals.real))
            if mu_max > 1.0:                                  # a real multiplier passed +1 without the orbit vanishing
                rec.update({"kind": "real +1 crossing", "between": [a.lam, b.lam], "mu_max": mu_max})
                break
            pts.append((b.lam, mu_max))
            a = b
        else:
            rec.update({"kind": "none within max_steps", "last": a.lam})
        log(json.dumps({k: v for k, v in rec.items() if k not in ("record", "fit_points")}))
        out.append(rec)
    return out


def orbits(args, loop, v, tok, esl):
    """Periodic orbits at given (log10 T, alpha, period) points: forward iteration from the opener state, a check that
    the attractor has the requested period, low-rank Newton and Ritz multipliers; the orbit's leading tokens."""
    rows = []
    for spec in args.points:
        p_s, lt_s, a_s = spec.split(":")                       # PERIOD first: argparse reads "-0.1..." as an option
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


def superstable(make_cell, X0, Q0, period, lam_a, lam_b, settle, tol=1e-12, half=None, lam_rel_tol=1e-13,
                max_eval=40, log=print):
    """Superstable parameter of the period-p orbit inside its band [lam_a, lam_b], where its critical (leading real)
    multiplier changes sign: + after the flip or fold that creates the orbit, - before the flip that ends it.

    Illinois method on the critical multiplier. Every evaluation first settles the state by `settle` forward iterations
    at the new parameter: inside the band the orbit attracts, so the state cannot slide onto the repelling half-period
    orbit (the failure of continuation near a flip). An evaluation counts as clean when the critical multiplier
    dominates (|mu_1| > 10 |mu_2|). Near the root the critical multiplier is smaller than the transverse ones; the search
    stops at the first unclean evaluation and returns the secant root of the closest clean points on either side (mu_c
    is smooth in the parameter), so the resolution is set by the clean pair, not by the transverse spectrum.
    Returns (root record, evaluation records)."""

    def ev(lam, X, Q):
        cell = make_cell(lam)
        Y = cell.iterate(X, settle)
        e = evaluate(make_cell, lam, Y, period, Q, tol, half=half)
        mu = e.vals[0]
        clean = bool(len(e.vals) < 2 or abs(mu) > 10.0 * abs(e.vals[1]))
        if clean and abs(mu.imag) > 1e-8:
            raise SearchError(f"critical multiplier of period {period} at {lam!r} is complex ({mu!r})")
        return e, float(mu.real), clean

    a, fa, ca = ev(lam_a, X0, Q0)
    b, fb, cb = ev(lam_b, a.X, a.Q)
    log(json.dumps({"period": period, "bracket": [lam_a, lam_b], "mu": [fa, fb]}))
    if not (ca and cb and fa * fb < 0):
        raise SearchError(f"no clean sign change of the critical multiplier of period {period} on "
                          f"[{lam_a!r}, {lam_b!r}]: {fa:.3g} ({ca}), {fb:.3g} ({cb})")
    plus, minus = ((a, fa), (b, fb)) if fa > 0 else ((b, fb), (a, fa))     # closest clean points on either side
    records, how = [], "converged"
    for _ in range(max_eval):
        if abs(b.lam - a.lam) <= lam_rel_tol * abs(b.lam):
            break
        lam = (a.lam * fb - b.lam * fa) / (fb - fa)
        start = a if abs(lam - a.lam) < abs(lam - b.lam) else b
        c, fc, cc = ev(lam, start.X, start.Q)
        records.append({"lambda": c.lam, "mu": fc, "clean": cc,
                        "next_modulus": float(abs(c.vals[1])) if len(c.vals) > 1 else None})
        log(json.dumps({"period": period, "lambda": c.lam, "mu": fc, "clean": cc}))
        if not cc:
            how = "stopped at the transverse spectrum; secant of the closest clean points"
            break
        if fc > 0:
            plus = (c, fc)
        else:
            minus = (c, fc)
        if abs(fc) < 1e-14:
            break
        if fc * fb < 0:
            a, fa = b, fb
        else:
            fa /= 2.0
        b, fb = c, fc
    else:
        raise SearchError(f"superstable parameter of period {period} not located within {max_eval} evaluations")
    (pe, pf), (me, mf) = plus, minus
    root = (pe.lam * mf - me.lam * pf) / (mf - pf)
    return {"period": period, "lambda": root, "how": how, "clean_bracket": [pe.lam, me.lam], "mu": [pf, mf],
            "record": pe.record()}, records


def superstable_cmd(args, loop, v, tok, esl):
    """Superstable parameters of a sequence of bands along alpha at fixed log10 T (points PERIOD:LOG10T:ALPHA_A:ALPHA_B);
    each band starts from the attractor reached from the opener state at ALPHA_A."""
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in args.points:
        p_s, lt_s, a_s, b_s = spec.split(":")
        period, log_t, lam_a, lam_b = int(p_s), float(lt_s), float(a_s), float(b_s)

        def make_cell(lam, log_t=log_t):
            return esl.SoftOrbit(loop, lam * v, 10.0 ** log_t, chunk=args.chunk)

        cell = make_cell(lam_a)
        X = cell.iterate(cell.start, args.transient)
        found = cell.detect_period(X, max(period, 32), 1e-6)
        rec = {"period": period, "log10_temperature": log_t, "bracket": [lam_a, lam_b], "detected_period": found}
        if found != period:
            rec["error"] = f"attractor period {found}, not {period}"
        else:
            Q0 = np.linalg.qr(np.random.default_rng(0).normal(size=(X.numel(), args.k)))[0]
            try:
                root, evals = superstable(make_cell, X, Q0, period, lam_a, lam_b, args.settle_factor * period,
                                          args.tol, half=(period // 2 if period % 2 == 0 else None), log=print)
                rec.update({"superstable": root["lambda"], "how": root["how"], "clean_bracket": root["clean_bracket"],
                            "mu": root["mu"], "record": root["record"], "evaluations": evals})
            except SearchError as err:
                rec["error"] = str(err)
        rows.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k not in ("record", "evaluations")}), flush=True)
        lams = [r["superstable"] for r in rows if "superstable" in r]
        (out / "superstable.json").write_text(json.dumps({"args": vars(args), "bands": rows, "superstable": lams,
                                                          "ratios": interval_ratios(lams) if len(lams) > 2 else []},
                                                         indent=1))


def window_edges(args, loop, v, tok, esl):
    """Fold and flip edges of periodic windows along alpha at fixed log10 T, from given window centres."""
    rows = []
    out = esl.DATA / "explore" / args.out
    out.mkdir(parents=True, exist_ok=True)
    for spec in args.points:
        p_s, lt_s, a_s, h_s = spec.split(":")
        period, log_t, alpha, step = int(p_s), float(lt_s), float(a_s), float(h_s)
        if args.free == "alpha":
            def make_cell(lam, log_t=log_t):
                return esl.SoftOrbit(loop, lam * v, 10.0 ** log_t, chunk=args.chunk)
            lam0 = alpha
        else:
            def make_cell(lam, alpha=alpha):
                return esl.SoftOrbit(loop, alpha * v, 10.0 ** lam, chunk=args.chunk)
            lam0 = log_t

        cell = make_cell(lam0)
        X = cell.iterate(cell.start, args.transient)
        found = cell.detect_period(X, max(period, 32), 1e-6)
        rec = {"period": period, "log10_temperature": log_t, "alpha": alpha, "free": args.free, "step": step,
               "detected_period": found}
        if found != period:
            rec["error"] = f"attractor period {found}, not {period}"
        else:
            Q0 = np.linalg.qr(np.random.default_rng(0).normal(size=(X.numel(), args.k)))[0]
            tic = time.perf_counter()
            try:
                dirs = {"both": (+1.0, -1.0), "up": (+1.0,), "down": (-1.0,)}[args.directions]
                rec["edges"] = edges(make_cell, X, Q0, period, lam0, step, args.tol, args.min_step, log=print,
                                     directions=dirs)
            except SearchError as err:
                rec["error"] = str(err)
            rec["seconds"] = round(time.perf_counter() - tic, 1)
        rows.append(rec)
        print(json.dumps({k: v for k, v in rec.items() if k != "edges"}), flush=True)
        (out / "edges.json").write_text(json.dumps({"args": vars(args), "windows": rows}, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ss = sub.add_parser("superstable", help="superstable parameters of a sequence of bands along alpha")
    ss.add_argument("--points", nargs="+", required=True, metavar="PERIOD:LOG10T:ALPHA_A:ALPHA_B")
    ss.add_argument("--settle-factor", type=int, default=20, help="forward iterations per evaluation, in periods")
    for name, kw in (("--model", dict(default="qwen0.5b")), ("--layer", dict(type=int, default=12)),
                     ("--threads", dict(type=int, default=8)), ("--device", dict(default="cpu")),
                     ("--prompt", dict(default="open")), ("--opener", dict(default=None)),
                     ("--window", dict(type=int, default=1)), ("--vectors", dict(default="llm-phase-map/vectors-l12")),
                     ("--concept", dict(default="philosophy")), ("--transient", dict(type=int, default=3000)),
                     ("--k", dict(type=int, default=6)), ("--tol", dict(type=float, default=1e-12)),
                     ("--chunk", dict(type=int, default=512)), ("--out", dict(required=True))):
        ss.add_argument(name, **kw)
    e = sub.add_parser("edges", help="fold and flip edges of windows along alpha (low-rank solver)")
    e.add_argument("--points", nargs="+", required=True, metavar="PERIOD:LOG10T:ALPHA:STEP")
    e.add_argument("--min-step", type=float, default=1e-10)
    e.add_argument("--free", choices=("alpha", "temperature"), default="alpha")
    e.add_argument("--directions", choices=("both", "up", "down"), default="both")
    for name, kw in (("--model", dict(default="qwen0.5b")), ("--layer", dict(type=int, default=12)),
                     ("--threads", dict(type=int, default=8)), ("--device", dict(default="cpu")),
                     ("--prompt", dict(default="open")), ("--opener", dict(default=None)),
                     ("--window", dict(type=int, default=1)), ("--vectors", dict(default="llm-phase-map/vectors-l12")),
                     ("--concept", dict(default="philosophy")), ("--transient", dict(type=int, default=900)),
                     ("--k", dict(type=int, default=6)), ("--tol", dict(type=float, default=1e-12)),
                     ("--chunk", dict(type=int, default=512)), ("--out", dict(required=True))):
        e.add_argument(name, **kw)
    o = sub.add_parser("orbits", help="periodic orbits at given points (low-rank Newton, Ritz multipliers, tokens)")
    o.add_argument("--points", nargs="+", required=True, metavar="PERIOD:LOG10T:ALPHA")
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
    if args.command == "edges":
        return window_edges(args, loop, v, tok, esl)
    if args.command == "superstable":
        return superstable_cmd(args, loop, v, tok, esl)

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
