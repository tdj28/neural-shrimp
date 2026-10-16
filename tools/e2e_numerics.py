"""Model-free numerics of run E2E (end-to-end explanation of the windows, their crossing and the return maps).

Everything here acts on numpy arrays and callables, so it is tested on maps with known answers
(tests/test_e2e_numerics.py); tools/e2e_model.py binds it to the soft-token map of Qwen2.5-0.5B-Instruct.

subspace_iteration  dominant invariant subspace and Ritz values of a monodromy M given only its action on blocks
newton_picard       periodic orbits of a strongly contracting map: Newton on the dominant subspace, Picard (plain
                    iteration) on the complement, with the coupling term (Lust et al. 1998)
classify            the type of a unit-circle crossing from the multipliers on its outer side (refuses ambiguity)
fold_extrapolate    a fold from multipliers on the stable branch: (1 - mu)^2 is linear in the parameter near a fold
flip_root           a flip from mu + 1 changing sign, by the secant/regula falsi on bracketing points
covariance_spectrum Sigma_p = E^T (diag p - p p^T) E on the support of p (tokens carrying all but tol of the mass)
interval map tools  for a scalar map sampled on a grid: fixed points, periodic orbits, derivative, Lyapunov exponent
"""
import numpy as np


# ----------------------------------------------------------------------------------------------- linear algebra
def orth(W):
    Q, _ = np.linalg.qr(W)
    return Q


def subspace_iteration(apply_M, n, k, iters=12, Q0=None, seed=0, tol=1e-9):
    """Block subspace iteration on M (n x n, known only through apply_M(V) = M V for V of shape (n, k)).

    Returns (Q, H, ritz, converged, history): Q orthonormal (n, k) spanning the approximate dominant invariant
    subspace, H = Q^T M Q (Rayleigh-Ritz), its eigenvalues sorted by modulus, whether the leading half of the moduli
    changed by less than tol (relative to the largest) over the last two iterations, and the modulus history."""
    rng = np.random.default_rng(seed)
    Q = rng.normal(size=(n, k)) if Q0 is None else np.asarray(Q0, dtype=np.float64).reshape(n, -1)
    if Q.shape[1] < k:
        Q = np.hstack([Q, rng.normal(size=(n, k - Q.shape[1]))])
    Q = orth(Q[:, :k])
    history = []
    converged = False
    for it in range(iters):
        W = np.asarray(apply_M(Q), dtype=np.float64)
        H = Q.T @ W                                  # Rayleigh-Ritz on the current (orthonormal) Q
        vals = np.linalg.eigvals(H)
        vals = vals[np.argsort(-np.abs(vals))]
        history.append(np.abs(vals[:max(1, k // 2)]))
        if len(history) >= 3:
            scale = max(history[-1][0], 1e-300)
            if max(np.abs(history[-1] - history[-2]).max(), np.abs(history[-2] - history[-3]).max()) <= tol * scale:
                converged = True
        if converged or it == iters - 1:
            break
        Q = orth(W)
    return Q, H, vals, converged, history


def dominant_block(Q, H, m):
    """Restrict a Rayleigh-Ritz pair (Q, H = Q^T M Q) to its m leading eigenvalues (a complex pair kept whole) by
    an ordered real Schur form: returns (V, Hm) with V = Q Z[:, :m'] orthonormal and Hm = Z^T H Z [:m', :m']. The
    leading Schur vectors of a block subspace iteration converge at the rate |mu_(k+1) / mu_m|, far faster than
    the whole block, so V is a much better invariant subspace than Q for Newton-Picard."""
    from scipy.linalg import schur

    vals = np.linalg.eigvals(H)
    mods = np.sort(np.abs(vals))[::-1]
    m = min(m, len(mods))
    cutoff = mods[m - 1] * (1 - 1e-12)
    Tm, Z, sdim = schur(H, output="real", sort=lambda re, im: abs(complex(re, im)) >= cutoff)
    return Q @ Z[:, :sdim], Tm[:sdim, :sdim]


def newton_picard(Fp, apply_M, x0, V, H, tol=1e-12, maxit=40, couple=True):
    """Solve Fp(x) = x when M = D Fp has a few dominant multipliers and contracts strongly elsewhere.

    V (n, k) orthonormal spans the dominant invariant subspace and H = V^T M V. Each iteration takes a Picard step
    on the complement, dq = P r with P = I - V V^T and r = Fp(x) - x, then a Newton step on the subspace,
    (H - I) a = -V^T (r + M dq) (the coupling term needs one product M dq; couple=False drops it). The relative
    residual |r| / |x| is computed from Fp itself, so the converged orbit does not depend on V or H.
    Returns (x, residual history, converged)."""
    x = np.asarray(x0, dtype=np.float64).copy()
    k = V.shape[1]
    A = H - np.eye(k)
    hist = []
    for _ in range(maxit + 1):
        r = np.asarray(Fp(x), dtype=np.float64) - x
        rel = float(np.linalg.norm(r) / max(np.linalg.norm(x), 1e-300))
        hist.append(rel)
        if rel < tol:
            return x, hist, True
        if len(hist) > 6 and hist[-1] > 0.9 * hist[-4]:     # stalled or diverging
            break
        if len(hist) == maxit + 1:
            break
        dq = r - V @ (V.T @ r)
        rhs = V.T @ r
        if couple and np.linalg.norm(dq) > 0:
            rhs = rhs + V.T @ np.asarray(apply_M(dq[:, None]), dtype=np.float64)[:, 0]
        a = np.linalg.solve(A, -rhs)
        x = x + dq + V @ a
    return x, hist, False


# ----------------------------------------------------------------------------------------------- crossings
def classify(vals, imag_tol=1e-9):
    """Type of the multiplier of largest modulus: 'complex pair' (Neimark-Sacker), 'real +1' (fold side) or
    'real -1' (flip side), with modulus, argument (turns) and the next-largest modulus (the conjugate excluded)."""
    vals = np.asarray(vals, dtype=complex)
    vals = vals[np.argsort(-np.abs(vals))]
    lead = vals[0]
    if abs(lead.imag) > imag_tol * max(1.0, abs(lead)):
        kind = "complex pair"
        rest = [v for v in vals[1:] if not np.isclose(v, np.conj(lead))]
    else:
        kind = "real +1" if lead.real > 0 else "real -1"
        rest = list(vals[1:])
    nxt = float(np.abs(rest[0])) if rest else None
    return {"type": kind, "multiplier": [float(lead.real), float(lead.imag)], "modulus": float(abs(lead)),
            "argument_turns": float(np.angle(lead) / (2 * np.pi)), "next_modulus": nxt}


def fold_extrapolate(params, mus, use=4):
    """Fold location from the leading real multiplier mu(lambda) < 1 on the stable branch approaching it: near a
    fold 1 - mu ~ 2 sqrt(ab |lambda_c - lambda|), so (1 - mu)^2 is linear in lambda. Fits the last `use` points;
    returns (lambda_c, slope, fit residual relative to the largest (1 - mu)^2). Refuses complex or negative mu."""
    params = np.asarray(params, dtype=np.float64)[-use:]
    mus = np.asarray(mus, dtype=complex)[-use:]
    if np.any(np.abs(mus.imag) > 1e-9) or np.any(mus.real <= 0) or np.any(mus.real >= 1):
        raise ValueError(f"fold extrapolation needs real multipliers in (0, 1), got {mus}")
    y = (1.0 - mus.real) ** 2
    c = np.polyfit(params, y, 1)
    if c[0] == 0:
        raise ValueError("flat (1 - mu)^2: no fold ahead")
    lam_c = -c[1] / c[0]
    res = float(np.abs(np.polyval(c, params) - y).max() / y.max())
    return float(lam_c), float(c[0]), res


def flip_root(l0, m0, l1, m1):
    """Linear interpolation of mu + 1 = 0 between two parameters whose real multipliers bracket -1."""
    f0, f1 = m0 + 1.0, m1 + 1.0
    if f0 * f1 > 0:
        raise ValueError("multipliers do not bracket -1")
    return l0 - f0 * (l1 - l0) / (f1 - f0)


# ----------------------------------------------------------------------------------------------- readout covariance
def support(p, tol=1e-12):
    """Indices of the most probable tokens holding all but `tol` of the probability."""
    order = np.argsort(-p)
    c = np.cumsum(p[order])
    m = int(np.searchsorted(c, 1.0 - tol)) + 1
    return order[:min(m, len(p))]


def covariance_spectrum(E, p, tol=1e-12, k=8):
    """Leading eigenvalues and eigenvectors of Sigma_p = Cov_{t ~ p}(e_t), computed on the support of p.
    Returns (eigenvalues (k,), eigenvectors (d, k), trace, number of support tokens)."""
    idx = support(p, tol)
    q = p[idx] / p[idx].sum()
    Es = E[idx]
    m = q @ Es
    C = (Es * q[:, None]).T @ Es - np.outer(m, m)
    w, U = np.linalg.eigh(C)
    order = np.argsort(-w)[:k]
    return w[order], U[:, order], float(np.trace(C)), len(idx)


def pair_alignment(u, ea, eb):
    """|cos| of the angle between a unit vector u and the difference of two embeddings e_a - e_b."""
    d = ea - eb
    return float(abs(u @ d) / (np.linalg.norm(u) * np.linalg.norm(d)))


# ----------------------------------------------------------------------------------------------- interval maps
class GridMap:
    """A scalar map f on [lo, hi] given by samples (s_i, f(s_i)) on an increasing grid; cubic-spline interpolation
    with its derivative (scipy CubicSpline, not-a-knot)."""

    def __init__(self, s, fs):
        from scipy.interpolate import CubicSpline

        s = np.asarray(s, dtype=np.float64)
        order = np.argsort(s)
        s, fs = s[order], np.asarray(fs, dtype=np.float64)[order]
        if np.any(np.diff(s) <= 0):
            raise ValueError("grid must be strictly increasing (single-valued coordinate)")
        self.s, self.fs = s, fs
        self.lo, self.hi = float(s[0]), float(s[-1])
        self.spline = CubicSpline(s, fs)
        self.dspline = self.spline.derivative()

    def __call__(self, x):
        return self.spline(x)

    def deriv(self, x):
        return self.dspline(x)

    def iterate(self, x, n):
        out = [float(x)]
        for _ in range(n):
            x = float(self(x))
            if not (self.lo - 1e-12 <= x <= self.hi + 1e-12):
                raise ValueError(f"orbit left the sampled interval at {x}")
            x = min(max(x, self.lo), self.hi)
            out.append(x)
        return np.array(out)

    def lyapunov(self, x0, transient=1000, steps=20000):
        xs = self.iterate(x0, transient + steps)[transient:-1]
        return float(np.mean(np.log(np.maximum(np.abs(self.deriv(xs)), 1e-300))))

    def periodic_orbits(self, period, grid=4000, tol=1e-12):
        """All solutions of f^period(x) = x found from sign changes of f^period(x) - x on a grid, refined by
        bisection; each returned once per orbit (smallest point), primitive period only, with its multiplier
        prod f'(x_i)."""
        from scipy.optimize import brentq

        def g(x):
            y = x
            for _ in range(period):
                y = float(self(y))
                y = min(max(y, self.lo), self.hi)
            return y - x

        xs = np.linspace(self.lo, self.hi, grid)
        gs = np.array([g(x) for x in xs])
        roots = []
        for i in range(grid - 1):
            if gs[i] == 0:
                roots.append(xs[i])
            elif gs[i] * gs[i + 1] < 0:
                roots.append(brentq(g, xs[i], xs[i + 1], xtol=tol))
        orbits = []
        for r in roots:
            pts = self.iterate(r, period - 1)
            prim = next(q for q in range(1, period + 1) if period % q == 0 and abs(
                self.iterate(r, q)[-1] - r) < 1e-8 * max(1.0, self.hi - self.lo))
            if prim != period:
                continue
            key = round(float(pts.min()), 9)
            if any(abs(key - o["min_point"]) < 1e-8 * max(1.0, self.hi - self.lo) for o in orbits):
                continue
            mult = float(np.prod(self.deriv(pts)))
            orbits.append({"min_point": key, "points": pts.tolist(), "multiplier": mult})
        return orbits

    def extrema(self):
        """Interior critical points of the spline (sign changes of f' on the sample grid, refined)."""
        from scipy.optimize import brentq

        d = self.deriv(self.s)
        out = []
        for i in range(len(self.s) - 1):
            if d[i] * d[i + 1] < 0:
                c = brentq(self.deriv, self.s[i], self.s[i + 1])
                out.append({"at": float(c), "value": float(self(c)), "kind": "max" if d[i] > 0 else "min"})
        return out


def logistic_grid(r, n=2001):
    """Samples of the logistic map x -> r x (1 - x) on [0, 1] (positive control for GridMap)."""
    s = np.linspace(0.0, 1.0, n)
    return s, r * s * (1 - s)
