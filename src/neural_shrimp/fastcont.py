"""Fast float64 periodic-orbit and fold continuation for two-rate GD maps (exploratory).

The map is ``F(theta) = theta - P grad L(theta)`` with ``P`` holding the hidden
rate on hidden-layer parameters and the output rate on the rest. Parameters
are written ``p = (lx, ly)`` with ``lx = log10 output rate`` and
``ly = log10 hidden rate``. Periodic orbits use single shooting on ``F^N``,
which suits orbits whose multipliers are moderate (this module is meant for
the S2 period-18 locking orbits). First derivatives are analytic
(``DF = I - P H``); fold and flip systems need derivatives of ``M q`` (with
``M`` the monodromy), taken by central differences of analytic products.
Results are numerical proposals, never proofs.
"""

import math

import numpy as np

LN10 = math.log(10.0)


class TwoRateMap:
    """Gradient-descent map of a ``TwoLayerNet`` on fixed data, one parameter vector."""

    def __init__(self, net, x, y):
        self.net = net
        self.x = np.asarray(x, dtype=np.float64)
        self.y = np.asarray(y, dtype=np.float64)
        self.n = net.n_parameters
        hidden = net.hidden_parameter_count
        self.mask_h = np.zeros(self.n)
        self.mask_h[:hidden] = 1.0
        self.mask_o = 1.0 - self.mask_h

    def rates(self, p):
        return 10.0 ** p[1] * self.mask_h + 10.0 ** p[0] * self.mask_o

    def grad(self, theta):
        return self.net.gradient(theta[None, :], self.x, self.y)[0]

    def hess(self, theta):
        return self.net.hessian(theta, self.x, self.y)

    def step(self, theta, p):
        return theta - self.rates(p) * self.grad(theta)

    def step_jac(self, theta, p):
        """F(theta), DF(theta) and dF/dp (n x 2) with p in log10 units."""
        g = self.grad(theta)
        r = self.rates(p)
        jac = np.eye(self.n) - r[:, None] * self.hess(theta)
        dp = np.stack([-(10.0 ** p[0]) * LN10 * self.mask_o * g,
                       -(10.0 ** p[1]) * LN10 * self.mask_h * g], axis=1)
        return theta - r * g, jac, dp

    def iterate(self, theta, p, steps):
        for _ in range(steps):
            theta = self.step(theta, p)
        return theta

    def orbit(self, theta, p, length):
        points = [np.asarray(theta, dtype=np.float64)]
        for _ in range(length):
            points.append(self.step(points[-1], p))
        return np.array(points)

    def period_map(self, theta, p, period):
        """F^N(theta), monodromy M, dF^N/dp (n x 2) and the N+1 orbit points."""
        m = np.eye(self.n)
        dp = np.zeros((self.n, 2))
        t = np.asarray(theta, dtype=np.float64)
        points = [t]
        for _ in range(period):
            t, jac, d = self.step_jac(t, p)
            m = jac @ m
            dp = jac @ dp + d
            points.append(t)
        return t, m, dp, np.array(points)

    def monodromy_vector(self, theta, p, period, q):
        """F^N(theta) and M q by forward Jacobian-vector products only."""
        t = np.asarray(theta, dtype=np.float64)
        v = np.asarray(q, dtype=np.float64)
        for _ in range(period):
            r = self.rates(p)
            g = self.grad(t)
            hv = self.net.hvp(t[None, :], self.x, self.y, v[None, :])[0]
            v = v - r * hv
            t = t - r * g
        return t, v


def divisor_separations(points, period):
    """Distance from the first point to the point after each proper divisor of the period."""
    first = points[0]
    return {k: float(np.linalg.norm(points[k] - first)) for k in range(1, period) if period % k == 0}


class OrbitCurve:
    """Solutions of F^N(theta) = theta with one free parameter (the other fixed).

    Unknowns z = (theta, s) where s is p[free]; the solution set is a curve,
    traced by pseudo-arclength. Closed curves through two folds arise at fixed
    output rate inside a thin locking interval.
    """

    def __init__(self, fmap, period, fixed, free=1):
        self.f, self.N, self.free = fmap, period, free
        self.fixed = float(fixed)

    def params(self, s):
        p = [0.0, 0.0]
        p[self.free] = float(s)
        p[1 - self.free] = self.fixed
        return p

    def residual_jacobian(self, z):
        n = self.f.n
        theta, s = z[:n], z[n]
        t, m, dp, points = self.f.period_map(theta, self.params(s), self.N)
        jac = np.concatenate([m - np.eye(n), dp[:, [self.free]]], axis=1)
        return t - theta, jac, m, points


class FoldCurve:
    """Two-parameter fold (sigma=+1) or flip (sigma=-1) curve of period-N orbits.

    Unknowns z = (theta, q, lx, ly); residual (F^N(theta)-theta, (M - sigma I) q,
    (q.q - 1)/2). Derivatives of M q come from central differences of analytic
    Jacobian-vector products with step ``h``.
    """

    def __init__(self, fmap, period, sigma=1.0, h=1e-6):
        self.f, self.N, self.sigma, self.h = fmap, period, float(sigma), h

    def residual_jacobian(self, z):
        n = self.f.n
        theta, q, p = z[:n], z[n:2 * n], z[2 * n:]
        t, m, dp, points = self.f.period_map(theta, p, self.N)
        mq = m @ q
        res = np.concatenate([t - theta, mq - self.sigma * q, [0.5 * (q @ q - 1.0)]])
        jac = np.zeros((2 * n + 1, 2 * n + 2))
        jac[:n, :n] = m - np.eye(n)
        jac[:n, 2 * n:] = dp
        for j in range(n + 2):
            plus, minus = theta.copy(), theta.copy()
            pp, pm = np.array(p, dtype=np.float64), np.array(p, dtype=np.float64)
            if j < n:
                step = self.h * max(1.0, abs(theta[j]))
                plus[j] += step
                minus[j] -= step
            else:
                step = self.h
                pp[j - n] += step
                pm[j - n] -= step
            _, vp = self.f.monodromy_vector(plus, pp, self.N, q)
            _, vm = self.f.monodromy_vector(minus, pm, self.N, q)
            column = (vp - vm) / (2.0 * step)
            if j < n:
                jac[n:2 * n, j] = column
            else:
                jac[n:2 * n, 2 * n + (j - n)] = column
        jac[n:2 * n, n:2 * n] = m - self.sigma * np.eye(n)
        jac[2 * n, n:2 * n] = q
        return res, jac, m, points


def null_tangent(jac, scale, reference=None):
    """Unit (scaled) null vector of a full-row-rank (k x k+1) Jacobian."""
    a = jac / scale[None, :]
    _, _, vt = np.linalg.svd(a)
    t = vt[-1] / scale
    t /= np.linalg.norm(t * scale)
    if reference is not None and t @ (reference * scale * scale) < 0:
        t = -t
    return t


def newton(system, z0, *, max_iter=12, tol=1e-11, extra=None):
    """Newton on system.residual_jacobian with optional extra rows (A z = b form callable)."""
    z = np.array(z0, dtype=np.float64)
    history = []
    for _ in range(max_iter):
        res, jac, m, points = system.residual_jacobian(z)
        if extra is not None:
            row, value = extra(z)
            res = np.concatenate([res, [value]])
            jac = np.vstack([jac, row[None, :]])
        norm = float(np.linalg.norm(res))
        history.append(norm)
        if not np.isfinite(norm):
            return None, history
        if norm < tol:
            return z, history
        try:
            dz = np.linalg.lstsq(jac, -res, rcond=None)[0]
        except np.linalg.LinAlgError:
            return None, history
        z = z + dz
    res, _, _, _ = system.residual_jacobian(z)
    if extra is not None:
        res = np.concatenate([res, [extra(z)[1]]])
    history.append(float(np.linalg.norm(res)))
    return (z if history[-1] < 10 * tol else None), history


def arclength(system, z0, tangent, scale, *, steps, h0, h_min, h_max, tol=1e-11,
              max_iter=10, stop=None, emit=None):
    """Pseudo-arclength continuation; returns accepted nodes (z, tangent, extras)."""
    nodes = []
    z, t, h = np.array(z0, dtype=np.float64), np.array(tangent, dtype=np.float64), h0
    for _ in range(steps):
        while True:
            prediction = z + h * t

            def extra(w, prediction=prediction, t=t):
                return t * scale * scale, float((w - prediction) @ (t * scale * scale))

            corrected, history = newton(system, prediction, max_iter=max_iter, tol=tol, extra=extra)
            if corrected is not None and np.linalg.norm((corrected - z) * scale) < 2.5 * h:
                break
            h *= 0.5
            if h < h_min:
                return nodes, "step below minimum"
        _, jac, m, points = system.residual_jacobian(corrected)
        t_new = null_tangent(jac, scale, reference=t)
        z, t = corrected, t_new
        node = {"z": z.copy(), "tangent": t.copy(), "monodromy": m, "points": points,
                "newton": history, "h": h}
        nodes.append(node)
        if emit is not None:
            emit(node)
        if stop is not None and stop(node):
            return nodes, "stop condition"
        if len(history) <= 4:
            h = min(h * 1.5, h_max)
    return nodes, "steps exhausted"
