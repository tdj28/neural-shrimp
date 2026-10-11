"""Minimal gradient-descent models for the Neimark-Sacker question (exploratory).

``TwoModeFactorization``: L = 1/2 z^T C z - d^T z + (lam/2)(|a|^2 + |b|^2) with
z = (a1 b1, a2 b2) and C = [[1, c], [c, 1]]. The target is d = C z* + lam 1, so
a = b = sqrt(z*) is a strict minimum for every |c| < 1. Gradient descent uses
rate ``eta_a`` on (a1, a2) and ``eta_b`` on (b1, b2); parameters are ordered
(a1, a2, b1, b2).

``EngineeredTwoCycle``: a planar gradient map x -> x - grad V(x) whose potential
is a quartic Hermite interpolant fixing the 2-cycle p0 = (1, 0) <-> p1 = (-1, 0)
with symmetric step derivatives S0 = diag(1, -1) and S1 = r R_phi S0, so that
the two-step derivative at p0 is r R_phi (multipliers r exp(+-i phi)). A term
kappa A^2 B^2, with A and B the squared distances to p0 and p1, leaves the
2-cycle and its first and second derivatives unchanged and adds cubic terms to
the two-step map; these are non-resonant, so the first Lyapunov coefficient of
the Neimark-Sacker bifurcation at r = 1 does not depend on kappa (d = -8.061
for every kappa tested, docs/explore-criticality-2026-10-07.md): the onset is
supercritical throughout. It is a positive control showing that two
coordinates suffice for a gradient-map 2-cycle to have complex multipliers.

``follow_orbit``: scalar-parameter predictor-corrector continuation of an
attracting periodic orbit (exact Newton corrector from
``neural_shrimp.orbits.certify_orbit``) that records every multiplier along
the path and types the boundary where the orbit stops attracting.
"""

from dataclasses import dataclass

import numpy as np

from .continuation import classify_multiplier
from .orbits import certify_orbit


@dataclass(frozen=True)
class TwoModeFactorization:
    c: float
    lam: float
    z_star: tuple = (1.0, 0.5)

    @property
    def C(self):
        return np.array([[1.0, self.c], [self.c, 1.0]])

    @property
    def d(self):
        return self.C @ np.asarray(self.z_star, dtype=np.float64) + self.lam

    @property
    def theta_star(self):
        s = np.sqrt(np.asarray(self.z_star, dtype=np.float64))
        return np.concatenate([s, s])

    def loss(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        a, b = theta[..., :2], theta[..., 2:]
        z = a * b
        quad = 0.5 * np.einsum("...i,ij,...j->...", z, self.C, z)
        return quad - z @ self.d + 0.5 * self.lam * np.sum(theta * theta, axis=-1)

    def gradient(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        a, b = theta[..., :2], theta[..., 2:]
        g = (a * b) @ self.C.T - self.d
        return np.concatenate([g * b + self.lam * a, g * a + self.lam * b], axis=-1)

    def hessian(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        a, b = theta[:2], theta[2:]
        C = self.C
        g = C @ (a * b) - self.d
        h = np.empty((4, 4))
        h[:2, :2] = C * np.outer(b, b) + self.lam * np.eye(2)
        h[2:, 2:] = C * np.outer(a, a) + self.lam * np.eye(2)
        h[:2, 2:] = C * np.outer(b, a) + np.diag(g)
        h[2:, :2] = h[:2, 2:].T
        return h

    @staticmethod
    def rate_vector(eta_a, eta_b):
        return np.array([eta_a, eta_a, eta_b, eta_b], dtype=np.float64)

    def step(self, theta, eta_a, eta_b):
        return np.asarray(theta, dtype=np.float64) - self.rate_vector(eta_a, eta_b) * self.gradient(theta)

    def jacobian(self, theta, eta_a, eta_b):
        return np.eye(4) - self.rate_vector(eta_a, eta_b)[:, None] * self.hessian(theta)

    def flip_rate(self, ratio):
        """Rate eta_a (with eta_b = ratio * eta_a) at which the minimum flips."""
        scale = np.sqrt(self.rate_vector(1.0, ratio))
        sym = scale[:, None] * self.hessian(self.theta_star) * scale[None, :]
        return 2.0 / float(np.linalg.eigvalsh(sym)[-1])

    def critical_direction(self, ratio):
        """Eigenvector of P H at the minimum whose eigenvalue reaches 2 first."""
        values, vectors = np.linalg.eig(self.rate_vector(1.0, ratio)[:, None] * self.hessian(self.theta_star))
        return np.real(vectors[:, int(np.argmax(values.real))])

    def map_pair(self, eta_a, eta_b):
        return (lambda x: self.step(x, eta_a, eta_b)), (lambda x: self.jacobian(x, eta_a, eta_b))


def _monomials(degree=4):
    return [(i, j) for total in range(1, degree + 1) for i in range(total + 1) for j in [total - i]]


class EngineeredTwoCycle:
    """Planar gradient map with a prescribed 2-cycle and two-step derivative r R_phi."""

    p0 = np.array([1.0, 0.0])
    p1 = np.array([-1.0, 0.0])

    def __init__(self, r, phi, kappa=0.0):
        self.r, self.phi, self.kappa = float(r), float(phi), float(kappa)
        self.terms = _monomials(4)
        s0 = np.diag([1.0, -1.0])
        reflection = np.array([[np.cos(phi), np.sin(phi)], [np.sin(phi), -np.cos(phi)]])
        s1 = self.r * reflection
        rows, rhs = [], []
        for point, gradient, step in ((self.p0, self.p0 - self.p1, s0), (self.p1, self.p1 - self.p0, s1)):
            hess = np.eye(2) - step
            rows += [self._row(point, (1, 0)), self._row(point, (0, 1)), self._row(point, (2, 0)),
                     self._row(point, (1, 1)), self._row(point, (0, 2))]
            rhs += [gradient[0], gradient[1], hess[0, 0], hess[0, 1], hess[1, 1]]
        self.coefficients = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]

    def _row(self, point, derivative):
        x, y = point
        dx, dy = derivative
        row = []
        for i, j in self.terms:
            if i < dx or j < dy:
                row.append(0.0)
                continue
            factor = np.prod(np.arange(i, i - dx, -1)) * np.prod(np.arange(j, j - dy, -1))
            row.append(factor * x ** (i - dx) * y ** (j - dy))
        return row

    def _poly_derivative(self, point, derivative):
        return float(np.dot(self._row(point, derivative), self.coefficients))

    def gradient(self, x):
        x = np.asarray(x, dtype=np.float64)
        g = np.array([self._poly_derivative(x, (1, 0)), self._poly_derivative(x, (0, 1))])
        if self.kappa:
            a, b = np.sum((x - self.p0) ** 2), np.sum((x - self.p1) ** 2)
            da, db = 2.0 * (x - self.p0), 2.0 * (x - self.p1)
            g = g + self.kappa * (2.0 * a * b * b * da + 2.0 * a * a * b * db)
        return g

    def hessian(self, x):
        x = np.asarray(x, dtype=np.float64)
        hxx, hxy, hyy = (self._poly_derivative(x, (2, 0)), self._poly_derivative(x, (1, 1)),
                         self._poly_derivative(x, (0, 2)))
        h = np.array([[hxx, hxy], [hxy, hyy]])
        if self.kappa:
            a, b = np.sum((x - self.p0) ** 2), np.sum((x - self.p1) ** 2)
            da, db = 2.0 * (x - self.p0), 2.0 * (x - self.p1)
            eye2 = 2.0 * np.eye(2)
            # d/dx of 2 a b^2 da + 2 a^2 b db
            h = h + self.kappa * (2.0 * b * b * np.outer(da, da) + 4.0 * a * b * np.outer(da, db)
                                  + 2.0 * a * b * b * eye2 + 4.0 * a * b * np.outer(db, da)
                                  + 2.0 * a * a * np.outer(db, db) + 2.0 * a * a * b * eye2)
        return h

    def step(self, x):
        return np.asarray(x, dtype=np.float64) - self.gradient(x)

    def jacobian(self, x):
        return np.eye(2) - self.hessian(x)


def dominant_pair(multipliers):
    """Return (dominant multiplier, whether the two largest moduli form a complex pair)."""
    mus = sorted((complex(m) for m in multipliers), key=abs, reverse=True)
    complex_pair = len(mus) > 1 and abs(mus[0].imag) > 1e-9 and abs(mus[0] - mus[1].conjugate()) < 1e-7
    return mus[0], complex_pair


def follow_orbit(family, x0, period, s_max, h0, h_min=None, h_max=None, bisection_tolerance=1e-10,
                 unit_band=0.02, cert_options=None, max_steps=4000):
    """Follow an attracting period-``period`` orbit for parameter s in [0, s_max].

    ``family(s)`` returns ``(map_fn, jacobian)``. Returns the path (s, all
    multipliers, first orbit point), the boundary bracket, its type and the
    dominant multiplier at the last attracting point.
    """
    h_min = h0 / 512.0 if h_min is None else h_min
    h_max = 8.0 * h0 if h_max is None else h_max
    options = {"candidate_tolerance": 0.05, **(cert_options or {})}

    def certify(s, seed):
        map_fn, jacobian = family(s)
        return certify_orbit(map_fn, jacobian, seed, period, **options)

    def multipliers(cert):
        return [complex(m["real"], m["imag"]) for m in cert["multipliers"]]

    start = certify(0.0, np.asarray(x0, dtype=np.float64))
    if not (start["verified"] and start["attracting"]):
        return {"type": "start_not_attracting", "reason": start.get("reason"), "path": [],
                "s_boundary": None, "bracket": None, "dominant": None}
    path = [(0.0, multipliers(start), np.asarray(start["orbit"][0]))]
    s, h, failure = 0.0, h0, None
    for _ in range(max_steps):
        if s >= s_max:
            break
        trial = min(s + h, s_max)
        state = path[-1][2]
        if len(path) > 1 and path[-1][0] > path[-2][0]:
            seed = state + (state - path[-2][2]) * (trial - s) / (path[-1][0] - path[-2][0])
        else:
            seed = state
        cert = certify(trial, seed)
        if cert["verified"] and cert["attracting"]:
            path.append((trial, multipliers(cert), np.asarray(cert["orbit"][0])))
            s, h = trial, min(1.5 * h, h_max)
            continue
        failure = trial
        if h > h_min:
            h *= 0.5
            continue
        break
    last_s, last_mus, last_state = path[-1]
    if failure is None and last_s >= s_max:
        mu, _ = dominant_pair(last_mus)
        return {"type": "none_within_range", "path": path, "s_boundary": None, "bracket": None,
                "dominant": [mu.real, mu.imag]}
    low, high = last_s, failure
    low_state, low_mus = last_state, last_mus
    while high - low > bisection_tolerance:
        mid = 0.5 * (low + high)
        cert = certify(mid, low_state)
        if cert["verified"] and cert["attracting"]:
            low, low_state, low_mus = mid, np.asarray(cert["orbit"][0]), multipliers(cert)
        else:
            high = mid
    mu, _ = dominant_pair(low_mus)
    return {"type": classify_multiplier(mu, unit_band=unit_band), "path": path,
            "s_boundary": 0.5 * (low + high), "bracket": [low, high], "dominant": [mu.real, mu.imag],
            "last_attracting_state": low_state, "last_multipliers": low_mus}


def iterate(step, x0, steps):
    """Iterate a map, returning the final state (None if it leaves the finite range)."""
    x = np.asarray(x0, dtype=np.float64).copy()
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(steps):
            x = step(x)
            if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 1e6:
                return None
    return x


@dataclass(frozen=True)
class DeepLinear2x2:
    """Depth-two linear network l = 1/2 ||W2 W1 - diag(s1, s2)||_F^2 + (lam/2)(|W1|^2 + |W2|^2).

    Parameters are ordered (W1 row-major, W2 row-major). By orthogonal
    invariance a general target U S V^T reduces to S = diag(s1, s2). Gradient
    descent may use separate rates for W1 and W2. Hidden-layer rotations
    W1 -> Q W1, W2 -> W2 Q^T are an exact symmetry of the map, so every orbit
    has a neutral direction (multiplier 1 for periodic orbits) that is removed
    with :meth:`symmetry_generator`.
    """

    s1: float
    s2: float
    lam: float = 0.0

    @property
    def target(self):
        return np.diag([self.s1, self.s2])

    @staticmethod
    def unpack(theta):
        theta = np.asarray(theta, dtype=np.float64)
        return theta[..., :4].reshape(theta.shape[:-1] + (2, 2)), theta[..., 4:].reshape(theta.shape[:-1] + (2, 2))

    @staticmethod
    def pack(w1, w2):
        return np.concatenate([np.reshape(w1, np.shape(w1)[:-2] + (4,)), np.reshape(w2, np.shape(w2)[:-2] + (4,))],
                              axis=-1)

    def loss(self, theta):
        w1, w2 = self.unpack(theta)
        r = w2 @ w1 - self.target
        return 0.5 * np.sum(r * r, axis=(-2, -1)) + 0.5 * self.lam * np.sum(np.asarray(theta) ** 2, axis=-1)

    def gradient(self, theta):
        w1, w2 = self.unpack(theta)
        r = w2 @ w1 - self.target
        g1 = np.swapaxes(w2, -1, -2) @ r
        g2 = r @ np.swapaxes(w1, -1, -2)
        return self.pack(g1, g2) + self.lam * np.asarray(theta, dtype=np.float64)

    def hvp(self, theta, vector):
        w1, w2 = self.unpack(theta)
        v1, v2 = self.unpack(vector)
        r = w2 @ w1 - self.target
        dr = v2 @ w1 + w2 @ v1
        d1 = np.swapaxes(v2, -1, -2) @ r + np.swapaxes(w2, -1, -2) @ dr
        d2 = dr @ np.swapaxes(w1, -1, -2) + r @ np.swapaxes(v1, -1, -2)
        return self.pack(d1, d2) + self.lam * np.asarray(vector, dtype=np.float64)

    def hessian(self, theta):
        return np.array([self.hvp(theta, e) for e in np.eye(8)]).T

    @staticmethod
    def rate_vector(a1, a2=None):
        a2 = a1 if a2 is None else a2
        return np.array([a1] * 4 + [a2] * 4, dtype=np.float64)

    def step(self, theta, a1, a2=None):
        return np.asarray(theta, dtype=np.float64) - self.rate_vector(a1, a2) * self.gradient(theta)

    def jacobian(self, theta, a1, a2=None):
        return np.eye(8) - self.rate_vector(a1, a2)[:, None] * self.hessian(theta)

    def map_pair(self, a1, a2=None):
        return (lambda x: self.step(x, a1, a2)), (lambda x: self.jacobian(x, a1, a2))

    @staticmethod
    def symmetry_generator(theta):
        """Tangent of the hidden-rotation orbit: (A W1, -W2 A), A = [[0, -1], [1, 0]]."""
        w1, w2 = DeepLinear2x2.unpack(theta)
        generator = np.array([[0.0, -1.0], [1.0, 0.0]])
        return DeepLinear2x2.pack(generator @ w1, -w2 @ generator)

    @staticmethod
    def invariants(theta):
        """Rotation-invariant observables: W2 W1, W1^T W1 and W2 W2^T (flattened, 12 numbers)."""
        w1, w2 = DeepLinear2x2.unpack(theta)
        parts = [w2 @ w1, np.swapaxes(w1, -1, -2) @ w1, w2 @ np.swapaxes(w2, -1, -2)]
        return np.concatenate([np.reshape(p, np.shape(p)[:-2] + (4,)) for p in parts], axis=-1)

    def aligned_cycle(self, a):
        """Balanced diagonal 2-cycle (equal rates): top mode on the cubic-map 2-cycle,
        second mode at its fixed point. Weight decay acts on the balanced diagonal
        manifold as s -> s - lam. Returns the two cycle points."""
        s = self.s1 - self.lam
        if self.s2 - self.lam <= 0:
            raise ValueError("weight decay removes the second mode")

        def f(w):
            return w - a * w * (w * w - s)

        # 2-cycle of the cubic map: f(f(w)) = w with f(w) != w; Newton from the
        # symmetric-difference seed around sqrt(s)
        w = np.sqrt(s) * (1.0 + 0.5 * np.sqrt(max(a * s - 1.0, 1e-12)))
        for _ in range(200):
            g = f(f(w)) - w
            fw = f(w)
            dg = (1 - a * (3 * fw * fw - s)) * (1 - a * (3 * w * w - s)) - 1.0
            step = g / dg
            w -= step
            if abs(step) < 1e-15 * max(1.0, abs(w)):
                break
        w_b = f(w)
        if abs(w_b - w) < 1e-9:
            raise ValueError("no 2-cycle at this rate (cubic map at its fixed point)")
        w2 = np.sqrt(self.s2 - self.lam)
        p = self.pack(np.diag([w, w2]), np.diag([w, w2]))
        q = self.pack(np.diag([w_b, w2]), np.diag([w_b, w2]))
        return p, q
