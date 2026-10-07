"""First Lyapunov coefficient of Neimark-Sacker bifurcations of maps (exploratory).

For a smooth map G with a fixed point x0 whose Jacobian A = DG(x0) has a simple
pair of multipliers mu, conj(mu) on (numerically: at) the unit circle, the
critical normal-form coefficient (Kuznetsov, Elements of Applied Bifurcation
Theory, 3rd ed., Section 5.4; Kuznetsov and Meijer 2019) is

    c1 = 1/2 <p, C(q, q, qbar) + 2 B(q, (I - A)^{-1} B(q, qbar))
                 + B(qbar, (mu^2 I - A)^{-1} B(q, q))>,

with A q = mu q, A^T p = conj(mu) p, <q, q> = 1 and <p, q> = 1 for the inner
product <u, v> = sum conj(u_i) v_i. The first Lyapunov coefficient is
d = Re(conj(mu) c1) with mu taken on the unit circle. d < 0: supercritical (an
attracting closed invariant curve of amplitude about sqrt(delta / -d) is born,
a soft loss of stability); d > 0: subcritical (a repelling curve exists before
the bifurcation and shrinks onto the orbit, a hard loss of stability). The value
of d depends on the normalization of q; its sign does not.

``B`` and ``C`` are the symmetric second and third derivative forms of G at x0,
supplied as callables on real vectors and extended to complex arguments by
multilinearity. ``finite_difference_forms`` builds them from an exact batched
Jacobian-vector product of G by Richardson-extrapolated central differences.
"""

import numpy as np


def complex_B(B, x, y):
    """B(x, y) for complex x, y from a real symmetric bilinear form."""
    x1, x2, y1, y2 = x.real, x.imag, y.real, y.imag
    return B(x1, y1) - B(x2, y2) + 1j * (B(x1, y2) + B(x2, y1))


def complex_C_qqqbar(C, q):
    """C(q, q, conj(q)) from a real symmetric trilinear form, q = a + i b."""
    a, b = q.real, q.imag
    return C(a, a, a) + C(a, b, b) + 1j * (C(a, a, b) + C(b, b, b))


def critical_pair(A, mu=None):
    """Critical multiplier (Im > 0), unit right eigenvector q and adjoint p with <p, q> = 1."""
    values, right = np.linalg.eig(A)
    if mu is None:
        candidates = [k for k in range(len(values)) if values[k].imag > 1e-12]
        if not candidates:
            raise ValueError("no complex multiplier pair")
        index = max(candidates, key=lambda k: abs(values[k]))
    else:
        index = int(np.argmin(np.abs(values - mu)))
    mu = values[index]
    q = right[:, index] / np.linalg.norm(right[:, index])
    left_values, left = np.linalg.eig(A.T)
    j = int(np.argmin(np.abs(left_values - np.conj(mu))))
    p = left[:, j]
    p = p / np.conj(np.vdot(p, q))
    others = np.delete(values, [index, int(np.argmin(np.abs(values - np.conj(mu))))])
    gap = float(np.min(np.abs(others - mu))) if len(others) else np.inf
    return mu, q, p, gap


def ns_coefficient(A, B, C, mu=None):
    """First Lyapunov coefficient d and the normal-form coefficient c1 at a Neimark-Sacker point."""
    A = np.asarray(A, dtype=np.float64)
    mu, q, p, gap = critical_pair(A, mu)
    unit = mu / abs(mu)
    identity = np.eye(len(A))
    qbar = np.conj(q)
    b_qqbar = complex_B(B, q, qbar)
    b_qq = complex_B(B, q, q)
    w1 = np.linalg.solve(identity - A, b_qqbar)
    w2 = np.linalg.solve(unit ** 2 * identity - A, b_qq)
    total = complex_C_qqqbar(C, q) + 2.0 * complex_B(B, q, w1) + complex_B(B, qbar, w2)
    c1 = 0.5 * np.vdot(p, total)
    theta = abs(np.angle(unit)) / (2 * np.pi)
    resonance = min(abs(theta - r) for r in (0.25, 1 / 3, 0.5))
    return {"d": float((np.conj(unit) * c1).real), "c1": [float(c1.real), float(c1.imag)],
            "mu": [float(mu.real), float(mu.imag)], "modulus": float(abs(mu)), "theta_over_2pi": float(theta),
            "distance_to_strong_resonance": float(resonance), "eigen_gap": gap,
            "imag_b_qqbar": float(np.max(np.abs(b_qqbar.imag)))}


def finite_difference_forms(jvp, x0, step_b=1e-4, step_c=1e-3):
    """Second and third derivative forms of G at x0 from an exact Jacobian-vector product.

    ``jvp(X, V)`` returns DG(X_k) V_k row-wise for arrays of shape (m, n). Central
    differences are Richardson-extrapolated (steps h and h/2), so B is accurate to
    O(h^4) plus rounding eps/h and C to O(h^4) plus rounding eps/h^2.
    """
    x0 = np.asarray(x0, dtype=np.float64)

    def B(u, v):
        out = []
        for h in (step_b, step_b / 2):
            X = np.stack([x0 + h * u, x0 - h * u])
            V = np.stack([v, v])
            d = jvp(X, V)
            out.append((d[0] - d[1]) / (2 * h))
        return (4 * out[1] - out[0]) / 3

    def C(u, v, w):
        out = []
        for h in (step_c, step_c / 2):
            X = np.stack([x0 + h * u + h * v, x0 + h * u - h * v, x0 - h * u + h * v, x0 - h * u - h * v])
            W = np.stack([w, w, w, w])
            d = jvp(X, W)
            out.append((d[0] - d[1] - d[2] + d[3]) / (4 * h * h))
        return (4 * out[1] - out[0]) / 3

    return B, C
