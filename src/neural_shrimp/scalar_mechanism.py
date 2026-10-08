"""Analytic derivatives for the scalar factorization map; exploratory float64 tools.

L(u,v) = (uv-target)^2/2 + ridge*(u^2+v^2)/2.  Rates are eta and
ratio*eta.  No interval certification or global bifurcation claim is made.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScalarFactorization:
    target: float = 1.2
    ridge: float = 0.2

    @property
    def minimum(self):
        if not 0 < self.ridge < self.target:
            raise ValueError("isolated positive balanced minimum requires 0 < ridge < target")
        return np.full(2, np.sqrt(self.target - self.ridge))

    def gradient(self, x):
        u, v = np.asarray(x)
        error = u * v - self.target
        return np.array([error * v + self.ridge * u, error * u + self.ridge * v])

    def hessian(self, x):
        u, v = np.asarray(x)
        return np.array([[v*v + self.ridge, 2*u*v-self.target],
                         [2*u*v-self.target, u*u+self.ridge]])

    def step(self, x, eta, ratio):
        return np.asarray(x) - eta * np.array([1.0, ratio]) * self.gradient(x)

    def jacobian(self, x, eta, ratio):
        return np.eye(2) - eta * np.array([1.0, ratio])[:, None] * self.hessian(x)

    def flip_rate(self, ratio):
        scale = np.sqrt([1.0, ratio])
        return 2.0 / np.linalg.eigvalsh(scale[:, None] * self.hessian(self.minimum) * scale)[-1]

    def second(self, x, a, b, eta, ratio):
        u, v = x
        cross = a[0]*b[1] + a[1]*b[0]
        return -2*eta*np.array([v*cross + u*a[1]*b[1],
                                ratio*(u*cross + v*a[0]*b[0])])

    def third(self, a, b, c, eta, ratio):
        return -2*eta*np.array([a[0]*b[1]*c[1] + a[1]*b[0]*c[1] + a[1]*b[1]*c[0],
                                ratio*(a[1]*b[0]*c[0] + a[0]*b[1]*c[0] + a[0]*b[0]*c[1])])

    def two_step_forms(self, x, eta, ratio):
        y = self.step(x, eta, ratio)
        A0, A1 = self.jacobian(x, eta, ratio), self.jacobian(y, eta, ratio)
        B0 = lambda a, b: self.second(x, a, b, eta, ratio)
        B1 = lambda a, b: self.second(y, a, b, eta, ratio)
        C = lambda a, b, c: self.third(a, b, c, eta, ratio)

        def B(a, b):
            return B1(A0@a, A0@b) + A1@B0(a, b)

        def third(a, b, c):
            return (C(A0@a, A0@b, A0@c) + B1(B0(a, b), A0@c)
                    + B1(B0(a, c), A0@b) + B1(B0(b, c), A0@a) + A1@C(a, b, c))

        return A1@A0, B, third

    def cycle_equations(self, mean, half_difference, eta, ratio):
        """Exact cubic mean/difference equations; zero iff the pair closes.

        Dividing the sum and difference of the two step equations by the
        rates gives these equations.  Nonzero half_difference excludes a fixed point.
        """
        a, b = mean
        p, q = half_difference
        average = np.array([a*(b*b+q*q+self.ridge) + 2*p*b*q-self.target*b,
                            b*(a*a+p*p+self.ridge) + 2*q*a*p-self.target*a])
        difference = np.array([p*(b*b+q*q+self.ridge) + (2*a*b-self.target)*q-2*p/eta,
                               q*(a*a+p*p+self.ridge) + (2*a*b-self.target)*p-2*q/(eta*ratio)])
        return np.concatenate([average, difference])


def planar_stability(matrix):
    """Jury margins, determinant and discriminant for a real planar map."""
    trace, determinant = float(np.trace(matrix)), float(np.linalg.det(matrix))
    margins = [1-trace+determinant, 1+trace+determinant, 1-determinant]
    return {"trace": trace, "determinant": determinant,
            "discriminant": trace*trace-4*determinant,
            "jury_margins": margins, "attracting": bool(min(margins) > 0)}
