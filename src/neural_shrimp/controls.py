"""Analytic maps used to validate the same diagnostics as target networks."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LogisticMap:
    """x -> r x (1-x); r=3.2 has a stable prime period-two orbit."""

    r: float = 3.2

    def __call__(self, state):
        x = np.asarray(state, dtype=np.float64)
        if x.shape != (1,):
            raise ValueError("Logistic state must have shape (1,)")
        return self.r * x * (1.0 - x)

    def jacobian(self, state):
        x = np.asarray(state, dtype=np.float64)
        return np.array([[self.r * (1.0 - 2.0 * x[0])]], dtype=np.float64)


@dataclass(frozen=True)
class HenonMap:
    """(x,y) -> (1 - a*x*x + y, b*x), the conventional Hénon map."""

    a: float = 1.4
    b: float = 0.3

    def __call__(self, state):
        x, y = np.asarray(state, dtype=np.float64)
        return np.array([1.0 - self.a * x * x + y, self.b * x], dtype=np.float64)

    def jacobian(self, state):
        x, _ = np.asarray(state, dtype=np.float64)
        return np.array([[-2.0 * self.a * x, 1.0], [self.b, 0.0]], dtype=np.float64)
