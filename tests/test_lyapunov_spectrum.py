import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import lyapunov_spectrum as ls  # noqa: E402


class FakeLoop:
    """Linear map x -> A x on (B, 1, d) states with a three-token readout and exact tangents; counts step calls."""

    def __init__(self, A):
        import torch

        self.torch, self.state_dtype, self.device = torch, torch.float64, torch.device("cpu")
        self.A = torch.as_tensor(A, dtype=torch.float64)
        self.rows = None

    def prepare(self, rows):
        self.rows = len(rows)
        return self.torch.ones((len(rows), 1, self.A.shape[0]), dtype=self.torch.float64)

    def step(self, X, temps):
        assert X.shape[0] == self.rows == len(temps)
        Y = X @ self.A.T
        y = Y[:, 0, 0]
        return Y, self.torch.softmax(self.torch.stack([y, -y, 0 * y], -1), -1)

    def step_jvp(self, X, temps, V):
        assert V.shape == X.shape
        Y, p = self.step(X, temps)
        return Y, p, V @ self.A.T


def test_qr_spectrum_recovers_eigenvalue_moduli_of_a_nonnormal_map():
    rng = np.random.default_rng(3)
    S = rng.normal(size=(5, 5)) + 3 * np.eye(5)
    lam = np.array([1.3, 0.9, 0.5, 0.2, 0.1])
    A = S @ np.diag(lam) @ np.linalg.inv(S)
    r = ls.spectrum(FakeLoop(A), np.zeros(5), 0.0, 3, 10, 3000, checkpoints=(100, 3000))
    assert np.allclose(r["exponents"], np.log(lam[:3]), atol=5e-3)
    assert r["checkpoints"][-1]["steps"] == 3000 and r["checkpoints"][-1]["exponents"] == r["exponents"]
    assert r["exponents"][0] >= r["exponents"][1] >= r["exponents"][2]


def test_k1_equals_single_tangent_exponent():
    A = np.diag([0.5, 0.25])
    r = ls.spectrum(FakeLoop(A), np.zeros(2), 0.0, 1, 5, 400)
    assert abs(r["exponents"][0] - np.log(0.5)) < 1e-2 and r["checkpoints"] == []
