import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import hidden_chaos as hc


def test_symbolic_period_finds_smallest_period():
    assert hc.symbolic_period([1, 2, 3] * 30) == 3
    assert hc.symbolic_period([5] * 50) == 1
    assert hc.symbolic_period([1, 2, 1, 3] * 20) == 4


def test_symbolic_period_zero_for_aperiodic_sequence():
    seq = [int(k * 0.6180339887 * 1000) % 7 for k in range(300)]
    assert hc.symbolic_period(seq) == 0


def test_classify():
    base = {"lyapunov": [0.05, 0.05], "period": 0, "lead_period": 7}
    assert hc.classify(base) == "hidden chaos"
    assert hc.classify({**base, "lead_period": 0}) == "visible chaos"
    assert hc.classify({**base, "period": 7}) == "periodic"
    assert hc.classify({**base, "lyapunov": [0.05, -0.01]}).startswith("irregular")


class FakeLoop:
    """Linear map x -> A x on (B, 1, d) states with a three-token readout, and exact tangents."""

    def __init__(self, A):
        import torch

        self.torch, self.state_dtype, self.device = torch, torch.float64, torch.device("cpu")
        self.A = torch.as_tensor(A, dtype=torch.float64)

    def prepare(self, rows):
        return self.torch.ones((len(rows), 1, self.A.shape[0]), dtype=self.torch.float64)

    def step(self, X, temps):
        Y = X @ self.A.T
        y = Y[:, 0, 0]
        return Y, self.torch.softmax(self.torch.stack([y, -y, 0 * y], -1), -1)

    def step_jvp(self, X, temps, V):
        Y, p = self.step(X, temps)
        return Y, p, V @ self.A.T


def test_tangent_exponent_matches_twins_and_probe_is_flat():
    import numpy as np

    loop = FakeLoop([[0.5, 0.0], [0.0, 0.25]])
    out = hc.run_batch(loop, np.zeros(2), [0.0, 0.1], [0.0, 0.0], 50, 400, [1e-4, 1e-6], 1e-8, tangent=True,
                       probe=True)
    for r in out:
        assert abs(r["lyapunov_tangent"] - np.log(0.5)) < 1e-2
        assert all(abs(l - np.log(0.5)) < 1e-2 for l in r["lyapunov"])
        ratio = np.array(r["fd_probe"]["central_ratio"])
        assert np.allclose(ratio, ratio[4], rtol=1e-3) and r["fd_probe"]["repeat_diff"] == 0.0


def test_solve_beta_hits_entropy_targets():
    import numpy as np
    import torch
    import explore_llm_soft_loop as esl

    torch.manual_seed(0)
    z = torch.randn(4, 5000, dtype=torch.float64) * 3
    targets = np.array([0.05, 1.0, 4.0, 8.0])
    beta = esl.solve_beta(z, targets)
    p = torch.softmax(beta[:, None] * z, -1)
    h = -(p * torch.log(p.clamp_min(1e-300))).sum(-1).numpy()
    assert np.allclose(h, targets, rtol=0, atol=1e-10)
    # smooth in z: a tiny change of the logits changes beta by a proportionally tiny amount
    beta2 = esl.solve_beta(z + 1e-9 * torch.randn_like(z), targets)
    assert ((beta2 - beta).abs() / beta).max() < 1e-7


def test_entropy_clamp_runs_with_probe():
    """The clamp wrapper through run_batch, including the probe's differently sized batch."""
    import numpy as np
    import torch

    class Tiny:
        def __init__(self):
            self.torch, self.state_dtype, self.device = torch, torch.float64, torch.device("cpu")
            g = torch.Generator().manual_seed(0)
            self.E = torch.randn(50, 3, dtype=torch.float64, generator=g)
            self.W = torch.randn(50, 3, dtype=torch.float64, generator=g) * 2

        def prepare(self, rows):
            return torch.zeros((len(rows), 1, 3), dtype=torch.float64) + 0.1

        def logits(self, X):
            return torch.tanh(X[:, -1, :]) @ self.W.T

    loop = hc.EntropyClamp(Tiny())
    out = hc.run_batch(loop, np.zeros(3), np.log10([1.0, 2.5]), [0.0, 0.0], 20, 50, [1e-5, 1e-6], 1e-8, probe=True)
    assert all(np.isfinite(r["mean_temperature"]) and "fd_probe" in r for r in out)
