import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import sync_coupled_loops as scl  # noqa: E402


class LogisticLoop:
    """F(x) = r x (1 - x) coordinatewise on (B, 1, d) states (r = 4: lambda_1 = ln 2), with a two-token readout and
    exact tangents; mimics the SoftLoop interface used by the sync tool."""

    def __init__(self, r=4.0):
        import torch

        self.torch, self.r = torch, r

    def step(self, X, temps):
        assert X.shape[0] == len(temps)
        Y = self.r * X * (1 - X)
        y = Y[:, 0, 0]
        return Y, self.torch.softmax(self.torch.stack([y, 1 - y], -1), -1)

    def step_jvp(self, X, temps, V):
        Y, p = self.step(X, temps)
        return Y, p, self.r * (1 - 2 * X) * V


def _pairs(a, b, n):
    import torch

    rows = []
    for _ in range(n):
        rows += [torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)]
    return torch.stack(rows)[:, None, :]


def test_c_star_and_prediction():
    assert scl.c_star(np.log(2)) == pytest.approx(0.25)
    assert scl.c_star(0.109) == pytest.approx(0.0516, abs=1e-4)
    assert scl.predicted_transverse(np.log(2), 0.25) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        scl.c_star(-0.1)


def test_couple_is_symmetric_and_preserves_sum():
    import torch

    F = torch.arange(12, dtype=torch.float64).reshape(4, 3)
    out = scl.couple(F, [0.0, 0.3])
    assert torch.equal(out[:2], F[:2])
    assert torch.allclose(out[2] + out[3], F[2] + F[3])
    assert torch.allclose(out[2] - out[3], (1 - 0.6) * (F[2] - F[3]))


def test_logistic_pairs_synchronize_above_c_star_only():
    loop = LogisticLoop()
    rng = np.random.default_rng(0)
    a = rng.uniform(0.1, 0.9, 3)
    b = a + 1e-3 * rng.normal(size=3)
    cs = [0.0, 0.1, 0.2, 0.3, 0.4]
    X = _pairs(a, b, len(cs))
    dist, tok, _ = scl.iterate_pairs(loop, X, np.ones(2 * len(cs)), cs, 400)
    synced = dist[:, -100:].max(1) < 1e-8
    assert list(synced) == [False, False, False, True, True]
    assert tok.shape == (10, 400)
    th = scl.threshold_interval(cs, synced)
    assert th["c_lo"] == 0.2 and th["c_hi"] == 0.3 and th["status"] == "bracketed"


def test_twin_exponent_matches_lambda1_plus_log_factor():
    loop = LogisticLoop()
    a = np.array([0.2345, 0.6789])
    cs = [0.0, 0.1, 0.35]
    X = _pairs(a, a + 1e-9, len(cs))
    logs = scl.twin_logs(loop, X, np.ones(2 * len(cs)), cs, 1e-9, 20000)
    lam = logs.mean(1)
    assert np.allclose(lam, scl.predicted_transverse(np.log(2), cs), atol=0.03)
    jl = scl.jvp_logs(loop, X[0::2].clone(), np.ones(len(cs)), 20000, np.random.default_rng(0))
    assert np.allclose(jl.mean(1), np.log(2), atol=0.03)


def test_twin_fails_loudly_on_collapse():
    loop = LogisticLoop()
    X = _pairs([0.5], [0.5], 1)          # F(0.5) = 1, F(1) = 0: identical rows, zero distance
    with pytest.raises(RuntimeError, match="collapsed"):
        scl.twin_logs(loop, X, np.ones(2), [0.1], 1e-8, 3)


def test_run_lengths_and_threshold_edge_cases():
    lens, censored = scl.run_lengths([1, 1, 0, 1, 0, 0, 1, 1, 1])
    assert list(lens) == [2, 1, 3] and censored
    lens, censored = scl.run_lengths([0, 0])
    assert len(lens) == 0 and not censored
    assert scl.threshold_interval([0.1, 0.2], [False, False])["c_hi"] is None
    th = scl.threshold_interval([0.1, 0.2, 0.3], [True, False, True])
    assert th["c_hi"] == 0.3 and th["non_monotone"]


def test_powerlaw_slope_recovers_minus_three_halves():
    rng = np.random.default_rng(0)
    # discrete Pareto with density ~ x^(-3/2): x = floor(xmin * u^(-2))
    x = np.floor(2 * rng.uniform(size=20000) ** (-2.0))
    fit = scl.powerlaw_slope(x[x < 1e5], min_len=2)
    assert fit is not None and abs(fit["slope"] + 1.5) < 0.1
    assert scl.powerlaw_slope([3, 4, 5]) is None


def test_closure_finds_primitive_period():
    import torch

    seq = [torch.tensor([[[1.0]], [[2.0]]]) * (1 + (t % 3)) for t in range(10)]
    ret = scl.closure(seq, 6)
    assert ret.shape == (1, 6)
    assert ret[0, 2] == 0 and ret[0, 5] == 0 and ret[0, 0] > 0 and ret[0, 1] > 0
