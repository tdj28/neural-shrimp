import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import soft_thinking_twins as stt


def test_growth_rate_exponential_then_saturating():
    t = np.arange(1, 301)
    d = np.minimum(1e-10 * np.exp(0.2 * t), 0.5)
    rate, first, pts = stt.growth_rate(d, 1e-10)
    assert abs(rate - 0.2) < 1e-6 and first == int(np.ceil(np.log(1e-3 / 1e-10) / 0.2)) and pts > 50


def test_growth_rate_decay():
    d = 1e-10 * np.exp(-0.5 * np.arange(1, 101))
    rate, first, _ = stt.growth_rate(d, 1e-10)
    assert abs(rate + 0.5) < 1e-6 and first is None


class FakeLoop:
    """Growing-context stand-in: logits are a fixed linear readout of the newest soft token plus the steering of
    each row (read from sm.steer at every call, as the steering hook does)."""

    def __init__(self, V=7, d=3, seed=0):
        import torch

        rng = np.random.default_rng(seed)
        self.torch, self.state_dtype, self.dtype, self.device = torch, torch.float64, torch.float64, torch.device("cpu")
        self.E = torch.as_tensor(rng.normal(size=(V, d)))
        self.W = torch.as_tensor(rng.normal(size=(d, V))) * 3
        self.sm = type("SM", (), {"steer": None})()
        self.cache, self.steer_rows, self.prepared, self.steered_calls = None, None, None, 0

    def prepare(self, rows):
        self.prepared = np.asarray(rows)
        self.steer_rows = self.torch.as_tensor(self.prepared)[:, None, :]
        return self.torch.ones((len(rows), 1, self.E.shape[1]), dtype=self.torch.float64)

    def model(self, inputs_embeds, past_key_values, use_cache, logits_to_keep):
        steer = self.sm.steer
        self.steered_calls += steer is not None
        z = (inputs_embeds + (0 if steer is None else steer)) @ self.W
        return type("Out", (), {"logits": z})()


def test_steered_rows_reach_prepare_and_every_step():
    loop = FakeLoop()
    v = np.array([1.0, -2.0, 0.5])
    out, dist, ent = stt.run(loop, [0.0, 0.1], 20, 1e-10, 2, alphas=[0.0, 0.3], v=v)
    expect = np.tile(np.array([0.0, 0.3])[:, None] * v[None, :], (3, 1))      # base rows, then each twin's block
    assert np.array_equal(loop.prepared, expect)
    assert loop.steered_calls == 20 and loop.sm.steer is None
    assert [c["alpha"] for c in out] == [0.0, 0.3] and dist.shape == (20, 2, 2)
    plain = FakeLoop()
    out0, _, ent0 = stt.run(plain, [0.0, 0.1], 20, 1e-10, 2)              # unsteered: alpha 0 rows match
    assert np.array_equal(plain.prepared, np.zeros((6, 3))) and np.array_equal(ent0[:, 0], ent[:, 0])
    assert not np.allclose(ent0[:, 1], ent[:, 1])


def test_steered_cells_without_vector_fail():
    import pytest

    with pytest.raises(ValueError):
        stt.run(FakeLoop(), [0.0], 5, 1e-10, 1, alphas=[0.2])
