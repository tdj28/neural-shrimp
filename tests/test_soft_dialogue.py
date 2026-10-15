"""PairLoop alternates the agents' caches and steering: one step is F_B o F_A, equal agents give the single loop
twice per step, and hidden_chaos.run_batch runs on it unchanged (a logistic map stands in for the language model)."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import hidden_chaos as hc  # noqa: E402
import soft_dialogue as sd  # noqa: E402

D = 4


class LogisticLoop:
    """x -> r x (1 - x) on the first coordinate, r from the row's steering (as a prefix cache would hold it)."""

    def __init__(self):
        self.torch, self.state_dtype, self.device = torch, torch.float64, "cpu"
        self.E = torch.eye(D, dtype=torch.float64)
        self.cache = self.steer_rows = self.extra_rows = None

    def prepare(self, rows):
        self.steer_rows = torch.as_tensor(np.asarray(rows), dtype=torch.float64)
        self.cache = {"r": self.steer_rows[:, 0].clone()}
        return torch.full((len(rows), 1, D), 0.3, dtype=torch.float64)

    def step(self, X, temps):
        r = self.cache["r"]
        assert torch.equal(r, self.steer_rows[:, 0])        # cache and steering always belong to one agent
        x = X[:, -1, 0]
        Y = X.clone()
        Y[:, -1, 0] = r * x * (1 - x)
        p = torch.softmax(torch.stack([Y[:, -1, 0], -Y[:, -1, 0], 0 * x, 0 * x], dim=1), dim=1)
        return Y, p


def rows(r):
    v = np.zeros(D)
    v[0] = 1.0
    return np.array([x * v for x in r])


def test_one_step_is_b_after_a():
    pair = sd.PairLoop(LogisticLoop())
    pair.set_b_cells(rows([3.2, 3.9]))
    X = pair.prepare(rows([3.9, 3.2]))
    for _ in range(5):
        X, _ = pair.step(X, np.ones(2))
    f = lambda r, x: r * x * (1 - x)
    for c, (ra, rb) in enumerate([(3.9, 3.2), (3.2, 3.9)]):
        x = 0.3
        for _ in range(5):
            x = f(rb, f(ra, x))
        assert abs(float(X[c, -1, 0]) - x) < 1e-15


def test_equal_agents_are_the_single_loop_twice():
    single = LogisticLoop()
    Xs = single.prepare(rows([3.7]))
    for _ in range(14):
        Xs, _ = single.step(Xs, np.ones(1))
    pair = sd.PairLoop(LogisticLoop())
    pair.set_b_cells(rows([3.7]))
    X = pair.prepare(rows([3.7]))
    for _ in range(7):
        X, _ = pair.step(X, np.ones(1))
    assert float(X[0, -1, 0]) == float(Xs[0, -1, 0])


def test_run_batch_labels_the_exchange_map():
    """r_A = r_B = 3.2 is a 2-cycle of the logistic map, a fixed point of G; 3.9 with 3.9 is chaotic."""
    pair = sd.PairLoop(LogisticLoop())
    v = np.zeros(D)
    v[0] = 1.0
    pair.set_b_cells(rows([3.2, 3.9]))
    out = hc.run_batch(pair, v, np.zeros(2), np.array([3.2, 3.9]), 300, 600, [1e-6, 1e-9], 1e-8)
    assert out[0]["period"] == 1 and hc.classify(out[0]) == "periodic"
    assert hc.classify(out[1]) in ("visible chaos", "hidden chaos") and out[1]["lyapunov"][0] > 0.5
