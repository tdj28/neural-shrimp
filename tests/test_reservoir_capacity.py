"""memory_capacity recovers the known capacities of linear reservoirs: about d for d distinct slow modes driven by
i.i.d. input (Jaeger 2001), and close to 1 for a single mode."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import reservoir_capacity as rc  # noqa: E402


def linear_states(a, N=6000, seed=0):
    rng = np.random.default_rng(seed)
    u = rng.uniform(-1, 1, N)
    x = np.zeros(len(a))
    S = np.zeros((N, len(a)))
    for t in range(N):
        x = a * (x + u[t])                      # input enters before the map, as in inject_run
        S[t] = x
    return S, u


def test_distinct_modes_give_capacity_near_their_number():
    S, u = linear_states(np.array([0.2, 0.5, 0.7, 0.85]))
    mc, curve, _ = rc.memory_capacity(S, u, 40)
    assert 3.5 < mc <= 4.2


def test_one_mode_gives_capacity_one():
    S, u = linear_states(np.array([0.8]))
    mc, curve, _ = rc.memory_capacity(S, u, 40)
    assert 0.9 < mc < 1.1 and curve[0] > 0.3
