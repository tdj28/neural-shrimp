import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import sheets_map as sm  # noqa: E402


def test_canonical_loop_is_rotation_invariant():
    assert sm.canonical_loop([5, 1, 3]) == sm.canonical_loop([1, 3, 5]) == sm.canonical_loop([3, 5, 1]) == (1, 3, 5)


def test_detect_period_finds_smallest_period():
    rng = np.random.default_rng(0)
    cyc = rng.normal(size=(3, 4))
    hist = np.stack([np.stack([cyc[t % 3], rng.normal(size=4)]) for t in range(40)])   # row 0 period 3, row 1 none
    per = sm.detect_period(hist, 32, 1e-8)
    assert per[0] == 3 and per[1] == 0


def test_settle_tracker_reports_end_of_transient():
    tr = sm.SettleTracker(torch, 1, 2, 8, 1e-6, torch.float64, "cpu")
    for t in range(30):
        x = torch.tensor([[float(t), 0.0]]) if t < 10 else torch.tensor([[1.0, 2.0]]) if t % 2 else torch.tensor([[3.0, 4.0]])
        tr.update(x)
    assert tr.settle()[0] == 12          # steps 10 and 11 start the 2-cycle; step 12 is the first to repeat a state
