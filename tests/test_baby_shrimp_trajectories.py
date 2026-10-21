import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from baby_shrimp_trajectories import recurrence


def test_primitive_candidate_uses_every_phase():
    s = np.tile(np.eye(3), (20, 1))
    assert recurrence(s, max_period=12)["period_candidate"] == 3
    s[-2] += .01
    assert recurrence(s, max_period=12)["period_candidate"] == 0


def test_early_accidental_return_does_not_pass_tail_gate():
    s = np.arange(180, dtype=float).reshape(60, 3)
    s[-1] = s[-4]
    assert recurrence(s, max_period=12)["period_candidate"] == 0


@pytest.mark.parametrize("s", [np.zeros((4, 3)), np.full((60, 3), np.nan)])
def test_incomplete_or_nonfinite_is_refused(s):
    with pytest.raises(ValueError):
        recurrence(s, max_period=12)
