import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import start_state as st


def test_entropy_profile_limits():
    z = np.random.default_rng(0).normal(size=5000) * 3.0
    H = st.entropy_profile(z, np.array([-3.0, 0.0, 4.0]))
    assert H[0] < 1e-3                                     # nearly greedy
    assert abs(H[2] - np.log(5000)) < 1e-3                 # nearly uniform
    assert 0 < H[1] < np.log(5000)


def test_breadth_and_crossing_width():
    log_t = np.linspace(-1, 1, 201)
    H = np.interp(log_t, [-1, -0.2, 0.3, 1], [0.0, 1.0, 10.0, 11.0])
    mid, width = st.breadth(log_t, H)
    assert abs(width - 0.5) < 0.011
    assert abs(mid - 0.5) < 0.02
    mid, width = st.breadth(log_t, np.full_like(log_t, 0.5))
    assert mid == 0 and np.isnan(width)
