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
