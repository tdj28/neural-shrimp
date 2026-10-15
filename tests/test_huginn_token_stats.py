"""token_stats splits a generation trace into the prefill block and one block per generated token and reads each block's
late decay rate."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import huginn_relax_eval as hre  # noqa: E402


def block(rate, steps=32, r0=0.5):
    return [(r0 * rate ** k, 0.0, [r0 * rate ** k]) for k in range(steps)]


def test_blocks_and_rates():
    trace = block(0.9) + block(0.85) + block(1.0) + block(1.05)
    s = hre.token_stats(trace, 32)
    assert s["tokens"] == 3
    assert abs(s["prefill_rate"] - 0.9) < 1e-9
    assert abs(s["median_rate"] - 1.0) < 1e-9                 # median of 0.85, 1.00, 1.05
    assert abs(s["share_not_converging"] - 2 / 3) < 1e-12     # 1.00 and 1.05 are at least 1 - 0.02
    assert np.isclose(s["median_final"], 0.5)                 # finals 0.5*0.85^31, 0.5, 0.5*1.05^31 -> median 0.5
