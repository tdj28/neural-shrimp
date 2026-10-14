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
