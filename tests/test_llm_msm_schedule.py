import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))


def test_schedule_factors():
    import explore_llm_msm as m

    assert np.allclose(m.schedule_factors(None, 5), 1.0)
    f = m.schedule_factors("0.5:1:0.5", 41)
    assert abs(f[0] - 0.5) < 1e-12 and abs(f[20] - 1.0) < 1e-12 and abs(f[40] - 0.5) < 1e-12 and abs(f[10] - 0.75) < 1e-12


def test_triggered_controller_counters_exactly_k_turns_after_a_flag():
    import explore_llm_msm as m

    remaining = np.zeros(2, dtype=int)
    applied = []
    flags = {1: [True, False], 5: [False, True]}           # dialogue 0 flagged at turn 1, dialogue 1 at turn 5
    for k in range(10):
        applied.append(m.counter_for_turn("triggered", -0.06, remaining).copy())
        remaining = m.update_remaining(remaining, np.array(flags.get(k, [False, False])), 3)
    a = np.array(applied)
    assert np.allclose(a[:, 0], [0, 0, -0.06, -0.06, -0.06, 0, 0, 0, 0, 0])
    assert np.allclose(a[:, 1], [0, 0, 0, 0, 0, 0, -0.06, -0.06, -0.06, 0])
    assert np.allclose(m.counter_for_turn("constant", -0.02, remaining), -0.02)
    assert np.allclose(m.counter_for_turn("none", -0.02, remaining), 0.0)


def test_regenerate_mode_counters_like_triggered():
    import explore_llm_msm as msm

    remaining = np.array([0, 2, 1])
    assert msm.counter_for_turn("regenerate", -0.06, remaining).tolist() == [0.0, -0.06, -0.06]
