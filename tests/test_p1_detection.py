import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import p1_detection as pd  # noqa: E402


def synthetic_arm(R=300, N=20, seed=0, informative=True):
    """Dialogues with a hidden risk that rises before capture; q carries it when informative."""
    rng = np.random.default_rng(seed)
    score = rng.uniform(0, 0.2, (R, N))
    risk = np.zeros((R, N))
    q = rng.uniform(0, 1, (R, N))
    for r in range(R):
        onset = rng.integers(4, N + 6)
        for t in range(N):
            risk[r, t] = 1.0 if onset - 3 <= t < onset else 0.0
            if t >= onset:
                score[r, t] = 0.9
        if informative:
            q[r] = 0.2 * rng.uniform(0, 1, N) + 0.8 * risk[r]
    return {"score": score, "q": q, "q_plus": q + 0.1 * rng.normal(size=(R, N)), "pre": rng.normal(size=(R, N, 30)),
            "tok": rng.normal(size=(R, N, 5)), "axis": rng.normal(size=(R, N)), "dose": 0.04, "name": "syn"}


def test_onsets_and_rows():
    h = np.array([[0, 0, 1, 0, 1, 1], [0, 0, 0, 0, 0, 0]], dtype=bool)
    assert list(pd.onsets(h, 1)) == [2, 6] and list(pd.onsets(h, 2)) == [4, 6]
    rows = pd.at_risk_rows(np.array([2, 6]), 6, 1)
    assert [tuple(x) for x in rows if x[0] == 0] == [(0, 0, 0), (0, 1, 1)]
    assert len([x for x in rows if x[0] == 1]) == 5 and all(x[2] == 0 for x in rows if x[0] == 1)


def test_incremental_auc_positive_control_and_null():
    arm = synthetic_arm(informative=True)
    tab = pd.table([arm], 3)
    ob, of = pd.crossfit(tab, "D1_forecast")
    res = pd.incremental(tab, ob, of, n_boot=200)
    assert res["increment"] > 0.1 and res["ci95"][0] > 0
    arm = synthetic_arm(informative=False, seed=1)
    tab = pd.table([arm], 3)
    ob, of = pd.crossfit(tab, "D1_forecast")
    res = pd.incremental(tab, ob, of, n_boot=200)
    assert res["ci95"][0] < 0.02


def test_no_event_sentinel_is_never_a_label_and_horizons_are_complete():
    turns = 40
    clean = np.zeros((1, turns), dtype=bool)                     # never hostile
    onset = pd.onsets(clean, 1)
    assert onset[0] == turns
    for h in (1, 3, 5):
        rows = pd.at_risk_rows(onset, turns, h)
        assert rows[:, 2].sum() == 0                             # no positive label without an event
        assert rows[:, 1].max() == turns - 1 - h                 # only complete horizons
    late = np.zeros((1, turns), dtype=bool)
    late[0, 38] = True                                           # event at turn 38
    rows = pd.at_risk_rows(pd.onsets(late, 1), turns, 3)
    assert [tuple(x) for x in rows if x[2] == 1] == [(0, 35, 1), (0, 36, 1)]
