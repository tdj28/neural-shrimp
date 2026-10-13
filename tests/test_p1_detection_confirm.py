import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import p1_detection_confirm as pc  # noqa: E402


def arm(seed, informative, R=300, N=20, d=40):
    """Dialogues whose pre-reply state carries the upcoming capture along one hidden direction when informative."""
    rng = np.random.default_rng(seed)
    score = rng.uniform(0, 0.2, (R, N))
    pre = rng.normal(size=(R, N, d))
    direction = np.zeros(d)
    direction[3] = 1.0
    for r in range(R):
        onset = rng.integers(4, N + 6)
        for t in range(N):
            if t >= onset:
                score[r, t] = 0.9
            if informative and onset - 3 <= t - 1 < onset:      # pre-reply state of turn t, seen after turn t - 1
                pre[r, t] += 2.5 * direction
    return {"score": score, "q": rng.uniform(size=(R, N)), "q_plus": rng.uniform(size=(R, N)), "pre": pre,
            "tok": rng.normal(size=(R, N, 4)), "axis": rng.normal(size=(R, N)), "dose": 0.04, "name": f"s{seed}"}


def test_frozen_models_transfer_to_fresh_dialogues_and_null():
    res = pc.evaluate([arm(0, True)], [arm(1, True)], 3, 1, n_boot=200)
    assert res["increment"] > 0.05 and res["ci95"][0] > 0
    assert res["flagged_at_train_10pct_fpr"]["with_state"] > res["flagged_at_train_10pct_fpr"]["baseline"]
    res = pc.evaluate([arm(2, False)], [arm(3, False)], 3, 1, n_boot=200)
    assert res["ci95"][0] < 0.02
