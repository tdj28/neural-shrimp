import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import p1_controller as pc  # noqa: E402


def write_arm(root, name, score, regenerated=None):
    src = root / "explore" / "msm" / f"p1-{name}"
    src.mkdir(parents=True)
    R, N = score.shape
    np.savez(src / "hostility.npz", classifier=score[:, :, None])
    (src / "sample.json").write_text(json.dumps({"texts": [["thank you"] * N for _ in range(R)]}))
    extra = {} if regenerated is None else {"regenerated": regenerated}
    np.savez(src / "turns.npz", counter=np.zeros((R, N)), length=np.full((R, N), 10), entropy=np.ones((R, N)), **extra)


def test_would_be_counts_replaced_replies_as_hostile(tmp_path, monkeypatch):
    import explore_llm_attractors as ela

    monkeypatch.setattr(ela, "DATA", tmp_path)
    score = np.zeros((2, 6))
    score[0, 5] = 0.9                                   # shown hostile, not replaced
    regen = np.zeros((2, 6), dtype=bool)
    regen[1, 2] = True                                  # original reply flagged and replaced (shown reply clean)
    write_arm(tmp_path, "regen", score, regen)
    write_arm(tmp_path, "plain", score)
    d = pc.load("msm", "regen")
    assert d["would_be"][0, 5] and d["would_be"][1, 2] and d["would_be"].sum() == 2
    assert d["hostile"].sum() == 1
    p = pc.load("msm", "plain")
    assert (p["would_be"] == p["hostile"]).all()
