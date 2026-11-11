"""Tests of tools/batch_fate_long.py with a small stand-in map (no language model)."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import batch_fate as bf  # noqa: E402
import batch_fate_long as bfl  # noqa: E402


class ToyLoop:
    """State (s, x, y). s is kept. For s > 0, x relaxes to 1 (a fixed point); for s < 0, x follows the chaotic
    logistic map. y alternates between two values, so a relaxed row has period 2."""
    torch = torch
    device = torch.device("cpu")
    state_dtype = torch.float64

    def prepare(self, steer):
        return torch.zeros((len(steer), 2, 3), dtype=torch.float64)

    def step(self, X, temps):
        s, x, y = X[:, -1, 0], X[:, -1, 1], X[:, -1, 2]
        x_new = torch.where(s > 0, 1.0 + 0.5 * (x - 1.0), 3.99 * x * (1.0 - x))
        Y = X.clone()
        Y[:, -1, 1] = x_new
        Y[:, -1, 2] = 3.0 - y
        return Y, torch.stack([y, 3.0 - y], dim=1)


def starts():
    # rows 0-2 relax from different distances, so they are captured at different steps; rows 3-4 stay chaotic
    return np.array([[1.0, 2.0, 1.0], [1.0, 1.0e4, 1.0], [1.0, 1.0e8, 2.0], [-1.0, 0.3, 1.0], [-1.0, 0.7, 2.0]])


def run(steps, segment, checkpoint=None):
    return bfl.run_long(ToyLoop(), np.zeros(3), 1.0, starts(), steps, segment, max_period=8, tol=1e-9,
                        record_every=10, checkpoint=checkpoint)


def test_segments_continue_one_trajectory():
    whole, cut = run(120, 120), run(120, 20)
    assert np.array_equal(whole["x"], cut["x"])
    assert np.array_equal(whole["rec"], cut["rec"]) and np.array_equal(whole["rec_steps"], cut["rec_steps"])
    assert whole["rec_steps"].tolist() == list(range(0, 121, 10))
    # the same rows are captured, with the same periods, at most max_period steps later in the cut run
    assert (whole["cap"] >= 0).tolist() == [True, True, True, False, False]
    assert np.array_equal(whole["per"], cut["per"]) and whole["per"][:3].tolist() == [2, 2, 2]
    late = cut["cap"][:3] - whole["cap"][:3]
    assert (late >= 0).all() and (late <= 8).all()
    assert whole["cap"][0] < whole["cap"][1] < whole["cap"][2]
    assert whole["left"] == [] and cut["left"] == []


def test_matches_run_chunk_of_batch_fate():
    cap, per, _ctok, cst, final, _mind, _rec, t_end = bf.run_chunk(ToyLoop(), np.zeros(3), 1.0, starts(), 120, 8, 1e-9,
                                                                 10, stop_alone=False)
    whole = run(120, 120)
    assert t_end == 120 and np.array_equal(cap, whole["cap"]) and np.array_equal(per, whole["per"])
    assert np.array_equal(final, whole["x"])
    assert all(np.array_equal(cst[r], whole["cyc_state"][r]) for r in cst)


def test_resume_from_checkpoint(tmp_path):
    ck = tmp_path / "checkpoint.npz"
    first = run(60, 20, checkpoint=ck)
    assert first["t"] == 60 and ck.exists()
    resumed = run(120, 20, checkpoint=ck)           # continues from step 60
    straight = run(120, 20)
    assert resumed["t"] == 120 and resumed["segments"] == 6
    for key in ("x", "cap", "per", "rec", "rec_steps"):
        assert np.array_equal(resumed[key], straight[key]), key
    assert resumed["cyc_tok"] == straight["cyc_tok"]
    assert all(np.array_equal(resumed["cyc_state"][r], straight["cyc_state"][r]) for r in straight["cyc_state"])


def test_stops_when_every_row_is_on_a_cycle():
    st = bfl.run_long(ToyLoop(), np.zeros(3), 1.0, starts()[:3], 400, 100, max_period=8, tol=1e-9, record_every=10)
    assert (st["cap"] >= 0).all() and st["t"] < 400


def test_reports_a_captured_row_that_leaves_its_cycle():
    class Kicked(ToyLoop):
        calls = 0

        def step(self, X, temps):
            Kicked.calls += 1
            Y, p = super().step(X, temps)
            if Kicked.calls == 150:                  # throw row 0 far from its fixed point, in the second segment
                Y[0, -1, 1] = 1.0e6
            return Y, p

    st = bfl.run_long(Kicked(), np.zeros(3), 1.0, starts(), 200, 100, max_period=8, tol=1e-9, record_every=10)
    # the kicked state is recorded at step 150 and has relaxed back by step 200
    assert {e["row"] for e in st["left"]} == {0} and {e["check"] for e in st["left"]} == {"record"}
    assert [e["step"] for e in st["left"]] == [150, 160, 170, 180, 190]


def test_reports_a_captured_row_that_is_off_its_cycle_at_a_segment_start():
    class Kicked(ToyLoop):
        calls = 0

        def step(self, X, temps):
            Kicked.calls += 1
            Y, p = super().step(X, temps)
            if Kicked.calls == 100:                  # the last step of the first segment
                Y[0, -1, 1] = 1.0e6
            return Y, p

    st = bfl.run_long(Kicked(), np.zeros(3), 1.0, starts(), 200, 100, max_period=8, tol=1e-9, record_every=10)
    assert any(e["check"] == "segment start" and e["row"] == 0 and e["step"] == 100 for e in st["left"])


def test_refuses_bad_segmenting():
    with pytest.raises(ValueError):
        run(120, 25)
    with pytest.raises(ValueError):
        bfl.run_long(ToyLoop(), np.zeros(3), 1.0, starts(), 16, 8, max_period=8, tol=1e-9, record_every=8)


def test_census_counts_rows_captured_by_each_horizon():
    cap = np.array([-1, 5, 50, 500, -1])
    assert bfl.census(cap, [10, 100, 1000]) == {"10": 1, "100": 2, "1000": 3}
