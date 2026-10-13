"""Tests for tools/p1_lead_time.py: AUC, onsets and horizon labels, the absorbing hit probability, rolling
statistics, and a synthetic positive control in which the latent state predicts the onset before the score does."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import p1_lead_time as lt  # noqa: E402


def test_auc_matches_definition():
    y = np.array([0, 0, 1, 1])
    assert lt.auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert lt.auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0
    assert lt.auc(y, np.array([0.5, 0.5, 0.5, 0.5])) == 0.5
    assert np.isnan(lt.auc(np.array([1, 1]), np.array([0.1, 0.2])))


def test_onsets_and_horizon_labels():
    score = np.array([[0.1, 0.2, 0.9, 0.9], [0.1, 0.1, 0.1, 0.1]])
    onset = lt.onsets(score, 0.5)
    assert onset.tolist() == [2, 4]
    rows = lt.horizon_labels(onset, 4, 1)
    # replica 0: turns 0, 1 before onset 2 -> labels 0 (onset not in t+1 = 1) and 1 (onset at t+1 = 2)
    assert [tuple(r) for r in rows if r[0] == 0] == [(0, 0, 0), (0, 1, 1)]
    assert all(r[2] == 0 for r in rows if r[0] == 1)


def test_hit_probability_on_a_chain():
    T = np.array([[0.5, 0.5, 0.0], [0.0, 0.5, 0.5], [0.0, 0.0, 1.0]])
    p = lt.hit_probability(T, np.array([False, False, True]), 2)
    assert np.allclose(p[1], [0.0, 0.5, 1.0])
    assert np.allclose(p[2], [0.25, 0.75, 1.0])


def test_rolling_stats():
    x = np.array([[1.0, 2.0, 3.0, 4.0]])
    var, ac = lt.rolling_stats(x, 3)
    assert np.isnan(var[0, 1]) and abs(var[0, 2] - np.var([1, 2, 3])) < 1e-12
    assert abs(ac[0, 3] - 1.0) < 1e-12


def test_latent_drift_predicts_onset_before_the_score():
    """Positive control: a latent coordinate drifts toward a threshold; the score only jumps at the onset."""
    rng = np.random.default_rng(0)
    R, N, d = 200, 30, 8
    acts = rng.normal(scale=0.1, size=(R, N, d))
    score = np.full((R, N), 0.01)
    for r in range(R):
        start = rng.integers(5, 25)
        ramp = np.clip((np.arange(N) - start) / 4.0, 0, None)
        acts[r, :, 0] += ramp
        onset = start + 4
        if onset < N:
            score[r, onset:] = 0.99
    model = lt.Model(acts[:100], score[:100], 0.5, dims=4, states=12)
    axis = np.zeros(d)
    axis[1] = 1.0                                          # an axis carrying no information about the drift
    base, fc = lt.predictors(model, acts[:100], score[:100], [2], 4, axis)
    comb = lt.Combined(base, fc, lt.onsets(score[:100], 0.5), N, [2])
    res = lt.evaluate(model, acts[100:], score[100:], 0.5, [2], 4, boot=20, axis=axis, combined=comb)
    hz = res["horizons"][2]
    a = hz["auc"]
    assert a["probe"] > 0.8 and a["forecast"] > 0.7 and a["classifier"] < 0.6
    # the dynamical features add value over the baselines when the latent drift carries the warning
    assert hz["incremental_auc"] > 0.1 and hz["incremental_auc_ci95"][0] > 0


def test_horizon_labels_sentinel_and_censoring():
    import p1_lead_time as lt
    turns = 40
    rows = lt.horizon_labels(np.array([turns]), turns, 5)
    assert rows[:, 2].sum() == 0 and rows[:, 1].max() == turns - 1 - 5
