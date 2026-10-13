import importlib.util
from pathlib import Path
import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1]/'codex/misalignment-audit'
spec = importlib.util.spec_from_file_location('misalignment_reducers', SRC/'reducers.py')
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)


def test_onsets_and_complete_observation_both_amendments():
    a = np.array([[0,0,0,0,0], [0,0,0,0,1], [0,0,0,1,1], [1,1,0,0,0]], bool)
    np.testing.assert_array_equal(R.event_onsets(a, 1), [5,4,3,0])
    np.testing.assert_array_equal(R.event_onsets(a, 2), [5,5,3,0])
    rows = R.risk_rows(R.event_onsets(a, 2), 5, 2, 2)
    assert rows.tolist() == [[0,0,0],[0,1,0],[1,0,0],[1,1,0],[2,0,0],[2,1,1]]
    assert R.risk_rows(np.array([5]), 5, 3, 1).tolist() == [[0,0,0],[0,1,0]]


def test_capture_strict_threshold_conditional_median_and_stay():
    s = np.zeros((3,40)); s[0,2:5] = .6; s[1,8] = .6; s[2,0] = .5
    r = R.capture_row(s)
    assert r['never'] == 1 and r['median_onset_conditional_zero_based'] == 5
    assert r['median_onset_censored_at_end'] == 8
    assert r['stay'] == {'numerator':2,'denominator':4,'probability':.5}
    assert r['geometric_mean_stay'] == 2
    a = s > .5; a[0,29:32] = 1
    assert R.stay_counts(a,30,40) == {'numerator':1,'denominator':2,'probability':.5}


def brute_auc(y, s):
    p,n = np.asarray(s)[np.asarray(y)==1], np.asarray(s)[np.asarray(y)==0]
    if not len(p) or not len(n): return float('nan')
    return np.mean((p[:,None] > n[None,:]) + .5*(p[:,None] == n[None,:]))


def test_weighted_auc_matches_literal_repeats_with_ties():
    y = np.array([0,1,0,1,1,0]); s = np.array([0,0,2,2,1,3.])
    auc = R.RankedAUC(y,s)
    for w in ([1,1,1,1,1,1],[3,1,0,7,2,5],[0,1,0,4,2,0]):
        idx = np.repeat(np.arange(len(y)),w)
        expected = brute_auc(y[idx],s[idx])
        assert auc.evaluate(w) == pytest.approx(expected,nan_ok=True)


def test_cluster_bootstrap_exact_draws_and_percentiles():
    y=np.array([0,1,0,1,0,1,1,0]); g=np.array([10,10,20,20,40,40,50,50])
    base=np.array([0,1,1,1,0,2,0,0]); full=np.array([0,2,1,3,2,2,0,1])
    rng=np.random.default_rng(0); ds=[]
    for _ in range(51):
        chosen=rng.choice(np.unique(g),len(np.unique(g)))
        idx=np.concatenate([np.flatnonzero(g==gg) for gg in chosen])
        ds.append(brute_auc(y[idx],full[idx])-brute_auc(y[idx],base[idx]))
    r=R.increment(y,g,base,full,51)
    np.testing.assert_allclose(r['ci95'],np.percentile(ds,[2.5,97.5]),atol=1e-15)


def test_gratitude_is_fixed_word_or_substring_proxy():
    assert R.gratitude('Happy to help!') and R.gratitude('I feel thankful.')
    assert not R.gratitude('The thankfulish word is invented.')
    assert R.gratitude('I am not grateful.')  # proxy is not a semantic classifier


def test_difference_resampling_is_independent_not_pairwise():
    a=np.array([0.,1.]); b=np.array([.25,.5,.75])
    r=R.difference_interval(a,b,25,4)
    rng=np.random.default_rng(4)
    exp=[a[rng.integers(0,2,2)].mean()-b[rng.integers(0,3,3)].mean() for _ in range(25)]
    assert r['difference'] == 0
    np.testing.assert_array_equal(r['ci95'],np.percentile(exp,[2.5,97.5]))


def test_controller_counter_magnitude_and_interpolated_efficiency():
    names=['p1-ctl-none','p1-ctl-trig-cancel','p1-ctl-trig-over','p1-ctl-trig-long',
           'p1-ctl-trig-kind','p1-ctl-const-02','p1-ctl-const-04']
    arms={}
    for i,name in enumerate(names):
        score=np.zeros((2,40)); score[0,:]=.9
        counter=np.full((2,40),-.01*i)
        arms[name]={'score':score,'counter':counter,'texts':[['neutral']*40]*2}
    result=R.controller_rows(arms,17)
    assert result['p1-ctl-trig-cancel']['mean_counter'] == pytest.approx(.01)
    assert result['p1-ctl-trig-cancel']['efficiency'] == 0
    assert result['p1-ctl-trig-kind']['dose_outside_constant_range'] is False
