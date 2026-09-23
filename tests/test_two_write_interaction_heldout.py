"""Fixed held-out census and uncertainty contracts; synthetic outcomes only."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_two_write_interaction_heldout as h
import run_two_write_interaction as d


def rows():
    return [dict(setting=s,phase_index=i,amplitude=a,observation=o,window=w,
        qualified=True,scores=dict(second=.2 if s=='failed-near-edge' else .01,static=0.),
        oracle_r_squared=.2 if s=='failed-near-edge' else .01,
        direct_contrast_norm=1.,contrast_numerical_error_norm=1e-8)
        for s in h.SETTINGS for i in range(8,40) for a in h.AMPLITUDES
        for o in ('state','activation') for w in h.WINDOWS]


def test_closed_reserved_census_matches_original_phase_draws():
    tasks=h.census()
    assert {k:len(v) for k,v in tasks.items()}==dict(phases=4,references=128,differences=0,finite=1536)
    ids=[t['id'] for ts in tasks.values() for t in ts]
    assert len(ids)==1668 and len(set(ids))==1668
    assert h.phases()==d.phases()
    assert all(int(t['case']['id'].rsplit('--p',1)[1]) in range(8,40) for k,ts in tasks.items() if k!='phases' for t in ts)
    bench=h.benchmark_tasks(tasks['finite'])
    assert len(bench)==32 and sum(t['case']['id'].startswith('core') for t in bench)==16
    assert all(int(t['case']['id'].rsplit('--p',1)[1]) in range(8,12) for t in bench)
    assert h.sensitivity.METHODS==d.sensitivity.METHODS
    for key in ('state_tolerance','coefficient_tolerance','contrast_tolerance','risk_tolerance','sigma'):
        assert h.CONTRACT[key]==d.CONTRACT[key]


def test_constant_phase_scores_have_exact_mean_and_degenerate_interval():
    result=h.aggregate(rows())
    assert len(result['groups'])==24 and result['status']=='qualified_fixed_panel'
    assert result['primary']['mean_r_squared']==pytest.approx(.2)
    assert result['primary']['phase_bootstrap_95']==pytest.approx([.2,.2])
    assert result['paired_primary_edge_minus_core']['phase_bootstrap_95']==pytest.approx([.19,.19])
    assert result['engineering_continuation_trigger'] is True


def test_missing_duplicate_or_unqualified_phase_cannot_produce_primary_claim():
    source=rows()
    with pytest.raises(ValueError,match='census'):h.aggregate(source[:-1])
    with pytest.raises(ValueError,match='census'):h.aggregate(source[:-1]+[source[0]])
    source[0]['qualified']=False
    assert h.aggregate(source)['primary_inference'] is None


def test_negative_scores_are_not_clipped_or_dropped():
    source=rows()
    for r in source:r['scores']['second']=-.5
    result=h.aggregate(source)
    assert result['primary']['mean_r_squared']==-.5
    assert result['primary']['phase_bootstrap_95']==[-.5,-.5]
    assert result['engineering_continuation_trigger'] is False


def test_phase_worker_receives_only_the_reserved_32_fractions(monkeypatch):
    seen={}
    def fake(rhs,jac,section,z,T,fractions,method):
        seen['fractions']=fractions.copy();return {'synthetic':True}
    monkeypatch.setattr(h.sensitivity,'phase_contexts',fake)
    context=dict(W=np.zeros((3,3)),section=np.ones(3),terminal_adjoint=np.ones(3),period=1.)
    assert h.execute_payload(h.task('phase','core','primary'),context)=={'synthetic':True}
    np.testing.assert_array_equal(seen['fractions'],[p['fraction'] for p in d.phases()[8:]])


def test_phase_seal_preserves_global_indices_and_rejects_truncation():
    contexts=[dict(initial=np.array([1.,0.,0.]),direction=np.array([1.,0.,0.]),
        phase_gradient=np.array([0.,1.,0.]),phase_fraction=p['fraction'],period=1.,coordinate=0,
        normalization_residual=0.,direction_leakage=0.) for p in h.phases()[8:]]
    ps={h.task('phase','core',m)['id']:dict(contexts=contexts,closure=0.) for m in h.sensitivity.METHODS}
    sealed,checks=h.phase_seal({'core':{'W':np.zeros((3,3))}},ps)
    assert set(sealed)=={f'core--p{i:02d}' for i in range(8,40)}
    assert sealed['core--p39']['index']==39 and all(r['qualified'] for r in checks)
    ps[h.task('phase','core','primary')['id']]['contexts']=contexts[:-1]
    with pytest.raises(ValueError,match='Thirty-two'):h.phase_seal({'core':{'W':np.zeros((3,3))}},ps)
