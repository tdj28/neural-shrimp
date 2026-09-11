import copy
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_hopfield_holdout as h

def plan():
    return dict(experiment_id='EXP-007',anchors=[{'id':f'holdout-{i:02}'} for i in range(12)],
       arms=h.ARMS,radii=h.RADII,phases=h.PHASES,dt=[.01,.005],primary_dt=.005,decision_dt=.005,
       preview_duration=75,probe_duration=300,sample_dt=.1,phase_grid=128,workers=4,wall_clock_ceiling_seconds=1800,
       initial_state=[.1,.1,.1],burn_time=10000,sample_time=2000,event_capacity=128,max_peak_period=32,minimum_repeats=4,
       recurrence_atol=2e-6,recurrence_rtol=2e-5,hybrid_local_through_radius=.004,bootstrap_samples=10000,
       bootstrap_seed=20260912,paid_compute=False,paid_review=False)

def test_exact_census_and_disjoint_phase_starts():
    p=plan();rows=list(h.specs(p))
    assert h.validate(p)==len(rows)==2784
    assert len({h.trial_id(s) for s in rows})==2784
    assert not set(h.PHASES)&{0,675,1350,2025}
    p['preview_duration']=76
    with pytest.raises(AssertionError):h.validate(p)

def preview_rows():
    return [dict(candidate=i,status='complete',score=float(21-i)/100) for i in range(21)]

def test_signs_axes_search_ties_and_hybrid():
    rows=preview_rows();choice=h.choose_policies(rows,.004)
    assert choice['local']['candidate']==17 and choice['tangent']['candidate']==19
    assert choice['axes']['candidate']==12 and choice['search']['candidate']==20
    assert choice['hybrid']['candidate']==17
    assert h.choose_policies(rows,.016)['hybrid']['candidate']==20
    for r in rows:r['score']=1.
    choice=h.choose_policies(rows,.001)
    assert choice['local']['candidate']==16 and choice['search']['candidate']==0
    rows[16]['status']='failed';rows[16]['score']=None
    choice=h.choose_policies(rows,.001)
    assert choice['local']['status']==choice['search']['status']==choice['hybrid']['status']=='selector_failed'
    assert choice['tangent']['status']=='complete' and choice['atlas']['candidate']==20

def test_candidate_metric_native_units_exact_axes():
    a={'tangent':[0.,1.],'atlas_margin':{'.001':{'direction':[1.,0.]}}}
    # JSON uses str(radius), which includes the leading zero.
    a['atlas_margin']={'0.001':{'direction':[1.,0.]}}
    ds=h.candidate_directions(a,np.array([.6,.8]),.001)
    assert ds.shape==(21,2)
    np.testing.assert_allclose(np.linalg.norm(ds,axis=1),1,rtol=0,atol=1e-15)
    np.testing.assert_array_equal(ds[[0,4,8,12]],[[1,0],[0,1],[-1,0],[0,-1]])
    np.testing.assert_array_equal(ds[16:20],[[.6,.8],[-.6,-.8],[0,1],[0,-1]])
    np.testing.assert_array_equal(ds[0],ds[20]) # Deliberate duplicate retained.

def complete_records(p):
    rows=[]
    for s in h.specs(p):
        value=0.2 if s['arm']=='hybrid' else .3
        rows.append({**s,'id':h.trial_id(s),'status':'complete','tail_metrics':{'aligned_rms':{'aggregate':value}}})
    return rows

def test_equal_weight_summary_and_no_success_only_deletion():
    p=plan();rows=complete_records(p);eligible=[a['id'] for a in p['anchors']]
    summary=h.summarize(rows,p,eligible)
    panel=summary['steps']['0.005']['panels']['pooled']
    assert panel['comparator_minus_hybrid']['local']['benefit']==pytest.approx(.1)
    assert panel['comparator_minus_hybrid']['local']['ci95']==pytest.approx([.1,.1])
    row=next(r for r in rows if r['arm']=='local' and r['dt']==.005)
    row['status']='failed'
    panel=h.summarize(rows,p,eligible)['steps']['0.005']['panels']['pooled']
    assert panel['means']['local'] is None
    assert panel['comparator_minus_hybrid']['local']['status']=='undefined_incomplete'
    with pytest.raises(AssertionError):h.summarize(rows[:-1],p,eligible)

def test_unevaluable_anchors_retained_not_replaced():
    p=plan();rows=complete_records(p);eligible=[a['id'] for a in p['anchors']][1:]
    for r in rows:
        if r['anchor']=='holdout-00':r['status']='baseline_not_evaluable'
    result=h.summarize(rows,p,eligible)
    assert result['status_counts']['baseline_not_evaluable']==232
    assert len(result['steps']['0.005']['panels']['pooled']['per_anchor']['hybrid'])==11

def test_target_excludes_all_preview_samples(monkeypatch,tmp_path):
    p=plan();s=dict(anchor='a',phase=337,radius=.001,arm='atlas',dt=.005)
    times=np.arange(3001)*.1;states=np.zeros((3001,3));initial=np.zeros(3)
    monkeypatch.setattr(h,'integrate_trace',lambda *a,**k:(times,states))
    monkeypatch.setattr(h,'atomic_npz',lambda *a,**k:None)
    received=[]
    def metric(ref,t,act,T,**kw):
        received.append(t.copy());return {'aligned_rms':{'aggregate':.1}}
    monkeypatch.setattr(h,'waveform_errors',metric)
    row=h.run_target(s,{'alpha':1.,'gamma':2.},initial,np.zeros((2700,3)),10.,3.,np.array([1.,0.]),None,p,tmp_path)
    assert row['status']=='complete'
    assert len(received[0])==2250 and received[0][0]==pytest.approx(78.1) and received[0][-1]==303.
    assert np.all(received[0]-3>75) and len(received[1])==3001


def test_failed_zero_has_no_success_only_floor():
    p=plan();rows=complete_records(p);eligible=[a['id'] for a in p['anchors']]
    next(r for r in rows if r['arm']=='zero' and r['dt']==.005)['status']='failed'
    zero=h.summarize(rows,p,eligible)['steps']['0.005']['zero']
    assert zero['complete']==47 and zero['planned']==48
    assert zero['mean'] is None and zero['maximum'] is None


@pytest.mark.parametrize('bad',[None,True,False,-.1,2.1,float('nan'),float('inf'),'0.1'])
def test_malformed_preview_invalidates_only_affected_selectors(bad):
    rows=preview_rows();rows[16]['score']=bad
    choice=h.choose_policies(rows,.001)
    assert choice['local']['status']==choice['search']['status']==choice['hybrid']['status']=='selector_failed'
    assert choice['atlas']['status']==choice['tangent']['status']==choice['uniform']['status']=='complete'
