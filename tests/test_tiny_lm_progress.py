"""Synthetic controls only; no development corpus or target training outcomes."""
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
import pytest
import torch

from neural_shrimp.tiny_lm_progress import (CORPUS_BYTES,TRAIN_STARTS,PROBE_STARTS,ar_progress,
    baseline_series,fit_scale,load_fixed_data,progress_targets,ridge_fit,ridge_predict,score,
    select_progress_model,select_windows,split_indices,token_hash,validate_intervals)

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))


def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/filename)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


runner=module('broader_runner_test','explore_tiny_lm_broader_data.py')
analysis=module('broader_analysis_test','analyze_tiny_lm_broader_data.py')


def test_data_intervals_token_alignment_and_identity_fail_closed(tmp_path):
    assert len(validate_intervals())==288
    raw=(b'abcdefg'*(CORPUS_BYTES//7+1))[:CORPUS_BYTES]
    x,y,identity=select_windows(raw,list('abcdefg')+['<UNK>'],TRAIN_STARTS)
    assert x.shape==y.shape==(256,32)
    np.testing.assert_equal(x[:,1:],y[:,:-1])
    assert identity['byte_character_intervals'][-1]==[130816,130849]
    assert identity['targets_per_evaluation']==8192
    rows=np.c_[x,y[:,-1]]
    assert token_hash(rows)==identity['window_tokens_sha256']
    cx,cy,ci=select_windows(raw,list('abcdefg')+['<UNK>'],TRAIN_STARTS[:8])
    np.testing.assert_equal(cx,x[:8]);np.testing.assert_equal(cy,y[:8])
    assert ci['targets_per_evaluation']*800==identity['targets_per_evaluation']*25
    assert PROBE_STARTS[-1]+33==204577
    (tmp_path/'input.txt').write_bytes(raw);(tmp_path/'receipt.json').write_text('{"corpus_sha256":"bad"}')
    with pytest.raises(ValueError,match='identity'):load_fixed_data(tmp_path)
    changed=bytearray(raw);changed[TRAIN_STARTS[0]]=255
    with pytest.raises(UnicodeDecodeError):select_windows(changed,list('abcdefg'),TRAIN_STARTS[:1])


def test_progress_target_embargo_censoring_and_nonfinite_horizon():
    ce=2+.1*np.sin(np.arange(800)*.21);base=progress_targets(ce)
    assert len(split_indices(800,'fit'))==352
    assert len(split_indices(800,'tune'))==len(split_indices(800,'test'))==184
    other=ce.copy();other[400:]+=100
    np.testing.assert_equal(base[32:384],progress_targets(other)[32:384])
    other=ce.copy();other[600:]+=100
    np.testing.assert_equal(base[400:584],progress_targets(other)[400:584])
    assert base[783]==pytest.approx(ce[799]-ce[783])
    assert np.isnan(base[784:]).all()
    bad=ce.copy();bad[50]=np.nan
    assert np.isnan(progress_targets(bad)[34:51]).all()


def test_ar_forecast_is_causal_and_rank_failures_do_not_fallback(monkeypatch):
    t=np.arange(90);ce=3+np.cos(.2*t)
    p,status=ar_progress(ce[:32],2)
    assert status=='ok' and p==pytest.approx(ce[47]-ce[31],abs=1e-10)
    a,sa=baseline_series(ce,2);other=ce.copy();other[60:]+=1000
    b,sb=baseline_series(other,2)
    np.testing.assert_equal(a[:60],b[:60]);np.testing.assert_equal(sa[:60],sb[:60])
    for order in (1,2):
        p,status=ar_progress(np.ones(32),order)
        assert np.isnan(p) and status=='rank_deficient'
    p,status=ar_progress(np.r_[np.ones(31),np.nan],1)
    assert np.isnan(p) and status=='nonfinite_history'
    monkeypatch.setattr(np.linalg,'lstsq',lambda *a,**k:(np.array([0.,1e308]),None,2,None))
    p,status=ar_progress(np.arange(32,dtype=float),1)
    assert np.isnan(p) and status=='numerical_failure'


def synthetic_record(seed,arm):
    t=np.arange(800);ce=2+.03*np.sin(.13*t)+.01*np.cos(.29*t)-.0001*t
    X=np.c_[ce,np.sin(.13*t),np.log(np.repeat(arm*32,800))]
    return {'seed':seed,'arm':arm,'role':'collect','eta':.2,'initial_state_sha256':str(seed),
            'data_identity_sha256':'one-fixed-pool','X':X,'ce':ce,'y':progress_targets(ce),
            'groups':['B0','B0','common'],'outcome':'completed','completed_updates':800,
            'final_train_ce':float(ce[-1]),'probe':[{'step':s,'ce':2.,'status':'ok'} for s in (0,25,800)]}


def test_training_only_fit_scale_imputation_and_selection_ignore_test_changes():
    pair=[synthetic_record(28161,a) for a in (8,256)]
    before=select_progress_model(pair,'B0');scale=fit_scale(pair[0])
    for r in pair:
        r['X'][600:]+=1e8;r['ce'][600:]+=100;r['y']=progress_targets(r['ce'])
    assert before==select_progress_model(pair,'B0') and scale==fit_scale(pair[0])
    X=np.c_[np.arange(50.),np.full(50,np.nan)];y=np.arange(50.)*.2
    model=ridge_fit(X,y,.1)
    assert model['mean']==[24.5,0.]
    assert np.isfinite(ridge_predict(model,X)).all()
    X[0,1]=np.inf
    with pytest.raises(ValueError,match='infinite'):ridge_fit(X,y,.1)


def test_nonfinite_comparator_keeps_rows_and_invalidates_positive_interpretation(monkeypatch):
    s=score(np.array([0.,1.]),np.array([0.,np.nan]),.01,2)
    assert not s['complete'] and s['rows']==2 and s['invalid_row_indices']==[1] and s['normalized_mse'] is None
    records=[synthetic_record(seed,arm) for seed in (28161,28162) for arm in (8,256)]
    original=analysis.baseline_series
    def broken(ce,order):
        p,status=original(ce,order);p[620]=np.nan;status[620]='rank_deficient';return p,status
    monkeypatch.setattr(analysis,'baseline_series',broken)
    result=analysis.analyze(records)
    assert not result['positive_forecast_interpretation']
    for r in result['seeds'].values():
        assert not r['all_comparators_and_eligible_rows_valid']
        assert r['scores']['AR1']['256']['test']['failed_forecast_steps']==[620]
        assert len(r['scores']['AR1']['256']['test']['prediction'])==184
    records[0]['eta']=.3
    with pytest.raises(ValueError,match='exact rate'):analysis.analyze(records)


class Quadratic:
    ridge=0.
    targets=torch.zeros((1,32),dtype=torch.int64)
    def loss(self,x):return .5*(x*x).sum()
    def cross_entropy(self,x):return self.loss(x)


class Probe:
    def __init__(self,value):self.value=value
    def cross_entropy(self,x):return torch.as_tensor(self.value,dtype=x.dtype)


def states(folder):return np.concatenate([np.load(p) for p in sorted(folder.glob('states-*.npy'))])


def test_probe_is_descriptive_and_cannot_change_training_states(tmp_path):
    results=[]
    for index,value in enumerate((1.,99999.)):
        folder=tmp_path/str(index)
        results.append(runner.collect_case(Quadratic(),Probe(value),torch.tensor([1.],dtype=torch.float64),
            .1,28160,8,100,folder,time.monotonic()+10,'synthetic','shared-data'))
    np.testing.assert_equal(states(tmp_path/'0'),states(tmp_path/'1'))
    assert [p['step'] for p in results[0]['probe']]==[0,25,100]
    assert results[0]['state_rows']==101 and results[0]['completed_updates']==100
    assert results[1]['outcome']=='completed'


def test_interrupted_broader_collector_retains_terminal_state(tmp_path,monkeypatch):
    original=runner.save_json
    def broken(path,data):
        if path.name=='progress-000050.json':raise runner.CollectionInterrupted('synthetic stop')
        original(path,data)
    monkeypatch.setattr(runner,'save_json',broken)
    with pytest.raises(runner.CollectionInterrupted):
        runner.collect_case(Quadratic(),Probe(1.),torch.tensor([1.],dtype=torch.float64),
            .1,28160,8,100,tmp_path/'case',time.monotonic()+10,'synthetic','shared-data')
    saved=states(tmp_path/'case');result=json.loads((tmp_path/'case/result.json').read_text())
    assert result['completed_updates']==50 and result['state_rows']==51
    assert saved[-1,0]==pytest.approx(.9**50,rel=1e-13)


def test_expired_budget_and_profile_resource_gate_ignore_learning_quality(tmp_path):
    r=runner.collect_case(Quadratic(),Probe(1.),torch.tensor([1.]),.1,28160,8,100,
       tmp_path/'expired',time.monotonic()-1,'synthetic','shared-data')
    assert r['completed_updates']==0 and r['state_rows']==1 and r['outcome']=='wall_budget_stop'
    rows=[{'outcome':'completed','completed_updates':100,'elapsed_seconds':1.} for _ in range(2)]
    a=runner.profile_gate(rows,1.,1024,10000)
    for row in rows:row['final_train_ce']=1e9;row['probe']=[{'ce':1e20}]
    assert a==runner.profile_gate(rows,1.,1024,10000) and a['pass']
    assert not runner.profile_gate(rows,1.,7*1024**3,10000)['pass']
    rows[0]['elapsed_seconds']=100
    assert not runner.profile_gate(rows,1.,1024,10000)['pass']
    assert runner.validate_request('profile',60)==((28160,),100)
    assert runner.validate_request('collect',450)==((28161,28162),800)
    for mode,cap in [('profile',61),('collect',451),('assessment',30),('collect',np.nan)]:
        with pytest.raises(ValueError):runner.validate_request(mode,cap)


def test_profile_reference_requires_identical_source_data_and_device(tmp_path,monkeypatch):
    identity={'commit':'synthetic','sha256':{'source.py':'abc'}}
    monkeypatch.setattr(runner,'committed_identity',lambda:identity)
    results=[{'seed':28160,'arm':a,'outcome':'completed','completed_updates':100,'elapsed_seconds':1.} for a in (8,256)]
    values={'plan.json':{'mode':'profile','device':'cpu','source':{'source_identity':identity}},
            'data.json':{'data_identity_sha256':'pool'},
            'summary.json':{'results':results,'peak_cuda_allocated_bytes':0},
            'initial-ritz-28160.json':{'elapsed_seconds':1.},
            'feasibility.json':runner.profile_gate(results,1.,0,1000)}
    for name,value in values.items():runner.save_json(tmp_path/name,value)
    runner.manifest(tmp_path,{})
    assert runner.verify_profile(tmp_path,'pool','cpu')['recomputed_feasibility']['pass']
    with pytest.raises(ValueError,match='identity'):runner.verify_profile(tmp_path,'other','cpu')
    with pytest.raises(ValueError,match='identity'):runner.verify_profile(tmp_path,'pool','cuda')
    monkeypatch.setattr(runner,'committed_identity',lambda:{'commit':'changed','sha256':{'source.py':'different'}})
    with pytest.raises(ValueError,match='identity'):runner.verify_profile(tmp_path,'pool','cpu')
    (tmp_path/'unhashed.bin').write_bytes(b'no')
    with pytest.raises(ValueError,match='exact input'):runner.verify_profile(tmp_path,'pool','cpu')
