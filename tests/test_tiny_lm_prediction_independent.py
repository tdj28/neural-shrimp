"""Independent synthetic review controls; no corpus or calibration trajectory."""
import importlib.util
from pathlib import Path
import time
import sys
import json
import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'tools'))
from neural_shrimp.tiny_lm_prediction import CausalFeatures,burst_labels,fit_logistic,predict


def load(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'tools'/filename)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


analysis=load('independent_prediction_analysis','analyze_tiny_lm_prediction.py')
runner=load('independent_prediction_runner','explore_tiny_lm_prediction.py')


def test_label_embargo_is_disjoint_from_next_split():
    original=np.ones(800)
    baseline=burst_labels(original)
    changed=original.copy();changed[400:]=100
    altered=burst_labels(changed)
    np.testing.assert_equal(baseline[32:384],altered[32:384])
    changed=original.copy();changed[600:]=100
    altered=burst_labels(changed)
    np.testing.assert_equal(baseline[400:584],altered[400:584])
    # Deliberately change the final included future observation: it must matter.
    changed=original.copy();changed[399]=100
    assert burst_labels(changed)[383]==1 and baseline[383]==0
    changed=original.copy();changed[599]=100
    assert burst_labels(changed)[583]==1 and baseline[583]==0


def test_training_statistics_and_selection_ignore_test_outcomes():
    t=np.arange(800)
    ce=1+.2*((t%71)==0)
    X=np.c_[np.sin(.17*t),np.cos(.07*t),t.astype(float)]
    records=[dict(groups=['B0']*3,X=X.copy(),y=burst_labels(ce),ce=ce.copy(),seed=s,
                  factor=.8,guard_step=None,role='calibration') for s in (28151,28152)]
    first=analysis.select_model(records,'B0')
    expected=X[32:384].mean(0)
    np.testing.assert_allclose(first['model']['mean'],expected,rtol=0,atol=1e-12)
    for r in records:
        r['X'][600:]=1e10
        r['ce'][600:]=1e12
        r['y'][600:]=1-r['y'][600:]
    second=analysis.select_model(records,'B0')
    assert first==second
    # Applying an already fitted transform does not mutate its statistics.
    before=json.dumps(first,sort_keys=True)
    predict(first['model'],np.full((3,3),1e100))
    assert json.dumps(first,sort_keys=True)==before


def test_degenerate_geometric_features_abstain_and_prefix_is_exact():
    def stream(count,changes=False):
        monitor=CausalFeatures();rows=[];theta=np.ones(3);previous=None
        for t in range(count):
            gradient=np.array([1.,0,0])*(-1 if t%2 else 1)
            loss=1+.001*t
            if changes and t>=55:gradient=np.array([1.,2.,3.]);loss=999
            rows.append(monitor.observe(loss,gradient,theta,previous))
            previous=-.01*gradient;theta=theta+previous
        return monitor,np.asarray(rows)
    monitor,X=stream(80);_,changed=stream(80,True)
    np.testing.assert_equal(X[:55],changed[:55])
    for name in ('plane_coherence','plane_mean8','plane_std8'):
        assert np.isnan(X[:,monitor.names.index(name)]).all()
    for name in ('plane_unavailable','plane_history_unavailable'):
        assert (X[:,monitor.names.index(name)]==1).all()
    common=[g=='common' for g in monitor.groups]
    assert sum(common)>=2


class Quadratic:
    ridge=0.
    def loss(self,x):return .5*(x*x).sum()


def test_numerical_guard_retains_offending_state_and_uses_only_future_labels(tmp_path):
    folder=tmp_path/'guard'
    result=runner.collect_case(Quadratic(),torch.tensor([1e-16],dtype=torch.float64),4.,28151,.8,
                               100,folder,time.monotonic()+10,'synthetic-review')
    assert result['outcome']=='numerical_guard_stop'
    stop=result['guard_step'];assert 31<stop<100
    states=np.concatenate([np.load(p) for p in sorted(folder.glob('states-*.npy'))])
    assert len(states)==stop+1==result['state_rows']
    assert .5*(states[-1]**2).sum()>1e4
    with np.load(folder/'features.npz') as z:
        assert len(z['ce'])==stop
        assert (z['burst'][max(31,stop-16):stop]==1).all()
        np.testing.assert_equal(z['burst'],burst_labels(z['ce'],stop))
    assert json.loads((folder/'guard.json').read_text())['step']==stop


def test_exhausted_wall_budget_retains_initial_state_without_labeling_guard(tmp_path):
    folder=tmp_path/'wall'
    initial=torch.tensor([.25],dtype=torch.float64)
    result=runner.collect_case(Quadratic(),initial,.1,28151,.8,800,folder,time.monotonic()-1,'synthetic-review')
    assert result['outcome']=='wall_budget_stop'
    assert result['guard_step'] is None and result['completed_updates']==0
    states=np.concatenate([np.load(p) for p in sorted(folder.glob('states-*.npy'))])
    np.testing.assert_equal(states,initial.numpy()[None,:])
    with np.load(folder/'features.npz') as z:assert not len(z['burst'])


def test_nonfinite_or_out_of_scope_budget_requests_fail_closed():
    for value in (float('nan'),float('inf'),-1,0,286):
        with pytest.raises(ValueError):runner.validate_request('calibration',value)
    for mode in ('assessment','test','policy','29000'):
        with pytest.raises(ValueError):runner.validate_request(mode,60)


def test_all_retained_gram_histories_reconstruct_nonlinear_features():
    rng=np.random.default_rng(732)
    monitor=CausalFeatures();theta=np.ones(7);previous=None
    for t in range(90):
        gradient=rng.normal(size=7)
        row=monitor.observe(1+.03*np.sin(t),gradient,theta,previous)
        previous=-.01*gradient;theta=theta+previous
        if t<32:continue
        names={name:i for i,name in enumerate(monitor.names)}
        planes=[]
        for lag in range(8):
            prefix='gram' if lag==0 else f'gram_history_lag{lag}'
            ab,ac,bc=[row[names[f'{prefix}_{j}']] for j in (3,4,5)]
            planes.append(np.clip((ab*bc-ac)/np.sqrt((1-ab*ab)*(1-bc*bc)),-1,1))
        np.testing.assert_allclose(row[[names[n] for n in ('plane_coherence','plane_mean8','plane_std8')]],
                                   [planes[0],np.mean(planes),np.std(planes)],rtol=1e-12,atol=1e-14)


def test_manifest_exactness_digest_and_symlinks(tmp_path):
    import hashlib
    p=tmp_path/'features.npz';p.write_bytes(b'payload')
    good={'files':[{'path':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}]}
    analysis.verify_inventory(tmp_path,good)
    p.write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='manifest mismatch'):analysis.verify_inventory(tmp_path,good)
    p.write_bytes(b'payload')
    extra=tmp_path/'result.json';extra.write_bytes(b'{}')
    with pytest.raises(ValueError,match='exact collection'):analysis.verify_inventory(tmp_path,good)
    extra.unlink();p.unlink();outside=tmp_path.parent/'outside_review_payload';outside.write_bytes(b'payload');p.symlink_to(outside)
    with pytest.raises(ValueError,match='unsafe|mismatch'):analysis.verify_inventory(tmp_path,good)


def test_interruption_after_accepted_update_retains_terminal_state(tmp_path,monkeypatch):
    original=runner.save_json
    def fail_progress(path,data):
        if path.name=='progress-000050.json':raise RuntimeError('synthetic checkpoint interruption')
        return original(path,data)
    monkeypatch.setattr(runner,'save_json',fail_progress)
    folder=tmp_path/'interruption'
    with pytest.raises(RuntimeError,match='synthetic checkpoint'):
        runner.collect_case(Quadratic(),torch.tensor([1.],dtype=torch.float64),.1,28151,.8,
                            100,folder,time.monotonic()+10,'synthetic-review')
    result=json.loads((folder/'result.json').read_text())
    assert result['outcome']=='failed_or_interrupted' and result['completed_updates']==50
    states=np.concatenate([np.load(p) for p in sorted(folder.glob('states-*.npy'))])
    assert len(states)==51==result['state_rows']
    assert states[-1,0]==pytest.approx(.9**50,rel=1e-13)
