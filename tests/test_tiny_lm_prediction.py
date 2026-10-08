import importlib.util
from pathlib import Path

import numpy as np
import pytest

from neural_shrimp.tiny_lm_prediction import (CausalFeatures,plane_coherence,burst_labels,
    fit_logistic,predict,prediction_metrics,family_columns)


def monitor_sequence(losses, gradients):
    monitor=CausalFeatures();theta=np.ones(3);previous=None;rows=[]
    for loss,g in zip(losses,gradients):
        rows.append(monitor.observe(loss,g,theta,previous))
        previous=-.01*g;theta=theta+previous
    return monitor,np.asarray(rows)


def test_plane_coherence_matches_cross_products_and_masks_degeneracy():
    a=np.array([1.,0.,0.]);b=np.array([0.,1.,0.]);c=np.array([0.,0.,1.])
    expected=np.cross(a,b)@np.cross(b,c)
    assert plane_coherence(a,b,c)==pytest.approx(expected)
    angles=np.array([0.,.3,.6]);points=np.c_[np.cos(angles),np.sin(angles),np.zeros(3)]
    assert plane_coherence(*points)==pytest.approx(1.)
    assert np.isnan(plane_coherence(a,-a,a))
    assert np.isnan(plane_coherence(a,a,a))
    assert np.isnan(plane_coherence(a,(a+1e-7*b)/np.linalg.norm(a+1e-7*b),b))


def test_online_features_are_prefix_invariant_and_angle_is_redundant():
    t=np.arange(80);g=np.c_[np.cos(.2*t),np.sin(.2*t),.01*t]
    first,X=monitor_sequence(4-.01*t,g)
    _,prefix=monitor_sequence(4-.01*t[:50],g[:50])
    np.testing.assert_equal(X[:50],prefix)
    units=-g/np.linalg.norm(g,axis=1)[:,None]
    j=first.names.index('consecutive_cosine')
    np.testing.assert_allclose(X[1:,j],np.sum(units[1:]*units[:-1],axis=1))
    assert set(family_columns(first.groups,'B2'))<set(family_columns(first.groups,'B3'))


def test_labels_horizon_boundary_guard_and_censoring():
    losses=np.ones(100);losses[60]=2
    y=burst_labels(losses)
    assert np.isnan(y[:31]).all()
    assert y[43]==0 and y[44]==1 and y[59]==1 and y[60]==0
    assert np.isnan(y[84:]).all()
    guarded=burst_labels(np.ones(50),guard_step=50)
    assert guarded[33]==0 and (guarded[34:]==1).all()
    assert np.isnan(burst_labels(np.ones(50))[34:]).all()


def test_logistic_training_only_scaling_and_tied_scores():
    X=np.c_[np.linspace(-2,2,100),np.full(100,np.nan)]
    y=(X[:,0]>0).astype(float)
    model=fit_logistic(X[:80],y[:80],.1)
    assert model['mean'][0]==pytest.approx(X[:80,0].mean())
    assert model['mean'][1]==0
    p=predict(model,X)
    assert p[0]<p[-1] and np.isfinite(p).all()
    metrics=prediction_metrics(np.array([0,1,0,1]),np.array([.5,.5,.5,.5]),.6)
    assert metrics['average_precision']==pytest.approx(.5)
    assert metrics['false_alarms_per_100']==0


def test_prediction_collection_exposes_only_calibration_and_profile():
    path=Path(__file__).resolve().parents[1]/'tools/explore_tiny_lm_prediction.py'
    import sys
    sys.path.insert(0,str(path.parent))
    spec=importlib.util.spec_from_file_location('prediction_runner_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.validate_request('profile',285)==((28151,),(.8,),100,60)
    assert module.validate_request('calibration',285)==((28151,28152),(.8,1.6,2.5),800,285)
    for bad in ['assessment','policy','29000']:
        with pytest.raises(ValueError):module.validate_request(bad,285)
    with pytest.raises(ValueError):module.validate_request('calibration',301)


def test_prediction_fit_cannot_use_later_test_features_and_embargo_is_exact():
    path=Path(__file__).resolve().parents[1]/'tools/analyze_tiny_lm_prediction.py'
    spec=importlib.util.spec_from_file_location('prediction_analysis_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    records=[]
    for seed in (28151,28152):
        ce=np.ones(800)
        records.append({'groups':['B0'],'X':np.arange(800)[:,None].astype(float),
                        'y':burst_labels(ce),'ce':ce,'seed':seed,'factor':.8,'guard_step':None,'role':'calibration'})
    a=module.select_model(records,'B0')
    for r in records:r['X'][600:]+=1e12
    b=module.select_model(records,'B0')
    assert a==b
    assert np.flatnonzero(module.split_mask(records[0],'fit'))[-1]+16==399
    assert np.flatnonzero(module.split_mask(records[0],'tune'))[0]==400
    assert np.flatnonzero(module.split_mask(records[0],'test'))[-1]+16==799
    with pytest.raises(ValueError,match='exactly six'):module.analyze(records)


def test_b3_all_history_moments_are_functions_of_retained_b2_history():
    t=np.arange(80);g=np.c_[np.cos(.2*t+.0003*t*t),np.sin(.2*t+.0003*t*t),.01*t]
    monitor,X=monitor_sequence(4-.01*t,g)
    names={name:i for i,name in enumerate(monitor.names)}
    row=X[-1];planes=[]
    for lag in range(8):
        prefix='gram' if lag==0 else f'gram_history_lag{lag}'
        ab,ac,bc=[row[names[f'{prefix}_{j}']] for j in (3,4,5)]
        planes.append((ab*bc-ac)/np.sqrt((1-ab*ab)*(1-bc*bc)))
    assert row[names['plane_coherence']]==pytest.approx(planes[0])
    assert row[names['plane_mean8']]==pytest.approx(np.mean(planes))
    assert row[names['plane_std8']]==pytest.approx(np.std(planes))


def test_analysis_manifest_rejects_unhashed_consumed_files_and_unsafe_paths(tmp_path):
    path=Path(__file__).resolve().parents[1]/'tools/analyze_tiny_lm_prediction.py'
    spec=importlib.util.spec_from_file_location('prediction_inventory_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    (tmp_path/'features.npz').write_bytes(b'nonempty')
    with pytest.raises(ValueError,match='exact collection'):
        module.verify_inventory(tmp_path,{'files':[]})
    with pytest.raises(ValueError,match='duplicate'):
        module.verify_inventory(tmp_path,{'files':[{'path':'features.npz'},{'path':'features.npz'}]})
