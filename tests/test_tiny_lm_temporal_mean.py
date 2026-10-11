import numpy as np
import pytest
from neural_shrimp import tiny_lm_temporal_mean as T


def test_mean_is_power_of_mean_not_product_and_preserves_action_vector():
    matrices = [np.array([[1+k/16, .2], [.2, 2-k/32]]) for k in range(16)]
    reference = np.c_[np.arange(17), np.zeros(17)]
    u = np.array([.6,.8]); seen=[]; saved={}
    def hvp(x,v):
        seen.append((int(x[0]),v.copy()))
        return matrices[int(x[0])]@v
    answer=T.mean_power(reference,u,.2,hvp,lambda n,v:saved.update({n:v}),lambda:None)
    j=np.eye(2)-.2*np.mean(matrices,axis=0)
    for h in range(17): np.testing.assert_allclose(answer[h],np.linalg.matrix_power(j,h)@u,rtol=1e-14,atol=1e-15)
    assert len(seen)==256
    for h in range(16):
        assert [s[0] for s in seen[16*h:16*h+16]]==list(range(16))
        assert all(np.array_equal(s[1],answer[h]) for s in seen[16*h:16*h+16])
    product=u.copy()
    for m in matrices: product=(np.eye(2)-.2*m)@product
    assert np.linalg.norm(product-answer[-1])>1e-5


def test_control_counts_original_direction_and_no_extra_ce_gradient():
    q=np.diag([1.,2.]); ridge=.001; theta=np.array([.2,.3]); u=np.array([1.,0.])
    counts=dict.fromkeys(T.TARGET_COUNTS,0); saved={}
    def grad(x): counts['gradient']+=1; return q@x
    def hvp(x,v): counts['hvp']+=1; return q@v
    def ce(x): counts['ce']+=1; return .5*x@(q-ridge*np.eye(2))@x
    out=T.acquire_control(theta,u,.1,ridge,grad,hvp,ce,lambda n,v:saved.update({n:v}),lambda:None)
    assert counts==dict(gradient=81,hvp=32,ce=29)
    assert np.array_equal(out['direction'],u)
    assert out['small_step_monotone']
    np.testing.assert_allclose(out['readout_gradients'],[(q-ridge*np.eye(2))@out['reference'][h] for h in T.HORIZONS])
    mean=T.mean_power(out['reference'],u,.1,hvp,lambda n,v:None,lambda:None)
    rows=T.measurements(out,mean)
    assert all(r['mechanism_qualified'] and r['training_ce_slope_resolved'] for r in rows)
    assert all(r['comparisons']['mean']['vector_error']<1e-8 for r in rows)
    assert counts==dict(gradient=81,hvp=288,ce=29)
    assert dict(hvp=2*counts['hvp']+2*256,gradient=2*counts['gradient'],ce=2*counts['ce'])==T.TARGET_COUNTS


def test_control_monotonicity_not_relabelled_and_mean_not_a_qualification():
    theta=np.array([.2]); u=np.array([1.]); saved={}
    case=T.acquire_control(theta,u,3.,0.,lambda x:x,lambda x,v:v,lambda x:float(x@x/2),lambda n,v:saved.update({n:v}),lambda:None)
    assert not case['small_step_monotone']
    case['ordered']=case['ordered']*2
    rows=T.measurements(case,case['frozen'])
    assert not any(r['mechanism_qualified'] for r in rows)


def test_partial_branch_readout_and_nonfinite_mean_remain_saved():
    saved={}; calls=0
    def ce(x):
        nonlocal calls
        calls+=1
        if calls==19: raise RuntimeError('synthetic interrupted readout')
        return float(x@x/2)
    with pytest.raises(RuntimeError):
        T.acquire_control(np.array([.2]),np.array([1.]),.1,0.,lambda x:x,lambda x,v:v,ce,lambda n,v:saved.update({n:v}),lambda:None)
    assert saved['epsilon-0-plus-ce'].shape==(1,2)
    assert saved['epsilon-0-plus-ce'][0,0]==1
    seen=[]
    def bad(x,v): seen.append(v.copy()); return np.array([np.nan])
    with pytest.raises(ValueError,match='finite'):
        T.mean_power(np.ones((17,1)),np.ones(1),.1,bad,lambda n,v:saved.update({n:v}),lambda:None)
    assert len(seen)==1 and saved['mean-powers'].shape==(1,1)


def test_profile_complete_counts_storage_bound_and_fixed_gate():
    times={k:[.01,.02] for k in T.TARGET_COUNTS}
    sizes={k:[126336,126336] for k in T.TARGET_COUNTS};sizes['ce']=[136,136]
    f=T.forecast(times,sizes)
    assert f['seconds']==2*sum(T.TARGET_COUNTS.values())*.02+30
    assert f['bytes']<512*1024**2
    with pytest.raises(ValueError): T.forecast(dict(hvp=[.01,.02]),sizes)
    rows=[dict(horizon=16,mechanism_qualified=True,comparisons=dict(mean=dict(vector_error=.1),frozen=dict(vector_error=2.)))]
    cases=[dict(seed=s,rate_role='original',rows=rows) for s in (28161,28162)]
    assert T.descriptive_gate(cases)
    rows[0]['comparisons']['mean']['vector_error']=.2
    assert not T.descriptive_gate(cases)
