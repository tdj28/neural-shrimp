import numpy as np
import pytest
from neural_shrimp.tiny_lm_ordering import reverse_product,decompose,reciprocal,profile_forecast

def product(mats):
    p=np.eye(len(mats[0]))
    for j in mats:p=j@p
    return p

def test_reversal_transpose_and_equal_singular_spectrum():
    rng=np.random.default_rng(71);mats=[]
    for _ in range(16):
        a=rng.normal(size=(5,5));mats.append(np.eye(5)+(a+a.T)/30)
    dirs=np.linalg.qr(rng.normal(size=(5,2)))[0].T;states=np.arange(16)[:,None]*np.ones((1,5));revs=[]
    for u in dirs:revs.append(reverse_product(states,u,1.,16,lambda t,v:(np.eye(5)-mats[int(t[0])])@v))
    p=product(mats);r=product(mats[::-1]);assert np.allclose(r,p.T)and np.allclose(revs,dirs@p)
    assert np.allclose(np.linalg.svd(p)[1],np.linalg.svd(r)[1])
    assert reciprocal(dirs,dirs@p.T,np.asarray(revs))['passed']
    for u,rev in zip(dirs,revs):
        z,a,b=decompose(u,p@u,rev,np.linalg.matrix_power(mats[0],16)@u,np.ones(5));assert z['algebra_qualified']and z['odd_norm']>0
        assert np.isclose(z['ce_projections']['error'],z['ce_projections']['odd']+z['ce_projections']['even_remainder'])

def test_commuting_constant_product_and_zero_error():
    j=np.diag([.8,.9]);u=np.array([1.,0.]);r=reverse_product(np.zeros((16,2)),u,1.,4,lambda t,v:v-j@v);p=np.linalg.matrix_power(j,4)@u
    z,_,_=decompose(u,p,r,p,np.ones(2));assert z['algebra_qualified']and z['odd_norm']<1e-14 and z['frozen_error_norm']==0

def test_reversal_even_does_not_mean_order_independent():
    a=np.array([[1.,.2],[.2,.7]]);b=np.diag([.4,1.3]);p=a@b@a
    assert np.allclose(p,p.T)and not np.allclose(p,a@a@b)

def test_cancellation_not_percentage_and_ce_projection():
    p=np.array([[1.,3.],[-1.,1.]]);u=np.array([1.,0.]);f=np.array([.9,-.9]);z,a,b=decompose(u,p@u,p.T@u,f,np.array([2.,4.]))
    assert z['odd_to_error']>1 and z['twice_cross_term']<0 and z['algebra_qualified']

def test_failure_preserves_partial_reverse_trace():
    records=[];calls=[]
    def action(t,v):
        calls.append(int(t[0]))
        if len(calls)==3:raise RuntimeError('fixed interruption')
        return v/10
    with pytest.raises(RuntimeError):reverse_product(np.arange(16)[:,None]*np.ones((1,2)),np.ones(2),1.,4,action,lambda x,i:records.append((x,i)))
    assert calls==[3,2,1]and records[0][1]==[3,2]and records[0][0].shape==(3,2)

def test_wrong_reciprocal_and_fixed_costs_rejected():
    u=np.eye(2);assert not reciprocal(u,u,np.array([[1.,1.],[0.,1.]]))['passed']
    assert profile_forecast([dict(arm=8,times=[.1,.2]),dict(arm=256,times=[.3,.4])])==116.
    with pytest.raises(ValueError):profile_forecast([dict(arm=8,times=[.1,.2])])
