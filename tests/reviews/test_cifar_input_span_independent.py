"""Independent toy algebra and adversarial input controls; no target file reads."""
import copy
import importlib.util
import inspect
import os
import json
from pathlib import Path
import numpy as np
import pytest
from neural_shrimp import cifar_input_span as span
ROOT=Path(os.environ.get('CIFAR_SPAN_REVIEW_SOURCE_ROOT',str(Path(__file__).resolve().parents[2])))
spec=importlib.util.spec_from_file_location('span_independent_driver',ROOT/'tools/diagnose_cifar_input_span.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


def manual_gradient(theta,x,y,lam):
    # Independent explicit two-hidden/two-output toy network, no torch/model calls.
    w=theta[:8].reshape(4,2);b=theta[8:10];v=theta[10:14].reshape(2,2);c=theta[14:16]
    h=np.tanh(x@w+b);error=h@v+c-y
    back=(error@v.T)*(1-h*h)
    return np.r_[(x.T@back/len(x)).ravel(),back.mean(0),(h.T@error/len(x)).ravel(),error.mean(0)]+lam*theta


def test_independent_nonlinear_full_two_layer_direct_product_and_jvp():
    pk=span.Packing(4,2,2);x=np.array([[1.,0.,0.,0.],[0.,1.,0.,0.]])
    y=np.array([[.2,-.1],[.4,.3]]);q=np.eye(4,dtype=np.float64)[:,:2]
    theta=np.linspace(-.7,.8,16);direction=np.sin(np.arange(16)+.3)
    eta=.07;lam=.13;factor=(1-eta*lam)**2-1
    def update(t):return t-eta*manual_gradient(t,x,y,lam)
    def residual(t):return update(update(t))-t
    r=residual(theta);ap=residual(theta+1e-30j*direction).imag/1e-30
    vectors={'original-state':theta,'original-residual':r,'p':direction,'ap':ap,'e':ap+r}
    parts={name:span.project(v,q,pk) for name,v in vectors.items()}
    np.testing.assert_allclose(parts['original-residual'][1],factor*parts['original-state'][1],rtol=1e-12,atol=1e-15)
    np.testing.assert_allclose(parts['ap'][1],factor*parts['p'][1],rtol=1e-12,atol=1e-15)
    assert span.linear_identities(vectors,parts,factor)[2]['passed']
    projected_next=span.project(update(theta),q,pk)[0]
    np.testing.assert_allclose(projected_next,update(parts['original-state'][0]),atol=1e-15,rtol=1e-13)
    assert np.linalg.norm(update(theta)[10:]-theta[10:])>0


@pytest.mark.parametrize('x',[np.zeros((2,4)),np.array([[1.,2.,3.,4.],[2.,4.,6.,8.]])])
def test_all_columns_remain_under_zero_or_deficient_rank(x):
    q,r=span.thin_basis(x)
    assert q.shape==(4,2) and r.shape==(2,2) and span.basis_checks(x,q,r)['passed']
    assert span.Packing(4,2,2).dimensions(2)==dict(full=16,active=12,complement=4)


def test_nonzero_invisible_period_two_state_when_q_is_minus_one():
    x=np.zeros((2,4));y=np.zeros((2,2));theta=np.zeros(16);theta[4:8]=[1.,2.,3.,4.]
    def update(t):return t-2*manual_gradient(t,x,y,1.)
    assert np.array_equal(update(update(theta)),theta) and not np.array_equal(update(theta),theta)
    assert not span.decay(2.,1.,2)['elimination_valid']


def relations():
    theta=np.zeros(3);p=np.zeros(3);v={};m={};trials=[]
    for i,name in enumerate(driver.STATE_NAMES):
        v[name+'-state']=theta.copy();v[name+'-residual']=np.zeros(3);m[name]=np.zeros(3)
        if i:trials.append(dict(index=i-1,alpha=(1.,.5,.25,.125)[i-1]))
    v.update(p=p,ap=np.zeros(3),e=np.zeros(3))
    return theta,v,m,trials


@pytest.mark.parametrize('which',['original_state','trial_state','residual','e'])
def test_declared_bitwise_relations_reject_opposite_zero_bits(which):
    theta,v,m,trials=relations();span.saved_relations(theta,v,m,trials)
    if which=='original_state':v['original-state'][0]=-0.
    if which=='trial_state':v['trial-0-state'][0]=-0.
    if which=='residual':v['trial-2-residual'][0]=-0.
    if which=='e':v['e'][0]=-0.
    with pytest.raises(ValueError):span.saved_relations(theta,v,m,trials)


def test_linear_bitwise_flag_rejects_signed_zero_mismatch():
    pk=span.Packing(1,1,1);q=np.ones((1,1));v={name:np.zeros(4) for name in ('original-state','original-residual','p','ap','e')}
    v['e'][0]=-0.
    parts={name:span.project(a,q,pk) for name,a in v.items()}
    assert not span.linear_identities(v,parts,0.)[2]['passed']


def profile_fixture():
    source={'commit':'a'*40,'sha256':{name:driver.sha(b'committed') for name in driver.SOURCES}}
    env={'cpu':'synthetic'};review='c'*40
    checks=dict(passed=True,retained_columns=512,rank_truncation=False,orthogonality_frobenius=0.,
        x_i_minus_p_relative=0.,representation_relative=0.,implicit_idempotence_on_q=0.,qr_reconstruction_relative=0.)
    p=dict(status='profile_passed',shape=[3072,512],seed=28240,retained_columns=512,qr_seconds=.5,
           projected_target_seconds=6.,basis_checks=checks)
    for name,shape in [('x',[512,3072]),('q',[3072,512]),('r',[512,512])]:
        p[name]=dict(shape=shape,dtype='<f8',order='C',sha256='d'*64)
    start=dict(schema='cifar-input-span-v1',mode='profile',status='running',source=source,environment=env,
       review_commit=review,settings=driver.SETTINGS,pid=123,cpu_threads=1,nice=10,gpu=False,paid=False,
       dynamics_evaluations=0,started_utc='synthetic')
    result={**copy.deepcopy(start),'status':'profile_passed','finished_utc':'synthetic','elapsed_seconds':1.,'max_rss_kib':10000,'profile':p}
    return result,p,start,source,env,review


def save_profile(folder,result,p,start):
    for name,record in [('runtime-start.json',start),('profile.json',p),('result.json',result)]:
        (folder/name).write_bytes(driver.canonical(record))
    entries={p.name:driver.entry(p) for p in folder.iterdir() if p.name!='manifest.json'}
    (folder/'manifest.json').write_bytes(driver.canonical(entries))
    return driver.entry(folder/'manifest.json')['sha256']


def test_profile_must_have_an_external_pin_before_rehashed_replacement(tmp_path,monkeypatch):
    r,p,start,source,env,review=profile_fixture();folder=tmp_path/'profile-001';folder.mkdir()
    pin=save_profile(folder,r,p,start)
    monkeypatch.setattr(driver,'OUTROOT',tmp_path)
    monkeypatch.setattr(driver.subprocess,'check_output',lambda *a,**k:b'committed')
    parameters=inspect.signature(driver.profile_gate).parameters
    assert len(parameters)==5, 'collect must receive independently pinned profile SHA'
    def gate(expected):return driver.profile_gate(folder,source,env,review,expected)
    assert gate(pin)['projection_seconds']==6.
    # All mirrors and inventory hashes remain consistent, but this is a new receipt.
    p['qr_seconds']=.6;p['projected_target_seconds']=6.2;r['profile']=p
    assert save_profile(folder,r,p,start)!=pin
    with pytest.raises(ValueError):gate(pin)


def test_incomplete_vectors_never_pass_shape_or_full_layer_checks():
    pk=span.Packing(4,2,2);q=np.eye(4)[:,:2]
    with pytest.raises(ValueError):span.project(np.ones(15),q,pk)
    vector=np.arange(16,dtype=float);active,perp=span.project(vector,q,pk)
    active[-1]=0.
    with pytest.raises(ValueError):span.vector_checks(vector,active,perp,np.eye(4)[:2],q,pk)
