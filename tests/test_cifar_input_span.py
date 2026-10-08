"""Synthetic only. Never opens saved CIFAR arrays or runs the 3072x512 profile."""
import copy
import importlib.util
from pathlib import Path
import signal
import numpy as np
import pytest
from neural_shrimp.cifar_input_span import (Packing,basis_checks,decay,error_check,finite,identities,
    linear_identities,norm,project,saved_relations,thin_basis,vector_checks)

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('span_driver',ROOT/'tools/diagnose_cifar_input_span.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)

def fixture():
    rng=np.random.default_rng(28241);packing=Packing(5,3,2)
    x=rng.normal(size=(2,5));y=rng.normal(size=(2,2));theta=rng.normal(size=packing.size);p=rng.normal(size=packing.size)
    q,r=thin_basis(x)
    return rng,packing,x,y,theta,p,q,r

def test_exact_packing_and_dimensions():
    packing=Packing();assert packing.dimensions(512)==dict(full=98666,active=16746,complement=81920)
    small=Packing(5,3,2);v=np.arange(small.size,dtype=np.float64)
    assert small.weight(v)[2,1]==7
    from neural_shrimp.cifar import MLP
    import torch
    model=MLP(5,3,2);w1,b1,w2,b2=model.unflatten(torch.from_numpy(v))
    np.testing.assert_array_equal(w1.numpy(),small.weight(v))
    np.testing.assert_array_equal(b1.numpy(),v[15:18]);np.testing.assert_array_equal(w2.numpy(),v[18:24].reshape(3,2));assert model.n_parameters==small.size

def test_nonlinear_both_layers_ridge_and_period_two_identities():
    import torch
    from neural_shrimp.cifar import MLP,Problem
    _,pk,x,y,theta,p,q,r=fixture();eta=.13;lam=.17
    model=MLP(pk.d,pk.width,pk.k);problem=Problem(model,x,y,lam);plain=Problem(model,x,y,0.)
    t=torch.from_numpy(theta);v=torch.from_numpy(p)
    np.testing.assert_allclose((problem.grad(t)-plain.grad(t)).numpy(),lam*theta,rtol=1e-13,atol=1e-14)
    out=model.forward(t,torch.from_numpy(x));expected=.5*np.sum((out.numpy()-y)**2)/len(x)+.5*lam*np.dot(theta,theta)
    assert problem.loss(t).item()==pytest.approx(expected,rel=1e-14)
    def residual(t):return problem.update(problem.update(t,eta),eta)-t
    residual0=residual(t).numpy();ap=torch.func.jvp(residual,(t,),(v,))[1].numpy()
    vectors={'original-state':theta,'original-residual':residual0,'p':p,'ap':ap,'e':ap+residual0}
    parts={n:project(a,q,pk) for n,a in vectors.items()};factor=decay(eta,lam)['rounded_factor']
    for n,a in vectors.items():assert vector_checks(a,*parts[n],x,q,pk)['passed']
    assert identities(residual0,parts['original-residual'][1],parts['original-state'][1],factor)[1]['passed']
    assert linear_identities(vectors,parts,factor)[2]['passed']
    active=parts['original-state'][0];a=torch.from_numpy(active)
    np.testing.assert_allclose(model.forward(t,torch.from_numpy(x)).numpy(),model.forward(a,torch.from_numpy(x)).numpy(),atol=1e-13)
    reduced_next=problem.update(a,eta).numpy();full_next=problem.update(t,eta).numpy()
    projected_next,_=project(full_next,q,pk)
    np.testing.assert_allclose(projected_next,reduced_next,rtol=1e-12,atol=1e-13)
    assert norm(full_next[pk.first:]-theta[pk.first:])>1e-3  # second layer/biases really train
    corrupt={k:a.copy() for k,a in vectors.items()};corrupt['e'][0]+=1e-3
    assert not linear_identities(corrupt,{n:project(a,q,pk) for n,a in corrupt.items()},factor)[2]['passed']

def test_rank_deficient_completion_and_implicit_projector():
    x=np.array([[1.,2.,3.,4.,5.],[2.,4.,6.,8.,10.],[0.,0.,0.,0.,0.]])
    q,r=thin_basis(x);assert q.shape==(5,3) and basis_checks(x,q,r)['passed']
    pk=Packing(5,2,1);v=np.arange(pk.size,dtype=np.float64);a,n=project(v,q,pk)
    assert vector_checks(v,a,n,x,q,pk)['passed'];assert np.linalg.matrix_rank(x)==1
    np.testing.assert_array_equal(a[pk.first:],v[pk.first:]);np.testing.assert_array_equal(n[pk.first:],0)
    wrong=q.copy();wrong[:,0]*=1.01;assert not basis_checks(x,wrong,r)['passed']

def test_exceptional_decay_does_not_eliminate_invisible_coordinates():
    for eta,lam,period in [(1.,0.,2),(2.,1.,2)]:
        d=decay(eta,lam,period);assert not d['elimination_valid'] and not d['omitted_directions_attracting']
        n=np.array([1.,-2.]);np.testing.assert_array_equal(d['rounded_q']**period*n,n)
    d=decay(.00827753,.01);assert d['elimination_valid'] and d['omitted_directions_attracting']
    from fractions import Fraction
    assert Fraction(d['exact_q'])==1-Fraction(.00827753)*Fraction(.01)
    assert d['eta_hex']==float(.00827753).hex()

@pytest.mark.parametrize('a',[np.array([np.nan]),np.array([np.inf]),np.array([1.],dtype=np.float32),np.array([1j])])
def test_nonfinite_precision_rejected(a):
    with pytest.raises(ValueError):finite(a)

def saved_fixture():
    theta=np.array([1.,2.,3.]);p=np.array([.25,-.5,.75]);vectors={};maps={};trials=[]
    for i,label in enumerate(driver.STATE_NAMES):
        a=theta if i==0 else theta+(1.,.5,.25,.125)[i-1]*p
        maps[label]=a+np.array([1.,2.,3.]);vectors[label+'-state']=a;vectors[label+'-residual']=maps[label]-a
        if i:trials.append(dict(index=i-1,alpha=(1.,.5,.25,.125)[i-1]))
    vectors.update(p=p,ap=p*.1,e=p*.1+vectors['original-residual'])
    return theta,vectors,maps,trials

def test_saved_order_and_original_only_linear_binding():
    theta,vectors,maps,trials=saved_fixture();saved_relations(theta,vectors,maps,trials)
    for case in ('order','trial_alpha','trial_state','residual','e'):
        v=copy.deepcopy(vectors);m=copy.deepcopy(maps);t=copy.deepcopy(trials)
        if case=='order':v=dict(reversed(list(v.items())))
        if case=='trial_alpha':t[0]['alpha']=.5
        if case=='trial_state':v['trial-0-state'][0]+=.01
        if case=='residual':v['trial-2-residual'][0]+=.01
        if case=='e':v['e'][0]+=.01
        with pytest.raises(ValueError):saved_relations(theta,v,m,t)

def make_inventory(path):
    path.mkdir();(path/'value.json').write_text('{}')
    (path/'manifest.json').write_bytes(driver.canonical({'value.json':driver.entry(path/'value.json')}))
    return driver.entry(path/'manifest.json')['sha256']

@pytest.mark.parametrize('corruption',['hash','extra','omission','symlink','unsafe'])
def test_complete_inventory_rejects_corruption(tmp_path,corruption):
    folder=tmp_path/'inputs';pin=make_inventory(folder)
    assert driver.inventory(folder,pin,{'value.json'})
    if corruption=='hash':(folder/'value.json').write_text('{"changed":true}')
    if corruption=='extra':(folder/'extra.json').write_text('{}')
    if corruption=='omission':(folder/'manifest.json').write_text('{}');pin=driver.entry(folder/'manifest.json')['sha256']
    if corruption=='symlink':(folder/'value.json').unlink();(folder/'value.json').symlink_to(folder/'manifest.json')
    if corruption=='unsafe':(folder/'manifest.json').write_bytes(driver.canonical({'../value.json':{'bytes':2,'sha256':'0'*64}}));pin=driver.entry(folder/'manifest.json')['sha256']
    with pytest.raises(ValueError):driver.inventory(folder,pin,{'value.json'})

def profile_fixture():
    source={'commit':'a'*40,'sha256':{name:'b'*64 for name in driver.SOURCES}};env={'cpu':'synthetic','numpy':'test'};review='c'*40
    c=dict(orthogonality_frobenius=1e-14,x_i_minus_p_relative=1e-15,representation_relative=1e-15,implicit_idempotence_on_q=1e-14,qr_reconstruction_relative=1e-15,retained_columns=512,rank_truncation=False,passed=True)
    p=dict(status='profile_passed',shape=[3072,512],seed=28240,retained_columns=512,qr_seconds=.5,projected_target_seconds=6.,basis_checks=c)
    for name,shape in [('x',[512,3072]),('q',[3072,512]),('r',[512,512])]:p[name]=dict(shape=shape,dtype='<f8',order='C',sha256='d'*64)
    start=dict(schema='cifar-input-span-v1',mode='profile',status='running',source=source,environment=env,review_commit=review,settings=driver.SETTINGS,pid=123,cpu_threads=1,nice=10,gpu=False,paid=False,dynamics_evaluations=0,started_utc='2026-10-08T10:00:00+00:00')
    result={**copy.deepcopy(start),'status':'profile_passed','finished_utc':'2026-10-08T10:00:01+00:00','elapsed_seconds':1.,'max_rss_kib':10000,'profile':p}
    return result,p,start,source,env,review

@pytest.mark.parametrize('case',['nan','inf','negative','elapsed','projection','pid','source_commit','source_omission','basis','rank','action','gpu','array_shape'])
def test_profile_gate_rehashed_contradictions(case):
    r,p,start,source,env,review=copy.deepcopy(profile_fixture())
    assert driver.validate_profile_records(r,p,start,source,env,review)==6.
    if case=='nan':p['qr_seconds']=float('nan')
    if case=='inf':r['max_rss_kib']=float('inf')
    if case=='negative':p['qr_seconds']=-1
    if case=='elapsed':r['elapsed_seconds']=6
    if case=='projection':p['projected_target_seconds']=5
    if case=='pid':r['pid']+=1
    if case=='source_commit':r['source']['commit']='e'*40
    if case=='source_omission':r['source']['sha256'].pop(next(iter(r['source']['sha256'])))
    if case=='basis':p['basis_checks']['x_i_minus_p_relative']=1e-5
    if case=='rank':p['basis_checks']['rank_truncation']=True
    if case=='action':start['mode']='collect'
    if case=='gpu':r['gpu']=True
    if case=='array_shape':p['q']['shape']=[3072,511]
    with pytest.raises(ValueError):driver.validate_profile_records(r,p,start,source,env,review)

def test_cap_reserve_and_partial_array_retention(tmp_path,monkeypatch):
    store=driver.Store(tmp_path/'cap',cap=4096,reserve=2048,external_reserve=512)
    store.write('accepted.bin',b'x'*1900)
    with pytest.raises(ValueError):store.write('too-large.bin',b'x'*200)
    store.finalize({'status':'cap_stopped'});assert driver.read(store.folder/'manifest.json')['accepted.bin']['bytes']==1900
    other=driver.Store(tmp_path/'interrupted',cap=4096,reserve=2048,external_reserve=512)
    def interrupted(f,*a,**kw):f.write(b'partial-npy');f.flush();raise TimeoutError('synthetic interrupted save')
    monkeypatch.setattr(driver.np,'save',interrupted)
    with pytest.raises(TimeoutError):other.array('partial.npy',np.ones(2))
    other.finalize({'status':'interrupted'});assert driver.read(other.folder/'manifest.json')['partial.npy']['bytes']==11
    with pytest.raises(ValueError):other.array('../escape.npy',np.ones(2))

def test_profile_interrupt_retains_failure_and_runtime_without_target(tmp_path,monkeypatch):
    monkeypatch.setattr(driver,'OUTROOT',tmp_path);monkeypatch.setattr(driver,'source_identity',lambda:{'commit':'a'*40,'sha256':{}})
    monkeypatch.setattr(driver,'environment',lambda:{'cpu':'synthetic'});monkeypatch.setattr(driver.subprocess,'run',lambda *a,**kw:None)
    monkeypatch.setattr(driver.os,'getpriority',lambda *a:10)
    class NoTargetRandom:
        def normal(self,size):assert size==(512,3072);raise TimeoutError('synthetic interruption before QR')
    monkeypatch.setattr(driver.np.random,'default_rng',lambda seed:NoTargetRandom())
    alarm=signal.getsignal(signal.SIGALRM);term=signal.getsignal(signal.SIGTERM)
    assert driver.run('profile',tmp_path/'profile-001','a'*40)==2
    folder=tmp_path/'profile-001';result=driver.read(folder/'result.json')
    assert result['status']=='failed_or_interrupted' and result['dynamics_evaluations']==0
    assert set(driver.read(folder/'manifest.json'))=={'runtime-start.json','failure.json','result.json'}
    assert signal.getsignal(signal.SIGALRM)==alarm and signal.getsignal(signal.SIGTERM)==term

def test_profile_inventory_source_and_retained_bytes(tmp_path,monkeypatch):
    r,p,start,source,env,review=copy.deepcopy(profile_fixture())
    for obj in (r['source'],start['source'],source):obj['sha256']={name:driver.sha(b'committed') for name in driver.SOURCES}
    folder=tmp_path/'profile-001';folder.mkdir()
    for name,obj in [('runtime-start.json',start),('profile.json',p),('result.json',r)]:
        (folder/name).write_bytes(driver.canonical(obj))
    rows={p.name:driver.entry(p) for p in folder.iterdir()};(folder/'manifest.json').write_bytes(driver.canonical(rows))
    monkeypatch.setattr(driver,'OUTROOT',tmp_path);monkeypatch.setattr(driver.subprocess,'check_output',lambda *a,**kw:b'committed')
    pin=driver.entry(folder/'manifest.json')['sha256']
    result=driver.profile_gate(folder,source,env,review,pin)
    assert result['retained_bytes']==sum(p.stat().st_size for p in folder.iterdir())
    monkeypatch.setattr(driver.subprocess,'check_output',lambda *a,**kw:b'changed')
    with pytest.raises(ValueError):driver.profile_gate(folder,source,env,review,pin)
