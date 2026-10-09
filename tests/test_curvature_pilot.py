"""Synthetic only: adapters, native composition, retained failures and fixed gates."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from neural_shrimp import curvature_pilot as C
from neural_shrimp.gdmaps import TwoLayerNet
from neural_shrimp.period2_branch_io import Store,load_record
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('curvature_driver',ROOT/'tools/explore_curvature_pilot.py');D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
spec=importlib.util.spec_from_file_location('curvature_launcher',ROOT/'tools/launch_curvature_pilot.py');L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)

@pytest.mark.parametrize('bias,d,w',[(True,2,3),(False,3,4)])
def test_analytic_derivatives_independent_hvp_with_nonunit_bias_scales(bias,d,w):
    from dataclasses import asdict
    net=TwoLayerNet(d,w,in_scale=.37,out_scale=1.71,loss_scale=.63,ridge=.017,bias=bias)
    rng=np.random.default_rng(719);theta=rng.normal(size=net.n_parameters)*.2;x=rng.normal(size=(9,d));y=rng.normal(size=9)
    loss,g,H=C.loss_gradient_hessian(theta,x,y,asdict(net))
    oldg,oldloss=net.gradient(theta[None],x,y,return_loss=True)
    np.testing.assert_allclose(loss,oldloss[0],atol=2e-15,rtol=2e-14);np.testing.assert_allclose(g,oldg[0],atol=2e-15,rtol=2e-14)
    for _ in range(3):
        v=rng.normal(size=len(theta));reference=net.hvp(theta[None],x,y,v[None])[0]
        np.testing.assert_allclose(H@v,reference,atol=2e-14,rtol=2e-13)
    np.testing.assert_allclose(H,H.T,atol=1e-15)


def toy(theta,x,y,model):return float(theta@theta),theta**2,np.diag(2*theta)
def slot(pair,name='toy',stratum='stable-real'):
    return dict(slot=name,stratum=stratum,pair=np.asarray(pair,dtype=np.float64),rate_vector=np.ones(len(pair[0])))

def test_actual_native_intermediate_derivatives_and_exact_only_cache():
    rows=[];b=C.Budget(lambda k,r:rows.append((k,r)))
    theta=np.array([.2,.4]);other=theta-theta**2+1e-11
    C.evaluate_slot(slot([theta,other]),np.zeros((1,1)),np.zeros(1),{},b,derivative=toy)
    assembly=next(r for k,r in rows if k=='native_assembly');expected=[]
    for t in (theta,other):
        image=t-t*t;expected.append((np.eye(2)-np.diag(2*image))@(np.eye(2)-np.diag(2*t)))
    np.testing.assert_array_equal(assembly['monodromies'],np.stack(expected))
    assert b.returned['analytic']==4
    assert not np.array_equal(assembly['monodromies'][0],(np.eye(2)-np.diag(2*other))@(np.eye(2)-np.diag(2*theta)))


def test_single_phase_derives_other_phase_with_three_fused_calls():
    rows=[];b=C.Budget(lambda k,r:rows.append((k,r)))
    C.evaluate_slot(slot([[.2,.4]]),np.zeros((1,1)),np.zeros(1),{},b,derivative=toy)
    assert b.returned['analytic']==3 and any(k=='cache' for k,_ in rows)


def test_cache_is_acquisition_context_local_and_exact():
    b=C.Budget(lambda *_:None);t=np.array([.2,.4]);a=C.Acquisition(b,'a',np.zeros((1,1)),np.zeros(1),{},np.ones(2),toy)
    a.phase(t);a.phase(t.copy());assert b.returned['analytic']==1
    a.phase(np.nextafter(t,np.inf));assert b.returned['analytic']==2
    # Distinct model/data/rates never share the object cache.
    other=C.Acquisition(b,'b',np.ones((1,1)),np.ones(1),{'different':True},np.full(2,.5),toy)
    other.phase(t);assert b.returned['analytic']==3


def test_return_survives_nonfinite_and_deadline(tmp_path):
    store=Store(tmp_path/'raw');clock=[0.];b=C.Budget(lambda k,r:store.emit(k,D.safe(r)),clock=lambda:clock[0])
    def late(t,*_):clock[0]=61.;return float('nan'),np.full(2,np.nan),np.eye(2)
    a=C.Acquisition(b,'late',None,None,{},np.ones(2),late)
    with pytest.raises(C.PilotStop):a.phase(np.ones(2))
    assert b.returned['analytic']==1
    kind,row=load_record(store.root,'00001-operation_return.json')
    assert kind=='operation_return' and np.isnan(row['gradient']).all() and row['loss']=={'nonfinite_scalar':'nan'}


def test_all_six_qualification_failures_are_retained_without_replacement():
    b=C.Budget(lambda *_:None);rows=[]
    for i in range(6):rows.append(C.evaluate_slot(slot([[.2,.4],[.9,.8]],str(i)),None,None,{},b,derivative=toy))
    assert len(rows)==6 and all(not r['qualified'] for r in rows)
    assert b.returned['analytic']==24 and not C.aggregate(rows)['passed']


def passing_rows():
    rows=[]
    for i in range(6):
        real=i in (0,2);rows.append(dict(slot=str(i),qualified=True,stratum='stable-real' if real else 'stable-complex',
            target_kind='real_pair' if real else 'complex_conjugate_pair',modeled_kind='real_pair' if real else 'complex_conjugate_pair',
            pair_error=.005,commuting_error=.01,angle_error=None if real else .01,attribution=not real))
    return rows

def test_complex_attribution_excludes_real_controls_and_requires_all_four():
    rows=passing_rows();assert C.aggregate(rows)['passed']
    rows[1]['attribution']=False;assert not C.aggregate(rows)['passed']

@pytest.mark.parametrize('fit,control',[(.0095,.011),(.0001,.0002)])
def test_both_relative_and_absolute_oracle_mean_gains_required(fit,control):
    rows=passing_rows()
    for row in rows:row.update(pair_error=fit,commuting_error=control)
    assert not C.aggregate(rows)['gates']['mean_gain']

def test_false_complex_real_control_rejected():
    rows=passing_rows();rows[0]['modeled_kind']='complex_conjugate_pair';assert not C.aggregate(rows)['passed']


def test_profile_failure_preserves_first_phase_and_separate_synthetic_timings(monkeypatch):
    events=[];b=C.Budget(lambda k,r:events.append((k,r)),seconds=5.)
    class Fake:
        def __init__(self,budget,*args):self.b=budget
        def phase(self,t):
            self.b.begin('analytic',dict(theta=t));r=dict(theta=t,elapsed_seconds=100.)
            self.b.returned_event('analytic',r);return r
        def observer(self,kind,row):
            if kind=='decomposition_start':self.b.begin(row['kind'],dict(record=row))
            elif kind=='decomposition_return':self.b.returned_event(row['kind'],dict(record=row))
    monkeypatch.setattr(D,'Acquisition',Fake)
    data=dict(slots=[None,None,dict(slot='S1R-28002-stable-real',family='S1R',pair=np.ones((2,72)),rate_vector=np.ones(72))],models={'S1R':dict(x=None,y=None,model={},identity={})})
    r=D.profile_work(data,b);assert not r['profile_passed']
    assert b.returned==dict(analytic=1,eigh=2,eig=1,svd=1)
    assert any(k=='operation_return' and v['kind']=='analytic' for k,v in events)
    assert events[-1][0]=='profile_return'


def test_explicit_launcher_commands_and_caps():
    a=L.command(Path('/tmp/unused-profile'),'a'*40,'profile');b=L.command(Path('/tmp/unused-run'),'a'*40,'run',Path('/tmp/unused-profile'),'b'*64)
    assert '10s' in a and '75s' in b and '--profile-sha256' in b
    assert L.STORE_BYTES==16*1024**2 and L.RSS_BYTES==2*1024**3


def test_bad_profile_binding_rejected_before_launcher_output(tmp_path):
    with pytest.raises(ValueError,match='profile binding'):L.launch(tmp_path/'raw',tmp_path/'launcher','a'*40,'b'*40,'run')
    assert list(tmp_path.iterdir())==[]


def test_mutated_cache_context_fails_before_reuse():
    b=C.Budget(lambda *_:None);rates=np.ones(2);a=C.Acquisition(b,'a',None,None,{},rates,toy);a.phase(np.ones(2))
    rates[0]=2.
    with pytest.raises(ValueError,match='cache context'):a.phase(np.ones(2))
    assert b.returned['analytic']==1


def test_full_six_slot_storage_below_ordinary_reserve(tmp_path):
    store=Store(tmp_path/'six_slots',C.LIMITS['store_bytes']);emit=lambda k,r:store.emit(k,D.safe(r));budget=C.Budget(emit)
    rows=[]
    for i,n in enumerate([13,13,72,72,13,72]):
        u=np.eye(n)[0];v=.6*u+.8*np.eye(n)[1]
        H=[np.eye(n)-(.5*np.eye(n)-1.2*np.outer(z,z)) for z in (u,v)]
        def payload_derivative(t,*_):return float(t@t),2*t+1e-8,H[int(t[0]<0)]
        pair=np.stack((np.eye(n)[0],-np.eye(n)[0]));s=slot(pair,str(i),'stable-real' if i in (0,2) else 'stable-complex')
        emit('synthetic_input',dict(pair=pair,x=np.zeros((72,8)),y=np.zeros(72),rates=np.ones(n)))
        rows.append(C.evaluate_slot(s,None,None,{},budget,derivative=payload_derivative))
        emit('slot_complete',rows[-1])
    ordinary=store.used;assert budget.returned==dict(analytic=24,eigh=12,eig=6,svd=6)
    assert ordinary<12*1024**2
    store.finish(dict(status='synthetic_storage_only',rows=rows,aggregate=C.aggregate(rows)))
    final=sum(p.stat().st_size for p in store.root.iterdir())
    print(f'synthetic_storage_ordinary_bytes={ordinary}; final_bytes={final}; files={len(list(store.root.iterdir()))}')
    assert final<16*1024**2
