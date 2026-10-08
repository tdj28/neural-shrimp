"""Independent producer controls: toy polynomial only, never project data."""
import copy
import math
import numpy as np
import pytest
from neural_shrimp import s1r_pseudoarclength as P

def toy(x):
    u=x[0]
    return 1.5*u*u-.5*u**4+u**6/6,np.array([3*u-2*u**3+u**5]),np.array([[3-6*u*u+5*u**4]])

def setup(x=.998,carried=0):
    d=3-2*x*x+x**4;h=math.log10(2/d)
    t=np.array([1.,-(4*x**3-4*x)/(math.log(10)*d)])
    if carried:t[0]*=-1
    t/=np.linalg.norm(t)
    events=[];e=P.Engine(toy,1,0.,lambda k,v:events.append((k,copy.deepcopy(v))))
    return e,events,np.array([[x],[-x]]),h,t

@pytest.mark.parametrize('carried',[0,1])
def test_independent_nonsingular_curved_branch(carried):
    e,ev,pair,h,t=setup(.9,carried)
    r=e.run(pair,h,carried,t,float(t[-1]))
    assert r['status']=='two_edges_complete'
    assert e.nodes[e.last_witnessed]['tangent'][-1]>0
    for k,v in ev:
        if k=='armijo_trial' and v['accepted']:
            assert v['slope']<0
            assert v['merit']<=v['old_merit']+1e-4*v['alpha']*v['slope']
        if k=='phase_transport':assert abs(v['cosine']-1)<=1e-7

@pytest.mark.parametrize('kind',['analytic','map','solve','svd','eig'])
@pytest.mark.parametrize('boundary',['operation_start','operation_return'])
def test_each_operation_durability_boundary(kind,boundary):
    events=[];executed=[]
    def emit(k,v):
        if k==boundary:raise OSError('synthetic durability boundary')
        events.append((k,copy.deepcopy(v)))
    ops=P.Operations(emit)
    def call():executed.append(1);return np.ones(1)
    with pytest.raises(OSError):ops.call(kind,{},call)
    assert ops.attempted[kind]==1
    assert ops.returned[kind]==int(boundary=='operation_return')
    assert len(executed)==int(boundary=='operation_return')
    assert len(events)==int(boundary=='operation_return')

def test_inaccurate_finite_linear_step_rejected_before_trials(monkeypatch):
    e,ev,pair,h,t=setup();e.anchor=np.r_[pair[0],h]
    solve=np.linalg.solve
    monkeypatch.setattr(np.linalg,'solve',lambda A,b:1.01*solve(A,b))
    with pytest.raises(P.Reject,match='linear_equation_gate'):
        e.correct(e.anchor,e.anchor+.002*t,e.anchor,t,.002,'bad_solve')
    assert not any(k=='armijo_trial' for k,v in ev)
    assert [v for k,v in ev if k=='linear_checks'][-1]['forcing']>1e-6

def test_common_control_requalification_uses_reference_exact_float(monkeypatch):
    e,ev,pair,h,t=setup();a=e.qualify(pair,h,t,0,'anchor')
    b=copy.deepcopy(a);b['id']='synthetic_other';b['h']=np.nextafter(h,float('inf'))
    used=[];states=e.states
    def record(pair,control,purpose):
        used.append(control);return states(pair,control,purpose)
    monkeypatch.setattr(e,'states',record)
    old=e.ops.attempted['solve'];e.common(a,b)
    assert np.float64(used[0]).tobytes()==np.float64(a['h']).tobytes()
    assert e.ops.attempted['solve']==old

def test_no_promotion_after_reverse_common_failure(monkeypatch):
    e,ev,pair,h,t=setup();a=e.qualify(pair,h,t,0,'anchor')
    e.anchor=a['z'];e.last_witnessed=a['id'];common=e.common;calls=[]
    def reject_second(ref,candidate):
        calls.append(1)
        if len(calls)==2:raise P.Reject('synthetic reverse mismatch')
        return common(ref,candidate)
    monkeypatch.setattr(e,'common',reject_second)
    assert e.edge(a,.002) is None
    assert e.last_witnessed==a['id'] and e.durable_edges==[]
    assert len(e.nodes)==5

def test_local_box_failure_has_retained_corrector_end():
    e,ev,pair,h,t=setup();e.anchor=np.r_[pair[0],h]
    pred=e.anchor.copy();pred[-1]+=.002
    with pytest.raises(P.Reject,match='local_box'):
        e.correct(e.anchor,pred,e.anchor,t,.002,'outside')
    assert ev[-1][0]=='corrector_end'
    assert not ev[-1][1]['converged']
    assert e.ops.attempted['analytic']==0

def test_rank_ambiguity_abstains():
    e,ev,*_=setup()
    with pytest.raises(P.PilotStop,match='augmented_rank'):
        e.tangent(dict(A=np.zeros((1,1)),b=np.zeros(1)),np.ones(2),'zero')

def test_invalid_svd_order_rejected_after_return(monkeypatch):
    e,ev,*_=setup()
    monkeypatch.setattr(np.linalg,'svd',lambda A,full_matrices:(np.eye(2),np.array([1.,2.]),np.eye(3)))
    with pytest.raises(ValueError,match='ordered'):
        e.svd(np.array([[1.,0.,0.],[0.,2.,0.]]),'invalid_order')
    assert ev[-1][0]=='operation_return'

def test_all_phase_attempt_counters_exact_on_synthetic_fold():
    e,ev,pair,h,t=setup();r=e.run(pair,h,0,t,float(t[-1]))
    for kind in e.ops.attempted:
        starts=[v for k,v in ev if k=='operation_start' and v['kind']==kind]
        ends=[v for k,v in ev if k=='operation_return' and v['kind']==kind]
        assert len(starts)==len(ends)==e.ops.attempted[kind]==e.ops.returned[kind]
    assert r['accepted_edges']==[v['id'] for k,v in ev if k=='edge_end' and v['accepted']]

def test_trial_locality_failure_preserves_rejected_trial(monkeypatch):
    e,ev,pair,h,t=setup();e.anchor=np.r_[pair[0],h]
    calls=[];original=e.local
    def second_trial(z):
        calls.append(np.array(z).copy())
        if len(calls)==2:raise P.Reject('synthetic local_box at first trial')
        return original(z)
    monkeypatch.setattr(e,'local',second_trial)
    with pytest.raises(P.Reject,match='local_box'):
        e.correct(e.anchor,e.anchor+.002*t,e.anchor,t,.002,'trial_boundary')
    rows=[v for k,v in ev if k=='armijo_trial']
    assert len(rows)==1 and rows[0]['halving']==0 and rows[0]['alpha']==1.
    assert not rows[0]['accepted']
    assert np.array_equal(rows[0]['state'],calls[1])
