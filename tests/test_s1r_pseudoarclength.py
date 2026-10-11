"""Synthetic PAL controls only: analytic toy polynomial, no research model/data."""
import copy,importlib.util,json,math
from pathlib import Path
import numpy as np
import pytest
from neural_shrimp import s1r_pseudoarclength as P
from neural_shrimp.period2_branch_io import Store


def toy(x):
    t=x[0];return 1.5*t*t-.5*t**4+t**6/6,np.array([3*t-2*t**3+t**5]),np.array([[3-6*t*t+5*t**4]])

def start(x=.998):
    denominator=3-2*x*x+x**4;h=math.log10(2/denominator)
    slope=-(4*x**3-4*x)/(math.log(10)*denominator)
    tangent=np.array([1.,slope]);tangent/=np.linalg.norm(tangent)
    return np.array([[x],[-x]]),h,tangent

def make(emit=None):
    events=[];emit=emit or (lambda k,v:events.append((k,copy.deepcopy(v))))
    return P.Engine(toy,1,0.,emit),events


@pytest.mark.parametrize('x',[.8,.998])
def test_full_two_edge_synthetic_fold_with_witnesses(x):
    e,events=make();pair,h,t=start(x);result=e.run(pair,h,0,t,float(t[-1]))
    assert result['status']=='two_edges_complete' and len(result['accepted_edges'])==2
    assert (e.nodes[e.last_witnessed]['z'][0]>1.)==(x==.998)
    assert (e.nodes[e.last_witnessed]['tangent'][-1]<0)==(x==.998)
    assert e.ops.attempted==e.ops.returned
    assert all(v<=P.LIMITS[k] for k,v in e.ops.attempted.items())
    assert len([1 for k,v in events if k=='common_end' and v['passed']])==4
    for k,v in events:
        if k=='edge_end' and v['accepted']:assert len(v['nodes'])==4
    assert e.counts['qualifiers']==9 and e.ops.attempted['svd']==54 and e.ops.attempted['eig']==18


def test_same_state_derivative_cached_across_parameters_and_exact_map_hits():
    e,events=make();x=np.array([.8]);a=e.phase(x,0.);b=e.phase(x,.01);c=e.phase(x,0.)
    assert e.ops.attempted['analytic']==1 and e.ops.attempted['map']==2 and c['id']==a['id']
    assert a['id']!=b['id'];assert len([1 for k,v in events if k=='cache_hit'])==2


def test_actual_opposite_phase_monodromy_not_blind_swap():
    e,_=make();pair=np.array([[.998],[-.9981]]);h=start()[1]
    a=e.assembly(pair[0],h,'a');b=e.assembly(pair[1],h,'b')
    blind=e.phase(pair[0],h)['J']@e.phase(pair[1],h)['J']
    assert abs(float(b['M'][0,0]-blind[0,0]))>1e-6


def test_reverse_zero_plane_uses_positive_leg_length():
    e,_=make();pair,h,t=start();e.anchor=np.r_[pair[0],h]
    a=e.qualify(pair,h,t,0,'a');forward=e.correct(a['z'],a['z']+.002*a['tangent'],a['z'],a['tangent'],.002,'d')
    den=float(a['tangent']@forward['tangent']);pred=forward['z']-float(a['tangent']@(forward['z']-a['z']))/den*forward['tangent']
    reverse=e.correct(forward['z'],pred,a['z'],a['tangent'],0.,'r')
    assert np.linalg.norm(reverse['z']-a['z'])<1e-8


def test_nonfinite_analytic_return_persisted_before_reject():
    events=[];e=P.Engine(lambda x:(float('nan'),np.array([np.nan]),np.ones((1,1))),1,0.,lambda k,v:events.append((k,v)))
    with pytest.raises(ValueError):e.phase(np.array([1.]),0.)
    assert events[-1][0]=='operation_return' and events[-1][1]['value']['loss'] is None
    assert e.ops.returned['analytic']==1


def test_return_persisted_before_next_deadline_check():
    clock=[0.];events=[];b=P.Operations(lambda k,v:events.append((k,v)),seconds=1.,clock=lambda:clock[0])
    def returned():clock[0]=2.;return dict(step=np.ones(2))
    b.call('solve',dict(),returned)
    assert events[-1][0]=='operation_return'
    with pytest.raises(P.PilotStop):b.check()


def test_invalid_return_and_operation_caps_are_retained():
    events=[];b=P.Operations(lambda k,v:events.append((k,v)))
    b.attempted['svd']=P.LIMITS['svd']
    with pytest.raises(P.PilotStop):b.call('svd',dict(),lambda:None)
    assert events==[]
    with pytest.raises(RuntimeError):b.call('solve',dict(J=np.eye(1)),lambda:(_ for _ in ()).throw(RuntimeError('synthetic')))
    assert events[-1][0]=='operation_exception' and b.returned['solve']==0


def test_common_failure_has_end_and_no_fixed_parameter_solve():
    e,events=make();pair,h,t=start();a=e.qualify(pair,h,t,0,'a');b=dict(a,id='other',h=h+1e-5)
    before=e.ops.attempted['solve']
    with pytest.raises(P.Reject):e.common(a,b)
    assert events[-1][0]=='common_end' and not events[-1][1]['passed']
    assert e.ops.attempted['solve']==before


def test_durable_edge_failure_cannot_advance_witness(monkeypatch):
    e,_=make();pair,h,t=start();a=e.qualify(pair,h,t,0,'a');e.last_witnessed=a['id']
    def emit(k,v):
        if k=='edge_end' and v['accepted']:raise OSError('synthetic edge durability failure')
    e.emit=emit;e.ops.emit=emit
    with pytest.raises(OSError):e.edge(a,.002)
    assert e.last_witnessed==a['id'] and not e.durable_edges


def test_ambiguous_orientation_abstains():
    e,_=make();a=dict(A=np.array([[1.]]),b=np.array([0.]))
    with pytest.raises(P.PilotStop,match='orientation'):e.tangent(a,np.array([1.,0.]),'synthetic')


def test_native_phase_one_anchor_and_carry():
    e,_=make();pair,h,t=start();ref=np.array([-t[0],t[1]])
    a=e.qualify(pair,h,ref,1,'a');e.anchor=a['z'];e.last_witnessed=a['id']
    child=e.edge(a,.002)
    assert child is not None and child['carried_phase']==0
    match=P.phase_match(a['pair'],child['pair'],e.o)
    assert a['carried_phase']^match['phase']==child['carried_phase']


def test_preserve_source_and_no_legacy_hidden_svd_calls():
    import ast
    source=Path(P.__file__).read_text();tree=ast.parse(source)
    forbidden={'spectral_checks','eigen_residuals','cond','svdvals'}
    calls={n.func.attr if isinstance(n.func,ast.Attribute) else n.func.id if isinstance(n.func,ast.Name) else '' for n in ast.walk(tree) if isinstance(n,ast.Call)}
    assert not calls.intersection(forbidden)
    assert 'full_matrices=True' in source


def test_synthetic_full_size_records_fit_ordinary_cap(tmp_path):
    # Structural payload screen only: no model, norm, eigensolver or SVD.
    store=Store(tmp_path/'size',P.LIMITS['store_bytes']);n=72;theta=np.zeros(n);square=np.zeros((n,n));aug=np.zeros((n,n+1))
    payloads={
      'analytic':dict(theta=theta,gradient=theta,hessian=square,loss=0.),
      'map':dict(theta=theta,rates=theta,image=theta,J=square,c=theta),
      'assembly':dict(theta=theta,map1=theta,map2=theta,R=theta,M=square,A=square,b=theta),
      'solve':dict(z=np.zeros(n+1),G=np.zeros(n+1),J=np.zeros((n+1,n+1)),step=np.zeros(n+1)),
      'svd':dict(matrix=aug,U=square,s=theta,Vh=np.zeros((n+1,n+1))),
      'eig':dict(matrix=square,values=np.zeros(n,dtype=complex),left=np.zeros((n,n),dtype=complex),right=np.zeros((n,n),dtype=complex)),
    }
    counts=dict(analytic=256,map=256,assembly=256,solve=64,svd=102,eig=34)
    for k,v in payloads.items():
        before=store.used;store.emit(k,v);payloads[k]=store.used-before
    projected=sum(payloads[k]*counts[k] for k in counts)+8*1024**2
    assert projected<=P.LIMITS['store_bytes']*.75
    print(json.dumps(dict(projected_bytes=projected,ordinary_cap=int(P.LIMITS['store_bytes']*.75),per_record_bytes=payloads,counts=counts)))


def test_same_plane_different_seed_tangent_reference():
    e,events=make();pair,h,t=start();e.run(pair,h,0,t,float(t[-1]))
    nodes={v['id']:v for k,v in events if k=='qualifier_end' and v['accepted']}
    starts=[v for k,v in events if k=='corrector_start']
    direct=nodes['node-0001'];half=nodes['node-0002']
    endpoint=next(v for v in starts if v['purpose']=='half_endpoint');reverse=next(v for v in starts if v['purpose']=='reverse')
    np.testing.assert_array_equal(endpoint['orientation_reference'],half['tangent'])
    np.testing.assert_array_equal(reverse['orientation_reference'],direct['tangent'])
    assert not np.array_equal(endpoint['orientation_reference'],endpoint['plane_tangent'])


def test_assembly_cap_applies_even_when_all_derivatives_cached():
    e,_=make();pair,h,_=start();e.assembly(pair[0],h,'first');count=e.ops.attempted.copy()
    e.counts['assemblies']=P.LIMITS['assemblies']
    with pytest.raises(P.PilotStop,match='assemblies_cap'):e.assembly(pair[0],h,'cached')
    assert e.ops.attempted==count


def test_physical_closure_uses_actual_second_image(monkeypatch):
    e,events=make();pair,h,_=start();original=e.assembly
    def changed(*args):
        a=original(*args);a['R']=np.full_like(a['R'],100.);return a
    monkeypatch.setattr(e,'assembly',changed)
    _,check=e.states(pair,h,'synthetic')
    assert check['passed']


def test_out_of_box_trial_is_recorded_then_rejects_edge(monkeypatch):
    e,events=make();e.anchor=np.zeros(2)
    def artificial(theta,h,purpose):
        return dict(A=np.ones((1,1)),b=np.zeros(1),R=theta-1.,map2=2*theta-1.)
    monkeypatch.setattr(e,'assembly',artificial)
    with pytest.raises(P.Reject,match='local_box'):
        e.correct(np.zeros(2),np.array([0.,.0005]),np.zeros(2),np.array([0.,1.]),.0005,'synthetic')
    trials=[v for k,v in events if k=='armijo_trial']
    assert len(trials)==1 and trials[0]['error']=='local_box' and trials[0]['halving']==0
    np.testing.assert_array_equal(trials[0]['state'],np.array([1.,.0005]))


def test_launcher_command_limits_and_bad_priority_before_creation(monkeypatch,tmp_path):
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('pal_launcher',root/'tools/launch_s1r_pseudoarclength.py');L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)
    assert L.command(tmp_path/'out','a'*40)[:6]==['timeout','--signal=TERM','--kill-after=5s','75s','nice','-n']
    assert L.STORE_BYTES==127*1024**2 and L.SUPERVISOR_SECONDS==80.
    monkeypatch.setattr(L.subprocess,'check_output',lambda *args,**kwargs:'a'*40)
    monkeypatch.setattr(L.subprocess,'run',lambda *args,**kwargs:None)
    monkeypatch.setattr(L,'committed_sources',lambda:{})
    monkeypatch.setattr(L.os,'getpriority',lambda *_:0)
    with pytest.raises(ValueError,match='nice'):L.launch(tmp_path/'out',tmp_path/'launch','a'*40,'b'*40)
    assert not (tmp_path/'out').exists() and not (tmp_path/'launch').exists()
