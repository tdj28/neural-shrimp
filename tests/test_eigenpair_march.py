"""Synthetic native continuation, rollback, radius and source/resource controls."""
import copy,importlib.util
from fractions import Fraction
from pathlib import Path
import numpy as np,pytest
from flint import arb,ctx
import neural_shrimp.rigorous_eigenpair_march as M
from neural_shrimp.rigorous_eigenpair_path import quadratic_tube
from neural_shrimp.rigorous_orbits import ball_json,exact
from test_rigorous_eigenpair_locus import CoupledCycle,settings


def parent(sigma=1):
    model=CoupledCycle(sigma);ini,kw=settings(model,sigma=sigma);r,_=quadratic_tube(model,ini,**kw);assert r['passed'];return model,r


@pytest.mark.parametrize('sigma',[-1,1])
def test_four_real_coupled_charts_keep_native_sign_full_anchor_and_gauge(sigma):
    with ctx.workprec(256):
        model,r=parent(sigma);events=[];ticks=[]
        out=M.march(model,r,tick=lambda:ticks.append(1),observe=lambda k,v:events.append((k,v)))
        assert out['completed'] and out['calls']==4 and not out['rebox_used'] and len(ticks)==4*15
        accepted=[v for k,v in events if k=='march_accepted'];assert len(accepted)==4
        right=r['parameter_endpoints'][1]
        for v in accepted:
            c=v['chart'];assert c['sigma']==sigma and c['normal']==r['normal'] and c['origin']==r['origin']
            assert c['parameter_endpoints'][0]==right;right=c['parameter_endpoints'][1]
            assert all(c['gates'].values()) and max(v['diagnostic']['relative_residuals'])<1e-12
        assert out['last_chart']['parameter_endpoints'][1]==right


def test_parent_rejects_incomplete_gates_shape_or_unqualified_chart():
    with ctx.workprec(256):
        _,r=parent()
        for change in ({'passed':False},{'gates':{}},{'parameter_endpoints':['0','1']},{'center':[]}):
            bad=dict(r,**change)
            with pytest.raises(ValueError):M.next_inputs(bad)
        ini,kw=M.next_inputs(r)
        assert kw['anchor']['exact_parameter']==r['parameter_endpoints'][1]
        assert all(a.contains(M.decode(b)) for a,b in zip(kw['anchor']['box'],r['endpoint_root_boxes'][1]))


def test_one_rebox_only_then_stop_without_failed_tail_advancement(monkeypatch):
    with ctx.workprec(256):
        model,r=parent();calls=[];events=[];proposals=[]
        def fail(model,initial,**kw):
            calls.append(dict(kw));x=copy.deepcopy(r);x['passed']=False;x['gates']['uniform_normalized_eigenpair_locus']=False;return x,None
        def rebox(row,observe):proposals.append(1);return {'radius_exponent':-15,'weights':['1/2']*10}
        monkeypatch.setattr(M,'proposed_radii',rebox)
        out=M.march(model,r,tube=fail,observe=lambda k,v:events.append(k))
        assert not out['completed'] and out['calls']==2 and out['rebox_used'] and len(proposals)==1
        assert out['accepted']==[] and out['last_chart']==M.compact(r)
        assert calls[0]['parameter']==calls[1]['parameter'] and calls[1]['radius_exponent']==-15
        assert calls[1]['weights']==['1/2']*10


def test_anchor_or_primitive_failure_never_reboxes(monkeypatch):
    with ctx.workprec(256):
        model,r=parent()
        monkeypatch.setattr(M,'proposed_radii',lambda *a:pytest.fail('forbidden rebox'))
        for gate in ('anchor_identified','primitive_period'):
            def fail(*a,**kw):
                v=copy.deepcopy(r);v['passed']=False;v['gates'][gate]=False;return v,None
            out=M.march(model,r,tube=fail)
            assert out['calls']==1 and out['accepted']==[] and not out['rebox_used']


def test_write_failure_at_accept_does_not_begin_next_call():
    with ctx.workprec(256):
        model,r=parent();starts=[];returns=[]
        def observer(k,v):
            if k=='march_call_start':starts.append(v)
            if k=='chart_return':returns.append(v)
            if k=='march_accepted':raise OSError('injected durability failure')
        with pytest.raises(OSError):M.march(model,r,observe=observer)
        assert len(starts)==len(returns)==1 and returns[0]['passed']


def scalar_failure(D):
    v=lambda a:ball_json(arb(a))
    return {'gates':{'uniform_normalized_eigenpair_locus':False,'primitive_period':True,'anchor_identified':True},'preconditioner':[[v(1)]],'predictor_residual':[[v('0.125')]],'defect':[[v(D)]]}


def test_radius_solve_return_precedes_bad_admission_and_exact_float_encoding():
    events=[]
    with ctx.workprec(256):
        good=M.proposed_radii(scalar_failure('.25'),lambda k,v:events.append((k,v)))
        raw=Fraction(good['exact_radii'][0]);assert raw==Fraction(good['weights'][0])*Fraction(2)**good['radius_exponent']
        assert events[0][0]=='rebox_linear_return' and good['predicted_old_domain_ratios'][0]<1
        events=[]
        with pytest.raises(ValueError,match='nonnegative'):M.proposed_radii(scalar_failure(2),lambda k,v:events.append((k,v)))
        assert events[0][1]['solution']==[-.125] and len(events)==1


def test_native_product_order_and_period1_identity_adjustment():
    with ctx.workprec(256):
        model,r=parent();d=M.midpoint_spectrum(r)
        values=np.sort(np.real(np.array(d['eigenvalues'])[:,0]));np.testing.assert_allclose(values,[.25,1.],atol=1e-12)
        z=ball_json(arb(0));one=ball_json(arb(1));chart={'dimension':2,'period':1,'sigma':1,'center_jacobian':[[z]*6 for _ in range(6)],'center':[z,z,one,z,one,one]}
        d=M.midpoint_spectrum(chart);assert d['complementary_near_critical_flag'] and d['other_distance_to_critical']==0
        assert d['monodromy_midpoint']==[[1.,0.],[0.,1.]]


def driver():
    p=Path(__file__).parents[1]/'tools/continue_eigenpair_locus.py';s=importlib.util.spec_from_file_location('march_driver',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def test_cost_gate_and_fail_closed_resource_metadata():
    m=driver();o={'passed':True,'charts':[{},{}],'ticks':272,'elapsed_seconds':50.,'peak_rss_kib':1112968,'stored_bytes_before_final':135110309}
    r={'commit':m.PARENT_COMMIT,'settings':m.Q.SETTINGS,'mode':'target','nice':10,'threads':dict.fromkeys(m.THREADS,'1')}
    l={'commit':m.PARENT_COMMIT,'exit_code':0,'owned_live_pids':[],'stop_reason':None,'elapsed_seconds':51.};p={'passed':True,'native_calls':0,'elapsed_seconds':1.,'peak_rss_kib':100000}
    assert m.cost_gate(o,r,l,p)['seconds']==270
    for bad in ({'elapsed_seconds':-1.},{'elapsed_seconds':float('nan')},{'peak_rss_kib':float('inf')},{'ticks':271},{'passed':False}):
        with pytest.raises(ValueError):m.cost_gate(dict(o,**bad),r,l,p)
    for bad in ({'passed':False},{'native_calls':1},{'elapsed_seconds':20.},{'peak_rss_kib':512*1024}):
        with pytest.raises(ValueError):m.cost_gate(o,r,l,dict(p,**bad))
    with pytest.raises(ValueError):m.cost_gate(o,r,dict(l,owned_live_pids=[12]),p)


def test_complete_sources_and_fixed_supervisor_limits():
    m=driver();root=Path(__file__).parents[1]
    assert set(m.Q.SOURCES)<=set(m.SOURCES) and len(m.SOURCES)==len(set(m.SOURCES))
    assert all((root/n).is_file() for n in m.SOURCES)
    s=(root/'tools/launch_eigenpair_march.py').read_text()
    assert '315 if target else 25' in s and '(769 if target else 9)*1024**2' in s and 'profile-outer-manifest' in s


def test_bad_eig_return_is_retained_before_finite_rejection(monkeypatch):
    with ctx.workprec(256):
        _,r=parent();events=[]
        monkeypatch.setattr(np.linalg,'eig',lambda m:(np.array([complex(float('nan'),0),1]),np.eye(2,dtype=complex)))
        with pytest.raises(ValueError,match='finite eig'):M.midpoint_spectrum(r,lambda k,v:events.append((k,v)))
        assert events[0][0]=='midpoint_eig_return' and events[0][1]['eigenvalues'][0][0]=='nan'
