"""Independent synthetic controls for the forthcoming S1R bootstrap source review.

Run with the frozen producer's src on PYTHONPATH. No historical/target artifact,
S1R construction, training trajectory or engineering profile is evaluated.
"""
import ast
from copy import deepcopy
from pathlib import Path
import subprocess

import numpy as np
import pytest

from neural_shrimp import gdmaps
from neural_shrimp import period2_branch as pair
from neural_shrimp import s1r_bootstrap as bootstrap

BASE = '960ab535e1954e6f46dce21999338ddcf1bd23ee'


def frozen_screen():
    root = Path(gdmaps.__file__).resolve().parents[2]
    raw = subprocess.check_output(['git','show',BASE+':src/neural_shrimp/gdmaps.py'],cwd=root,text=True)
    tree = ast.parse(raw)
    functions = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('screen_tails','iterate_and_screen')]
    namespace = {'np':np,**{n:getattr(gdmaps,n) for n in dir(gdmaps) if n.startswith('STATUS_')}}
    exec(compile(ast.Module(body=functions,type_ignores=[]),'<frozen-prior-screen>','exec'),namespace)
    return namespace['iterate_and_screen']


def synthetic_operators(diverge=False):
    calls = {'map':0,'jvp':0,'loss_states':[]}
    def step(state,idx):
        calls['loss_states'].append((idx.copy(),state.copy()))
        new = .9*state+.01
        if diverge and calls['map']==5:
            new[np.flatnonzero(idx==1)] = np.nan
        calls['map'] += 1
        return new,np.sum(state*state,axis=1)
    def jvp(state,idx,tangent):
        calls['jvp'] += 1
        return .9*tangent
    return step,jvp,calls


@pytest.mark.parametrize('diverge',[False,True])
def test_observer_is_exact_prior_arithmetic_and_copies(diverge):
    initial=np.array([[.4,-.2],[.3,.6]])
    options=dict(tail_steps=24,ftle_steps=8,ftle_align=4,sd_horizons=(),max_period=4,min_repeats=6)
    step,jvp,old_calls=synthetic_operators(diverge)
    baseline=frozen_screen()(step,jvp,initial,64,**options)
    step,jvp,new_calls=synthetic_operators(diverge)
    terminal=[]
    def observer(kind,snapshot):
        if kind=='terminal':terminal.append(snapshot())
        # Even deliberate mutation of every copied numerical array cannot alter live arithmetic.
        for value in snapshot().values():
            if isinstance(value,np.ndarray):value.fill(7)
    observed=gdmaps.iterate_and_screen(step,jvp,initial,64,observer=observer,**options)
    assert set(observed)==set(baseline)
    for name in baseline:
        np.testing.assert_array_equal(observed[name],baseline[name])
    assert new_calls['map']==old_calls['map']==64 and new_calls['jvp']==old_calls['jvp']==12
    saved=terminal[0]
    assert saved['completed_loops']==64
    np.testing.assert_array_equal(saved['tail_state_updates'],np.arange(41,65))
    np.testing.assert_array_equal(saved['tail_loss_updates'],np.arange(40,64))
    np.testing.assert_array_equal(saved['tail'][0,-1],observed['final_state'][0])
    np.testing.assert_array_equal(saved['tail_losses'][0,-1],np.sum(new_calls['loss_states'][-1][1][0]**2))
    np.testing.assert_array_equal(saved['attempted_updates'],[64,6 if diverge else 64])
    np.testing.assert_array_equal(saved['accepted_updates'],[64,5 if diverge else 64])
    if diverge:
        assert not saved['alive'][1] and np.isnan(saved['state'][1]).all()
        assert np.isfinite(saved['last_valid_state'][1]).all()


def test_observer_interrupt_preserves_latest_completed_state():
    clock=[0.]
    budget=bootstrap.InterruptibleBudget(1,clock=lambda:clock[0])
    events=[]
    observer=bootstrap.WarmObserver(budget,lambda k,v:events.append((k,v)),'synthetic',cadence=1000)
    def step(state,idx):
        clock[0]=2.
        return state+.1,np.sum(state*state,axis=1)
    with pytest.raises(pair.PilotStop,match='deadline'):
        gdmaps.iterate_and_screen(step,lambda state,idx,tangent:tangent,np.zeros((1,2)),8,
            tail_steps=4,ftle_steps=2,ftle_align=1,sd_horizons=(),observer=observer)
    terminal=[v for k,v in events if k=='bootstrap_terminal']
    assert len(terminal)==1 and terminal[0]['completed_loops']==1
    np.testing.assert_array_equal(terminal[0]['accepted_updates'],[1])
    np.testing.assert_array_equal(terminal[0]['state'],[[.1,.1]])
    assert terminal[0]['tail'].shape==(1,0,2)


def translated_family(params):
    center=float(params[0]+.3*params[1])
    def f(x):
        z=np.asarray(x)-center
        return center-z+.2*z*(z*z-1)
    def jac(x):
        z=np.asarray(x)-center
        return np.diag(-1+.2*(3*z*z-1))
    return f,jac


def bridge(max_links=19):
    engine=bootstrap.HiddenBridge(translated_family,pair.Options(),pair.Budget(30,160),max_links=max_links)
    root=engine.node(np.array([0.,.2]),[np.array([-.94]),np.array([1.06])],'anchor')
    assert root['accepted']
    return engine,root


def test_hidden_axis_secants_preserve_carried_phase_despite_native_swaps():
    engine,root=bridge()
    original=engine.node
    def swap(params,seeds,purpose):
        result=original(params,seeds,purpose)
        if result['accepted']:
            result['pair']=result['pair'][::-1].copy()
            result['spectra']['phases']=result['spectra']['phases'][::-1]
            result['corrections']=result['corrections'][::-1]
        return result
    engine.node=swap
    previous=None;current=root
    for target in (-.001,-.002):
        segment,current,previous=engine.hidden_segment(current,np.array([target,.2]),previous)
        assert segment['status']=='connected'
        assert current['params'][0]==target and current['params'][1]==.2
        assert current['carried_phase']==1
        np.testing.assert_allclose(current['pair'],np.array([[-.94],[1.06]])+target,atol=1e-13,rtol=0)
    assert len(engine.edges)==2 and all(e['status']=='accepted' for e in engine.edges)


def test_hidden_gap_is_not_bridged_and_attempt_cap_precedes_new_edge():
    engine,root=bridge()
    prior=engine.family
    def gap(params):
        if params[0]<=-.00045:raise ValueError('synthetic branch gap')
        return prior(params)
    engine.family=gap
    segment,last,_=engine.hidden_segment(root,np.array([-.001,.2]))
    assert segment['status']=='unresolved_gap' and last['params'][0]>-.00045
    assert any(e['status']=='rejected' for e in engine.edges)
    assert all(engine.edges[int(e.rsplit('-',1)[1])]['status']=='accepted' for e in segment['path'])
    capped,root=bridge(1)
    capped.link(root,np.array([-.001,.2]))
    calls=capped.budget.calls
    with pytest.raises(pair.PilotStop,match='global_link_attempt_limit'):
        capped.link(root,np.array([-.002,.2]))
    assert len(capped.edges)==1 and capped.budget.calls==calls


def test_common_control_identity_precedes_phase_distance(monkeypatch):
    engine,node=bridge()
    other=deepcopy(node);other['params'][0]=np.nextafter(other['params'][0],1.)
    monkeypatch.setattr(bootstrap,'phase_match',lambda *a,**k:pytest.fail('phase distance evaluated before identity gate'))
    with pytest.raises(ValueError,match='identical binary64 controls'):
        bootstrap.compare_common_controls(node,other)
    other=deepcopy(node);other['rates'][0]=np.nextafter(other['rates'][0],2.)
    with pytest.raises(ValueError,match='identical binary64 controls'):
        bootstrap.compare_common_controls(node,other)


def test_numerical_coexistence_requires_distinct_qualified_attracting_pairs():
    engine,node=bridge()
    same=deepcopy(node);same['pair']=same['pair'][::-1]
    assert bootstrap.compare_common_controls(node,same)['status']=='same_pair_within_tolerance'
    intermediate=deepcopy(node);intermediate['pair']+=2e-6
    assert bootstrap.compare_common_controls(node,intermediate)['status']=='unresolved_match'
    distinct=deepcopy(node);distinct['pair']+=2e-3
    comparison=bootstrap.compare_common_controls(node,distinct)
    assert comparison['status']=='numerically_distinct_pairs' and comparison['numerically_coexisting_attractors']
    distinct['spectra']['phases'][0]['stability']='neutral_or_unresolved'
    assert not bootstrap.compare_common_controls(node,distinct)['numerically_coexisting_attractors']
    distinct['accepted']=False
    with pytest.raises(ValueError,match='two qualified nodes'):
        bootstrap.compare_common_controls(node,distinct)
