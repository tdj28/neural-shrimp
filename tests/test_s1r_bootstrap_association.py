"""Synthetic-only controls: never load the S1R data, profile state or target inputs."""
from copy import deepcopy
from dataclasses import dataclass
import importlib.util
from pathlib import Path
import subprocess
import sys
import types
import json
import numpy as np
import pytest
from neural_shrimp import gdmaps as g
from neural_shrimp import s1r_bootstrap as b
from neural_shrimp.period2_branch import Options,Budget,PilotStop
from neural_shrimp.period2_branch_io import Store,load_record,file_entry


def historical():
    name='frozen_gdmaps_s1r_test';module=types.ModuleType(name);sys.modules[name]=module
    source=subprocess.check_output(['git','show','960ab535e1954e6f46dce21999338ddcf1bd23ee:src/neural_shrimp/gdmaps.py'],text=True)
    exec(compile(source,'frozen_gdmaps.py','exec'),module.__dict__);return module


def assert_identical(a,b):
    assert set(a)==set(b)
    for key in a:
        assert a[key].dtype==b[key].dtype and a[key].shape==b[key].shape
        assert a[key].tobytes()==b[key].tobytes(),key


def test_observer_free_and_mutating_copied_observer_are_bit_identical():
    net=g.TwoLayerNet(2,2,bias=False,ridge=.01)
    grid=g.GridMap(net,np.array([[.1,.2],[-.4,.3],[.7,-.2]]),np.array([.2,-.1,.3]),[.02],[.03])
    theta=np.array([.2,-.3,.4,.5,-.2,.6]);options=dict(tail_steps=12,ftle_steps=5,ftle_align=3,max_period=2,min_repeats=4,sd_horizons=(20,))
    gold=historical().run_grid(grid,theta,40,**options)
    assert_identical(gold,g.run_grid(grid,theta,40,**options))
    seen=[]
    def observer(kind,get):
        snap=get();seen.append((kind,deepcopy(snap)))
        for value in snap.values():
            if isinstance(value,np.ndarray):value[:]=0
    result=g.run_grid(grid,theta,40,observer=observer,**options);assert_identical(gold,result)
    final=seen[-1][1];assert seen[-1][0]=='terminal'
    assert final['accepted_updates'].tolist()==[40] and final['attempted_updates'].tolist()==[40]
    assert final['jvp_attempts'].tolist()==[8] and final['jvp_successful_returns'].tolist()==[8]
    assert final['tail_state_updates'].tolist()==list(range(29,41))
    assert final['tail_loss_updates'].tolist()==list(range(28,40))
    states=[s['state'][0] for k,s in seen if k=='after_step']
    np.testing.assert_array_equal(final['tail'][0],states[-12:])
    prior=np.vstack([theta,*states[:-1]])
    np.testing.assert_array_equal(final['tail_losses'][0],np.array([net.gradient(t[None,:],grid.x,grid.y,return_loss=True)[1][0] for t in prior[-12:]]))


def test_deadline_after_update_before_observer_cadence_keeps_newest_state():
    clock=[0.];events=[];budget=b.InterruptibleBudget(1,clock=lambda:clock[0])
    observer=b.WarmObserver(budget,lambda k,v:events.append((k,v)),'synthetic',1000)
    def step(states,idx):clock[0]=2;return states+1,np.zeros(len(states))
    with pytest.raises(PilotStop,match='deadline'):
        g.iterate_and_screen(step,lambda s,i,v:v,np.zeros((1,2)),10,tail_steps=4,ftle_steps=2,ftle_align=0,sd_horizons=(),observer=observer)
    assert observer.loops==0
    terminal=events[-1][1];assert terminal['completed_loops']==1 and terminal['accepted_updates'].tolist()==[1]
    np.testing.assert_array_equal(terminal['state'],np.ones((1,2)))


def test_divergence_retains_last_valid_and_attempted_counts():
    calls=[0];snapshots=[]
    def step(states,idx):
        calls[0]+=1
        return (states+1 if calls[0]<3 else np.full_like(states,np.inf)),np.zeros(len(states))
    result=g.iterate_and_screen(step,lambda s,i,v:v,np.zeros((1,1)),10,tail_steps=4,ftle_steps=2,ftle_align=0,sd_horizons=(),observer=lambda k,s:snapshots.append((k,s())) if k=='terminal' else None)
    end=snapshots[-1][1]
    assert result['period'].tolist()==[-1] and end['attempted_updates'].tolist()==[3] and end['accepted_updates'].tolist()==[2]
    assert np.isnan(end['state']).all() and end['last_valid_state'][0,0]==2


def test_interrupted_jvp_has_attempt_but_no_successful_return():
    snapshots=[]
    def fail(*args):raise RuntimeError('synthetic interrupted JVP')
    with pytest.raises(RuntimeError,match='JVP'):
        g.iterate_and_screen(lambda s,i:(s+1,np.zeros(len(s))),fail,np.zeros((1,1)),5,tail_steps=2,ftle_steps=2,ftle_align=0,sd_horizons=(),observer=lambda k,s:snapshots.append(s()) if k=='terminal' else None)
    end=snapshots[-1];assert end['attempted_updates'].tolist()==[4] and end['accepted_updates'].tolist()==[3]
    assert end['jvp_attempts'].tolist()==[1] and end['jvp_successful_returns'].tolist()==[0]
    assert end['state'][0,0]==3


def test_terminal_tail_seeds_and_incomplete_or_wrong_period_abstain():
    tail=np.zeros((1,384,72));tail[0,-1]=1;tail[0,-2]=-1
    terminal=dict(accepted_updates=np.array([40000]),attempted_updates=np.array([40000]),state=np.ones((1,72)),alive=np.array([True]),tail=tail,tail_state_updates=np.arange(39617,40001))
    result=dict(period=np.array([2]),final_state=np.ones((1,72)))
    full,seeds=b.bootstrap_seeds(result,terminal);assert full
    np.testing.assert_array_equal(seeds,np.vstack([np.ones(72),-np.ones(72)]))
    assert b.bootstrap_seeds(dict(result,period=np.array([1])),terminal)[1] is None
    assert b.bootstrap_seeds(result,dict(terminal,accepted_updates=np.array([39999])))[1] is None
    with pytest.raises(ValueError,match='indexing'):b.bootstrap_seeds(result,dict(terminal,tail_state_updates=np.arange(39616,40000)))


def family(params):
    center=float(params[0]+2*params[1])
    def f(x):z=np.asarray(x)-center;return center-z+.2*z*(z*z-1)
    def jac(x):z=np.asarray(x)-center;return np.diag(-1+.2*(3*z*z-1))
    return f,jac


def engine():return b.HiddenBridge(family,Options(),Budget(20,160))
def anchor(e):return e.node(np.array([0.,0.]),[np.array([-1.]),np.array([1.])],'synthetic')


def test_hidden_axis_secant_exact_end_and_fixed_output():
    e=engine();root=anchor(e)
    segment,end,prev=e.hidden_segment(root,np.array([.002,0.]))
    assert segment['status']=='connected' and end['params'].tobytes()==np.array([.002,0.]).tobytes()
    segment,last,_=e.hidden_segment(end,np.array([.004,0.]),prev)
    assert segment['status']=='connected';np.testing.assert_allclose(last['pair'],np.array([[-.996],[1.004]]),atol=1e-14)
    with pytest.raises(ValueError,match='fixed output'):e.hidden_segment(last,np.array([.005,.001]))


def test_carried_swapped_phase_preserves_native_record():
    e=engine();root=anchor(e);root_native=root['pair'].copy();root={**root,'pair':root['pair'][::-1].copy(),'carried_phase':1}
    segment,end,_=e.hidden_segment(root,np.array([.001,0.]))
    assert segment['status']=='connected' and e.edges[0]['parent_phase']==1
    np.testing.assert_array_equal(e.nodes[root['id']]['pair'],root_native)
    assert end['pair'][0,0]>end['pair'][1,0]


def test_gap_stops_without_crossing_and_global_cap_before_20th_edge():
    e=engine();root=anchor(e);base=e.family
    e.family=lambda p: (_ for _ in ()).throw(ValueError('gap')) if p[0]>=.00045 else base(p)
    segment,last,_=e.hidden_segment(root,np.array([.002,0.]))
    assert segment['status']=='unresolved_gap' and last['params'][0]<.00045
    e.edges=[{} for _ in range(19)]
    with pytest.raises(PilotStop,match='global_link'):e.link(root,np.array([.001,0.]))
    assert len(e.edges)==19


def test_common_controls_only_match_distinct_or_unresolved():
    e=engine();a=anchor(e);z=deepcopy(a)
    assert b.compare_common_controls(a,z)['status']=='same_pair_within_tolerance'
    z['pair']+=.001
    assert b.compare_common_controls(a,z)['numerically_coexisting_attractors']
    z['pair']=a['pair']+1e-6
    assert b.compare_common_controls(a,z)['status']=='unresolved_match'
    z['rates']=z['rates'].copy();z['rates'][0]=np.nextafter(z['rates'][0],np.inf)
    with pytest.raises(ValueError,match='controls'):b.compare_common_controls(a,z)
    z=deepcopy(a);z['accepted']=False
    with pytest.raises(ValueError,match='qualified'):b.compare_common_controls(a,z)


def test_fixed_profile_uses_loss_path_and_never_advances_input():
    class Net:
        n_parameters=2
        def gradient(self,t,x,y,return_loss=False):
            assert return_loss and np.array_equal(t,[[1.,2.]])
            return np.ones_like(t),np.array([3.])
        def hvp(self,t,x,y,v):
            assert np.array_equal(t,[[1.,2.]]) and np.array_equal(v,[[1.,0.]])
            return np.ones_like(t)
    grid=types.SimpleNamespace(net=Net(),x=None,y=None,rates=np.ones((1,2)))
    events=[];r=b.fixed_state_profile(grid,[1.,2.],Budget(10),lambda k,v:events.append((k,v)),5,3)
    assert r['map_calls']==5 and r['jvp_calls']==3 and events[-1][0]=='profile_counts'


def test_final_snapshot_can_use_reserved_storage(tmp_path):
    store=Store(tmp_path/'reserved',65536)
    store.emit('fill',{'array':np.zeros(5300)})
    with pytest.raises(PilotStop):store.emit('nonfinal',{'array':np.zeros(1200)})
    store.emit('bootstrap_terminal',{'state':np.zeros(1200)},final=True)
    store.finish(dict(status='resource_stop'))
    assert (store.root/'manifest.json').exists() and sum(p.stat().st_size for p in store.root.iterdir())<=65536


def runner_module():
    spec=importlib.util.spec_from_file_location('bootstrap_runner',Path(__file__).parents[1]/'tools/explore_s1r_bootstrap_association.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_profile_gate_rejects_wrong_hash_counts_or_nonfinite_time(tmp_path):
    runner=runner_module();source={'fake':{'sha256':'abc'}}
    folder=tmp_path/'profile';store=Store(folder)
    runtime=dict(sources=source,input_manifest_sha256=runner.INPUT_MANIFEST,device='cpu',nice=10,thread_env={k:'1' for k in runner.THREADS})
    store.json('runtime-start.json',runtime)
    timing=dict(map_calls=2000,jvp_calls=128,map_seconds=.01,jvp_seconds=.001)
    timing['projected_collection_seconds']=2*(80000*.01/2000+2400*.001/128)+30;timing['fits_internal_100_seconds']=True
    result=dict(mode='profile',status='profile_complete',timing=timing);store.finish(result)
    assert runner.qualified_profile(folder,source)['projected_collection_seconds']<100
    for key,value in [('map_calls',1999),('map_seconds',float('nan')),('jvp_seconds',-1),('projected_collection_seconds',99.)]:
        changed=deepcopy(result);changed['timing'][key]=value
        (folder/'result.json').write_text(json.dumps(changed));m=json.loads((folder/'manifest.json').read_text());m['files']['result.json']=file_entry(folder/'result.json');(folder/'manifest.json').write_text(json.dumps(m))
        with pytest.raises(ValueError):runner.qualified_profile(folder,source)


def test_collection_synthetic_full_bridge_and_no_lower_association(tmp_path,monkeypatch):
    runner=runner_module()
    @dataclass
    class Model:n_parameters:int=1
    class Grid:
        def __init__(self,net,x,y,hidden,output):self.params=np.log10([hidden[0],output[0]])
        def cell_map(self,index):return family(self.params)
    central_center=b.CENTRAL_HIDDEN+2*b.OUTPUT_LOG
    central=dict(params=np.array([b.CENTRAL_HIDDEN,b.OUTPUT_LOG]),pair=np.array([[central_center-1],[central_center+1]]))
    monkeypatch.setattr(runner,'sources',lambda:{'synthetic':{}})
    monkeypatch.setattr(runner,'qualified_profile',lambda *a:{'synthetic':True})
    monkeypatch.setattr(runner,'bound_inputs',lambda:(Model(),None,None,np.zeros(1),central,{}))
    monkeypatch.setattr(runner,'GridMap',Grid)
    def bootstrap(grid,theta,budget,emit,label):
        center=sum(grid.params*np.array([1.,2.]));return dict(period=np.array([2]),status=np.array([2])),np.array([[center-1],[center+1]])
    monkeypatch.setattr(runner,'run_bootstrap',bootstrap)
    result=runner.run('collect',tmp_path/'collection','960ab535e1954e6f46dce21999338ddcf1bd23ee',Path('synthetic'))
    assert result['status']=='completed_common_control_comparison' and result['phase_corrector_calls']==102 and result['link_attempts']==12
    assert result['association']['status']=='same_pair_within_tolerance' and result['lower_start_association']=='not_attempted'
    assert result['historical_family_association']=='unresolved'


def test_historical_ftle_window_makes_1200_preupdate_jvps():
    snapshots=[];calls=[]
    def jvp(states,idx,vector):calls.append(float(states[0,0]));return vector
    result=g.iterate_and_screen(lambda s,i:(s+1,np.zeros(len(s))),jvp,np.zeros((1,1)),1300,
        tail_steps=384,ftle_steps=1000,ftle_align=200,sd_horizons=(),
        observer=lambda k,s:snapshots.append(s()) if k=='terminal' else None)
    assert calls==list(range(100,1300))
    assert snapshots[0]['jvp_attempts'].tolist()==[1200]
    assert snapshots[0]['jvp_successful_returns'].tolist()==[1200]
    assert result['ftle'].tolist()==[0.]


@pytest.mark.parametrize('bad_stage',['map','jvp'])
def test_nonfinite_profile_retains_completed_call_counts(bad_stage):
    class Net:
        n_parameters=1
        def gradient(self,t,x,y,return_loss=False):
            return np.array([[np.nan if bad_stage=='map' else 1.]]),np.array([0.])
        def hvp(self,t,x,y,v):return np.array([[np.inf]])
    grid=types.SimpleNamespace(net=Net(),x=None,y=None,rates=np.ones((1,1)))
    events=[]
    with pytest.raises(FloatingPointError):
        b.fixed_state_profile(grid,[1.],Budget(10),lambda k,v:events.append((k,v.copy())),2,2)
    assert events[-1]==('profile_counts',dict(map_calls=0 if bad_stage=='map' else 2,jvp_calls=0))


def test_external_signal_stops_before_new_map_but_retains_terminal():
    budget=b.InterruptibleBudget(10,stop_requested=lambda:True);events=[]
    observer=b.WarmObserver(budget,lambda k,v:events.append((k,v)),'synthetic')
    def forbidden(*args):raise AssertionError('no map call allowed')
    with pytest.raises(PilotStop,match='external_signal'):
        g.iterate_and_screen(forbidden,forbidden,np.zeros((1,1)),10,tail_steps=4,ftle_steps=2,ftle_align=0,sd_horizons=(),observer=observer)
    assert len(events)==1 and events[0][0]=='bootstrap_terminal'
    assert events[0][1]['completed_loops']==0 and events[0][1]['attempted_updates'].tolist()==[0]
