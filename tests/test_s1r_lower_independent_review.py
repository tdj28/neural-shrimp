"""Independent synthetic controls for the lower-bootstrap comparison contract.

Never load a development anchor or call a numerical target map/corrector.
"""
from copy import deepcopy
import numpy as np
import pytest
from neural_shrimp import period2_branch as core
from neural_shrimp.s1r_bootstrap import HiddenBridge,compare_common_controls
from neural_shrimp.period2_branch_io import Store,file_entry,load_record
from pathlib import Path
import json

OUTPUT=.6271428571428572
UPPER=np.array([1.1,OUTPUT]);LOWER=np.array([.95,OUTPUT])


def ideal(params):
    x=np.zeros(72);x[0]=1.;shift=np.zeros(72);shift[1]=.003*params[0]**2
    return np.stack([x+shift,-x+shift])


def independent_match(a,b,replay=False):
    scale=max(1.,abs(a).max(),abs(b).max())
    costs=np.array([abs(a-b).max()/scale,abs(a-b[::-1]).max()/scale]);phase=int(costs[1]<costs[0])
    sep=abs(a[0]-a[1]).max()/max(1.,abs(a).max());cap=min(.01,.25*sep,1e-7 if replay else np.inf)
    return phase,bool(costs[phase]<=cap and costs[1-phase]-costs[phase]>=.5*sep and sep>=1e-6)


def engine(monkeypatch,fail_at=None,unstable=False):
    events=[];counter=[0];budget=core.Budget(100,644,clock=lambda:0.)
    def emit(k,p):events.append((k,deepcopy(p)))
    def fake(family,params,seeds,options,budget,emit):
        counter[0]+=1
        for _ in range(2):budget.corrector()
        if fail_at is not None and counter[0]>=fail_at:return dict(accepted=False,reason='first_phase_corrector')
        pair=ideal(params)
        if counter[0]%3==0:pair=pair[::-1]
        return dict(accepted=True,reason='qualified',pair=pair,spectra=dict(phases=[dict(stability='unstable' if unstable else 'attracting')]*2))
    monkeypatch.setattr(core,'qualify_pair',fake)
    e=HiddenBridge(lambda p:pytest.fail('target family called'),core.Options(),budget,emit,max_links=80)
    return e,events,budget


def run_nominal(e):
    upper=e.node(UPPER,ideal(UPPER),'upper_requalification')
    lower=e.node(LOWER,ideal(LOWER),'lower_requalification')
    current=upper;previous=None;segments=[]
    for i in range(1,76):
        endpoint=LOWER.copy() if i==75 else np.array([UPPER[0]+(LOWER[0]-UPPER[0])*i/75,OUTPUT])
        segment,current,previous=e.hidden_segment(current,endpoint,previous)
        segments.append(segment)
        if segment['status']!='connected':break
    return upper,lower,current,segments


def test_full_75_link_plan_native_phases_secants_and_exact_endpoint(monkeypatch):
    e,events,budget=engine(monkeypatch);upper,lower,current,segments=run_nominal(e)
    assert len(segments)==75 and len(e.edges)==75 and budget.calls==604
    assert current['params'].tobytes()==lower['params'].tobytes()==LOWER.tobytes()
    pending={p['id']:p for k,p in events if k=='node_pending'}
    previous=None;parent=upper
    nonzero_bits=0
    for i,edge in enumerate(e.edges,1):
        target=LOWER.copy() if i==75 else np.array([UPPER[0]+(LOWER[0]-UPPER[0])*i/75,OUTPUT])
        assert edge['target'].tobytes()==target.tobytes()
        prediction=parent['pair'] if previous is None else parent['pair']+(target[0]-parent['params'][0])/(parent['params'][0]-previous['params'][0])*(parent['pair']-previous['pair'])
        np.testing.assert_array_equal(np.stack(pending[edge['direct']]['seeds']),prediction)
        direct=e.nodes[edge['direct']];phase,passed=independent_match(parent['pair'],direct['pair']);assert passed
        assert edge['parent_phase']==parent.get('carried_phase',0) and phase==edge['parent_match']['phase']
        nonzero_bits+=phase
        for key in ('half','endpoint','reverse'):assert e.nodes[edge[key]]['accepted']
        np.testing.assert_array_equal(e.nodes[edge['half']]['params'],(parent['params']+target)/2)
        np.testing.assert_array_equal(e.nodes[edge['reverse']]['params'],parent['params'])
        previous=parent;parent=dict(direct,carried_phase=phase,pair=direct['pair'][::-1] if phase else direct['pair'])
    assert nonzero_bits>0
    assert compare_common_controls(current,lower)['status']=='same_pair_within_tolerance'


def test_qualified_instability_does_not_sever_branch_or_become_attractor(monkeypatch):
    e,events,budget=engine(monkeypatch,unstable=True);_,lower,current,segments=run_nominal(e)
    r=compare_common_controls(current,lower)
    assert len(segments)==75 and r['status']=='same_pair_within_tolerance'
    assert not r['both_numerically_attracting'] and not r['numerically_coexisting_attractors']


def test_80_attempt_cap_prevents_81_and_phase_cap_is_exact(monkeypatch):
    e,events,budget=engine(monkeypatch)
    parent=e.node(UPPER,ideal(UPPER),'upper_requalification');e.node(LOWER,ideal(LOWER),'lower_requalification')
    for i in range(80):assert e.link(parent,np.array([1.099,OUTPUT]),parent['pair'],'synthetic')['status']=='accepted'
    assert budget.calls==644 and len(e.edges)==80
    before=len(events)
    with pytest.raises(core.PilotStop,match='global_link_attempt_limit'):e.link(parent,np.array([1.098,OUTPUT]))
    assert len(events)==before and budget.calls==644


def test_failed_witness_cannot_jump_gap(monkeypatch):
    e,events,budget=engine(monkeypatch,fail_at=3);upper,lower,current,segments=run_nominal(e)
    assert len(segments)==1 and segments[0]['status']=='unresolved_gap'
    assert len(e.edges)==5 and all(edge['status']=='rejected' for edge in e.edges)
    assert current['id']==upper['id'] and not segments[0]['path']
    assert all(edge['target'][1]==OUTPUT for edge in e.edges)
    assert not any(k=='association' for k,p in events)


def candidates(distance=0.,attracting=True):
    a=dict(accepted=True,params=LOWER.copy(),rates=np.array([10.**float(v) for v in LOWER]),pair=ideal(LOWER),
      spectra=dict(phases=[dict(stability='attracting' if attracting else 'unstable')]*2))
    b=deepcopy(a);b['pair'][:,2]+=distance
    return a,b


@pytest.mark.parametrize('field',['params','rates'])
def test_binary64_common_controls_precede_pair_matching(monkeypatch,field):
    a,b=candidates();b[field][0]=np.nextafter(b[field][0],np.inf)
    import neural_shrimp.s1r_bootstrap as module
    monkeypatch.setattr(module,'phase_match',lambda *args,**kw:pytest.fail('pair matching before exact controls'))
    with pytest.raises(ValueError,match='identical binary64'):compare_common_controls(a,b)


@pytest.mark.parametrize('distance,expected',[(0.,'same_pair_within_tolerance'),(1e-6,'unresolved_match'),(2e-5,'numerically_distinct_pairs')])
def test_same_distinct_and_intermediate_labels(distance,expected):
    a,b=candidates(distance);r=compare_common_controls(a,b)
    assert r['status']==expected
    assert r['numerically_coexisting_attractors']==(expected=='numerically_distinct_pairs')


def test_distinct_solution_is_not_automatically_coexisting_attractors():
    a,b=candidates(2e-5,False);r=compare_common_controls(a,b)
    assert r['status']=='numerically_distinct_pairs' and not r['numerically_coexisting_attractors']


def test_resource_interruption_keeps_pending_attempt_and_prior_state(monkeypatch):
    e,events,budget=engine(monkeypatch);parent=e.node(UPPER,ideal(UPPER),'upper');e.node(LOWER,ideal(LOWER),'lower')
    budget.limit=5
    with pytest.raises(core.PilotStop,match='phase_corrector_limit'):e.hidden_segment(parent,np.array([1.098,OUTPUT]))
    assert budget.calls==5 and len(e.edges)==1 and e.edges[0]['status']=='pending'
    assert np.array_equal(e.nodes[parent['id']]['pair'],ideal(UPPER))
    assert any(k=='node_pending' and p['purpose']=='hidden_bridge:direct' for k,p in events)
    assert not any(k=='edge' for k,p in events)


def test_exclusive_store_retains_pending_full_state_with_hashes(tmp_path):
    raw=tmp_path/'attempt';store=Store(raw,256*1024**2)
    store.emit('node_pending',dict(id='node-0000',params=UPPER,seeds=ideal(UPPER),source='synthetic'))
    store.finish(dict(status='resource_stop',reason='synthetic_phase_cap',last_accepted=None))
    manifest=json.loads((raw/'manifest.json').read_text())
    assert all(file_entry(raw/name)==entry for name,entry in manifest['files'].items())
    kind,p=load_record(raw,'00000-node_pending.json');np.testing.assert_array_equal(p['seeds'],ideal(UPPER))
    with pytest.raises(FileExistsError):Store(raw,256*1024**2)
    assert store.limit==256*1024**2 and store.reserve==64*1024**2


def lower_module(name):
    import importlib.util,os
    source=Path(os.environ.get('S1R_LOWER_REVIEW_SOURCE_ROOT',Path(__file__).resolve().parents[1]))
    path=source/'tools'/(name+'.py')
    spec=importlib.util.spec_from_file_location('independent_'+name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('failed_kind',['node','edge'])
def test_terminal_pending_ids_follow_durable_journal_not_mutated_object(failed_kind):
    from neural_shrimp.s1r_lower_association import Progress
    class FailingStore:
        def __init__(self):self.events=[]
        def emit(self,kind,payload,final=False):
            if kind==failed_kind:raise core.PilotStop('raw_byte_limit')
            self.events.append((kind,deepcopy(payload)))
    store=FailingStore();progress=Progress(store)
    node=dict(id='node-0000',params=UPPER,rates=np.array([10.**float(v) for v in UPPER]),pair=ideal(UPPER),reason='pending',accepted=False)
    edge=dict(id='edge-0000',status='pending',direct=node['id'],parent_match=dict(phase=1))
    class Engine:pass
    engine=Engine();engine.nodes={node['id']:node};engine.edges=[edge];progress.engine=engine
    progress.emit('node_pending',node)
    if failed_kind=='node':
        node.update(reason='qualified',accepted=True)
        with pytest.raises(core.PilotStop):progress.emit('node',node)
        assert progress.terminal()['pending_nodes']==['node-0000']
    else:
        node.update(reason='qualified',accepted=True);progress.emit('node',node)
        progress.emit('edge_pending',edge);edge['status']='accepted'
        with pytest.raises(core.PilotStop):progress.emit('edge',edge)
        assert progress.terminal()['pending_edges']==['edge-0000']
        assert progress.terminal()['last_witnessed'] is None
    assert not progress.accepted_edges


def test_driver_rejects_input_failure_before_any_correction_and_retains_attempt(tmp_path,monkeypatch):
    runner=lower_module('explore_s1r_lower_association')
    monkeypatch.setattr(runner,'sources',lambda:{'synthetic':{}})
    def failure():raise ValueError('independent synthetic input mismatch')
    monkeypatch.setattr(runner,'bound_inputs',failure)
    monkeypatch.setattr(runner,'HiddenBridge',lambda *args,**kwargs:pytest.fail('corrector built before valid inputs'))
    import subprocess
    review=subprocess.check_output(['git','rev-parse','HEAD'],cwd=runner.ROOT,text=True).strip()
    with pytest.raises(ValueError,match='synthetic input mismatch'):runner.run(tmp_path/'failed',review)
    raw=tmp_path/'failed';result=json.loads((raw/'result.json').read_text());manifest=json.loads((raw/'manifest.json').read_text())
    assert result['status']=='failed_or_interrupted' and result['phase_corrector_calls']==result['link_attempts']==0
    assert result['association'] is None
    assert all(file_entry(raw/name)==entry for name,entry in manifest['files'].items())
    assert not raw.stat().st_mode&0o222 and all(not p.stat().st_mode&0o222 for p in raw.iterdir())


def test_source_mutation_is_rejected_without_output_creation(tmp_path,monkeypatch):
    runner=lower_module('explore_s1r_lower_association')
    monkeypatch.setattr(runner,'ROOT',tmp_path);monkeypatch.setattr(runner,'SOURCES',('synthetic.py',))
    (tmp_path/'synthetic.py').write_bytes(b'changed')
    monkeypatch.setattr(runner.subprocess,'check_output',lambda *args,**kwargs:b'committed')
    with pytest.raises(ValueError,match='uncommitted source'):runner.sources()
    assert not (tmp_path/'attempt').exists()


def test_historical_input_manifest_requires_exact_inventory_and_rejects_symlink(tmp_path):
    runner=lower_module('explore_s1r_lower_association');raw=tmp_path/'raw';store=Store(raw)
    store.emit('bound_inputs',dict(x=np.ones((2,2))));store.finish(dict(status='synthetic'))
    sha=file_entry(raw/'manifest.json')['sha256'];runner.check_manifest(raw,sha)
    p=next(raw.glob('*.npz'));outside=tmp_path/'same-bytes';outside.write_bytes(p.read_bytes());p.unlink();p.symlink_to(outside)
    with pytest.raises(ValueError,match='binding'):runner.check_manifest(raw,sha)


def test_launcher_bounds_owned_descendant_after_leader_exit(tmp_path,monkeypatch):
    import os,subprocess,sys
    launcher=lower_module('launch_s1r_lower_association')
    monkeypatch.setattr(launcher,'committed_sources',lambda:{'synthetic':{}})
    monkeypatch.setattr(launcher,'SUPERVISOR_SECONDS',.15)
    pidfile=tmp_path/'owned-descendant.pid'
    code="import os,time,pathlib; pid=os.fork(); pathlib.Path(__import__('sys').argv[1]).write_text(str(os.getpid())) if pid==0 else None; time.sleep(20) if pid==0 else os._exit(0)"
    monkeypatch.setattr(launcher,'command',lambda out,review:[sys.executable,'-c',code,str(pidfile)])
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=launcher.ROOT,text=True).strip()
    receipt=launcher.launch(tmp_path/'raw',tmp_path/'launch',head,head)
    assert receipt['failure']=='supervisor_timeout_or_inherited_pipe' and receipt['elapsed_seconds']<3
    assert receipt['process_exit']==0  # direct leader exit cannot conceal inherited-pipe failure
    child=int(pidfile.read_text());status=Path(f'/proc/{child}/stat')
    assert not status.exists() or status.read_text().split()[2] in ('Z','X')
    manifest=json.loads((tmp_path/'launch'/'manifest.json').read_text())
    assert all(file_entry(tmp_path/'launch'/name)==entry for name,entry in manifest['files'].items())
