"""Prospective lower saved-data audit controls: synthetic artifacts only."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import audit_s1r_lower_outcome as a
from neural_shrimp import period2_branch as core
from neural_shrimp.s1r_bootstrap import HiddenBridge
from neural_shrimp.s1r_lower_association import Progress,compare_along_bridge
from neural_shrimp.period2_branch_io import Store,file_entry


def pair(params,offset=0):
    x=np.zeros(72);x[0]=1.;s=np.zeros(72);s[1]=.003*params[0]**2;s[2]=offset
    return np.stack([x+s,-x+s])


def endpoint(hidden):
    params=np.array([hidden,a.OUTPUT]);return dict(id='saved',params=params,rates=np.array([10.**float(v) for v in params]),pair=pair(params))


def fixture(monkeypatch,*,reject_from=None,offset=0.,unstable=False,fail_event=None,cap=644):
    class Memory:
        def __init__(self):self.events=[]
        def emit(self,k,p,final=False):
            if k==fail_event:raise core.PilotStop('raw_byte_limit')
            self.events.append((k,deepcopy(p)))
    store=Memory();progress=Progress(store);calls=[0];budget=core.Budget(100,cap,clock=lambda:0.)
    def qualify(family,params,seeds,options,budget,emit):
        calls[0]+=1
        for _ in range(2):budget.corrector()
        if reject_from is not None and calls[0]>=reject_from:return dict(accepted=False,reason='first_phase_corrector')
        p=pair(params,offset if calls[0]==2 else 0.)
        if calls[0]%3==0:p=p[::-1]
        return dict(accepted=True,reason='qualified',pair=p,spectra=dict(phases=[dict(stability='unstable' if unstable else 'attracting')]*2))
    monkeypatch.setattr(core,'qualify_pair',qualify)
    engine=HiddenBridge(lambda p:pytest.fail('target model requested'),core.Options(),budget,progress.emit,80);progress.engine=engine
    inputs=dict(endpoints=dict(upper=endpoint(1.1),lower=endpoint(.95)))
    result=dict(status='running',bridge=[],association=None)
    try:compare_along_bridge(engine,inputs['endpoints']['upper'],inputs['endpoints']['lower'],result,progress)
    except core.PilotStop as e:result.update(status='resource_stop',reason=str(e),association=None)
    terminal=progress.terminal();store.emit('terminal',terminal,final=True)
    result.update(phase_corrector_calls=budget.calls,link_attempts=len(engine.edges),processed_macro_segments=len(result['bridge']),
      completed_macro_links=sum(s['status']=='connected' for s in result['bridge']),witnessed_edge_ids=terminal['accepted_edges'],
      last_witnessed_node=None if terminal['last_witnessed'] is None else terminal['last_witnessed']['id'])
    return store.events,result,inputs


def verify(ev,r,inputs):
    parser=a.LowerStructure(ev,r);result=parser.run(inputs);terminal=a.terminal_audit(ev,r,parser)
    return result,terminal


def test_complete_75_macro_graph_native_phase_secants_and_terminal(monkeypatch):
    ev,r,inputs=fixture(monkeypatch);g,t=verify(ev,r,inputs)
    assert g['completed_nodes']==302 and g['accepted_links']==75 and t['pending_nodes']==t['pending_edges']==[]
    assert r['phase_corrector_calls']==604 and g['association']['status']=='same_pair_within_tolerance'


@pytest.mark.parametrize('offset,unstable,label,coexist',[(0.,False,'same_pair_within_tolerance',False),(1e-6,False,'unresolved_match',False),(2e-5,False,'numerically_distinct_pairs',True),(2e-5,True,'numerically_distinct_pairs',False)])
def test_common_control_outcomes_keep_stability_scope(monkeypatch,offset,unstable,label,coexist):
    ev,r,inputs=fixture(monkeypatch,offset=offset,unstable=unstable);g,t=verify(ev,r,inputs)
    assert g['association']['status']==label and g['association']['numerically_coexisting_attractors']==coexist


@pytest.mark.parametrize('field',['seed','phase','target','half','reverse','verdict','macro','terminal_pair','pending','unpersisted'])
def test_hash_valid_semantic_mutations_fail(monkeypatch,field):
    ev,r,inputs=fixture(monkeypatch)
    select=lambda kind,predicate=lambda p:True:next(p for k,p in ev if k==kind and predicate(p))
    if field=='seed':select('node_pending',lambda p:p['purpose']=='hidden_bridge:direct')['seeds'][0][2]+=1e-5
    if field=='phase':select('edge_pending')['parent_phase']=1
    if field=='target':select('edge_pending')['target'][0]=np.nextafter(select('edge_pending')['target'][0],np.inf)
    if field=='half':select('node_pending',lambda p:p['purpose']=='hidden_bridge:half')['params'][0]+=.001
    if field=='reverse':select('node_pending',lambda p:p['purpose']=='hidden_bridge:reverse')['seeds'][0][1]+=.001
    if field=='verdict':r['association']['status']='numerically_distinct_pairs'
    if field=='macro':r['macro_targets'][-1][0]=np.nextafter(r['macro_targets'][-1][0],np.inf)
    if field=='terminal_pair':select('terminal')['last_witnessed']['pair'][0,0]+=.001
    if field=='pending':select('terminal')['pending_edges']=['edge-0000']
    if field=='unpersisted':select('terminal')['unpersisted_edge_attempts']=['edge-0075']
    with pytest.raises(ValueError):verify(ev,r,inputs)


@pytest.mark.parametrize('reject_from',[1,2,3,4,5,6])
def test_failed_endpoint_and_each_witness_preserve_actual_gap(monkeypatch,reject_from):
    ev,r,inputs=fixture(monkeypatch,reject_from=reject_from);g,t=verify(ev,r,inputs)
    assert g['association'] is None
    assert r['status']==('endpoint_requalification_failed' if reject_from<=2 else 'bridge_gap')
    if reject_from>2:assert r['link_attempts']==5 and g['accepted_links']==0


@pytest.mark.parametrize('event',['node_pending','node','edge_pending','edge','bridge_anchor','bridge_segment','association'])
def test_failed_emission_does_not_invent_finished_state(monkeypatch,event):
    ev,r,inputs=fixture(monkeypatch,fail_event=event);g,t=verify(ev,r,inputs)
    assert r['status']=='resource_stop' and g['association'] is None
    if event=='node':assert t['pending_nodes']==['node-0000']
    if event=='node_pending':assert t['unpersisted_node_attempts']==['node-0000']
    if event=='edge':assert t['pending_edges']==['edge-0000'] and g['accepted_links']==0
    if event=='edge_pending':assert t['unpersisted_edge_attempts']==['edge-0000']
    if event=='association':assert g['accepted_links']==75


def test_mid_phase_cap_is_incomplete_not_nonexistence(monkeypatch):
    ev,r,inputs=fixture(monkeypatch,cap=5);g,t=verify(ev,r,inputs)
    assert t['pending_nodes']==['node-0002'] and t['pending_edges']==['edge-0000']
    assert g['association'] is None and r['phase_corrector_calls']==5


def history(calls):
    events=[];state=np.zeros(72)
    for call in range(1,calls+1):
        node=f'node-{call-1:04d}'
        events.append(('node_pending',dict(id=node)))
        start=dict(call=call,state=state,seed=state)
        it=dict(call=call,iteration=1,state=state,step=state,residual=state,final_relative_correction=0.)
        end=dict(call=call,state=state,seed=state,converged=True,reason='converged',iterations=1,final_relative_correction=0.,candidate_distance=0.)
        events.extend([('corrector_start',start),('corrector_iteration',it),('corrector_end',end),('node',dict(id=node,corrections=[end]))])
    return events


def test_explicit_644_history_cap_preserves_old_160_default():
    events=history(644);result=dict(phase_corrector_calls=644,status='completed_common_control_comparison')
    assert a.prior.corrector_history(events,result,max_calls=644,complete_statuses=a.COMPLETE)['recorded_ends']==644
    with pytest.raises(ValueError,match='phase call budget'):a.prior.corrector_history(events,result)
    result['phase_corrector_calls']=645
    with pytest.raises(ValueError):a.prior.corrector_history(events,result,max_calls=644,complete_statuses=a.COMPLETE)


def test_lower_parser_rejects_bootstrap_and_unjournaled_events(tmp_path):
    raw=tmp_path/'raw';store=Store(raw);store.emit('node_pending',dict(id='node-0000'));store.finish(dict(status='resource_stop'))
    assert len(a.saved_events(raw)[0])==1
    obj=json.loads((raw/'00000-node_pending.json').read_text());obj['kind']='bootstrap_start';(raw/'00000-node_pending.json').write_text(json.dumps(obj))
    with pytest.raises(ValueError):a.saved_events(raw)


def test_exact_controls_before_distance_evaluation(monkeypatch):
    n=dict(accepted=True,params=np.array([.95,a.OUTPUT]),rates=np.array([10.**.95,10.**a.OUTPUT]),pair=pair([.95]),spectra=dict(phases=[dict(stability='attracting')]*2))
    other=deepcopy(n);other['rates'][0]=np.nextafter(other['rates'][0],np.inf)
    monkeypatch.setattr(a.prior,'pair_match',lambda *args,**kwargs:pytest.fail('matched changed rates'))
    with pytest.raises(ValueError):a.prior.common(n,other)


def test_stop_between_bound_inputs_and_schedule_is_incomplete():
    r=dict(status='failed_or_interrupted',bridge=[],association=None,witnessed_edge_ids=[],last_witnessed_node=None,
      processed_macro_segments=0,completed_macro_links=0,link_attempts=0)
    inputs=dict(endpoints=dict(upper=endpoint(1.1),lower=endpoint(.95)))
    terminal=dict(last_witnessed=None,accepted_edges=[],latest_corrector=None,pending_nodes=[],pending_edges=[],
      durably_finished_nodes=[],durably_finished_edges=[],unpersisted_node_attempts=[],unpersisted_edge_attempts=[])
    g,t=verify([('terminal',terminal)],r,inputs)
    assert g['completed_nodes']==0 and g['association'] is None
    r['status']='completed_common_control_comparison'
    with pytest.raises(ValueError):verify([('terminal',terminal)],r,inputs)


def synthetic_launcher(tmp_path,monkeypatch,mutation=None):
    raw=tmp_path/'raw';raw.mkdir();launch=tmp_path/'launch';launch.mkdir()
    result=dict(status='completed_common_control_comparison',elapsed_seconds=39.)
    (raw/'result.json').write_bytes(a.canonical(result))
    (raw/'manifest.json').write_bytes(a.canonical(dict(files={'result.json':a.entry(raw/'result.json')})))
    files={p.name:a.entry(p) for p in raw.iterdir()}
    source=b'# synthetic launcher\n'
    sources={'tools/launch_s1r_lower_association.py':dict(bytes=len(source),sha256=a.digest(source))}
    rt=dict(schema='s1r-lower-launcher-v1',source_commit=a.EXECUTION,source_review_commit=a.REVIEW,sources=sources,
      output=str(raw),launcher=str(launch),cwd=a.EXECUTION_ROOT,command=['timeout','--signal=TERM','--kill-after=5s','115s','nice','-n','10',
      '/data2/neural-shrimp/.venv/bin/python','tools/explore_s1r_lower_association.py','--out',str(raw),'--review-commit',a.REVIEW],
      nice=10,thread_env=dict.fromkeys(a.THREADS,'1'),launcher_pid=10,child_pid=11,started_utc='2026-10-08T10:00:00+00:00')
    rt.update({k:a.LIMITS[k] for k in ('external_seconds','kill_grace_seconds','store_bytes','launcher_bytes','total_raw_bytes')})
    runtime=dict(pid=12,started_utc='2026-10-08T10:00:01+00:00')
    stdout=a.canonical(result)
    receipt=dict(source_commit=a.EXECUTION,source_review_commit=a.REVIEW,child_pid=11,status='process_finished',failure=None,
      log_truncated=False,within_store_cap=True,process_exit=0,raw_file_count=len(files),raw_bytes=sum(x['bytes'] for x in files.values()),
      raw_inventory_sha256=a.digest(a.canonical(files)),raw_manifest=a.entry(raw/'manifest.json'),raw_result=a.entry(raw/'result.json'),
      stdout=dict(bytes=len(stdout),sha256=a.digest(stdout)),elapsed_seconds=40.,finished_utc='2026-10-08T10:00:40+00:00')
    if mutation:mutation(rt,receipt,runtime)
    for name,value in (('launcher-source.py',source),('runtime-start.json',a.canonical(rt)),('receipt.json',a.canonical(receipt)),('stdout.log',stdout)):
        (launch/name).write_bytes(value)
    (launch/'manifest.json').write_bytes(a.canonical(dict(files={p.name:a.entry(p) for p in launch.iterdir()},status='process_finished')))
    monkeypatch.setattr(a,'LAUNCHER_MANIFEST',a.entry(launch/'manifest.json')['sha256'])
    for p in launch.iterdir():p.chmod(0o444)
    launch.chmod(0o555)
    return launch,raw,runtime,result,sources


def test_independent_launcher_inventory_and_process_binding(tmp_path,monkeypatch):
    data=synthetic_launcher(tmp_path,monkeypatch)
    result=a.launch_binding(*data)
    assert result['exit_code']==0 and result['child_pid']==11


@pytest.mark.parametrize('mutation',[
 lambda rt,r,p:rt.update(source_commit='0'*40),
 lambda rt,r,p:r.update(source_review_commit='0'*40),
 lambda rt,r,p:rt['command'].__setitem__(3,'116s'),
 lambda rt,r,p:rt['thread_env'].update(OMP_NUM_THREADS='2'),
 lambda rt,r,p:rt.update(cwd='/tmp/other-checkout'),
 lambda rt,r,p:rt.update(nice=0),
 lambda rt,r,p:rt.update(external_seconds=116),
 lambda rt,r,p:r.update(child_pid=13),
 lambda rt,r,p:p.update(pid=11),
 lambda rt,r,p:p.update(started_utc='2026-10-08T10:00:41+00:00'),
 lambda rt,r,p:r.update(process_exit=2),
 lambda rt,r,p:r.update(raw_inventory_sha256='0'*64),
 lambda rt,r,p:r.update(raw_file_count=999),
 lambda rt,r,p:r.update(raw_bytes=999),
 lambda rt,r,p:r.update(raw_result={'bytes':0,'sha256':'0'*64}),
 lambda rt,r,p:r.update(stdout={'bytes':0,'sha256':'0'*64}),
 lambda rt,r,p:r.update(log_truncated=True),
 lambda rt,r,p:r.update(failure='supervisor_timeout_or_inherited_pipe'),
 lambda rt,r,p:r.update(elapsed_seconds=130.),
 lambda rt,r,p:r.update(finished_utc='2026-10-08T10:10:00+00:00'),
])
def test_hash_valid_launcher_mutations_reject(tmp_path,monkeypatch,mutation):
    data=synthetic_launcher(tmp_path,monkeypatch,mutation)
    with pytest.raises(ValueError):a.launch_binding(*data)


@pytest.mark.parametrize('where',['basis','execution','local'])
def test_reviewed_execution_and_local_source_must_agree(tmp_path,monkeypatch,where):
    name='control.py';(tmp_path/name).write_bytes(b'bad' if where=='local' else b'good')
    monkeypatch.setattr(a,'ROOT',tmp_path);monkeypatch.setattr(a,'SOURCE_FILES',(name,))
    def show(command,**kwargs):
        reference=command[-1]
        return b'bad' if (where=='basis' and reference.startswith(a.BASIS)) or (where=='execution' and reference.startswith(a.EXECUTION)) else b'good'
    monkeypatch.setattr(a.subprocess,'check_output',show)
    with pytest.raises(ValueError):a.source_inventory()


@pytest.mark.parametrize('status',[' M tools/audit_s1r_lower_outcome.py\n','?? untracked.py\n','M  src/neural_shrimp/controls.py\n'])
def test_clean_checkout_guard_rejects_before_any_output(monkeypatch,status):
    monkeypatch.setattr(a.subprocess,'check_output',lambda *args,**kwargs:status)
    with pytest.raises(ValueError,match='clean audit checkout'):a.clean_checkout()


def test_clean_checkout_guard_and_indirect_import_binding(monkeypatch):
    monkeypatch.setattr(a.subprocess,'check_output',lambda *args,**kwargs:'')
    a.clean_checkout()
    assert 'src/neural_shrimp/controls.py' in a.AUDIT_FILES
    assert 'src/neural_shrimp/controls.py' not in a.SOURCE_FILES
    assert len(a.SOURCE_FILES)==12
