"""Synthetic EXP018 controller contracts; no historical arrays or target solves."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
HEAD='a'*40

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True)+'\n')

@pytest.fixture
def runner(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('synthetic_periodic_reference_refinement_runner',TOOLS/'run_periodic_reference_refinement.py')
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    r.ROOT=tmp_path/'checkout';r.ROOT.mkdir()
    r.PREVIOUS=tmp_path/'previous';r.PREVIOUS.mkdir()
    old=tmp_path/'old';old.mkdir();monkeypatch.setattr(r.prior,'OLD',old)
    def binding(path,root=None):
        p=Path(path);b=p.read_bytes()
        return dict(path=str(p.relative_to(r.ROOT if root is None else root)),bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
    r.binding=binding;r.SOURCE_PATHS=['synthetic-source.py']
    (r.ROOT/'synthetic-source.py').write_text('# metadata fixture\n')
    olditems=[]
    for name in ['receipt.json','references/payloads/core/packet.json','references/payloads/core/packet.npz','references/payloads/failed-near-edge/packet.json','references/payloads/failed-near-edge/packet.npz']:
        p=old/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'opaque nonnumerical fixture')
        olditems.append(binding(p,old))
    monkeypatch.setattr(r.prior,'old_inputs',lambda:copy.deepcopy(olditems))
    write(r.PREVIOUS/'receipt.json',dict(source_commit=r.PREVIOUS_SOURCE,artifacts=[]))
    r.PREVIOUS_SHA=r.legacy.sha(r.PREVIOUS/'receipt.json')
    write(r.ROOT/r.PREVIOUS_SUMMARY,dict(source_commit=r.PREVIOUS_SOURCE,raw_receipt=dict(sha256=r.PREVIOUS_SHA)))
    def forbidden(*a,**k):pytest.fail('Target acquisition forbidden in controller tests')
    for name in ('refine_cycle','crossverify_reference','acquire_refined_event'):monkeypatch.setattr(r.acquisition,name,forbidden)
    return r

def source_fixture(r,monkeypatch):
    path=r.ROOT/r.PLAN_REL;write(path,{'synthetic':True})
    write(r.ROOT/r.REVIEW_REL,dict(status='ready_for_exploratory_execution',plan_sha256=r.legacy.sha(path)))
    state=dict(dirty='',head=HEAD,branch='codex/synthetic',remote=HEAD,origin='git@github-personal:tdj28/neural-shrimp.git')
    committed={name:(r.ROOT/name).read_bytes() for name in [*r.SOURCE_PATHS,r.PLAN_REL,r.REVIEW_REL]}
    def git(*args):
        key={'status':'dirty','rev-parse':'head','branch':'branch','remote':'origin','ls-remote':'remote'}[args[0]]
        return state[key]
    monkeypatch.setattr(r.legacy,'git',git)
    monkeypatch.setattr(r.subprocess,'check_output',lambda command:committed[command[-1].split(':',1)[1]])
    return path,state,committed

def test_complete_census_and_nested_slot_limit(runner):
    r=runner;phases=r.prior.phase_rows();ledger=r.stages(phases);nested=r.nested_ledger(phases)
    assert {k:len(v) for k,v in ledger.items()}==dict(shooting=4,references=2,events=96)
    assert len({t['id'] for rows in ledger.values() for t in rows})==102
    assert len(nested)==len({r['id'] for r in nested})==208
    assert dict(r.Counter(x['kind'] for x in nested))==dict(shooting_trial=100,cross_forward=8,adjoint=4,event=96)
    assert {t['method']['id'] for t in ledger['shooting']}==set(r.METHODS)
    for setting in r.SETTINGS:
        for index,phase in enumerate(phases):
            rows=[t for t in ledger['events'] if t['case']['setting']==setting and t['case']['phase_index']==index]
            assert len(rows)==2 and {t['method']['id'] for t in rows}==set(r.METHODS)
            assert all(t['case']['phase_fraction']==phase['phase_fraction'] and t['case']['split']==phase['split'] for t in rows)
    assert len([t for t in ledger['events'] if t['case']['split']=='development'])==32
    assert len([t for t in ledger['events'] if t['case']['split']=='heldout'])==64


def test_metadata_plan_never_decodes_old_arrays_or_integrates(runner,monkeypatch):
    def forbidden(*a,**k):pytest.fail('Metadata preparation decoded a packet')
    monkeypatch.setattr(runner.codec,'load_packet',forbidden)
    plan=runner.build_plan()
    assert len(plan['historical_inputs'])==7
    assert runner.validate(plan)==dict(total_jobs=102,contexts=48,nested_slots=208,historical_inputs=7)
    assert not any('EXP-016' in x['root'] for x in plan['historical_inputs'])
    changed=copy.deepcopy(plan);changed['phases'][0]['phase_fraction']+=.0001
    with pytest.raises(ValueError):runner.validate(changed)


def test_metadata_rejects_changed_provenance(runner):
    write(runner.ROOT/runner.PREVIOUS_SUMMARY,dict(source_commit='wrong',raw_receipt=dict(sha256=runner.PREVIOUS_SHA)))
    with pytest.raises(ValueError):runner.historical_inputs()


def test_pending_protocol_is_preparable_but_not_executable(runner):
    runner.SOURCE_PATHS.append(runner.PROTOCOL_REL)
    plan=runner.build_plan();assert plan['source_files'][-1]['status']=='pending_protocol'
    with pytest.raises(ValueError,match='Protocol'):runner.validate(plan)


def test_source_guard_accepts_exact_private_review(runner,monkeypatch):
    path,_,_=source_fixture(runner,monkeypatch)
    assert runner.guard({},path,HEAD)['source_commit']==HEAD

@pytest.mark.parametrize('field,value',[('dirty','?? unknown.py'),('head','b'*40),('branch','main'),('remote','c'*40),('origin','https://example.org/wrong.git')])
def test_source_guard_rejects_inexact_source_or_origin(runner,monkeypatch,field,value):
    path,state,_=source_fixture(runner,monkeypatch);state[field]=value
    with pytest.raises(ValueError):runner.guard({},path,HEAD)

@pytest.mark.parametrize('key',['status','plan_sha256','source_bytes'])
def test_source_guard_rejects_wrong_review_or_source_bytes(runner,monkeypatch,key):
    path,_,committed=source_fixture(runner,monkeypatch)
    if key=='source_bytes':committed[runner.SOURCE_PATHS[0]]=b'changed'
    else:
        value=runner.read(runner.ROOT/runner.REVIEW_REL);value[key]='wrong'
        write(runner.ROOT/runner.REVIEW_REL,value);committed[runner.REVIEW_REL]=(runner.ROOT/runner.REVIEW_REL).read_bytes()
    with pytest.raises(ValueError):runner.guard({},path,HEAD)

def test_attempt_exclusive_and_protected_paths(runner,tmp_path):
    directory=runner.external_attempt(tmp_path/'raw','attempt-001');(directory/'keep').write_text('evidence')
    with pytest.raises(FileExistsError):runner.external_attempt(tmp_path/'raw','attempt-001')
    assert (directory/'keep').read_text()=='evidence'
    for protected in (runner.ROOT,runner.prior.OLD,runner.PREVIOUS):
        with pytest.raises(ValueError):runner.external_attempt(protected/'nested','attempt-002')
    with pytest.raises(ValueError):runner.external_attempt(tmp_path/'raw','../escape')

def test_attempt_rejects_experiment_symlink(runner,tmp_path):
    raw=tmp_path/'raw';raw.mkdir();(raw/'EXP-018').symlink_to(runner.ROOT,target_is_directory=True)
    with pytest.raises(ValueError):runner.external_attempt(raw,'attempt-001')

def test_prepare_is_exclusive(runner,monkeypatch,tmp_path):
    path=tmp_path/'plan.json';path.write_text('keep original')
    monkeypatch.setattr(runner,'build_plan',lambda:{'new':True})
    monkeypatch.setattr(sys,'argv',['runner','prepare','--output',str(path)])
    with pytest.raises(FileExistsError):runner.main()
    assert path.read_text()=='keep original'


def setup_run(r,monkeypatch,tmp_path):
    plan=dict(phases=r.prior.phase_rows());path=tmp_path/'plan.json';write(path,plan)
    monkeypatch.setattr(r,'validate',lambda p:{})
    monkeypatch.setattr(r,'guard',lambda *a:dict(source_commit=HEAD,pushed_branch='codex/synthetic'))
    monkeypatch.setattr(r.importlib.metadata,'version',lambda k:r.CONTRACT['required_packages'][k])
    for key in r.legacy.THREAD_ENV:monkeypatch.setenv(key,'1')
    monkeypatch.setattr(r.legacy,'git',lambda *a:HEAD if a[0]=='rev-parse' else '')
    monkeypatch.setattr(r.prior,'resource_check',lambda *a,**k:{})
    monkeypatch.setattr(r.signal,'setitimer',lambda *a:None)
    monkeypatch.setattr(r,'inputs',lambda:dict(references={s:{'setting':s} for s in r.SETTINGS}))
    monkeypatch.setattr(r,'check_packet_bindings',lambda *a:None)
    return path


def fake_packet(r,t,*,bad_shoot=None,bad_reference=None):
    p=dict(id=t['id'],kind=t['kind'],method=t['method']['id'],status='complete',complete=True,qualified=True)
    if t['kind']=='shoot':
        p.update(converged=t['id']!=bad_shoot,trials=[dict(id=t['id']+f'--trial-{i:02d}',index=i,status='complete' if i==0 else 'not_needed',attempted=i==0,ivp_started=i==0,ivp_completed=i==0) for i in range(25)])
    elif t['kind']=='reference':
        s=t['case']['setting']
        p.update(qualified=t['id']!=bad_reference,bases={m:dict(id=s+'--'+m,qualified=True) for m in r.METHODS},
            verifications={f'{c}--{m}':dict(id=f'{s}--verify--{c}--{m}',status='complete',ivp_started=True,ivp_completed=True) for c in r.METHODS for m in r.METHODS},
            adjoints={m:dict(id=f'{s}--adjoint--{m}',status='complete',ivp_started=True,ivp_completed=True) for m in r.METHODS})
    else:p.update(ivp_started=True,ivp_completed=True)
    return p


def synthetic_execution(r,monkeypatch,*,bad_shoot=None,bad_reference=None,analysis_complete=True,qualified=False):
    seen=[]
    def execute(tasks,contexts,directory,libraries,**kwargs):
        directory=Path(directory);directory.mkdir();(directory/'records').mkdir();seen.append((directory.name,len(tasks)))
        events=[json.loads(x) for x in (directory.parent/'events.jsonl').read_text().splitlines()]
        if directory.name=='references':
            assert (directory.parent/'shooting-seal-index.json').is_file()
            assert events[-2]['event']=='shooting_sealed'
            assert all(c['candidates'][m]['converged'] for c in contexts.values() for m in r.METHODS)
        if directory.name=='events':
            assert (directory.parent/'reference-seal-index.json').is_file()
            assert events[-2]['event']=='references_sealed'
        assert kwargs['workers']==4 and kwargs['per_task']==180
        rows=[]
        for t in tasks:
            packet=fake_packet(r,t,bad_shoot=bad_shoot,bad_reference=bad_reference)
            row=dict(id=t['id'],kind=t['kind'],case_id=t['case']['id'],method=t['method']['id'],status='complete',qualified=packet['qualified'])
            seal=r.codec.save_packet(directory/'payloads'/t['id'],'packet',packet)
            row.update(packet_path=str(Path(seal['path']).relative_to(directory)),packet_sha256=seal['sha256'],packet_semantic_sha256=seal['semantic_sha256'])
            rows.append(row)
        return rows
    monkeypatch.setattr(r.runtime,'execute',execute)
    monkeypatch.setattr(r,'analyze',lambda *a:dict(complete=analysis_complete,qualified=qualified))
    return seen


def test_102_complete_workers_can_be_scientifically_negative(runner,monkeypatch,tmp_path):
    r=runner;path=setup_run(r,monkeypatch,tmp_path);seen=synthetic_execution(r,monkeypatch)
    result=r.run(path,tmp_path/'raw','attempt-001',HEAD);directory=Path(result['directory'])
    assert seen==[('shooting',4),('references',2),('events',96)]
    assert result['complete'] is True
    summary=r.read(directory/'summary.json');assert summary['qualified'] is False
    nested=r.read(directory/'nested-terminal-records.json')
    assert len(nested)==208 and sum(x['ivp_started'] is True for x in nested)==112
    assert sum(x['status']=='not_needed' for x in nested)==96


@pytest.mark.parametrize('kind', ['shoot','reference'])
def test_failed_scientific_prerequisite_blocks_whole_setting_not_other(runner,monkeypatch,tmp_path,kind):
    r=runner;path=setup_run(r,monkeypatch,tmp_path)
    kwargs={'bad_shoot':'core--tight-rk45'} if kind=='shoot' else {'bad_reference':'core--reference-pair'}
    seen=synthetic_execution(r,monkeypatch,**kwargs)
    result=r.run(path,tmp_path/'raw','attempt-001',HEAD);directory=Path(result['directory'])
    assert seen==[('shooting',4),('references',1 if kind=='shoot' else 2),('events',48)]
    assert result['complete'] is False and result['error'] is None
    rows=r.read(directory/'all-terminal-records.json')
    blocked=[x for x in rows['events'] if x['status']=='blocked_reference_prerequisite']
    assert len(blocked)==48 and all(x['id'].startswith('core--') for x in blocked)
    assert len(r.read(directory/'nested-terminal-records.json'))==208


def test_completed_raw_jobs_cannot_mask_failed_analysis(runner,monkeypatch,tmp_path):
    r=runner;path=setup_run(r,monkeypatch,tmp_path);synthetic_execution(r,monkeypatch,analysis_complete=False)
    assert r.run(path,tmp_path/'raw','attempt-001',HEAD)['complete'] is False


def test_before_input_failure_retains_full_denominators(runner,monkeypatch,tmp_path):
    r=runner;path=setup_run(r,monkeypatch,tmp_path)
    def fail(*a,**k):raise RuntimeError('synthetic RAM unavailable')
    monkeypatch.setattr(r.prior,'resource_check',fail)
    out=r.run(path,tmp_path/'raw','attempt-001',HEAD);directory=Path(out['directory'])
    assert out['complete'] is False and 'RAM unavailable' in out['error']
    assert sum(len(x) for x in r.read(directory/'all-terminal-records.json').values())==102
    assert len(r.read(directory/'nested-terminal-records.json'))==208
    assert not (directory/'input-seal.json').exists()
    assert len(r.codec.load_packet(directory/'analysis.json')['contexts'])==48


def test_worker_retains_failed_shooting_payload(runner,monkeypatch,tmp_path):
    r=runner;t=r.stages(r.prior.phase_rows())['shooting'][0]
    payload=fake_packet(r,t);payload.update(status='failed',complete=False,qualified=False)
    class Failure(Exception):pass
    def fail(*a,**k):
        e=Failure('synthetic partial solve');e.payload=payload;raise e
    monkeypatch.setattr(r.acquisition,'refine_cycle',fail);monkeypatch.setattr(r.signal,'signal',lambda *a:None)
    directory=tmp_path/'worker';(directory/'records').mkdir(parents=True);(directory/'payloads').mkdir()
    r.worker(t,dict(reference={}),directory,{},0)
    row=r.read(directory/'records'/f'{t["id"]}.json')
    assert row['status']=='failed' and len(r.packet_record(directory,t,row)['trials'])==25


def test_parent_timeout_keeps_saved_child_prefix_but_cannot_promote(runner,tmp_path):
    r=runner;t=r.stages(r.prior.phase_rows())['shooting'][0];directory=tmp_path/'shooting';packet=fake_packet(r,t)
    seal=r.codec.save_packet(directory/'payloads'/t['id'],'packet',packet)
    child=dict(id=t['id'],kind=t['kind'],case_id=t['case']['id'],method=t['method']['id'],status='complete',pid=123,
        packet_path=str(Path(seal['path']).relative_to(directory)),packet_sha256=seal['sha256'],packet_semantic_sha256=seal['semantic_sha256'])
    write(directory/'records'/f'{t["id"]}.json',child)
    parent={k:v for k,v in child.items() if not k.startswith('packet_')};parent['status']='task_timeout'
    assert r.packet_record(directory,t,parent)['complete'] is True
    assert r.eligible_references(r.stages(r.prior.phase_rows())['references'],
        [parent,*[dict(id=x['id'],status='complete') for x in r.stages(r.prior.phase_rows())['shooting'][1:]]],{t['id']:packet})==[]
    parent['pid']=124
    with pytest.raises(ValueError,match='identity'):r.packet_record(directory,t,parent)


def partial_fixture(r,monkeypatch,tmp_path,*,sealed=False,dispatch=False):
    directory=tmp_path/'partial';directory.mkdir();phases=r.prior.phase_rows();ledger=r.stages(phases)
    plan=dict(phases=phases,source_files=[r.binding(r.ROOT/p) for p in r.SOURCE_PATHS]);encoded=json.dumps(plan,sort_keys=True).encode();plan_sha=hashlib.sha256(encoded).hexdigest()
    review=r.ROOT/r.REVIEW_REL;write(review,{'synthetic':True})
    committed={r.PLAN_REL:encoded,r.REVIEW_REL:review.read_bytes(),**{name:(r.ROOT/name).read_bytes() for name in r.SOURCE_PATHS}}
    monkeypatch.setattr(r.subprocess,'check_output',lambda cmd:committed[cmd[-1].split(':',1)[1]])
    monkeypatch.setattr(r,'validate',lambda p:{});monkeypatch.setattr(r.work,'identity',lambda pid:None)
    input_value=dict(references={'synthetic':'input'});monkeypatch.setattr(r,'inputs',lambda:copy.deepcopy(input_value))
    manifest=dict(source_commit=HEAD,plan_sha256=plan_sha,plan=plan,review=r.binding(review),producer_pid=99999999,producer_identity='parent')
    write(directory/'manifest.json',manifest);write(directory/'planned-ledger.json',ledger);write(directory/'planned-nested-ledger.json',r.nested_ledger(phases))
    rows={k:r.prior.fill_rows(v,[],'not_started_interruption') for k,v in ledger.items()}
    if sealed:
        seal=r.codec.save_packet(directory,'input-seal',input_value)
        r.prior.event(directory,'inputs_sealed',sha256=seal['sha256'],semantic_sha256=seal['semantic_sha256'])
    if dispatch:
        assert sealed;r.prior.event(directory,'shooting_started',planned=4)
        stage=directory/'shooting';stage.mkdir();row=rows['shooting'][0];row.update(status='interrupted',pid=99999998,identity='child',error='stop')
        owned=dict(task_id=row['id'],pid=row['pid'],identity=row['identity'],started_at=r.legacy.now())
        write(stage/'terminal-records.json',[row]);(stage/'journal.jsonl').write_text(json.dumps(row)+'\n');(stage/'processes.jsonl').write_text(json.dumps(owned)+'\n')
        write(stage/'cleanup.json',dict(all_registered_workers_joined=True,all_original_identities_absent=True,workers=[dict(owned,original_identity_absent=True)]))
    write(directory/'all-terminal-records.json',rows)
    nested=r.nested_records(plan,rows,{k:{} for k in ledger});write(directory/'nested-terminal-records.json',nested)
    error='synthetic interruption';write(directory/'producer-error.json',dict(stage='shooting' if dispatch else 'input_seal',error=error))
    analysis=r.unavailable_analysis(plan,error);r.codec.save_packet(directory,'analysis',analysis)
    write(directory/'postflight.json',dict(source_bindings_unchanged=True,error=None))
    write(directory/'summary.json',dict(complete=False,qualified=False,error=error,postflight_error=None,source_valid=True,analysis=analysis,
        status_counts={k:dict(r.Counter(x['status'] for x in v)) for k,v in rows.items()},nested_status_counts=dict(r.Counter(x['status'] for x in nested))))
    def reseal():
        artifacts=r.prior.inventory(directory);write(directory/'receipt.json',dict(source_commit=HEAD,plan_sha256=plan_sha,experiment_id='EXP-018',status='incomplete_or_invalid',artifacts=artifacts,artifact_bytes=sum(x['bytes'] for x in artifacts)))
    reseal();return directory,rows,reseal


@pytest.mark.parametrize('sealed,dispatch',[(False,False),(True,False),(True,True)])
def test_partial_verifier_valid_prefix(runner,monkeypatch,tmp_path,sealed,dispatch):
    directory,_,_=partial_fixture(runner,monkeypatch,tmp_path,sealed=sealed,dispatch=dispatch)
    result=runner.verify(directory);assert not result['complete'] and not result['saved_array_replay']
    assert result['nested_slots']==208 and result['all_original_owned_identities_absent_now'] is True


@pytest.mark.parametrize('tamper',['nested','terminal','cleanup','duplicate_journal','postflight','premature_start'])
def test_rehashed_partial_tampering_rejected(runner,monkeypatch,tmp_path,tamper):
    r=runner;directory,rows,reseal=partial_fixture(r,monkeypatch,tmp_path,sealed=True,dispatch=True)
    if tamper=='nested':
        p=directory/'nested-terminal-records.json';x=r.read(p);x[0]['id']='wrong';write(p,x)
    elif tamper=='terminal':rows['shooting'][0]['method']='wrong';write(directory/'all-terminal-records.json',rows)
    elif tamper=='cleanup':
        p=directory/'shooting/cleanup.json';x=r.read(p);x['all_original_identities_absent']=False;write(p,x)
    elif tamper=='duplicate_journal':
        p=directory/'shooting/terminal-records.json';x=r.read(p)*2;write(p,x);(directory/'shooting/journal.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in x))
    elif tamper=='postflight':write(directory/'postflight.json',dict(source_bindings_unchanged=False,error=None))
    else:
        p=directory/'shooting/processes.jsonl';x=json.loads(p.read_text());x['started_at']='1900';p.write_text(json.dumps(x)+'\n')
        q=directory/'shooting/cleanup.json';y=r.read(q);y['workers'][0]['started_at']='1900';write(q,y)
    reseal()
    with pytest.raises((ValueError,KeyError)):r.verify(directory)


@pytest.mark.parametrize('dispatch',[False,True])
def test_input_seal_written_before_chronology_is_only_valid_without_dispatch(runner,monkeypatch,tmp_path,dispatch):
    r=runner;directory,_,reseal=partial_fixture(r,monkeypatch,tmp_path,sealed=True,dispatch=dispatch)
    p=directory/'events.jsonl';events=[json.loads(x) for x in p.read_text().splitlines()];p.write_text(''.join(json.dumps(x)+'\n' for x in events if x['event']!='inputs_sealed'));reseal()
    if dispatch:
        with pytest.raises(ValueError):r.verify(directory)
    else:assert r.verify(directory)['complete'] is False


def test_owned_scheduler_timeout_preserves_child_partial_packet(runner,monkeypatch,tmp_path):
    """Actual fork/lifecycle, fabricated payload, no dynamics or historical reads."""
    import time
    r=runner;t=r.stages(r.prior.phase_rows())['shooting'][0]
    payload=fake_packet(r,t);payload.update(status='failed',complete=False,qualified=False,converged=False)
    def delayed(*args,**kwargs):
        try:time.sleep(5)
        except TimeoutError as error:
            error.payload=payload;raise
    monkeypatch.setattr(r.acquisition,'refine_cycle',delayed)
    directory=tmp_path/'owned'
    rows=r.runtime.execute([t],{t['case']['id']:dict(reference={})},directory,{},workers=1,
        deadline=time.monotonic()+10,per_task=.1,worker_fn=r.worker)
    assert rows[0]['status']=='task_timeout'
    assert r.packet_record(directory,t,rows[0])['complete'] is False
    cleanup=r.read(directory/'cleanup.json')
    assert cleanup['all_registered_workers_joined'] and cleanup['all_original_identities_absent']
    assert len(cleanup['workers'])==1


def test_parent_baseexception_still_seals_full_attempt(runner,monkeypatch,tmp_path):
    r=runner;path=setup_run(r,monkeypatch,tmp_path)
    def stop(tasks,contexts,directory,*a,**k):
        directory=Path(directory);directory.mkdir()
        t=tasks[0]
        row=dict(id=t['id'],kind=t['kind'],case_id=t['case']['id'],method=t['method']['id'],status='interrupted')
        write(directory/'terminal-records.json',[row])
        raise r.prior.ProducerStop('synthetic parent deadline')
    monkeypatch.setattr(r.runtime,'execute',stop)
    result=r.run(path,tmp_path/'raw','attempt-001',HEAD);directory=Path(result['directory'])
    assert not result['complete'] and 'parent deadline' in result['error']
    assert sum(len(x) for x in r.read(directory/'all-terminal-records.json').values())==102
    assert len(r.read(directory/'nested-terminal-records.json'))==208
    assert r.read(directory/'all-terminal-records.json')['shooting'][0]['status']=='interrupted'


def test_reference_requires_exact_candidate_copy_not_replay_tolerance(runner,monkeypatch):
    r=runner;np=r.prior.np;old=dict(alpha=1.,gamma=2.,period=3.,certificate=dict(section_state=np.zeros(3)))
    common=dict(setting='core',alpha=1.,gamma=2.,original_period=3.,original_state=np.zeros(3))
    shoots={f'core--{m}':dict(common,id=f'core--{m}',method=m,complete=True,converged=True,
        candidate=dict(state=np.ones(3),period=3.)) for m in r.METHODS}
    ref=dict(common,id='core--reference-pair',status='failed',complete=False,
        candidates={m:copy.deepcopy(shoots[f'core--{m}']['candidate']) for m in r.METHODS})
    monkeypatch.setattr(r.evaluation,'validate_shoot_packet',lambda *a,**k:None)
    monkeypatch.setattr(r.evaluation,'validate_partial_packet',lambda *a,**k:dict(partial_replay_pass=True))
    args=(dict(phases=r.prior.phase_rows()),dict(references=dict(core=old)),dict(shooting=shoots,references={ref['id']:ref},events={}))
    r.check_packet_bindings(*args)
    ref['candidates'][r.METHODS[0]]['period']=float(np.nextafter(3.,4.))
    with pytest.raises(ValueError,match='candidate copy'):r.check_packet_bindings(*args)
