"""Synthetic EXP017 controller contracts; no historical arrays or target solves."""
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
    spec=importlib.util.spec_from_file_location('synthetic_reference_diagnostic_runner',TOOLS/'run_phase_reference_diagnostic.py')
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
    items=[]
    for setting in r.SETTINGS:
        for i in range(24):
            for suffix in ('json','npz'):
                p=r.PREVIOUS/f'seals/{setting}--p{i:02d}.{suffix}';p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'opaque prior seal')
                items.append(binding(p,r.PREVIOUS))
    write(r.PREVIOUS/'receipt.json',dict(source_commit='9904368b7b13721292efdc84dcf31efd7add30c7',artifacts=items))
    r.PREVIOUS_SHA=r.legacy.sha(r.PREVIOUS/'receipt.json')
    def forbidden(*a,**k):pytest.fail('Target acquisition forbidden in controller tests')
    monkeypatch.setattr(r.acquisition,'acquire_base',forbidden)
    monkeypatch.setattr(r.acquisition,'acquire_event',forbidden)
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

def test_complete_census_inherits_all_phases_and_both_fine_methods(runner):
    r=runner;phases=r.prior.phase_rows();ledger=r.stages(phases)
    assert {k:len(v) for k,v in ledger.items()}==dict(bases=6,events=96)
    assert len({t['id'] for rows in ledger.values() for t in rows})==102
    assert {t['method']['id'] for t in ledger['bases']}=={'baseline-dop853','fine-dop853','fine-rk45'}
    for setting in r.SETTINGS:
        for index,phase in enumerate(phases):
            rows=[t for t in ledger['events'] if t['case']['setting']==setting and t['case']['phase_index']==index]
            assert len(rows)==2 and {t['method']['id'] for t in rows}==set(r.FINE)
            assert all(t['case']['phase_fraction']==phase['phase_fraction'] and t['case']['split']==phase['split'] for t in rows)
    assert len([t for t in ledger['events'] if t['case']['split']=='development'])==32
    assert len([t for t in ledger['events'] if t['case']['split']=='heldout'])==64

def test_metadata_plan_never_decodes_old_arrays_or_integrates(runner,monkeypatch):
    def forbidden(*a,**k):pytest.fail('Metadata preparation decoded a packet')
    monkeypatch.setattr(runner.codec,'load_packet',forbidden)
    plan=runner.build_plan()
    assert len(plan['historical_inputs'])==102
    assert runner.validate(plan)==dict(total_jobs=102,contexts=48)
    changed=copy.deepcopy(plan);changed['phases'][0]['phase_fraction']+=.0001
    with pytest.raises(ValueError):runner.validate(changed)

def test_metadata_rejects_changed_historical_seal(runner):
    (runner.PREVIOUS/'seals/core--p00.npz').write_bytes(b'changed opaque bytes')
    with pytest.raises(ValueError):runner.historical_inputs()

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
    raw=tmp_path/'raw';raw.mkdir();(raw/'EXP-017').symlink_to(runner.ROOT,target_is_directory=True)
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
    synthetic=dict(references={s:{'setting':s} for s in r.SETTINGS},oldseals={})
    monkeypatch.setattr(r,'inputs',lambda:copy.deepcopy(synthetic))
    return path

def test_pre_input_failure_preserves_full_102_denominator(runner,monkeypatch,tmp_path):
    r=runner;path=setup_run(r,monkeypatch,tmp_path)
    def fail(*a,**k):raise RuntimeError('synthetic resource failure')
    monkeypatch.setattr(r.prior,'resource_check',fail)
    result=r.run(path,tmp_path/'raw','attempt-001',HEAD)
    directory=Path(result['directory']);rows=r.read(directory/'all-terminal-records.json')
    assert {k:len(v) for k,v in rows.items()}==dict(bases=6,events=96)
    assert result['complete'] is False and 'synthetic resource failure' in result['error']
    assert r.read(directory/'receipt.json')['status']=='incomplete_or_invalid'
    assert not (directory/'input-seal.json').exists()

def synthetic_execution(r,monkeypatch,*,analysis_complete=True,failed_base=None):
    seen=[]
    def execute(tasks,contexts,directory,libraries,**kwargs):
        directory=Path(directory);directory.mkdir();(directory/'records').mkdir()
        if directory.name=='events':
            assert (directory.parent/'base-seal-index.json').is_file()
            ev=[json.loads(x) for x in (directory.parent/'events.jsonl').read_text().splitlines()]
            assert [x['event'] for x in ev].index('bases_sealed')<[x['event'] for x in ev].index('events_started')
            assert all(contexts[t['case']['id']]['base']['qualified'] is False for t in tasks)
        assert kwargs['workers']==4 and kwargs['per_task']==180
        seen.append((directory.name,len(tasks)));rows=[]
        for t in tasks:
            packet=dict(id=t['id'],kind=t['kind'],method=t['method']['id'],complete=True,status='complete',qualified=False)
            row=dict(id=t['id'],kind=t['kind'],case_id=t['case']['id'],method=t['method']['id'],status='complete',qualified=False)
            if t['id']==failed_base:row.update(status='failed',error='synthetic base failure')
            else:
                dest=directory/'payloads'/t['id'];seal=r.codec.save_packet(dest,'packet',packet)
                row.update(packet_path=str(Path(seal['path']).relative_to(directory)),packet_sha256=seal['sha256'],packet_semantic_sha256=seal['semantic_sha256'])
            rows.append(row)
        return rows
    monkeypatch.setattr(r.runtime,'execute',execute)
    monkeypatch.setattr(r,'analyze',lambda *a:dict(diagnostic_complete=analysis_complete,all_numerical_gates_pass=False,original_study_requalified=False))
    return seen

def test_complete_bases_are_sealed_before_events_without_numerical_withholding(runner,monkeypatch,tmp_path):
    r=runner;path=setup_run(r,monkeypatch,tmp_path);seen=synthetic_execution(r,monkeypatch)
    result=r.run(path,tmp_path/'raw','attempt-001',HEAD)
    assert seen==[('bases',6),('events',96)]
    assert result['complete'] is True
    assert r.read(Path(result['directory'])/'summary.json')['analysis']['all_numerical_gates_pass'] is False

def test_failed_base_blocks_only_its_dependent_phase_jobs(runner,monkeypatch,tmp_path):
    r=runner;path=setup_run(r,monkeypatch,tmp_path);seen=synthetic_execution(r,monkeypatch,failed_base='core--fine-dop853')
    result=r.run(path,tmp_path/'raw','attempt-001',HEAD)
    assert seen==[('bases',6),('events',72)]
    rows=r.read(Path(result['directory'])/'all-terminal-records.json')
    assert len(rows['events'])==96
    blocked=[x for x in rows['events'] if x['status']=='not_started_base_unavailable']
    assert len(blocked)==24 and all(x['id'].startswith('core--') and x['method']=='fine-dop853' for x in blocked)
    assert result['complete'] is False

def test_complete_jobs_cannot_mask_incomplete_analysis(runner,monkeypatch,tmp_path):
    r=runner;path=setup_run(r,monkeypatch,tmp_path);synthetic_execution(r,monkeypatch,analysis_complete=False)
    result=r.run(path,tmp_path/'raw','attempt-001',HEAD)
    assert result['complete'] is False
    assert r.read(Path(result['directory'])/'receipt.json')['status']=='incomplete_or_invalid'

def test_worker_preserves_acquisition_failure_payload(runner,monkeypatch,tmp_path):
    r=runner;task=r.stages(r.prior.phase_rows())['bases'][0]
    class Failed(Exception):
        payload=dict(status='failed',complete=False,partial_values=[1.,2.],failed_stage='backward')
    def fail(*a,**k):raise Failed('synthetic incomplete solver')
    monkeypatch.setattr(r.acquisition,'acquire_base',fail)
    monkeypatch.setattr(r.signal,'signal',lambda *a:None)
    directory=tmp_path/'worker';(directory/'records').mkdir(parents=True);(directory/'payloads').mkdir()
    r.worker(task,dict(reference={},phases=[]),directory,{},0)
    row=r.read(directory/'records'/f'{task["id"]}.json')
    assert row['status']=='failed' and 'synthetic incomplete solver' in row['error']
    saved=r.codec.load_packet(directory/row['packet_path'])
    assert saved==Failed.payload

def partial_fixture(r,monkeypatch,tmp_path,*,sealed=False,dispatch=False):
    directory=tmp_path/'partial';directory.mkdir()
    phases=r.prior.phase_rows();ledger=r.stages(phases)
    plan=dict(phases=phases,source_files=[r.binding(r.ROOT/p) for p in r.SOURCE_PATHS])
    encoded=json.dumps(plan,sort_keys=True).encode();plan_sha=hashlib.sha256(encoded).hexdigest()
    review=r.ROOT/r.REVIEW_REL;write(review,{'synthetic':True})
    committed={r.PLAN_REL:encoded,r.REVIEW_REL:review.read_bytes(),**{name:(r.ROOT/name).read_bytes() for name in r.SOURCE_PATHS}}
    monkeypatch.setattr(r.subprocess,'check_output',lambda cmd:committed[cmd[-1].split(':',1)[1]])
    monkeypatch.setattr(r,'validate',lambda p:{})
    monkeypatch.setattr(r.work,'identity',lambda pid:None)
    inputs=dict(references={'synthetic':'input'},oldseals={})
    monkeypatch.setattr(r,'inputs',lambda:copy.deepcopy(inputs))
    manifest=dict(source_commit=HEAD,plan_sha256=plan_sha,plan=plan,review=r.binding(review),producer_pid=99999999,producer_identity='synthetic-parent')
    write(directory/'manifest.json',manifest);write(directory/'planned-ledger.json',ledger)
    rows={k:r.prior.fill_rows(v,[],'not_started_interruption') for k,v in ledger.items()}
    if sealed:
        seal=r.codec.save_packet(directory,'input-seal',inputs)
        r.prior.event(directory,'inputs_sealed',sha256=seal['sha256'],semantic_sha256=seal['semantic_sha256'])
    if dispatch:
        assert sealed
        r.prior.event(directory,'bases_started',planned=6)
        stage=directory/'bases';stage.mkdir()
        row=rows['bases'][0];row.update(status='interrupted',pid=99999998,identity='synthetic-child',error='synthetic stop')
        owned=dict(task_id=row['id'],pid=row['pid'],identity=row['identity'],started_at=r.legacy.now())
        write(stage/'terminal-records.json',[row]);(stage/'journal.jsonl').write_text(json.dumps(row)+'\n')
        (stage/'processes.jsonl').write_text(json.dumps(owned)+'\n')
        write(stage/'cleanup.json',dict(all_registered_workers_joined=True,all_original_identities_absent=True,workers=[dict(owned,original_identity_absent=True)]))
    write(directory/'all-terminal-records.json',rows)
    error='synthetic interruption';write(directory/'producer-error.json',dict(stage='bases' if dispatch else 'input_seal',error=error))
    analysis=dict(status='unavailable',error=error);r.codec.save_packet(directory,'analysis',analysis)
    write(directory/'postflight.json',dict(source_bindings_unchanged=True,error=None))
    write(directory/'summary.json',dict(complete=False,error=error,postflight_error=None,source_valid=True,analysis=analysis,
        status_counts={k:dict(r.Counter(x['status'] for x in v)) for k,v in rows.items()}))
    def reseal():
        artifacts=r.prior.inventory(directory)
        write(directory/'receipt.json',dict(source_commit=HEAD,plan_sha256=plan_sha,experiment_id='EXP-017',status='incomplete_or_invalid',artifacts=artifacts,artifact_bytes=sum(x['bytes'] for x in artifacts)))
    reseal();return directory,rows,reseal

@pytest.mark.parametrize('sealed,dispatch',[(False,False),(True,False),(True,True)])
def test_partial_verifier_preserves_valid_prefixes(runner,monkeypatch,tmp_path,sealed,dispatch):
    directory,_,_=partial_fixture(runner,monkeypatch,tmp_path,sealed=sealed,dispatch=dispatch)
    result=runner.verify(directory)
    assert result['complete'] is False and result['saved_array_replay'] is False
    assert result['all_original_owned_identities_absent_now'] is True

def test_partial_verifier_checks_decoded_input_even_without_analysis(runner,monkeypatch,tmp_path):
    directory,_,_=partial_fixture(runner,monkeypatch,tmp_path,sealed=True)
    monkeypatch.setattr(runner,'inputs',lambda:dict(references={'synthetic':'changed'},oldseals={}))
    with pytest.raises(ValueError):runner.verify(directory)

@pytest.mark.parametrize('dispatch',[False,True])
def test_input_seal_before_event_interruption_is_valid_only_without_dispatch(runner,monkeypatch,tmp_path,dispatch):
    r=runner;directory,_,reseal=partial_fixture(r,monkeypatch,tmp_path,sealed=True,dispatch=dispatch)
    p=directory/'events.jsonl';events=[json.loads(x) for x in p.read_text().splitlines()]
    p.write_text(''.join(json.dumps(x)+'\n' for x in events if x['event']!='inputs_sealed'))
    reseal()
    if dispatch:
        with pytest.raises(ValueError):r.verify(directory)
    else:assert r.verify(directory)['complete'] is False

@pytest.mark.parametrize('tamper',['duplicate_journal','cleanup','task_identity','postflight','pre_seal_start'])
def test_partial_failure_does_not_skip_ledger_cleanup_or_chronology(runner,monkeypatch,tmp_path,tamper):
    r=runner;directory,rows,reseal=partial_fixture(r,monkeypatch,tmp_path,sealed=True,dispatch=True)
    if tamper=='duplicate_journal':
        rr=[rows['bases'][0]]*2;write(directory/'bases/terminal-records.json',rr)
        (directory/'bases/journal.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in rr))
    elif tamper=='cleanup':
        v=r.read(directory/'bases/cleanup.json');v['all_original_identities_absent']=False;write(directory/'bases/cleanup.json',v)
    elif tamper=='task_identity':
        rows['events'][0]['case_id']='wrong-context';write(directory/'all-terminal-records.json',rows)
    elif tamper=='postflight':write(directory/'postflight.json',dict(source_bindings_unchanged=False,error=None))
    else:
        p=directory/'bases/processes.jsonl';v=json.loads(p.read_text());v['started_at']='1900-01-01T00:00:00+00:00';p.write_text(json.dumps(v)+'\n')
        cleanup=r.read(directory/'bases/cleanup.json');cleanup['workers'][0]['started_at']=v['started_at'];write(directory/'bases/cleanup.json',cleanup)
    reseal()
    with pytest.raises(ValueError):r.verify(directory)
