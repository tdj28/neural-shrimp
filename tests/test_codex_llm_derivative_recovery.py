"""Synthetic whole-receipt recovery controls: no model, network or Git operation."""
import copy
import json
from pathlib import Path
import pytest
from test_codex_llm_derivative_campaign import C, H, SOURCE, save, frozen_fixture


def seal(attempt, **fields):
    artifacts=[dict(path=str(p.relative_to(attempt)),bytes=p.stat().st_size,sha256=C.sha(p))
               for p in sorted(attempt.rglob('*')) if p.is_file() and p.name!='receipt.json']
    return save(attempt/'receipt.json',dict(artifacts=artifacts,**fields))


def identity(n):
    return dict(pid=9000000+n,start=100+n,uid=1000,sid=9000000+n,pgid=9000000+n,
                ppid=9000000,state='R',nice=10,rss=1,command=['synthetic-python'])


@pytest.fixture
def recovered(frozen_fixture):
    repo,owned,base=frozen_fixture
    first=base['first']; first_attempt=owned/first['attempt']
    for rel in C.FILES:
        dst=first_attempt/'source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((repo/rel).read_bytes())
    planrel='codex/certify-llm/'+first['plan']['name']
    (first_attempt/'source'/planrel).write_bytes((repo/planrel).read_bytes())
    save(first_attempt/'manifest.json',dict(source_commit=SOURCE))
    save(first_attempt/'run/inputs.json',dict(source_commit=SOURCE,plan_sha256=first['plan']['sha256'],plan=C.shard_plan(first['plan'],repo)))
    dh=save(first_attempt/'run/derivative-shard.json',{'synthetic':'first no target arrays'})
    save(first_attempt/'run/status.json',dict(status='final_derivative_shard_completed',certificate_passed=False,
        phase_index=0,direction_indices=list(range(64)),columns_completed=64,full_state_dimension=896,
        full_vocabulary=151936,precision=1024,shard_sha256=dh,max_rss_kib=100))
    first['receipt_sha256']=seal(first_attempt,source_commit=SOURCE,exit_code=0,timed_out=False,elapsed_seconds=1.0)
    first['audit']['sha256']=save(Path(first['audit']['path']),dict(schema=C.AUDIT_SCHEMA,passed=True,certificate_passed=False,
        attempt=first['attempt'],receipt_sha256=first['receipt_sha256'],plan_sha256=first['plan']['sha256'],phase_index=0,start_column=0,
        **{k:base['common'][k] for k in ('definition_sha256','map_sha256','backend_sha256','evaluator_sha256')},
        measured_elapsed_seconds=1.0,measured_max_rss_kib=100,reviewer='independent synthetic',review_source_sha256=H))
    old=copy.deepcopy(base); c=old['common']
    prior=owned/'prior-campaign'; rows=[]
    for i,row in enumerate(old['assignments'][:21]):
        attempt=owned/row['attempt']; source=attempt/'source'
        for rel in C.FILES:
            dst=source/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((repo/rel).read_bytes())
        planrel='codex/certify-llm/'+row['plan']['name']
        (source/planrel).write_bytes((repo/planrel).read_bytes())
        manifest=dict(source_commit=SOURCE,owned_pid=identity(2*i+2)['pid'],command=C.expected_command(attempt,row))
        save(attempt/'manifest.json',manifest)
        save(attempt/'run/inputs.json',dict(source_commit=SOURCE,plan_sha256=row['plan']['sha256'],plan=C.shard_plan(row['plan'],repo)))
        if i<17:
            dh=save(attempt/'run/derivative-shard.json',{'synthetic':'no derivative arrays'})
            save(attempt/'run/status.json',dict(status='final_derivative_shard_completed',certificate_passed=False,
                phase_index=row['phase_index'],direction_indices=list(range(row['start_column'],row['start_column']+64)),
                columns_completed=64,full_state_dimension=896,full_vocabulary=151936,precision=1024,
                shard_sha256=dh,max_rss_kib=100))
        seal(attempt,source_commit=SOURCE,exit_code=0 if i<17 else -15,timed_out=False,elapsed_seconds=2.0)
        result=C.completed(attempt,row,c,SOURCE) if i<17 else None
        rows.append(dict(row=row,state='completed' if i<17 else 'failed',registered=True,
            worker_termination_verified=True,launcher_exit=0 if i<17 else 1,receipt=result,
            launcher_identity=identity(2*i+1),worker_identity=identity(2*i+2)))
    original_hash=save(prior/'original-plan.json',old)
    status=dict(status='campaign_failed',completed=17,started=21,undispatched=90,uncertain_ownership=[],
                certificate_passed=False,stop_reason='original synthetic status failure')
    save(prior/'admission.json',dict(source_commit=SOURCE,plan_sha256=original_hash,plan=old))
    save(prior/'admission-started.json',dict(source_commit=SOURCE,plan_sha256=original_hash,dispatcher_identity=identity(0)))
    save(prior/'run/state.json',dict(uncertain_ownership=[],stop_reason=status['stop_reason'],rows=rows))
    save(prior/'run/result.json',status);save(prior/'status.json',status)
    ph=seal(prior,status='campaign_failed',certificate_passed=False)
    failures=[]
    for rec in rows[17:]:
        row=rec['row'];attempt=owned/row['attempt']
        failures.append(dict(attempt=row['attempt'],plan=row['plan'],worker_termination_verified=True,launcher_exit=1,
                            **{n:dict(sha256=C.sha(attempt/n),bytes=(attempt/n).stat().st_size) for n in ('receipt.json','manifest.json')}))
    identities=[dict(identity(0),role='controller')]
    for rec in rows:
        identities.extend(dict(rec[k],role=k,attempt=rec['row']['attempt']) for k in ('launcher_identity','worker_identity'))
    close=owned/'closure/closure.json'
    ch=save(close,dict(source_commit=SOURCE,plan_sha256=original_hash,status=status,original_receipt_sha256=ph,
                     controller_artifacts_verified=6,all43_original_pid_and_session_groups_absent=True,
                     tool_reaped_exit=1,identities=identities,failed_shards=failures))
    close_source=owned/'closure/source.py';close_source.write_text('# synthetic independently reviewed closure\n')
    new=copy.deepcopy(old);new['schema']=C.RECOVERY_SCHEMA
    new['resources']['timeout_seconds']=7000
    new['assignments']=copy.deepcopy(old['assignments'][17:])
    for row in new['assignments'][:4]:
        original=repo/'codex/certify-llm'/row['plan']['name']
        row['attempt']+='-recovery';row['plan']['name']='recovery-'+row['plan']['name']
        (repo/'codex/certify-llm'/row['plan']['name']).write_bytes(original.read_bytes())
    new['recovery']=dict(attempt=prior.name,source_commit=SOURCE,receipt_sha256=ph,
                         closure=dict(path=str(close),sha256=ch),closure_source=dict(path=str(close_source),sha256=C.sha(close_source)))
    return repo,owned,new


def test_real_v2_validation_and_whole_receipt_recovery(recovered,monkeypatch):
    repo,owned,plan=recovered
    assert C.validate(plan,repo,owned) is plan
    monkeypatch.setattr(C,'process_table',lambda:[])
    result=C.recovery_authority(plan,repo,owned)
    assert len(result['retained'])==17 and len(result['interrupted'])==4
    assert result['known_completed_total_including_first']==18
    assert result['planned_new_assignments']==94 and result['undispatched_unchanged']==90
    assert result['existing_bytes']>0 and result['certificate_passed'] is False
    assert result['scientific_derivative_audit_pending'] is True


@pytest.mark.parametrize('fault',['retain_rerun','omit','duplicate','rename_unattempted','reuse_failed','alter_replacement','alter_common','resource','new_deadline'])
def test_canonical_recovery_refusals(recovered,fault):
    repo,owned,plan=recovered
    if fault=='retain_rerun':plan['assignments'][0]=copy.deepcopy(C.read(owned/'prior-campaign/original-plan.json')['assignments'][0])
    elif fault=='omit':plan['assignments'].pop()
    elif fault=='duplicate':plan['assignments'][1]=plan['assignments'][0]
    elif fault=='rename_unattempted':plan['assignments'][4]['attempt']+='-other'
    elif fault=='reuse_failed':plan['assignments'][0]['attempt']=C.read(owned/'prior-campaign/original-plan.json')['assignments'][17]['attempt']
    elif fault=='alter_replacement':
        row=plan['assignments'][0];p=repo/'codex/certify-llm'/row['plan']['name'];d=C.read(p);d['purpose']='changed scientific payload';row['plan']['sha256']=save(p,d)
    elif fault=='alter_common':plan['common']['map_sha256']='c'*64
    elif fault=='resource':plan['resources']['cpu_total']+=1
    elif fault=='new_deadline':plan['resources']['timeout_seconds']=8000
    with pytest.raises(ValueError):C.validate(plan,repo,owned)


@pytest.mark.parametrize('fault',['closure_missing_identity','closure_live','prior_controller','retained_partial','failed_changed','old_operational_source'])
def test_original_evidence_cannot_be_substituted(recovered,monkeypatch,fault):
    repo,owned,plan=recovered
    monkeypatch.setattr(C,'process_table',lambda:[])
    if fault=='closure_missing_identity':
        p=Path(plan['recovery']['closure']['path']);d=C.read(p);d['identities'].pop();plan['recovery']['closure']['sha256']=save(p,d)
    elif fault=='closure_live':monkeypatch.setattr(C,'process_table',lambda:[identity(0)])
    elif fault=='prior_controller':(owned/'prior-campaign/run/state.json').write_text('{}')
    elif fault=='retained_partial':(owned/'shard-0-64/run/status.json').write_text('{}')
    elif fault=='failed_changed':
        row=C.read(owned/'prior-campaign/original-plan.json')['assignments'][17]
        (owned/row['attempt']/'manifest.json').write_text('{}')
    elif fault=='old_operational_source':(owned/'shard-0-64/source/codex/certify-llm/launch.py').write_text('# changed')
    with pytest.raises(ValueError):C.recovery_authority(plan,repo,owned)


def test_scoped_git_status_argument_keeps_clean_dirty_meaning(monkeypatch,tmp_path):
    commands=[]; result=['']
    def checked(cmd,**kw):commands.append(cmd);return result[0]
    monkeypatch.setattr(C.subprocess,'check_output',checked)
    assert C.git(tmp_path,'status','--porcelain','--untracked-files=all')==''
    assert commands[-1]==['git','-c','core.preloadIndex=false','status','--porcelain','--untracked-files=all']
    result[0]=' M tracked\n?? extra\n'
    assert C.git(tmp_path,'status','--porcelain','--untracked-files=all')=='M tracked\n?? extra'
    C.git(tmp_path,'rev-parse','HEAD')
    assert commands[-1]==['git','rev-parse','HEAD']


def test_recovery_storage_forecast_charges_retained_history(frozen_fixture,monkeypatch):
    repo,owned,plan=frozen_fixture;plan['schema']=C.RECOVERY_SCHEMA;plan['assignments']=plan['assignments'][:94]
    result=dict(elapsed_seconds=1.0,max_rss_kib=100,bytes=4096)
    monkeypatch.setattr(C,'completed',lambda *a,**k:result)
    f=plan['first'];a=dict(schema=C.AUDIT_SCHEMA,passed=True,certificate_passed=False,attempt=f['attempt'],
        receipt_sha256=f['receipt_sha256'],plan_sha256=f['plan']['sha256'],phase_index=0,start_column=0,
        **{k:plan['common'][k] for k in ('definition_sha256','map_sha256','backend_sha256','evaluator_sha256')},
        measured_elapsed_seconds=1.0,measured_max_rss_kib=100,reviewer='synthetic',review_source_sha256=H)
    f['audit']['sha256']=save(Path(f['audit']['path']),a)
    base=C.first_authority(plan,owned,existing_bytes=1000)
    assert base['existing_history_bytes']==5096
    plan['resources']['raw_bytes']=base['forecast_bytes']-1
    with pytest.raises(ValueError,match='retained history'):C.first_authority(plan,owned,existing_bytes=1000)


def test_v2_integrated_admission_then_dispatches_exact94_only(recovered,monkeypatch):
    repo,owned,plan=recovered
    C.validate(plan,repo,owned)
    monkeypatch.setattr(C,'process_table',lambda:[])
    recovered_authority=C.recovery_authority(plan,repo,owned)
    measured=C.first_authority(plan,owned,existing_bytes=recovered_authority['existing_bytes'])
    assert measured['forecast_seconds']<plan['resources']['timeout_seconds']
    dispatcher=C.Dispatcher(plan,owned/'synthetic-dispatch',SOURCE,repo=repo,owned=owned,
                            existing_bytes=measured['existing_history_bytes'])
    selected=[]
    def fake_start(row):
        selected.append(row)
        dispatcher.children.append(dict(row=row,state='completed',registered=True,
            stdout=open('/dev/null','w'),stderr=open('/dev/null','w'),leader=None,worker=None))
    monkeypatch.setattr(dispatcher,'start',fake_start)
    monkeypatch.setattr(dispatcher,'sample',lambda:None)
    monkeypatch.setattr(dispatcher,'inspect',lambda:None)
    monkeypatch.setattr(C.time,'sleep',lambda _:None)
    result=dispatcher.run()
    assert result['status']=='campaign_completed' and result['completed']==94
    assert selected==plan['assignments']
    old=C.read(owned/'prior-campaign/original-plan.json')
    assert not {r['attempt'] for r in selected}&{r['attempt'] for r in [old['first']]+old['assignments'][:17]}
    assert selected[4:]==old['assignments'][21:]


def test_retained_domain_or_evaluator_source_rejected_even_with_resealed_receipt(recovered,monkeypatch):
    repo,owned,plan=recovered
    monkeypatch.setattr(C,'process_table',lambda:[])
    row=C.read(owned/'prior-campaign/original-plan.json')['assignments'][0]
    attempt=owned/row['attempt']
    # Direct completed() verifies semantic source/domain independently of inventory.
    for fault in ('domain','source'):
        inputs=attempt/'run/inputs.json';original=inputs.read_bytes();d=json.loads(original)
        if fault=='domain':d['plan']['map_sha256']='c'*64
        else:d['source_commit']='c'*40
        save(inputs,d); rh=seal(attempt,source_commit=SOURCE,exit_code=0,timed_out=False,elapsed_seconds=2.0)
        with pytest.raises(ValueError):C.completed(attempt,row,plan['common'],SOURCE,rh)
        inputs.write_bytes(original)


@pytest.mark.parametrize('fault',['missing','duplicate'])
def test_resealed_controller_cannot_omit_or_duplicate_retained_history(recovered,fault):
    repo,owned,plan=recovered;prior=owned/'prior-campaign'
    state=C.read(prior/'run/state.json')
    if fault=='missing':state['rows'].pop(0)
    else:state['rows'][1]=copy.deepcopy(state['rows'][0])
    save(prior/'run/state.json',state)
    h=seal(prior,status='campaign_failed',certificate_passed=False)
    plan['recovery']['receipt_sha256']=h
    closure=Path(plan['recovery']['closure']['path']);d=C.read(closure);d['original_receipt_sha256']=h
    plan['recovery']['closure']['sha256']=save(closure,d)
    with pytest.raises(ValueError,match='all21|dispatched prefix'):C.validate(plan,repo,owned)
