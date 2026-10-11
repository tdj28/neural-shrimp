"""Synthetic metadata, arrays and mocked ownership only; no target solves."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import signal
import sys
import time
import numpy as np
import pytest
_checkout=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(Path(__file__).resolve().parent),str(_checkout/'tools'),str(_checkout/'tests')]
import run_phase_memory_refinement as r
import test_phase_memory_refinement_evaluation as f


def synthetic_plan():
    return dict(**r.CONTRACT,scope=r.SCOPE,source_files=[],selection=dict(cases=f.census(),original_failed_comparisons=[dict(qualified=False,ids=['original-left','original-right'])]),original_setting_qualification={'core':False,'failed-near-edge':False})


def rewrite_receipt(directory):
    receipt=r.read(directory/'receipt.json');receipt['artifacts']=r.inventory(directory)
    receipt['artifact_bytes']=sum(x['bytes'] for x in receipt['artifacts'])
    (directory/'receipt.json').write_text(json.dumps(receipt))


@pytest.fixture
def lifecycle(tmp_path,monkeypatch):
    plan=synthetic_plan();planpath=tmp_path/'plan.json';planpath.write_text(json.dumps(plan));head='a'*40
    old=tmp_path/'old';old.mkdir();r.codec.save_packet(old,'reference-seal',{})
    monkeypatch.setattr(r,'OLD',old);monkeypatch.setattr(r,'validate',lambda *a,**k:None)
    review=tmp_path/'review.json';review.write_text('{}');reviewbind=dict(path='review.json',bytes=2,sha256=r.legacy.sha(review))
    monkeypatch.setattr(r,'source_guard',lambda *a:dict(source_commit=head,pushed_branch='codex/test',review=reviewbind))
    monkeypatch.setattr(r.importlib.metadata,'version',lambda name:r.CONTRACT['required_packages'][name])
    monkeypatch.setattr(r.legacy,'git',lambda *a:head if a==('rev-parse','HEAD') else '')
    monkeypatch.setattr(r.codec,'resource_check',lambda *a,**k:None)
    monkeypatch.setattr(r.work,'build_libraries',lambda *a:{})
    monkeypatch.setattr(r.work,'identity',lambda pid:None)
    monkeypatch.setattr(r.platform,'platform',lambda:'synthetic-platform')
    def git_show(args):return review.read_bytes() if args[-1].endswith(':review.json') else planpath.read_bytes()
    monkeypatch.setattr(r.subprocess,'check_output',git_show)
    contexts={case['id']:f.context(case) for case in plan['selection']['cases']}
    monkeypatch.setattr(r,'make_contexts',lambda *a:contexts)
    packets={};scores={}
    for case in plan['selection']['cases']:
        for method in r.acquisition.METHODS:
            packet=f.packet(case,method,contexts[case['id']]);ident=packet['task']['id']
            packets[ident]=packet;scores[ident]=f.fabricated_score(case,method,contexts[case['id']])
    monkeypatch.setattr(r,'original_packets',lambda *a,**k:(dict(packets),dict(scores)))
    monkeypatch.setattr(r.evaluate,'score_packet',lambda packet,case,ctx:f.fabricated_score(case,packet['task']['method'],ctx))
    def workers(specs,contexts,directory,libraries,*,workers,deadline,per_task,worker_fn):
        assert workers==4 and per_task==60 and 0<deadline-time.monotonic()<=900
        directory=Path(directory);directory.mkdir();(directory/'records').mkdir();rows=[];owned=[]
        for index,task in enumerate(specs):
            dest=directory/'payloads'/task['id'];dest.mkdir(parents=True)
            packet=f.packet(task['case'],task['method']['id'],contexts[task['case']['id']]);seal=r.codec.save_packet(dest,'packet',packet)
            row=dict(id=task['id'],kind=task['kind'],case_id=task['case']['id'],method=task['method']['id'],status='complete',qualified=True,pid=1000+index,rmse=None,
                packet_path=str(Path(seal['path']).relative_to(directory)),packet_sha256=seal['sha256'],packet_semantic_sha256=seal['semantic_sha256'])
            owner=dict(task_id=task['id'],pid=row['pid'],identity=dict(pid=row['pid'],start_ticks=str(index),boot_id='synthetic'),started_at='synthetic')
            owned.append(owner);rows.append(row);r.work.write(directory/'records'/f'{row["id"]}.json',row)
            r.work.append(directory/'processes.jsonl',owner);r.work.append(directory/'journal.jsonl',row)
        r.work.write(directory/'terminal-records.json',rows)
        r.work.write(directory/'cleanup.json',dict(workers=[dict(**p,current_identity=None,original_identity_absent=True) for p in owned],all_registered_workers_joined=True,all_original_identities_absent=True))
        return rows
    monkeypatch.setattr(r.work,'execute',workers)
    return plan,planpath,head,tmp_path,contexts


def test_exact_census_methods_and_comparison_groups():
    plan=synthetic_plan();tasks=r.tasks(plan)
    assert len(tasks)==425 and len({t['id'] for t in tasks})==425
    assert [x['id'] for x in r.NEW_METHODS]==list(r.evaluate.NEW_METHODS)
    assert r.Counter(c['setting'] for c in plan['selection']['cases'])=={'core':26,'failed-near-edge':59}
    result=r.placeholder(plan,'synthetic failure')['summary']
    assert len(result['terminal_rows'])==680 and len(result['comparisons'])==2380 and len(result['paired_readouts'])==3744
    assert result['comparison_groups']=={'old-old':dict(required=255,complete=0,passed=0),'old-new':dict(required=1275,complete=0,passed=0),'new-new':dict(required=850,complete=0,passed=0)}
    assert result['setting_means'] is None and not result['original_study_requalified']


def test_output_exclusive_outside_repo_original_and_aliases(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir();old=tmp_path/'immutable';old.mkdir()
    monkeypatch.setattr(r,'ROOT',root);monkeypatch.setattr(r,'OLD',old)
    for target in (root,old,old/'nested'):
        with pytest.raises(ValueError):r.external_attempt(target,'attempt-001')
    alias=tmp_path/'alias';alias.symlink_to(old,target_is_directory=True)
    with pytest.raises(ValueError):r.external_attempt(alias,'attempt-001')
    out=r.external_attempt(tmp_path/'safe','attempt-001');assert out.exists()
    with pytest.raises(FileExistsError):r.external_attempt(tmp_path/'safe','attempt-001')
    with pytest.raises(ValueError):r.external_attempt(tmp_path/'safe','../escape')


def test_inventory_exact_and_no_aliases(tmp_path):
    (tmp_path/'receipt.json').write_text('excluded');(tmp_path/'nested').mkdir();(tmp_path/'nested/receipt.json').write_text('{}')
    assert [x['path'] for x in r.inventory(tmp_path)]==['nested/receipt.json']
    (tmp_path/'alias').symlink_to(tmp_path/'nested/receipt.json')
    with pytest.raises(ValueError):r.inventory(tmp_path)


@pytest.mark.parametrize('stage',['resource','build','execute','seal_event','cancellation'])
def test_partial_failure_retains_all_rows_and_verifies(lifecycle,monkeypatch,stage):
    plan,path,head,tmp,contexts=lifecycle
    def fail(*a,**k):raise RuntimeError('synthetic '+stage)
    if stage=='resource':monkeypatch.setattr(r.codec,'resource_check',fail)
    elif stage=='build':monkeypatch.setattr(r.work,'build_libraries',fail)
    elif stage=='execute':monkeypatch.setattr(r.work,'execute',fail)
    elif stage=='seal_event':monkeypatch.setattr(r,'event',fail)
    else:
        def cancelled(*a,**k):raise r.ProducerStop('synthetic deadline')
        monkeypatch.setattr(r.work,'build_libraries',cancelled)
    out=r.run(path,tmp/'raw','attempt-001',head);directory=Path(out['directory'])
    assert not out['complete'] and out['error']
    summary=r.read(directory/'summary.json')
    assert len(r.read(directory/'all-terminal-records.json'))==425
    assert len(summary['analysis']['comparisons'])==2380 and len(summary['analysis']['paired_readouts'])==3744
    verified=r.verify(directory)
    assert not verified['complete'] and verified['partial_placeholder_verified']
    assert r.read(directory/'receipt.json')['artifacts']==r.inventory(directory)
    if stage=='seal_event':assert (directory/'input-seal.json').exists() and not (directory/'events.jsonl').exists()


def test_cancellation_cannot_be_swallowed_as_oserror():
    assert not issubclass(r.ProducerStop,Exception)
    with pytest.raises(r.ProducerStop):
        try:raise r.ProducerStop('deadline')
        except (OSError,ValueError):pass


def test_full_controller_saved_packet_and_verifier_roundtrip(lifecycle):
    plan,path,head,tmp,contexts=lifecycle
    out=r.run(path,tmp/'raw','attempt-001',head);directory=Path(out['directory'])
    assert out['complete'] and out['error'] is None
    verified=r.verify(directory)
    assert verified['complete'] and verified['saved_array_replay'] and verified['all_original_owned_identities_absent_now']
    summary=r.read(directory/'summary.json')
    assert summary['analysis']['complete'] and summary['analysis']['setting_means'] is None
    assert len(summary['analysis']['comparisons'])==2380
    # Duplicate journal entries must not collapse through dictionary equality.
    terminal=directory/'traces/terminal-records.json';journal=directory/'traces/journal.jsonl'
    rows=r.read(terminal);rows.append(rows[0]);terminal.write_text(json.dumps(rows));journal.write_text(''.join(json.dumps(x)+'\n' for x in rows));rewrite_receipt(directory)
    with pytest.raises(ValueError,match='census'):r.verify(directory)


def test_failed_score_with_complete_raw_tasks_cannot_claim_complete(lifecycle,monkeypatch):
    plan,path,head,tmp,contexts=lifecycle;original=r.evaluate.score_packet;first=r.tasks(plan)[0]['id']
    def score(packet,case,ctx):
        if packet['task']['id']==first:return dict(task=packet['task'],case_id=case['id'],status='failed',error='synthetic scoring failure')
        return original(packet,case,ctx)
    monkeypatch.setattr(r.evaluate,'score_packet',score)
    out=r.run(path,tmp/'raw','attempt-001',head);directory=Path(out['directory'])
    assert out['status_counts']=={'complete':425} and out['error'] is None and not out['complete']
    analysis=r.codec.load_packet(directory/'analysis.json');assert not analysis['summary']['complete'] and not analysis['packet_errors']
    assert not r.verify(directory)['complete']
    assert r.read(directory/'receipt.json')['status']=='incomplete_or_invalid'


def test_source_guard_requires_committed_plan_review_clean_branch_and_remote(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir();head='a'*40;monkeypatch.setattr(r,'ROOT',root)
    path=root/r.PLAN_REL;path.parent.mkdir(parents=True);path.write_text('{}')
    review=root/r.REVIEW_REL;review.parent.mkdir();review.write_text(json.dumps(dict(status='ready_for_exploratory_execution',plan_sha256=r.legacy.sha(path))))
    def git(*args):
        if args[0]=='status':return ''
        if args==('rev-parse','HEAD'):return head
        if args==('branch','--show-current'):return 'codex/test'
        return head+'\trefs/heads/codex/test'
    monkeypatch.setattr(r.legacy,'git',git)
    monkeypatch.setattr(r.subprocess,'check_output',lambda args:(root/args[-1].split(':',1)[1]).read_bytes())
    assert r.source_guard({'source_files':[]},path,head)['source_commit']==head
    with pytest.raises(ValueError):r.source_guard({'source_files':[]},path,'b'*40)
    monkeypatch.setattr(r.legacy,'git',lambda *a:'dirty' if a[0]=='status' else git(*a))
    with pytest.raises(ValueError):r.source_guard({'source_files':[]},path,head)
    monkeypatch.setattr(r.legacy,'git',lambda *a:'b'*40+'\tx' if a[0]=='ls-remote' else git(*a))
    with pytest.raises(ValueError):r.source_guard({'source_files':[]},path,head)


@pytest.mark.parametrize('method',r.NEW_METHODS)
def test_truthful_acquisition_wrapper_with_mock_integrators(method,monkeypatch):
    case=f.case();ctx=f.context(case);task=dict(id=case['id']+'--'+method['id'],case=case,method=method)
    states=np.broadcast_to(ctx['pulse']['initial'],(12801,3)).copy()
    monkeypatch.setattr(r,'integrate_trace',lambda *a,**k:(r.acquisition.TIMES.copy(),states.copy()))
    class Result:success=True;nfev=7;njev=0;nlu=0;message='synthetic';t=r.acquisition.TIMES;y=states.T
    monkeypatch.setattr(r,'solve_ivp',lambda *a,**k:Result())
    packet=r.acquire_new(task,ctx,object())
    assert packet['complete'] and packet['task']['method']==method['id']
    assert r.evaluate.score_packet(packet,case,ctx)['status']=='complete'


def test_adaptive_failure_retains_last_rhs_without_claiming_accepted_state(monkeypatch):
    case=f.case();ctx=f.context(case);method=r.NEW_METHODS[-1];task=dict(id=case['id']+'--'+method['id'],case=case,method=method)
    def failed(fn,*a,**k):fn(.125,np.array([1.,2.,3.]));raise RuntimeError('synthetic integrator failure')
    monkeypatch.setattr(r,'solve_ivp',failed)
    packet=r.acquire_new(task,ctx)
    assert not packet['complete'] and packet['status']=='failed'
    assert packet['last_rhs']['accepted_solver_state'] is False and packet['last_rhs']['time']==.125


def test_packet_hash_path_and_operational_timeout_are_enforced(tmp_path):
    plan=synthetic_plan();task=r.tasks(plan)[0];case=task['case'];ctx=f.context(case)
    dest=tmp_path/'payloads'/task['id'];dest.mkdir(parents=True);seal=r.codec.save_packet(dest,'packet',f.packet(case,task['method']['id'],ctx))
    row=dict(id=task['id'],kind=task['kind'],case_id=case['id'],method=task['method']['id'],status='task_timeout',packet_path=str(Path(seal['path']).relative_to(tmp_path)),packet_sha256=seal['sha256'],packet_semantic_sha256=seal['semantic_sha256'])
    packets,errors=r.load_new_packets(tmp_path,[row],[task]);assert not errors and packets[row['id']]['complete'] is False
    row['packet_sha256']='0'*64
    packets,errors=r.load_new_packets(tmp_path,[row],[task]);assert not packets and len(errors)==1
    row['packet_path']='../escape';packets,errors=r.load_new_packets(tmp_path,[row],[task]);assert not packets and len(errors)==1
