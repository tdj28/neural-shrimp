"""Synthetic EXP019 integrity tests; no historical arrays or neural integration.

Real packet codecs and full metadata denominators are exercised. Numerical
acquisition/evaluation have their own actual synthetic integration suite.
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
HEAD = 'a'*40


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True)+'\n')


def histories(phases):
    return [dict(id=f'{setting}--p{p["phase_index"]:02d}--f{a}--s{b}--{arm}',
        context_id=f'{setting}--p{p["phase_index"]:02d}',setting=setting,**p,
        first_sign=a,second_sign=b,arm=arm,epsilon=1e-4)
        for setting in ('core','failed-near-edge') for p in phases
        for a in (-1,0,1) for b in (-1,0,1) for arm in ('identity','timing','transverse')]


@pytest.fixture
def runner(tmp_path, monkeypatch):
    spec=importlib.util.spec_from_file_location('synthetic_refined_runner',TOOLS/'run_refined_two_write_memory.py')
    r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    r.ROOT=tmp_path/'checkout';r.ROOT.mkdir();r.OLD=tmp_path/'prior';r.OLD.mkdir()
    r.binding.__defaults__=(r.ROOT,)
    r.SOURCE_PATHS=['synthetic-source.py']
    (r.ROOT/r.SOURCE_PATHS[0]).write_text('# synthetic source\n')
    def forbidden(*a,**k):pytest.fail('Scientific kernel called in controller test')
    r.acquisition=SimpleNamespace(TANGENT_METHODS={'dop853':{},'rk45':{}},METHODS={},
        acquire_tangent=forbidden,acquire_trace=forbidden,seal_context=forbidden)
    r.evaluation=SimpleNamespace(history_census=histories,contract=lambda:{'synthetic':True})
    artifacts=[]
    for setting in ('core','failed-near-edge'):
        for ext in ('json','npz'):
            p=r.OLD/f'references/payloads/{setting}--reference-pair/packet.{ext}'
            p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'opaque synthetic bytes')
            artifacts.append(r.binding(p,r.OLD))
    summary=dict(complete=True,qualified=True,source_valid=True,error=None,postflight_error=None,
        analysis=dict(counts=dict(shoots_complete=4,shoots_converged=4,references_complete=2,
        references_qualified=2,events_complete=96,events_qualified=96,contexts_complete=48,contexts_qualified=48)))
    write_json(r.OLD/'summary.json',summary)
    write_json(r.OLD/'manifest.json',dict(source_commit=r.OLD_SOURCE,plan_sha256=r.OLD_PLAN_SHA))
    artifacts += [r.binding(r.OLD/x,r.OLD) for x in ('summary.json','manifest.json')]
    write_json(r.OLD/'receipt.json',dict(status='complete',source_commit=r.OLD_SOURCE,
        plan_sha256=r.OLD_PLAN_SHA,artifacts=artifacts))
    r.OLD_RECEIPT_SHA=r.legacy.sha(r.OLD/'receipt.json')
    compact=dict(source_commit=r.OLD_SOURCE,plan_sha256=r.OLD_PLAN_SHA,
        raw_receipt=dict(sha256=r.OLD_RECEIPT_SHA),frozen_verification=dict(complete=True,qualified=True,
        saved_array_replay=True,all_original_owned_identities_absent_now=True,receipt_sha256=r.OLD_RECEIPT_SHA),
        reuse_qualification=dict(complete_cohort_qualified=True,frozen_saved_replay_pass=True,
        independent_evidence_decisions_pass=True,alternate_strict_arithmetic_pass=False))
    write_json(r.ROOT/r.OLD_SUMMARY,compact);r.OLD_SUMMARY_SHA=r.legacy.sha(r.ROOT/r.OLD_SUMMARY)
    return r


def terminal(task,status='complete',**extra):
    return dict(id=task['id'],kind=task['kind'],case_id=task['case']['id'],method=task['method']['id'],
        status=status,qualified=status=='complete',**extra)


def good_seals(stages):
    return {t['case']['context_id']:dict(context_id=t['case']['context_id'],status='complete',qualified=True,
        acquisition={'id':t['case']['context_id']},evaluation=dict(complete=True,qualified=True))
        for t in stages['references']}


def test_phase_and_full_census_match_original_without_new_holdout(runner):
    phases=runner.phase_rows()
    assert phases==runner.inherited.phase_rows()
    assert len(phases)==24 and len({x['phase_fraction'] for x in phases})==24
    assert {x['phase_index'] for x in phases if x['split']=='development'}==set(range(8))
    assert {x['phase_index'] for x in phases if x['split']=='heldout'}==set(range(8,24))
    stages=runner.stage_tasks(phases);rows=sum(stages.values(),[])
    assert [len(stages[k]) for k in ('references','development','heldout')]==[96,918,1782]
    assert len(rows)==len({r['id'] for r in rows})==2796
    assert 2*len(rows)==runner.CONTRACT['maximum_segment_integrations']==5592
    traces=stages['development']+stages['heldout']
    assert len({r['case']['id'] for r in traces})==1296
    panel=[r for r in traces if r['method']['id']=='dop853-independent']
    assert len(panel)==108 and {r['case']['phase_index'] for r in panel}=={0,8}
    assert runner.CONTRACT['fresh_holdout'] is False


def test_metadata_plan_hashes_exact_eight_inputs_without_array_decode(runner,monkeypatch):
    def forbidden(*a,**k):pytest.fail('Metadata plan decoded arrays')
    monkeypatch.setattr(runner.codec,'load_packet',forbidden)
    monkeypatch.setattr(runner.np,'load',forbidden)
    plan=runner.build_plan()
    assert len(plan['historical_inputs'])==8
    assert runner.validate(plan)['total_jobs']==2796
    changed=copy.deepcopy(plan);changed['phases'][0]['phase_fraction']+=.001
    with pytest.raises(ValueError):runner.validate(changed)


@pytest.mark.parametrize('change',['raw_bytes','receipt','compact','qualified_count','audit_fail'])
def test_historical_bindings_and_qualification_fail_closed(runner,change):
    r=runner
    if change=='raw_bytes':(r.OLD/'references/payloads/core--reference-pair/packet.npz').write_bytes(b'changed')
    elif change=='receipt':write_json(r.OLD/'receipt.json',{})
    elif change=='compact':write_json(r.ROOT/r.OLD_SUMMARY,{})
    elif change=='audit_fail':
        value=r.read(r.ROOT/r.OLD_SUMMARY);value['reuse_qualification']['independent_evidence_decisions_pass']=False
        write_json(r.ROOT/r.OLD_SUMMARY,value);r.OLD_SUMMARY_SHA=r.legacy.sha(r.ROOT/r.OLD_SUMMARY)
    else:
        p=r.OLD/'summary.json';value=r.read(p);value['analysis']['counts']['contexts_qualified']=47;write_json(p,value)
        receipt=r.read(r.OLD/'receipt.json');receipt['artifacts']=[r.binding(p,r.OLD) if a['path']=='summary.json' else a for a in receipt['artifacts']]
        write_json(r.OLD/'receipt.json',receipt);r.OLD_RECEIPT_SHA=r.legacy.sha(r.OLD/'receipt.json')
    with pytest.raises((ValueError,KeyError)):r.old_inputs()


def source_fixture(r,monkeypatch):
    path=r.ROOT/r.PLAN_REL;write_json(path,{'synthetic':True})
    write_json(r.ROOT/r.REVIEW_REL,dict(status='ready_for_exploratory_execution',plan_sha256=r.legacy.sha(path)))
    state=dict(dirty='',head=HEAD,branch='codex/synthetic',remote=HEAD,origin='git@github-personal:tdj28/neural-shrimp.git')
    committed={name:(r.ROOT/name).read_bytes() for name in [*r.SOURCE_PATHS,r.PLAN_REL,r.REVIEW_REL]}
    def git(*args):
        return state[{'status':'dirty','rev-parse':'head','branch':'branch','remote':'origin','ls-remote':'remote'}[args[0]]]
    monkeypatch.setattr(r.legacy,'git',git)
    monkeypatch.setattr(r.subprocess,'check_output',lambda cmd:committed[cmd[-1].split(':',1)[1]])
    return path,state,committed


def test_source_guard_accepts_exact_committed_review(runner,monkeypatch):
    path,_,_=source_fixture(runner,monkeypatch)
    assert runner.source_guard({},path,HEAD)['source_commit']==HEAD


@pytest.mark.parametrize('field,value',[('dirty','?? extra.py'),('head','b'*40),('branch','main'),('remote','c'*40),('origin','https://elsewhere/repo')])
def test_source_guard_rejects_unfrozen_or_wrong_origin(runner,monkeypatch,field,value):
    path,state,_=source_fixture(runner,monkeypatch);state[field]=value
    with pytest.raises(ValueError):runner.source_guard({},path,HEAD)


def test_source_guard_checks_exact_plan_review_and_committed_source(runner,monkeypatch):
    path,_,committed=source_fixture(runner,monkeypatch)
    review=runner.read(runner.ROOT/runner.REVIEW_REL);review['plan_sha256']='f'*64
    write_json(runner.ROOT/runner.REVIEW_REL,review)
    committed[runner.REVIEW_REL]=(runner.ROOT/runner.REVIEW_REL).read_bytes()
    with pytest.raises(ValueError,match='Exact-plan'):runner.source_guard({},path,HEAD)
    committed[runner.SOURCE_PATHS[0]]=b'wrong'
    with pytest.raises(ValueError,match='Uncommitted'):runner.source_guard({},path,HEAD)


def test_external_attempt_exclusive_and_rejects_checkout_aliases_and_prior_attempts(runner,tmp_path):
    raw=tmp_path/'raw';path=runner.external_attempt(raw,'attempt-001')
    with pytest.raises(FileExistsError):runner.external_attempt(raw,'attempt-001')
    for bad in [runner.ROOT,runner.ROOT/'raw',runner.OLD,runner.OLD/'nested',path,path/'nested']:
        with pytest.raises(ValueError):runner.external_attempt(bad,'attempt-002')
    prior=tmp_path/'EXP-017'/'attempt-001';prior.mkdir(parents=True)
    with pytest.raises(ValueError):runner.external_attempt(prior/'new','attempt-002')
    assert not (prior/'new').exists()
    raw2=tmp_path/'aliases';raw2.mkdir();(raw2/'EXP-019').symlink_to(runner.ROOT,target_is_directory=True)
    with pytest.raises(ValueError):runner.external_attempt(raw2,'attempt-001')


def test_prepare_is_exclusive(runner,monkeypatch,tmp_path):
    target=tmp_path/'plan.json';target.write_text('preserve')
    monkeypatch.setattr(runner,'build_plan',lambda:{})
    monkeypatch.setattr(sys,'argv',['runner','prepare','--output',str(target)])
    with pytest.raises(FileExistsError):runner.main()
    assert target.read_text()=='preserve'


def test_gate_requires_all_96_and_all_48_with_exact_identities(runner):
    r=runner;stages=r.stage_tasks(r.phase_rows());rows=[terminal(t) for t in stages['references']];seals=good_seals(stages)
    gate=lambda:r.reference_gate(stages,rows,seals,input_sha256='a',index_sha256='b')
    assert gate()['passed']
    rows[0]['status']='task_timeout'
    assert not gate()['passed'] and gate()['complete_references']==95
    rows[0]['status']='complete';cid=next(iter(seals))
    seals[cid]['qualified']=False;seals[cid]['evaluation']['qualified']=False
    assert not gate()['passed'] and gate()['qualified_contexts']==47
    seals[cid]['evaluation']['qualified']=True
    with pytest.raises(ValueError,match='qualification differs'):gate()
    seals=good_seals(stages);rows[0]=rows[1]
    with pytest.raises(ValueError,match='Exact96'):gate()


def mock_run(r,monkeypatch,tmp_path,*,failed_reference=False,failed_seal=False,setup_error=False,index_corrupt=False,gate_corrupt=False,analysis_complete=True):
    plan={'phases':r.phase_rows()};path=tmp_path/'plan.json';write_json(path,plan)
    monkeypatch.setattr(r,'validate',lambda *a,**k:{})
    monkeypatch.setattr(r,'source_guard',lambda *a:dict(source_commit=HEAD,pushed_branch='codex/synthetic'))
    monkeypatch.setattr(r.importlib.metadata,'version',lambda k:r.CONTRACT['required_packages'][k])
    for key in r.legacy.THREAD_ENV:monkeypatch.setenv(key,'1')
    monkeypatch.setattr(r.legacy,'git',lambda *a:HEAD if a[0]=='rev-parse' else '')
    def resource(*a,**k):
        if setup_error:raise RuntimeError('synthetic setup failure')
        return {}
    monkeypatch.setattr(r,'resource_check',resource);monkeypatch.setattr(r.signal,'setitimer',lambda *a:None)
    stages=r.stage_tasks(plan['phases']);contexts={t['case']['context_id']:{'id':t['case']['context_id']} for t in stages['references']}
    monkeypatch.setattr(r,'initial_contexts',lambda p:contexts)
    monkeypatch.setattr(r.work,'build_libraries',lambda *a:{})
    calls=[]
    def execute(specs,inputs,directory,libraries,**kwargs):
        directory=Path(directory);calls.append((directory.name,len(specs)))
        if directory.name!='references':
            gate=r.read(directory.parent/'global-reference-gate.json')
            assert gate['passed'] and gate['complete_references']==96 and gate['qualified_contexts']==48
            index=r.read(directory.parent/'reference-seal-index.json');assert len(index)==48
            events=[json.loads(x) for x in (directory.parent/'events.jsonl').read_text().splitlines()]
            assert events[-1]['event']==directory.name+'_started'
            assert any(e['event']=='global_reference_gate_sealed' and e['passed'] for e in events[:-1])
        rows=[terminal(t) for t in specs]
        if directory.name=='references' and failed_reference:rows[0]['status']='task_timeout'
        return rows
    monkeypatch.setattr(r.runtime,'execute',execute)
    def seals(directory,stages,rows,contexts,*,persist):
        saved=good_seals(stages)
        if failed_seal:
            cid=next(iter(saved));saved[cid]['qualified']=False;saved[cid]['evaluation']['qualified']=False
        for cid,packet in saved.items():r.codec.save_packet(Path(directory)/'seals',cid,packet)
        # Corrupt the returned in-memory copy only. The saved objects govern.
        returned=copy.deepcopy(saved)
        for packet in returned.values():packet['qualified']=False
        return returned
    monkeypatch.setattr(r,'make_reference_seals',seals)
    monkeypatch.setattr(r,'analyze',lambda *a:dict(packet_errors=[],summary=dict(complete=analysis_complete)))
    if index_corrupt or gate_corrupt:
        original=r.read
        def read(p):
            result=original(p)
            if index_corrupt and Path(p).name=='reference-seal-index.json':result.pop(next(iter(result)))
            if gate_corrupt and Path(p).name=='global-reference-gate.json':result['passed']=not result['passed']
            return result
        monkeypatch.setattr(r,'read',read)
    result=r.run(path,tmp_path/'raw','attempt-001',HEAD)
    return result,calls


@pytest.mark.parametrize('failed', ['reference','seal'])
def test_one_failed_prerequisite_blocks_every_finite_task(runner,monkeypatch,tmp_path,failed):
    result,calls=mock_run(runner,monkeypatch,tmp_path,failed_reference=failed=='reference',failed_seal=failed=='seal')
    assert calls==[('references',96)]
    assert not result['summary']['complete']
    rows=runner.read(Path(result['directory'])/'all-terminal-records.json')
    assert sum(map(len,rows.values()))==2796
    assert all(row['status']=='blocked_global_reference_gate' for s in ('development','heldout') for row in rows[s])
    assert result['summary']['global_reference_gate']['passed'] is False


def test_all_disk_seals_and_gate_readback_precede_complete_finite_census(runner,monkeypatch,tmp_path):
    result,calls=mock_run(runner,monkeypatch,tmp_path)
    assert calls==[('references',96),('development',918),('heldout',1782)]
    assert result['summary']['complete'] is True


def test_corrupted_gate_readback_cannot_promote_failed_prerequisites(runner,monkeypatch,tmp_path):
    result,calls=mock_run(runner,monkeypatch,tmp_path,failed_seal=True,gate_corrupt=True)
    assert calls==[('references',96)] and not result['summary']['complete']
    assert 'Global reference gate readback differs' in result['summary']['error']
    rows=runner.read(Path(result['directory'])/'all-terminal-records.json')
    assert sum(len(rows[s]) for s in ('development','heldout'))==2700
    assert all(r['status']!='complete' and r.get('pid') is None for s in ('development','heldout') for r in rows[s])


@pytest.mark.parametrize('mode',['setup','index','analysis'])
def test_failed_setup_readback_or_analysis_preserves_full_denominator(runner,monkeypatch,tmp_path,mode):
    result,calls=mock_run(runner,monkeypatch,tmp_path,setup_error=mode=='setup',index_corrupt=mode=='index',analysis_complete=mode!='analysis')
    assert result['summary']['complete'] is False
    if mode!='analysis':assert not any(s!='references' for s,n in calls)
    rows=runner.read(Path(result['directory'])/'all-terminal-records.json')
    assert [len(rows[k]) for k in ('references','development','heldout')]==[96,918,1782]
    assert runner.read(Path(result['directory'])/'receipt.json')['status']=='incomplete_or_invalid'


def save_child(r,directory,task,packet,*,status='failed',pid=123):
    directory.mkdir(parents=True,exist_ok=True);(directory/'records').mkdir(exist_ok=True)
    saved=r.codec.save_packet(directory/'payloads'/task['id'],'packet',packet)
    row=terminal(task,status,pid=pid,packet_path=str(Path(saved['path']).relative_to(directory)),
        packet_sha256=saved['sha256'],packet_semantic_sha256=saved['semantic_sha256'])
    write_json(directory/'records'/f'{task["id"]}.json',row)
    return row


@pytest.mark.parametrize('kind',['reference','trace'])
def test_parent_timeout_retains_real_packet_schema_without_promotion(runner,tmp_path,kind):
    r=runner;stages=r.stage_tasks(r.phase_rows());stage='references' if kind=='reference' else 'development';task=stages[stage][0]
    packet=dict(context_id=task['case']['context_id'],status='failed',complete=False,failed_stage='post',
        pre={'states':np.zeros((2,3),dtype=np.float64)},partial_failure={'last_state':np.ones(3)})
    if kind=='reference':packet['method']=task['method']['id']
    else:packet['task']=dict(task['case'],id=task['id'],history_id=task['case']['id'],method=task['method']['id'])
    directory=tmp_path/'partial'/stage;save_child(r,directory,task,packet)
    parent=terminal(task,'task_timeout',pid=123)
    assert r.load_task_packet(directory,task,parent) is None
    retained=r.load_task_packet(directory,task,parent,include_partial=True)
    assert retained['failed_stage']=='post'
    rows={key:[] for key in stages};rows[stage]=[parent]
    census=r.partial_packets(directory.parent,stages,rows)
    assert len(census)==1 and census[0]['parent_status']=='task_timeout' and not census[0]['qualified']
    assert len(census[0]['array_census'])==2
    parent['pid']=999
    with pytest.raises(ValueError,match='worker identity'):r.load_task_packet(directory,task,parent,include_partial=True)


def test_completed_child_after_parent_timeout_is_not_scored(runner,tmp_path):
    r=runner;task=r.stage_tasks(r.phase_rows())['references'][0];directory=tmp_path/'refs'
    packet=dict(context_id=task['case']['context_id'],method=task['method']['id'],status='complete',complete=True)
    save_child(r,directory,task,packet,status='complete')
    parent=terminal(task,'task_timeout',pid=123)
    assert r.load_task_packet(directory,task,parent) is None
    assert r.load_task_packet(directory,task,parent,include_partial=True)['complete'] is True


def test_packet_context_or_method_cannot_alias_another_task(runner,tmp_path):
    r=runner;task=r.stage_tasks(r.phase_rows())['references'][0];directory=tmp_path/'refs'
    packet=dict(context_id=task['case']['context_id'],method='wrong',status='complete',complete=True)
    row=save_child(r,directory,task,packet,status='complete')
    with pytest.raises(ValueError,match='method identity'):r.load_task_packet(directory,task,row)


def test_owned_worker_timeout_preserves_catchable_prefix_and_cleanup(runner,monkeypatch,tmp_path):
    r=runner;task=r.stage_tasks(r.phase_rows())['references'][0]
    class PartialFailure(Exception):
        def __init__(self,context):
            self.payload=dict(context_id=context['id'],method='dop853',status='failed',complete=False,
                failed_stage='post',pre={'states':np.zeros((2,3),dtype=np.float64)})
    def cancelled_acquire(context,method):
        try:
            while True:time.sleep(.005)
        except TimeoutError:raise PartialFailure(context)
    monkeypatch.setattr(r.acquisition,'acquire_tangent',cancelled_acquire)
    directory=tmp_path/'owned'
    rows=r.runtime.execute([task],{task['case']['id']:{'id':task['case']['context_id']}},directory,{},
        workers=1,deadline=time.monotonic()+5,per_task=.08,worker_fn=r.worker)
    assert rows[0]['status']=='task_timeout'
    assert r.load_task_packet(directory,task,rows[0]) is None
    saved=r.load_task_packet(directory,task,rows[0],include_partial=True)
    assert saved['failed_stage']=='post' and saved['pre']['states'].shape==(2,3)
    cleanup=r.read(directory/'cleanup.json')
    assert cleanup['all_registered_workers_joined'] and cleanup['all_original_identities_absent']
    assert r.work.identity(rows[0]['pid']) is None


def chronology_fixture(r,monkeypatch,tmp_path):
    directory=tmp_path/'chronology';directory.mkdir()
    plan=dict(phases=r.phase_rows());stages=r.stage_tasks(plan['phases'])
    rows={s:[terminal(t,'complete' if s=='references' else 'not_started') for t in tasks] for s,tasks in stages.items()}
    rows['development'][0]=terminal(stages['development'][0],pid=991)
    contexts={t['case']['id']:{'id':t['case']['id']} for t in stages['references']}
    seals=good_seals(stages)
    monkeypatch.setattr(r,'initial_contexts',lambda p:contexts)
    monkeypatch.setattr(r,'make_reference_seals',lambda *a,**k:copy.deepcopy(seals))
    monkeypatch.setattr(r,'load_task_packet',lambda *a,**k:{})
    monkeypatch.setattr(r,'partial_packets',lambda *a:[])
    monkeypatch.setattr(r.evaluation,'_tangent',lambda *a:None,raising=False)
    saved=r.codec.save_packet(directory,'input-seal',contexts)
    for cid,p in seals.items():r.codec.save_packet(directory/'seals',cid,p)
    write_json(directory/'reference-seal-index.json',r.reference_index(directory,seals))
    gate=r.reference_gate(stages,rows['references'],seals,input_sha256=r.legacy.sha(directory/'input-seal.json'),
        index_sha256=r.legacy.sha(directory/'reference-seal-index.json'))
    write_json(directory/'global-reference-gate.json',gate)
    events=[dict(event='inputs_sealed',sha256=saved['sha256'],semantic_sha256=saved['semantic_sha256']),
        dict(event='references_started',planned=96),dict(event='reference_seals_complete',index_sha256=r.legacy.sha(directory/'reference-seal-index.json')),
        dict(event='global_reference_gate_sealed',sha256=r.legacy.sha(directory/'global-reference-gate.json'),passed=True),
        dict(event='development_started',planned=918,eligible=918)]
    for i,e in enumerate(events):e.update(monotonic_ns=i+1,at=f'2026-09-17T00:00:0{i}Z')
    def save_events():
        (directory/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    save_events()
    return directory,plan,stages,rows,events,save_events


def test_saved_gate_chronology_accepts_complete_prerequisites(runner,monkeypatch,tmp_path):
    d,p,s,r,_,_=chronology_fixture(runner,monkeypatch,tmp_path)
    assert len(runner.verify_seals(d,p,s,r,validate_trace_prefix=False))==48


@pytest.mark.parametrize('bad',['missing_gate_event','early_stage','selective_stage','early_owned_worker','gate_file_tamper'])
def test_saved_verifier_rejects_unsealed_early_or_selective_finite_dispatch(runner,monkeypatch,tmp_path,bad):
    d,p,s,r,events,save=chronology_fixture(runner,monkeypatch,tmp_path)
    if bad=='missing_gate_event':events[:]=[e for e in events if e['event']!='global_reference_gate_sealed']
    elif bad=='early_stage':events[-1]['monotonic_ns']=events[-2]['monotonic_ns']
    elif bad=='selective_stage':events[-1]['eligible']=917
    elif bad=='early_owned_worker':
        (d/'development').mkdir()
        (d/'development/processes.jsonl').write_text(json.dumps(dict(started_at='2026-09-17T00:00:02Z'))+'\n')
    else:
        gate=runner.read(d/'global-reference-gate.json');gate['qualified_contexts']=47
        write_json(d/'global-reference-gate.json',gate)
    save()
    with pytest.raises(ValueError):runner.verify_seals(d,p,s,r,validate_trace_prefix=False)


def partial_fixture(r,monkeypatch,tmp_path,*,input_seal=False,stage_journal=False):
    directory=tmp_path/'partial-attempt';directory.mkdir()
    case=dict(id='synthetic-context',context_id='synthetic-context')
    stages={s:[dict(id=s+'-0',kind='reference' if s=='references' else 'trace',case=case,
        method=dict(id='dop853' if s=='references' else 'rk4-fine'))] for s in ('references','development','heldout')}
    plan=dict(phases=[],source_files=[r.binding(r.ROOT/n) for n in r.SOURCE_PATHS])
    plan_bytes=json.dumps(plan,sort_keys=True).encode();plan_sha=hashlib.sha256(plan_bytes).hexdigest()
    write_json(r.ROOT/r.REVIEW_REL,{'synthetic_review':True});review=r.binding(r.ROOT/r.REVIEW_REL)
    committed={r.PLAN_REL:plan_bytes,r.REVIEW_REL:(r.ROOT/r.REVIEW_REL).read_bytes(),
        **{n:(r.ROOT/n).read_bytes() for n in r.SOURCE_PATHS}}
    monkeypatch.setattr(r.subprocess,'check_output',lambda c:committed[c[-1].split(':',1)[1]])
    monkeypatch.setattr(r,'validate',lambda *a,**k:{})
    monkeypatch.setattr(r,'stage_tasks',lambda p:stages)
    contexts={'synthetic-context':{'metadata_only':1}}
    monkeypatch.setattr(r,'initial_contexts',lambda p:contexts)
    write_json(directory/'manifest.json',dict(plan=plan,source_commit=HEAD,plan_sha256=plan_sha,review=review))
    write_json(directory/'planned-ledger.json',stages)
    rows={s:r.fill_rows(tasks,[],'not_started_interruption') for s,tasks in stages.items()}
    write_json(directory/'all-terminal-records.json',rows)
    error='synthetic interruption';write_json(directory/'producer-error.json',dict(stage='input_seal',error=error))
    r.codec.save_packet(directory,'analysis',dict(status='unavailable',reason=error,packet_errors=[],summary=None))
    write_json(directory/'postflight.json',dict(source_bindings_unchanged=True,error=None))
    write_json(directory/'summary.json',dict(complete=False,source_valid=True,error=error,postflight_error=None,
        analysis=None,global_reference_gate=None,status_counts={s:dict(r.Counter(x['status'] for x in v)) for s,v in rows.items()}))
    if input_seal:r.codec.save_packet(directory,'input-seal',contexts)
    if stage_journal:
        stage=directory/'references';write_json(stage/'terminal-records.json',rows['references'])
        (stage/'journal.jsonl').write_text(json.dumps(rows['references'][0])+'\n')
        write_json(stage/'cleanup.json',dict(workers=[],all_registered_workers_joined=True,all_original_identities_absent=True))
    def reseal():
        artifacts=r.inventory(directory)
        write_json(directory/'receipt.json',dict(source_commit=HEAD,plan_sha256=plan_sha,experiment_id='EXP-019',
            artifacts=artifacts,artifact_bytes=sum(x['bytes'] for x in artifacts),status='incomplete_or_invalid'))
    reseal()
    return directory,rows,reseal


@pytest.mark.parametrize('seal',[False,True])
def test_pre_dispatch_prefix_replays_without_claiming_complete(runner,monkeypatch,tmp_path,seal):
    d,_,_=partial_fixture(runner,monkeypatch,tmp_path,input_seal=seal)
    result=runner.verify(d)
    assert not result['complete'] and not result['saved_array_replay'] and not result['global_reference_gate_passed']


@pytest.mark.parametrize('bad',['identity','cleanup','duplicate_journal','postflight','input_seal'])
def test_rehashed_partial_corruption_is_rejected(runner,monkeypatch,tmp_path,bad):
    r=runner;d,rows,reseal=partial_fixture(r,monkeypatch,tmp_path,stage_journal=True,input_seal=True)
    if bad=='identity':rows['heldout'][0]['case_id']='wrong';write_json(d/'all-terminal-records.json',rows)
    elif bad=='cleanup':
        p=d/'references/cleanup.json';v=r.read(p);v['all_original_identities_absent']=False;write_json(p,v)
    elif bad=='duplicate_journal':
        duplicate=rows['references']*2;write_json(d/'references/terminal-records.json',duplicate)
        (d/'references/journal.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in duplicate))
    elif bad=='postflight':write_json(d/'postflight.json',dict(source_bindings_unchanged=False,error=None))
    else:
        p=d/'input-seal.json';v=json.loads(p.read_text().replace('"metadata_only": 1','"metadata_only": 2'));write_json(p,v)
    reseal()
    with pytest.raises(ValueError):r.verify(d)
