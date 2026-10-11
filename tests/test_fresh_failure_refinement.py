"""Synthetic arrays, mocked lifecycle and metadata only; no target integration."""
from __future__ import annotations
import copy
import json
from pathlib import Path
import sys
import time
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_fresh_failure_refinement as r


def synthetic_plan():
    cases=[dict(id=r.CASE_IDS[0],anchor='fresh-11',phase=1350,radius=.016,candidate=7),
           dict(id=r.CASE_IDS[1],anchor='fresh-11',phase=1350,radius=0.)]
    return dict(**r.CONTRACT,scope=r.SCOPE,selected_cases=cases,
        comparison_counts=dict(all_new_method_pairs=20,old_fine_against_new=10,original_adjacent_native=4,unique_pairs=34,new_native_adjacent_subset=6),
        original_numerical_gates={r.CASE_IDS[0]:dict(native_waveform_pass=False,qualified=False),r.CASE_IDS[1]:dict(native_waveform_pass=True,qualified=True)})


def synthetic_context():
    times=np.arange(2700)/2700;theta=2*np.pi*times
    reference=np.tanh(np.column_stack((np.sin(theta),.2*np.cos(theta),.3*np.sin(2*theta))))
    return dict(alpha=1.,gamma=10.,delta=np.array([.01,-.02]),initial=np.array([0.,.2,0.]),reference=reference,period=10.,epoch=0.)


def synthetic_state(offset=0.):
    theta=2*np.pi*r.work.TIMES/10
    state=np.column_stack((np.sin(theta),.2*np.cos(theta),.3*np.sin(2*theta)))
    state[1:,0]+=offset;state[0]=[0.,.2,0.]
    return state


def save_row(task,directory,context,offset=0.):
    directory=Path(directory);dest=directory/'payloads'/task['id'];dest.mkdir(parents=True)
    path=dest/'trace.npz';state=synthetic_state(offset)
    r.legacy.atomic_npz(path,times=r.work.TIMES,states=state,initial=context['initial'],alpha=context['alpha'],gamma=context['gamma'],delta=context['delta'])
    method=task['method'];solver=dict(method)
    if method['kind']=='RK4':solver['steps']=round(75/method['dt'])
    else:solver.update(success=True,nfev=17)
    row=dict(id=task['id'],case_id=task['case']['id'],kind='preview',method=method['id'],status='complete',solver=solver,
        raw=str(path.relative_to(directory)),raw_sha256=r.legacy.sha(path),raw_bytes=path.stat().st_size,pid=123)
    row.update(r.work.score(context,r.work.TIMES,state))
    return row,state


@pytest.fixture()
def synthetic_evidence(tmp_path):
    plan=synthetic_plan();contexts={case:synthetic_context() for case in r.CASE_IDS};records=[];old_records={};old_arrays={}
    for task in r.tasks(plan):
        row,_=save_row(task,tmp_path/'previews',contexts[task['case']['id']],offset=1e-9*r.METHODS.index(task['method']))
        records.append(row)
    for case in plan['selected_cases']:
        for method in r.OLD_METHODS:
            task=dict(id=case['id']+'--'+method['id'],case=case,kind='preview',method=method)
            row,state=save_row(task,tmp_path/'old',contexts[case['id']]);old_records[row['id']]=row;old_arrays[row['id']]=state
    return plan,records,tmp_path,contexts,old_records,old_arrays


def test_exact_task_method_comparison_censuses():
    plan=synthetic_plan();tasks=r.tasks(plan);pairs=r.comparison_specs(plan)
    assert len(tasks)==10 and len({t['id'] for t in tasks})==10
    assert [m['dt'] for m in r.METHODS[:3]]==[.000625,.0003125,.00015625]
    assert r.METHODS[-1]==dict(id='dop853-1e-13',kind='DOP853',rtol=1e-13,atol=1e-15,max_step=.0125)
    assert len(pairs)==34 and len({(x['case_id'],*x['pair']) for x in pairs})==34
    assert r.Counter(x['group'] for x in pairs)==dict(all_new_method_pairs=20,old_fine_against_new=10,original_adjacent_native=4)


def test_synthetic_saved_raw_full_score_and_comparison_replay(synthetic_evidence):
    result=r.summarize(*synthetic_evidence)
    assert result['complete'] and len(result['terminal'])==10 and len(result['comparisons'])==34
    assert all(x['raw_score_replay'] for x in result['terminal'])
    assert all(x['passed'] for x in result['comparisons'])
    for case in result['cases'].values():
        assert case['original_fine_resolved_by_diagnostic'] and len(case['native_adjacent'])==3
    assert result['cases'][r.CASE_IDS[0]]['original_numerical_gates']['native_waveform_pass'] is False
    assert result['all_original_labels_and_qualifications_unchanged']


@pytest.mark.parametrize('mutation',['missing','duplicate','unknown','score','solver','boolsteps','adaptive','rawhash','float32','shape','grid','nonfinite','extraarray','delta','initial'])
def test_bad_or_missing_evidence_retains_full_denominator(synthetic_evidence,mutation):
    plan,rows,directory,contexts,old_records,old_arrays=synthetic_evidence
    row=rows[0]
    if mutation=='missing':rows.pop(0)
    elif mutation=='duplicate':rows.append(copy.deepcopy(row))
    elif mutation=='unknown':rows.append(dict(id='unplanned'))
    elif mutation=='score':row['metrics']['windows'][0]['aligned_rms']['aggregate']+=1e-5
    elif mutation=='solver':row['solver']['dt']*=2
    elif mutation=='boolsteps':row['solver']['steps']=True
    elif mutation=='adaptive':rows[3]['solver']['success']=False
    elif mutation=='rawhash':row['raw_sha256']='0'*64
    else:
        path=directory/'previews'/row['raw']
        with np.load(path,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
        if mutation=='float32':arrays['states']=arrays['states'].astype(np.float32)
        elif mutation=='shape':arrays['states']=arrays['states'][:-1]
        elif mutation=='grid':arrays['times'][4]+=.001
        elif mutation=='nonfinite':arrays['states'][3,2]=np.nan
        elif mutation=='extraarray':arrays['extra']=np.array([1.])
        elif mutation=='delta':arrays['delta'][0]+=.001
        else:arrays['states'][0,0]+=.001
        path.unlink();r.legacy.atomic_npz(path,**arrays);row.update(raw_sha256=r.legacy.sha(path),raw_bytes=path.stat().st_size)
    result=r.summarize(plan,rows,directory,contexts,old_records,old_arrays)
    assert not result['complete'] and len(result['terminal'])==10 and len(result['comparisons'])==34
    assert not result['cases'][r.CASE_IDS[0]]['original_fine_resolved_by_diagnostic']


@pytest.mark.parametrize('amount,expected',[(1e-4,True),(1.0000001e-4,False)])
def test_scalar_gate_unchanged_and_inclusive(synthetic_evidence,amount,expected):
    plan,rows,directory,contexts,old_records,old_arrays=synthetic_evidence
    case=r.CASE_IDS[0];a=case+'--rk4-0.005';b=case+'--rk4-0.0025'
    old_records[a]['rmse']=0.;old_records[b]['rmse']=amount
    spec=dict(case_id=case,group='original_adjacent_native',pair=['rk4-0.005','rk4-0.0025'])
    result=r.compare_pair(spec,old_records,old_arrays,contexts[case])
    assert result['scalar_pass'] is expected and result['passed'] is expected


def test_no_scalar_agreement_waives_waveform_failure(synthetic_evidence):
    plan,rows,directory,contexts,old_records,old_arrays=synthetic_evidence
    case=r.CASE_IDS[0];a=case+'--rk4-0.005';b=case+'--rk4-0.0025'
    old_records[a]['rmse']=old_records[b]['rmse']=.2;old_arrays[b]=old_arrays[b].copy();old_arrays[b][100,0]+=.1
    result=r.compare_pair(dict(case_id=case,group='original_adjacent_native',pair=['rk4-0.005','rk4-0.0025']),old_records,old_arrays,contexts[case])
    assert result['scalar_pass'] and not result['maximum_pass'] and not result['passed']
    assert result['maximum_location']==[100,0] and result['maximum_time']==10.


def test_external_attempt_exclusive_and_symlink_resolution(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir();monkeypatch.setattr(r,'ROOT',root)
    with pytest.raises(ValueError):r.external_attempt(root,'attempt-001')
    link=tmp_path/'alias';link.symlink_to(root,target_is_directory=True)
    with pytest.raises(ValueError):r.external_attempt(link,'attempt-001')
    out=r.external_attempt(tmp_path/'raw','attempt-001');assert out.exists()
    with pytest.raises(FileExistsError):r.external_attempt(tmp_path/'raw','attempt-001')
    with pytest.raises(ValueError):r.external_attempt(tmp_path/'raw','../attempt')


def test_fill_never_silently_drops_unknown_duplicate_rows():
    plan=synthetic_plan();first=r.tasks(plan)[0]
    row=dict(id=first['id'],status='failed')
    assert len(r._fill(plan,[row],'unavailable'))==10
    with pytest.raises(ValueError):r._fill(plan,[row,row],'unavailable')
    with pytest.raises(ValueError):r._fill(plan,[dict(id='unknown')],'unavailable')


@pytest.mark.parametrize('mode',['resource','build','execute'])
def test_run_failure_seals_receipt_all_ten_rows_and_thirty_four_pairs(tmp_path,monkeypatch,mode):
    plan=synthetic_plan();planpath=tmp_path/'plan.json';planpath.write_text(json.dumps(plan));head='a'*40
    monkeypatch.setattr(r,'validate',lambda *a,**k:None)
    monkeypatch.setattr(r,'source_guard',lambda *a:dict(source_commit=head,pushed_branch='codex/test'))
    monkeypatch.setattr(r.importlib.metadata,'version',lambda name:r.CONTRACT['required_packages'][name])
    monkeypatch.setattr(r.legacy,'git',lambda *args:head if args==('rev-parse','HEAD') else '')
    monkeypatch.setattr(r.original,'resources',lambda *a:dict(available_ram_bytes=0 if mode=='resource' else 10**12,disk_free_bytes=10**12))
    monkeypatch.setattr(r,'load_original',lambda p:({}, {}, {}))
    def fail(*a,**k):raise RuntimeError('synthetic failure')
    monkeypatch.setattr(r.work,'build_libraries',fail if mode=='build' else lambda *a:{})
    monkeypatch.setattr(r.work,'execute',fail)
    result=r.run(planpath,tmp_path/'raw','attempt-001',head)
    directory=Path(result['directory']);receipt=r.read(directory/'receipt.json');summary=r.read(directory/'summary.json')
    assert result['error'] and not result['complete'] and len(summary['terminal'])==10 and len(summary['comparisons'])==34
    assert len(r.read(directory/'all-terminal-records.json'))==10
    assert receipt['status']=='incomplete_or_invalid' and receipt['artifacts']==r.work.inventory(directory)


def test_source_guard_requires_exact_head_clean_source_and_live_remote(tmp_path,monkeypatch):
    plan={'source_files':[]};head='a'*40;root=tmp_path/'repo';root.mkdir();p=root/'plan.json';p.write_text('{}')
    monkeypatch.setattr(r,'ROOT',root)
    def git(*args):
        if args[0]=='status':return ''
        if args==('rev-parse','HEAD'):return head
        if args==('branch','--show-current'):return 'codex/test'
        return head+'\trefs/heads/codex/test'
    monkeypatch.setattr(r.legacy,'git',git)
    monkeypatch.setattr(r.subprocess,'check_output',lambda *a,**k:b'{}')
    assert r.source_guard(plan,p,head)['source_commit']==head
    with pytest.raises(ValueError):r.source_guard(plan,p,'b'*40)
    monkeypatch.setattr(r.legacy,'git',lambda *a:'dirty' if a[0]=='status' else git(*a))
    with pytest.raises(ValueError):r.source_guard(plan,p,head)
    monkeypatch.setattr(r.legacy,'git',lambda *a:'b'*40+'\tx' if a[0]=='ls-remote' else git(*a))
    with pytest.raises(ValueError):r.source_guard(plan,p,head)


def test_inventory_covers_nested_receipt_names_and_rejects_aliases(tmp_path):
    (tmp_path/'nested').mkdir();(tmp_path/'nested/receipt.json').write_text('{}')
    (tmp_path/'receipt.json').write_text('root excluded')
    assert [r['path'] for r in r.inventory(tmp_path)]==['nested/receipt.json']
    (tmp_path/'alias').symlink_to(tmp_path/'nested/receipt.json')
    with pytest.raises(ValueError):r.inventory(tmp_path)


def test_run_invalid_terminal_census_still_preserves_receipt(tmp_path,monkeypatch):
    plan=synthetic_plan();planpath=tmp_path/'plan.json';planpath.write_text(json.dumps(plan));head='a'*40
    monkeypatch.setattr(r,'validate',lambda *a,**k:None)
    monkeypatch.setattr(r,'source_guard',lambda *a:dict(source_commit=head,pushed_branch='codex/test'))
    monkeypatch.setattr(r.importlib.metadata,'version',lambda name:r.CONTRACT['required_packages'][name])
    monkeypatch.setattr(r.legacy,'git',lambda *args:head if args==('rev-parse','HEAD') else '')
    monkeypatch.setattr(r.original,'resources',lambda *a:dict(available_ram_bytes=10**12,disk_free_bytes=10**12))
    monkeypatch.setattr(r,'load_original',lambda p:({}, {}, {}))
    monkeypatch.setattr(r.work,'build_libraries',lambda *a:{})
    monkeypatch.setattr(r.work,'execute',lambda *a,**k:[dict(id='unplanned',status='failed')])
    result=r.run(planpath,tmp_path/'raw','attempt-001',head)
    path=Path(result['directory'])
    assert result['error'] and (path/'invalid-terminal-records.json').exists()
    assert r.read(path/'receipt.json')['artifacts']==r.inventory(path)
    assert len(r.read(path/'all-terminal-records.json'))==10


def test_actual_run_saved_packet_verify_path_with_synthetic_workers(tmp_path,monkeypatch):
    """Exercise real controller/receipt/verify; only worker and provenance mocked."""
    plan=synthetic_plan();planpath=tmp_path/'plan.json';planpath.write_text(json.dumps(plan));head='a'*40
    contexts={case:synthetic_context() for case in r.CASE_IDS};old_records={};old_arrays={}
    for case in plan['selected_cases']:
        for method in r.OLD_METHODS:
            task=dict(id=case['id']+'--'+method['id'],case=case,kind='preview',method=method)
            row,state=save_row(task,tmp_path/'old',contexts[case['id']]);old_records[row['id']]=row;old_arrays[row['id']]=state
    monkeypatch.setattr(r,'validate',lambda *a,**k:None)
    monkeypatch.setattr(r,'source_guard',lambda *a:dict(source_commit=head,pushed_branch='codex/test'))
    monkeypatch.setattr(r.importlib.metadata,'version',lambda name:r.CONTRACT['required_packages'][name])
    monkeypatch.setattr(r.legacy,'git',lambda *args:head if args==('rev-parse','HEAD') else '')
    monkeypatch.setattr(r.original,'resources',lambda *a:dict(available_ram_bytes=10**12,disk_free_bytes=10**12))
    monkeypatch.setattr(r,'load_original',lambda p:(contexts,old_records,old_arrays))
    monkeypatch.setattr(r.work,'build_libraries',lambda *a:{})
    monkeypatch.setattr(r.work,'identity',lambda pid:None)
    def synthetic_workers(specs,contexts,directory,libraries,*,workers,deadline,per_task):
        assert workers==4 and per_task==30 and 0<deadline-time.monotonic()<=120
        directory=Path(directory);directory.mkdir();(directory/'records').mkdir();rows=[];owned=[]
        for index,task in enumerate(specs):
            row,_=save_row(task,directory,contexts[task['case']['id']]);row['pid']=1000+index
            process=dict(task_id=task['id'],pid=row['pid'],identity=dict(pid=row['pid'],start_ticks=str(index),boot_id='synthetic'),started_at='synthetic')
            owned.append(process);rows.append(row)
            r.work.write(directory/'records'/f'{row["id"]}.json',row)
            r.work.append(directory/'processes.jsonl',process);r.work.append(directory/'journal.jsonl',row)
        r.work.write(directory/'terminal-records.json',rows)
        r.work.write(directory/'cleanup.json',dict(workers=[dict(**p,current_identity=None,original_identity_absent=True) for p in owned],all_registered_workers_joined=True,all_original_identities_absent=True))
        return rows
    monkeypatch.setattr(r.work,'execute',synthetic_workers)
    monkeypatch.setattr(r.subprocess,'check_output',lambda *a,**k:planpath.read_bytes())
    result=r.run(planpath,tmp_path/'raw','attempt-001',head)
    assert result['complete'] and result['error'] is None
    verified=r.verify(result['directory'])
    assert verified['saved_array_replay'] and verified['complete'] and verified['new_raw_scores_replayed']==10
    assert verified['comparisons']==34 and verified['summary_replay']=='exact_full_saved_summary'
    assert verified['all_original_identities_absent_now']
    # Rehashed duplicate stage rows cannot collapse through dictionary equality.
    directory=Path(result['directory']);terminalfile=directory/'previews/terminal-records.json';journalfile=directory/'previews/journal.jsonl'
    original_terminal=terminalfile.read_text();original_journal=journalfile.read_text();original_receipt=(directory/'receipt.json').read_text()
    duplicated=json.loads(original_terminal);duplicated.append(duplicated[0]);terminalfile.write_text(json.dumps(duplicated))
    journalfile.write_text(''.join(json.dumps(row)+'\n' for row in duplicated))
    packet=json.loads(original_receipt);packet['artifacts']=r.inventory(directory);packet['artifact_bytes']=sum(x['bytes'] for x in packet['artifacts']);(directory/'receipt.json').write_text(json.dumps(packet))
    with pytest.raises(ValueError,match='cardinality'):r.verify(directory)
    terminalfile.write_text(original_terminal);journalfile.write_text(original_journal);(directory/'receipt.json').write_text(original_receipt)
    # Scores can be complete while source postflight invalidates the attempt.
    summaryfile=directory/'summary.json';postfile=directory/'postflight.json'
    original_summary=summaryfile.read_text();original_post=postfile.read_text()
    packet=json.loads(original_summary);packet.update(postflight_error='synthetic drift',source_valid=False);summaryfile.write_text(json.dumps(packet))
    postfile.write_text(json.dumps(dict(error='synthetic drift',source_bindings_unchanged=False)))
    packet=json.loads(original_receipt);packet.update(status='incomplete_or_invalid',artifacts=r.inventory(directory));packet['artifact_bytes']=sum(x['bytes'] for x in packet['artifacts']);(directory/'receipt.json').write_text(json.dumps(packet))
    assert not r.verify(directory)['complete']
    summaryfile.write_text(original_summary);postfile.write_text(original_post);(directory/'receipt.json').write_text(original_receipt)
    # Immutable raw receipt rejects an extra artifact, not just hash changes.
    (Path(result['directory'])/'extra.json').write_text('{}')
    with pytest.raises(ValueError,match='artifact census'):r.verify(result['directory'])


def test_producer_cancellation_crosses_per_record_oserror_handler(synthetic_evidence,monkeypatch):
    monkeypatch.setattr(r,'_row',lambda *a,**k:(_ for _ in ()).throw(r.ProducerCancelled('synthetic alarm')))
    with pytest.raises(r.ProducerCancelled):r.summarize(*synthetic_evidence)


@pytest.mark.parametrize('tamper',['identity','placeholder','error','postflight','receipt','dispatched_without_cleanup','gitplan'])
def test_partial_attempt_verifier_checks_full_contract(tmp_path,monkeypatch,tamper):
    plan=synthetic_plan();planpath=tmp_path/'plan.json';planpath.write_text(json.dumps(plan));head='a'*40
    monkeypatch.setattr(r,'validate',lambda *a,**k:None)
    monkeypatch.setattr(r,'source_guard',lambda *a:dict(source_commit=head,pushed_branch='codex/test'))
    monkeypatch.setattr(r.importlib.metadata,'version',lambda name:r.CONTRACT['required_packages'][name])
    monkeypatch.setattr(r.legacy,'git',lambda *args:head if args==('rev-parse','HEAD') else '')
    monkeypatch.setattr(r.original,'resources',lambda *a:dict(available_ram_bytes=0,disk_free_bytes=10**12))
    monkeypatch.setattr(r.subprocess,'check_output',lambda *a,**k:planpath.read_bytes())
    result=r.run(planpath,tmp_path/'raw','attempt-001',head);directory=Path(result['directory'])
    verified=r.verify(directory);assert verified['partial_attempt'] and verified['comparisons']==34
    if tamper=='identity':
        file=directory/'all-terminal-records.json';packet=r.read(file);packet[0]['method']='wrong';file.write_text(json.dumps(packet))
    elif tamper in ('placeholder','error','postflight'):
        file=directory/'summary.json';packet=r.read(file)
        if tamper=='placeholder':packet['comparisons'].pop()
        elif tamper=='error':packet['error']=None
        else:packet['postflight_error']='forged'
        file.write_text(json.dumps(packet))
    elif tamper=='dispatched_without_cleanup':
        (directory/'previews').mkdir();(directory/'previews/processes.jsonl').write_text('{}\n')
    elif tamper=='gitplan':monkeypatch.setattr(r.subprocess,'check_output',lambda *a,**k:b'{}')
    receipt=r.read(directory/'receipt.json');receipt['artifacts']=r.inventory(directory);receipt['artifact_bytes']=sum(x['bytes'] for x in receipt['artifacts'])
    if tamper=='receipt':receipt['status']='complete'
    (directory/'receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError):r.verify(directory)
