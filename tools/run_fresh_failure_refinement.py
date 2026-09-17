#!/usr/bin/env python3
"""Outcome-selected EXP013 numerical diagnosis. Only run() acquires new traces.

Reuse frozen CPU acquisition/scoring/lifecycle without changing original raw,
labels, policies, gates or cohort statistics. All output is exclusive/external.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import subprocess
import sys
import time

import fresh_preview_runtime as work  # Sets BLAS/OpenMP limits before NumPy.
import run_hopfield_refinement as legacy
import run_hopfield as base
import run_fresh_preview as original
from hopfield_probe_selector import direction_candidates
import numpy as np

ROOT=work.ROOT
OLD=Path('/data2/neural-shrimp-data/raw/EXP-013/attempt-002')
OLD_RECEIPT_SHA='058aa80de824ecdfa00d8abc77d85ff231353a2af60e8db932f39eb07936552e'
OLD_SOURCE='e5864f93ec6e822ba4fc6670640149f4a7c07b7b'
CASE_IDS=('fresh-11-p1350-r0.016-c07','fresh-11-p1350-sham')
METHODS=[dict(id=f'rk4-{dt:g}',kind='RK4',dt=dt) for dt in (.000625,.0003125,.00015625)]+[
    dict(id='dop853-1e-12',kind='DOP853',rtol=1e-12,atol=1e-14,max_step=.025),
    dict(id='dop853-1e-13',kind='DOP853',rtol=1e-13,atol=1e-15,max_step=.0125)]
OLD_METHODS=work.METHODS[:3]
NEW_SOURCE_PATHS=('tools/run_fresh_failure_refinement.py','tests/test_fresh_failure_refinement.py',
    'docs/EXP-013-refinement-protocol.md')
CONTRACT=dict(schema_version='EXP013-refinement-v1',experiment_id='EXP-013-refinement',
    outcome_selected=True,new_traces=10,cases=2,methods=METHODS,duration=75.,sample_dt=.1,samples=751,
    phase_grid=128,workers=4,thread_limit=1,wall_clock_ceiling_seconds=120,per_trace_seconds=30,
    ram_reserve_bytes=8*1024**3,disk_reserve_bytes=2*1024**3,
    gates=dict(rmse_absolute_difference=1e-4,activation_rms_difference=1e-3,activation_max_difference=1e-2),
    required_packages=dict(numpy='2.5.3',scipy='1.18.1'),paid_compute=False,paid_review=False)
class ProducerCancelled(BaseException):
    """Whole-attempt cancellation must cross per-record OSError handlers."""


SCOPE=('Outcome-selected numerical diagnosis of one EXP013 failed case and its preselected matched sham. '
       'Original failed waveform gate remains failed; no policy refit, label replacement, new origin, '
       'cohort geometry means, deployment inference or retrospective qualification of EXP013.')


def read(path):return json.loads(Path(path).read_text())

def inventory(directory):
    directory=Path(directory);result=[]
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():raise ValueError('Symlink is not an owned immutable artifact')
        if path.is_file() and path!=directory/'receipt.json':result.append(_binding(path,directory))
    return result


def _binding(path,root):return dict(path=str(Path(path).relative_to(root)),sha256=legacy.sha(path),bytes=Path(path).stat().st_size)

def _source_file(relative):
    # Draft preparation only: exact future source path is still recorded. run()
    # insists every source exists at ROOT and is committed at expected HEAD.
    target=ROOT/relative
    local=Path(__file__).resolve().parent/Path(relative).name
    if ROOT not in Path(__file__).resolve().parents and relative in NEW_SOURCE_PATHS and local.exists():return local
    if target.exists():return target
    if relative in NEW_SOURCE_PATHS and local.exists():return local
    raise FileNotFoundError(target)


def _old_paths():
    paths=['manifest.json','receipt.json','numerical-audit.json','policy-seal.json','winner-seal.json',
           'access-audit.json','postflight.json','baselines/records/fresh-11.json',
           'baselines/payloads/fresh-11/baselines/fresh-11/result.json']
    for case in CASE_IDS:
        for method in OLD_METHODS:
            ident=case+'--'+method['id']
            paths.extend([f'previews/records/{ident}.json',f'previews/payloads/{ident}/trace.npz'])
    return paths


def _validate_old_inventory():
    if legacy.sha(OLD/'receipt.json')!=OLD_RECEIPT_SHA:raise ValueError('Original receipt hash differs')
    receipt=read(OLD/'receipt.json');items=receipt['artifacts'];index={x['path']:x for x in items}
    if len(items)!=len(index) or receipt['source_commit']!=OLD_SOURCE:raise ValueError('Original receipt census/source differs')
    for relative in _old_paths():
        if relative=='receipt.json':continue
        if _binding(OLD/relative,OLD)!=index.get(relative):raise ValueError('Original receipt artifact binding differs: '+relative)
    manifest=read(OLD/'manifest.json')
    if manifest['source_commit']!=OLD_SOURCE or manifest['plan_sha256']!=receipt['plan_sha256']:
        raise ValueError('Original manifest source/plan identity differs')
    return manifest


def build_plan():
    manifest=_validate_old_inventory();old_plan=manifest['plan']
    cases={c['id']:c for c in old_plan['census']['cases']+old_plan['census']['shams']}
    selected=[cases[key] for key in CASE_IDS]
    if (selected[0]['candidate']!=7 or selected[0]['radius']!=.016 or selected[1]['radius']!=0.
            or any(c['phase']!=1350 or c['anchor']!='fresh-11' for c in selected)):
        raise ValueError('Exact outcome-selected case/sham identity differs')
    audit={r['id']:r for r in read(OLD/'numerical-audit.json')['details']}
    if audit[CASE_IDS[0]]['native_waveform_pass'] is not False or audit[CASE_IDS[1]]['qualified'] is not True:
        raise ValueError('Frozen selection reason differs')
    source_paths=list(dict.fromkeys([r['path'] for r in old_plan['source_files']]+list(NEW_SOURCE_PATHS)))
    sources=[dict(path=p,sha256=legacy.sha(_source_file(p)),bytes=_source_file(p).stat().st_size) for p in source_paths]
    anchor=next(a for a in old_plan['selection']['anchors'] if a['id']=='fresh-11')
    return dict(**CONTRACT,scope=SCOPE,original_directory=str(OLD),original_source=OLD_SOURCE,
        original_receipt_sha256=OLD_RECEIPT_SHA,selected_cases=selected,anchor=anchor,
        original_numerical_gates={key:audit[key] for key in CASE_IDS},source_files=sources,
        input_files=[_binding(OLD/p,OLD) for p in _old_paths()],
        comparison_counts=dict(all_new_method_pairs=20,old_fine_against_new=10,original_adjacent_native=4,
                               unique_pairs=34,new_native_adjacent_subset=6),
        failure_policy='Retain all10 planned terminal statuses and34 planned comparisons; no retry/replacement.',
        selection_reason='Sole EXP013 candidate failing original native waveform gates; fixed same-origin/phase zero-dose sham.')


def tasks(plan):
    return [dict(id=case['id']+'--'+method['id'],kind='preview',case=case,method=method)
            for case in plan['selected_cases'] for method in plan['methods']]


def validate(plan,*,bindings=True,reconstruct=False):
    if any(plan.get(k)!=v for k,v in CONTRACT.items()) or plan.get('scope')!=SCOPE:
        raise ValueError('Exact refinement contract differs')
    if [c['id'] for c in plan['selected_cases']]!=list(CASE_IDS) or len(tasks(plan))!=10:
        raise ValueError('Exact two-case/ten-trace census differs')
    if plan['original_directory']!=str(OLD) or plan['original_source']!=OLD_SOURCE or plan['original_receipt_sha256']!=OLD_RECEIPT_SHA:
        raise ValueError('Original attempt identity differs')
    if [x['path'] for x in plan['input_files']]!=_old_paths():raise ValueError('Original input path census differs')
    if bindings:
        for item in plan['input_files']:
            if _binding(OLD/item['path'],OLD)!=item:raise ValueError('Original input drift: '+item['path'])
        for item in plan['source_files']:
            path=_source_file(item['path'])
            if legacy.sha(path)!=item['sha256'] or path.stat().st_size!=item['bytes']:raise ValueError('Source drift: '+item['path'])
        _validate_old_inventory()
    if reconstruct and plan!=build_plan():raise ValueError('Exact plan reconstruction differs')
    return dict(cases=2,new_traces=10,old_raw_traces=6,comparisons=34)


def source_guard(plan,plan_path,expected_head):
    if not re.fullmatch('[0-9a-f]{40}',expected_head):raise ValueError('Exact expected frozen commit required')
    if legacy.git('status','--porcelain','--untracked-files=all'):raise ValueError('Clean source checkout required')
    head=legacy.git('rev-parse','HEAD');branch=legacy.git('branch','--show-current')
    if head!=expected_head or not branch.startswith('codex/'):raise ValueError('Expected source branch/HEAD differs')
    relative=str(Path(plan_path).resolve().relative_to(ROOT))
    for rel in [r['path'] for r in plan['source_files']]+[relative]:
        data=subprocess.check_output(['git','-C',str(ROOT),'show',f'{head}:{rel}'])
        if hashlib.sha256(data).hexdigest()!=legacy.sha(ROOT/rel):raise ValueError('Uncommitted source: '+rel)
    if legacy.git('ls-remote','--exit-code','origin',f'refs/heads/{branch}').split()[0]!=head:
        raise ValueError('Exact private source freeze has not been pushed')
    return dict(source_commit=head,pushed_branch=branch,remote_head_verified_at=legacy.now())


def _raw(path,context):
    with np.load(path,allow_pickle=False) as z:
        if set(z.files)!={'times','states','initial','alpha','gamma','delta'}:raise ValueError('Raw array member census differs')
        arrays={k:z[k].copy() for k in z.files}
    for name,array in arrays.items():
        if array.dtype!=np.float64 or not np.isfinite(array).all():raise ValueError('Raw dtype/nonfinite: '+name)
    if arrays['times'].shape!=(751,) or not np.array_equal(arrays['times'],work.TIMES) or arrays['states'].shape!=(751,3):
        raise ValueError('Raw shape/sample grid differs')
    for name in ('initial','alpha','gamma','delta'):
        if not np.array_equal(arrays[name],context[name]) or arrays[name].shape!=np.shape(context[name]):
            raise ValueError('Realized intervention differs: '+name)
    if not np.array_equal(arrays['states'][0],context['initial']):raise ValueError('Initial saved state differs')
    return arrays['states']


def _row(task,row,directory,context,*,replay=True):
    expected=dict(id=task['id'],kind='preview',case_id=task['case']['id'],method=task['method']['id'])
    if any(type(row.get(k)) is not type(v) or row.get(k)!=v for k,v in expected.items()):raise ValueError('Terminal task identity differs')
    if row['status']!='complete':raise ValueError('Trace did not complete: '+str(row['status']))
    method=task['method'];solver=row['solver']
    if any(type(solver.get(k)) is not type(v) or solver.get(k)!=v for k,v in method.items()):raise ValueError('Solver metadata differs')
    if method['kind']=='RK4':
        if type(solver.get('steps')) is not int or solver['steps']!=round(75/method['dt']):raise ValueError('Native step count differs')
    elif solver.get('success') is not True or type(solver.get('nfev')) is not int or solver['nfev']<0:
        raise ValueError('Independent solver status/work differs')
    raw=f'payloads/{task["id"]}/trace.npz'
    if row['raw']!=raw:raise ValueError('Raw path differs')
    path=Path(directory)/raw
    if legacy.sha(path)!=row['raw_sha256'] or path.stat().st_size!=row['raw_bytes']:raise ValueError('Raw hash/bytes differ')
    states=_raw(path,context)
    if replay:
        scored=work.score(context,work.TIMES,states)
        legacy.compare_scoring({k:row[k] for k in scored},scored)
    return states


def load_original(plan):
    packet=read(OLD/'baselines/payloads/fresh-11/baselines/fresh-11/result.json');cert=packet['certificate']
    if not work.certificate_ok(cert,plan['anchor']):raise ValueError('Original unchanged baseline certificate invalid')
    contexts={};records={};states={}
    for case in plan['selected_cases']:
        delta=np.zeros(2) if case['radius']==0 else case['radius']*np.asarray(direction_candidates('uniform_16')[case['candidate']]['direction'])
        parameters=np.array([plan['anchor']['alpha'],plan['anchor']['gamma']])+delta
        context=dict(alpha=float(parameters[0]),gamma=float(parameters[1]),delta=np.asarray(delta),
            initial=np.asarray(cert['orbit_states'],dtype=np.float64)[case['phase']].copy(),
            reference=np.asarray(cert['orbit_activations'],dtype=np.float64)[:-1],period=cert['period'],epoch=case['phase']*cert['period']/2700)
        contexts[case['id']]=context
        for method in OLD_METHODS:
            task=dict(id=case['id']+'--'+method['id'],kind='preview',case=case,method=method)
            row=read(OLD/'previews/records'/f'{task["id"]}.json')
            states[task['id']]=_row(task,row,OLD/'previews',context);records[task['id']]=row
    return contexts,records,states


def comparison_specs(plan):
    result=[]
    for case in plan['selected_cases']:
        new=[m['id'] for m in METHODS];old=[m['id'] for m in OLD_METHODS]
        groups=[('all_new_method_pairs',list(itertools.combinations(new,2))),
                ('old_fine_against_new',[(old[-1],m) for m in new]),
                ('original_adjacent_native',list(zip(old[:-1],old[1:])))]
        for group,pairs in groups:
            for a,b in pairs:result.append(dict(case_id=case['id'],group=group,pair=[a,b]))
    return result


def compare_pair(spec,records,arrays,context):
    ids=[spec['case_id']+'--'+m for m in spec['pair']]
    result=dict(**spec,ids=ids,complete=False,passed=False,error=None)
    try:
        if any(i not in arrays or records[i]['status']!='complete' for i in ids):raise ValueError('Missing/invalid required trace')
        a,b=[arrays[i] for i in ids];scores=[records[i]['rmse'] for i in ids]
        if any(type(x) not in (float,int) or not np.isfinite(x) for x in scores):raise ValueError('Invalid scalar score')
        delta=np.tanh(a)-np.tanh(b);rms=float(np.sqrt(np.mean(delta**2)));maximum=float(np.max(np.abs(delta)))
        difference=abs(scores[0]-scores[1]);location=np.unravel_index(np.abs(delta).argmax(),delta.shape)
        phase=records[spec['case_id']+'--rk4-0.00125']['metrics']['phase_offset_cycles']
        common=work.periodic_interpolate(context['reference'],(work.TIMES+context['epoch'])/context['period']+phase)
        values=[float(np.sqrt(np.mean((np.tanh(x)-common)**2))) for x in (a,b)]
        result.update(complete=True,rmse_difference=difference,activation_rms_difference=rms,activation_max_difference=maximum,
            activation_component_rms=np.sqrt(np.mean(delta**2,axis=0)).tolist(),
            maximum_location=[int(v) for v in location],maximum_time=float(work.TIMES[location[0]]),
            common_original_fine_phase_cycles=phase,common_phase_rmse=values,
            potential_rms_difference=float(np.sqrt(np.mean((a-b)**2))),potential_max_difference=float(np.max(np.abs(a-b))),
            scalar_pass=difference<=1e-4,rms_pass=rms<=1e-3,maximum_pass=maximum<=1e-2)
        result['passed']=result['scalar_pass'] and result['rms_pass'] and result['maximum_pass']
    except (ValueError,KeyError,TypeError) as error:result['error']=repr(error)
    return result


def summarize(plan,records,directory,contexts,old_records,old_arrays):
    expected=tasks(plan);indexed={};invalid=[]
    for row in records:
        ident=row.get('id') if isinstance(row,dict) else None
        if ident not in {t['id'] for t in expected} or ident in indexed:invalid.append(dict(id=ident,error='unknown_or_duplicate_record'));continue
        indexed[ident]=row
    all_records=dict(old_records);arrays=dict(old_arrays);terminal=[]
    for task in expected:
        row=indexed.get(task['id']);status='missing' if row is None else row.get('status','invalid')
        result=dict(id=task['id'],case_id=task['case']['id'],method=task['method']['id'],status=status,raw_score_replay=False)
        try:
            if row is None:raise ValueError('Missing planned terminal trace')
            arrays[task['id']]=_row(task,row,Path(directory)/'previews',contexts[task['case']['id']])
            all_records[task['id']]=row;result['raw_score_replay']=True
        except (ValueError,KeyError,TypeError,OSError) as error:
            result.update(status='invalid_or_incomplete',error=repr(error))
        terminal.append(result)
    comparisons=[compare_pair(spec,all_records,arrays,contexts[spec['case_id']]) for spec in comparison_specs(plan)]
    cases={}
    for case in plan['selected_cases']:
        rows=[r for r in comparisons if r['case_id']==case['id']]
        new=[r for r in rows if r['group']=='all_new_method_pairs'];oldfine=[r for r in rows if r['group']=='old_fine_against_new']
        chain=['rk4-0.00125']+[m['id'] for m in METHODS[:3]]
        adjacent=[next(r for r in rows if r['pair']==[a,b]) for a,b in zip(chain[:-1],chain[1:])]
        cases[case['id']]=dict(all_new_methods_agree=not invalid and all(r['passed'] for r in new),
            original_fine_agrees_with_all_new=not invalid and all(r['passed'] for r in oldfine),
            original_fine_resolved_by_diagnostic=not invalid and all(r['passed'] for r in new+oldfine),
            native_adjacent=adjacent,original_numerical_gates=plan['original_numerical_gates'][case['id']])
    complete=all(r['raw_score_replay'] for r in terminal) and not invalid
    return dict(experiment_id=CONTRACT['experiment_id'],complete=complete,terminal=terminal,
        invalid_records=invalid,comparisons=comparisons,cases=cases,scope=SCOPE,
        required_counts=plan['comparison_counts'],all_original_labels_and_qualifications_unchanged=True)


def external_attempt(data_root,attempt):
    if not re.fullmatch('attempt-[0-9]{3}',attempt):raise ValueError('Named immutable attempt required')
    target=(Path(data_root)/CONTRACT['experiment_id']/attempt).resolve()
    if target==ROOT or ROOT in target.parents:raise ValueError('All output must be external to repository')
    target.parent.mkdir(parents=True,exist_ok=True);target.mkdir()
    return target


def _fill(plan,records,reason):
    result=[];index={r['id']:r for r in records}
    if len(index)!=len(records) or set(index)-{t['id'] for t in tasks(plan)}:
        raise ValueError('Cannot silently omit duplicate or unknown terminal records')
    for task in tasks(plan):result.append(index.get(task['id'],dict(id=task['id'],case_id=task['case']['id'],kind='preview',method=task['method']['id'],status=reason,rmse=None)))
    return result


def unavailable_summary(plan,records):
    return dict(experiment_id=CONTRACT['experiment_id'],complete=False,scope=SCOPE,
        terminal=[dict(id=r['id'],status=r['status']) for r in records],required_counts=plan['comparison_counts'],
        comparisons=[dict(**spec,complete=False,passed=False,error='producer_failure') for spec in comparison_specs(plan)],
        all_original_labels_and_qualifications_unchanged=True)


def run(plan_path,data_root,attempt,expected_head):
    plan=read(plan_path);validate(plan,reconstruct=True);provenance=source_guard(plan,plan_path,expected_head)
    if any(importlib.metadata.version(k)!=v for k,v in CONTRACT['required_packages'].items()):raise ValueError('Unqualified numerical versions')
    directory=external_attempt(data_root,attempt);start=time.monotonic();deadline=start+120
    manifest=dict(**provenance,plan=plan,plan_sha256=legacy.sha(plan_path),started_at=legacy.now(),identity=work.identity(os.getpid()),
        executable=sys.executable,argv=list(sys.argv),cwd=str(Path.cwd()),python=sys.version,platform=platform.platform(),
        packages={k:importlib.metadata.version(k) for k in CONTRACT['required_packages']},
        thread_env={k:os.environ.get(k) for k in legacy.THREAD_ENV})
    work.write(directory/'manifest.json',manifest);records=[];error=None;summary=None;contexts={};old_records={};old_arrays={}
    handlers={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM)}
    def stop(signum,frame):raise ProducerCancelled('Frozen120-second deadline or external interruption')
    for sig in handlers:signal.signal(sig,stop)
    signal.setitimer(signal.ITIMER_REAL,120.)
    try:
        resource=original.resources(directory);work.write(directory/'resources.json',resource)
        if resource['available_ram_bytes']<CONTRACT['ram_reserve_bytes'] or resource['disk_free_bytes']<CONTRACT['disk_reserve_bytes']:
            raise RuntimeError('Shared resource reserve unavailable')
        contexts,old_records,old_arrays=load_original(plan)
        libraries=work.build_libraries(directory/'build',deadline)
        records=work.execute(tasks(plan),contexts,directory/'previews',libraries,workers=4,deadline=deadline,per_task=30)
        summary=summarize(plan,records,directory,contexts,old_records,old_arrays)
    except BaseException as exc:
        error=repr(exc);work.write(directory/'producer-error.json',dict(error=error,at=legacy.now()))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig,handler in handlers.items():signal.signal(sig,handler)
        if not records and (directory/'previews/terminal-records.json').exists():records=read(directory/'previews/terminal-records.json')
        try:records=_fill(plan,records,'not_started_prerequisite_or_producer_failure')
        except (ValueError,KeyError,TypeError) as exc:
            work.write(directory/'invalid-terminal-records.json',dict(records=records,error=repr(exc)))
            error=error or repr(exc);summary=None
            records=_fill(plan,[],'invalid_terminal_census')
        work.write(directory/'all-terminal-records.json',records)
        if summary is None:summary=unavailable_summary(plan,records)
        postflight_error=None
        try:
            validate(plan,reconstruct=False)
            if legacy.git('rev-parse','HEAD')!=expected_head or legacy.git('status','--porcelain','--untracked-files=all'):
                raise ValueError('Source checkout changed during attempt')
        except BaseException as exc:postflight_error=repr(exc)
        work.write(directory/'postflight.json',dict(error=postflight_error,source_bindings_unchanged=postflight_error is None))
        summary.update(error=error,postflight_error=postflight_error,source_valid=postflight_error is None,
            status_counts=dict(Counter(r['status'] for r in records)),producer_and_analysis_seconds=time.monotonic()-start,ended_at=legacy.now())
        work.write(directory/'summary.json',summary)
        artifacts=inventory(directory)
        receipt=dict(**provenance,experiment_id=CONTRACT['experiment_id'],plan_sha256=legacy.sha(plan_path),
            status='complete' if summary['complete'] and error is None and postflight_error is None else 'incomplete_or_invalid',
            artifacts=artifacts,artifact_bytes=sum(r['bytes'] for r in artifacts),inventory_sealed_at=legacy.now(),directory=str(directory))
        work.write(directory/'receipt.json',receipt)
    return dict(directory=str(directory),complete=summary['complete'],error=error,postflight_error=postflight_error,
                receipt_sha256=legacy.sha(directory/'receipt.json'))


def verify(directory):
    """Full complete/partial saved-evidence replay, with no new integrations."""
    directory=Path(directory).resolve();receipt=read(directory/'receipt.json');manifest=read(directory/'manifest.json');plan=manifest['plan']
    if receipt['artifacts']!=inventory(directory):raise ValueError('Immutable receipt artifact census/hash differs')
    if (receipt['source_commit']!=manifest['source_commit'] or receipt['plan_sha256']!=manifest['plan_sha256']
            or receipt.get('experiment_id')!=CONTRACT['experiment_id']):
        raise ValueError('Receipt/manifest source, plan or experiment mismatch')
    source=receipt['source_commit']
    if not re.fullmatch('[0-9a-f]{40}',source):raise ValueError('Invalid frozen source commit')
    committed=subprocess.check_output(['git','-C',str(ROOT),'show',f'{source}:experiments/hopfield/EXP-013-refinement-plan.json'])
    if hashlib.sha256(committed).hexdigest()!=receipt['plan_sha256'] or json.loads(committed)!=plan:
        raise ValueError('Committed original plan differs')
    for item in plan.get('source_files',[]):
        content=subprocess.check_output(['git','-C',str(ROOT),'show',f'{source}:{item["path"]}'])
        if len(content)!=item['bytes'] or hashlib.sha256(content).hexdigest()!=item['sha256']:
            raise ValueError('Original committed source binding differs: '+item['path'])
    validate(plan,reconstruct=True)
    rows=read(directory/'all-terminal-records.json');specs=tasks(plan);index={t['id']:t for t in specs}
    if len(rows)!=10 or {r.get('id') for r in rows}!=set(index):raise ValueError('Exact terminal census differs')
    for row in rows:
        task=index[row['id']]
        required=dict(id=task['id'],kind='preview',case_id=task['case']['id'],method=task['method']['id'])
        if any(type(row.get(k)) is not type(v) or row.get(k)!=v for k,v in required.items()):raise ValueError('Terminal case/method identity differs')
        if type(row.get('status')) is not str or not row['status']:raise ValueError('Terminal status missing')
    saved=read(directory/'summary.json');post=read(directory/'postflight.json')
    producer=read(directory/'producer-error.json') if (directory/'producer-error.json').exists() else None
    invalid=read(directory/'invalid-terminal-records.json') if (directory/'invalid-terminal-records.json').exists() else None
    expected_error=producer['error'] if producer else invalid['error'] if invalid else None
    if saved.get('error')!=expected_error:raise ValueError('Producer error conclusion differs')
    if (saved.get('postflight_error')!=post['error'] or saved.get('source_valid') is not (post['error'] is None)
            or post.get('source_bindings_unchanged') is not (post['error'] is None)):
        raise ValueError('Postflight source conclusion differs')
    if saved.get('status_counts')!=dict(Counter(r['status'] for r in rows)):raise ValueError('Terminal status counts differ')
    elapsed=saved.get('producer_and_analysis_seconds')
    if type(elapsed) not in (int,float) or not np.isfinite(elapsed) or elapsed<0:raise ValueError('Invalid elapsed-time receipt')
    if type(saved.get('complete')) is not bool:raise ValueError('Strict completion flag required')
    expected_status='complete' if saved['complete'] and expected_error is None and post['error'] is None else 'incomplete_or_invalid'
    if receipt['status']!=expected_status or receipt['artifact_bytes']!=sum(r['bytes'] for r in receipt['artifacts']):
        raise ValueError('Receipt conclusion/byte count differs')
    placeholder=unavailable_summary(plan,rows)
    placeholder_matches=expected_error is not None and saved['complete'] is False and {key:saved.get(key) for key in placeholder}==placeholder
    stage=directory/'previews';processes=[]
    if stage.exists():
        # Even a failed/dispatched worker needs cleanup. No early return based
        # solely on absence of a completed trace.
        if not (stage/'cleanup.json').exists():raise ValueError('Dispatched stage lacks owned cleanup')
        cleanup=read(stage/'cleanup.json')
        if cleanup['all_registered_workers_joined'] is not True or cleanup['all_original_identities_absent'] is not True:
            raise ValueError('Original owned cleanup incomplete')
        processes=[json.loads(line) for line in (stage/'processes.jsonl').read_text().splitlines()] if (stage/'processes.jsonl').exists() else []
        if (len({p['task_id'] for p in processes})!=len(processes) or any(p['task_id'] not in index for p in processes)
                or len(cleanup['workers'])!=len(processes)):
            raise ValueError('Owned process census differs')
        owned={p['task_id']:p for p in processes}
        for entry,check in zip(processes,cleanup['workers'],strict=True):
            if any(check[k]!=v for k,v in entry.items()) or check['original_identity_absent'] is not True:
                raise ValueError('Cleanup identity differs')
        terminal=read(stage/'terminal-records.json')
        journal=[json.loads(line) for line in (stage/'journal.jsonl').read_text().splitlines()] if (stage/'journal.jsonl').exists() else []
        if terminal!=journal:raise ValueError('Stage terminal journal differs')
        if len(terminal)!=10 or len({r.get('id') for r in terminal})!=10 or {r.get('id') for r in terminal}!=set(index):
            raise ValueError('Exact stage-terminal cardinality/identity differs')
        if invalid is None and {r['id']:r for r in terminal}!={r['id']:r for r in rows}:
            raise ValueError('Stage and parent terminal ledgers differ')
        for row in rows:
            if row.get('pid') is not None or row['status']=='complete':
                if row['id'] not in owned or row.get('pid')!=owned[row['id']]['pid']:
                    raise ValueError('Dispatched terminal trace lacks owned process')
            if row['status']=='complete' and read(stage/'records'/f'{row["id"]}.json')!=row:
                raise ValueError('Completed worker record differs')
    elif any(r.get('pid') is not None or r['status']=='complete' for r in rows) or not placeholder_matches:
        raise ValueError('No acquisition stage requires exact failed full-denominator placeholder')
    if not stage.exists():
        # Still validates every terminal identity,34 comparisons, source/plan,
        # error/postflight/status/receipt fields before returning a partial result.
        return dict(saved_array_replay=False,complete=False,partial_attempt=True,receipt_sha256=legacy.sha(directory/'receipt.json'),
            terminal_status_counts=saved['status_counts'],comparisons=34,reason='No acquisition worker stage started')
    contexts,old_records,old_arrays=load_original(plan);replay=summarize(plan,rows,directory,contexts,old_records,old_arrays)
    if placeholder_matches:summary_replay='verified_failure_placeholder; available arrays independently rescored'
    elif {key:saved.get(key) for key in replay}!=replay:raise ValueError('Full comparison/scoring summary replay differs')
    else:summary_replay='exact_full_saved_summary'
    current=[dict(**p,current_identity=work.identity(p['pid'])) for p in processes]
    return dict(saved_array_replay=True,complete=replay['complete'] and expected_error is None and post['error'] is None,summary_replay=summary_replay,
        new_raw_scores_replayed=sum(t['raw_score_replay'] for t in replay['terminal']),old_raw_scores_replayed=6,comparisons=34,
        receipt_sha256=legacy.sha(directory/'receipt.json'),owned_processes=current,
        all_original_identities_absent_now=all(p['current_identity'] is None or p['current_identity']!=p['identity'] for p in current))


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--out',required=True)
    check=sub.add_parser('validate');check.add_argument('--plan',required=True)
    runp=sub.add_parser('run');runp.add_argument('--plan',required=True);runp.add_argument('--data-root',required=True);runp.add_argument('--attempt',required=True);runp.add_argument('--expected-head',required=True)
    verify_p=sub.add_parser('verify');verify_p.add_argument('--directory',required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        plan=build_plan();work.write(args.out,plan);result=dict(plan=str(args.out),sha256=legacy.sha(args.out))
    elif args.command=='validate':result=validate(read(args.plan),reconstruct=True)
    elif args.command=='run':result=run(args.plan,args.data_root,args.attempt,args.expected_head)
    else:result=verify(args.directory)
    print(json.dumps(base.safe(result),indent=2,allow_nan=False),flush=True)

if __name__=='__main__':main()
