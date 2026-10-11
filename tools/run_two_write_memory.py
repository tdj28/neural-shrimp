#!/usr/bin/env python3
"""EXP-016 lifecycle: freeze, reference seal, complete two-write census, saved replay.

prepare/validate access metadata and hashes only. Neural integration is confined
 to the guarded run command; verify reads immutable arrays without solving.
"""
from __future__ import annotations
import argparse
from collections import Counter
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import resource
import shutil
import signal
import subprocess
import sys
import time

import fresh_preview_runtime as work  # sets one numerical thread before NumPy
import run_hopfield_refinement as legacy
import run_phase_memory_recovery as codec
import two_write_runtime as runtime
import numpy as np
import two_write_acquisition as acquisition
import two_write_evaluation as evaluation

ROOT = work.ROOT
OLD = Path('/data2/neural-shrimp-data/raw/EXP-014/attempt-002')
OLD_SOURCE = '406b89553b956bac1b97ef05413c799e383a9103'
OLD_RECEIPT_SHA = '2494966914cb68d8c5286a5dca6710212ade171ead42c9cfaed82cebd44c12a8'
PLAN_REL = 'experiments/hopfield/EXP-016-plan.json'
REVIEW_REL = 'docs/EXP-016-local-review.json'
NEW_SOURCES = ['tools/run_two_write_memory.py', 'tools/two_write_memory.py',
    'tools/two_write_acquisition.py', 'tools/two_write_evaluation.py', 'tools/two_write_runtime.py',
    'tests/test_two_write_runtime.py',
    'tests/test_two_write_runner.py', 'tests/test_two_write_memory.py',
    'tests/test_two_write_acquisition.py', 'tests/test_two_write_evaluation.py',
    'docs/EXP-016-protocol.md', 'docs/two-write-memory-design.md',
    'docs/consults/2026-09-17-pro-adjudication.md',
    'docs/consults/2026-09-17-pro-receipt.json',
    'docs/consults/2026-09-17-pro-response.md',
    'docs/consults/2026-09-17-two-write-independent-adjudication.md']
SOURCE_PATHS = list(dict.fromkeys(NEW_SOURCES + codec.SOURCE_PATHS))
GIB = 1024**3
CONTRACT = dict(experiment_id='EXP-016', schema='two-write-memory-v1', exploratory=True,
    contexts=48, development_phases=8, heldout_phases=16, histories=1296,
    reference_jobs=96, development_traces=918, heldout_traces=1782,
    traces=2700, total_jobs=2796, comparisons=1404, phase_seed=2026091701,
    phase_rng='numpy.Generator(PCG64)', bootstrap_seed=20260917, bootstrap_draws=20000,
    epsilon=1e-4, sigma=1e-4, windows=['immediate','k2','k4','k8'],
    first_signs=[-1,0,1], second_signs=[-1,0,1], arms=['identity','timing','transverse'],
    native_steps_per_period=[32768,65536], samples_per_period=4096,
    reference_methods=['dop853','rk45'], independent_panel_phases=[0,8],
    workers=4, thread_limit=1, producer_seconds=3600, per_reference_seconds=90,
    per_trace_seconds=60, sealing_seconds=180, launch_ram_bytes=16*GIB,
    reserve_ram_bytes=8*GIB, owned_rss_bytes=12*GIB, raw_budget_bytes=8*GIB,
    potential_threshold=1e-6, activation_threshold=1e-7,
    standardized_margin_threshold=.01, accuracy_threshold=.001,
    adjoint_normalization_threshold=1e-7, paid_compute=False, paid_review=False,
    required_packages=dict(numpy='2.5.3',scipy='1.18.1'),
    scope='Prespecified linear observer, known phase and external clock, two selected fixed-weight cycles; no capacity, information-erasure, geometry-causal or trained-network claim.')

class ProducerStop(BaseException):
    """Parent deadline/resource cancellation bypasses per-case handlers."""


def read(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def binding(path, root=ROOT):
    path=Path(path)
    return dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=legacy.sha(path))


def inventory(directory):
    directory=Path(directory);result=[]
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():raise ValueError('Evidence symlinks are forbidden')
        if path.is_file() and path != directory/'receipt.json':result.append(binding(path,directory))
    return result


def phase_rows():
    values=np.random.Generator(np.random.PCG64(CONTRACT['phase_seed'])).random(24)
    if len(set(values.tolist()))!=24 or any(float(v)*8 in range(8) for v in values):
        raise ValueError('New unique phase fractions required')
    return [dict(phase_index=i,phase_fraction=float(v),split='development' if i<8 else 'heldout') for i,v in enumerate(values)]


def old_inputs():
    if legacy.sha(OLD/'receipt.json')!=OLD_RECEIPT_SHA:raise ValueError('Historical receipt changed')
    receipt=read(OLD/'receipt.json')
    if receipt['source_commit']!=OLD_SOURCE:raise ValueError('Historical source differs')
    index={r['path']:r for r in receipt['artifacts']}
    if len(index)!=len(receipt['artifacts']):raise ValueError('Duplicate historical paths')
    entries=[binding(OLD/'receipt.json',OLD)]
    for setting in ('core','failed-near-edge'):
        for suffix in ('json','npz'):
            relative=f'references/payloads/{setting}/packet.{suffix}'
            item=binding(OLD/relative,OLD)
            if item!=index[relative]:raise ValueError('Historical reference changed: '+relative)
            entries.append(item)
    return entries


def stage_tasks(phases):
    histories=evaluation.history_census(phases)
    refs=[]
    for setting in ('core','failed-near-edge'):
        for phase in phases:
            cid=f'{setting}--p{phase["phase_index"]:02d}'
            case=dict(id=cid,context_id=cid,setting=setting,**phase)
            for method in CONTRACT['reference_methods']:
                refs.append(dict(id=f'{cid}--reference--{method}',kind='reference',case=case,method=dict(id=method)))
    stages=dict(references=refs,development=[],heldout=[])
    for history in histories:
        methods=['rk4-coarse','rk4-fine']
        if history['phase_index'] in CONTRACT['independent_panel_phases']:methods.append('dop853-independent')
        for method in methods:
            task=dict(id=f'{history["id"]}--{method}',kind='trace',case=history,method=dict(id=method))
            stages[history['split']].append(task)
    return stages


def build_plan():
    phases=phase_rows();stages=stage_tasks(phases)
    return dict(**CONTRACT,date='2026-09-17',phases=phases,
        reference_root=str(OLD),reference_source_commit=OLD_SOURCE,
        historical_inputs=old_inputs(),source_files=[binding(ROOT/p) for p in SOURCE_PATHS],
        acquisition_methods=dict(reference=acquisition.TANGENT_METHODS,trace=acquisition.METHODS),
        evaluation_contract=evaluation.contract(),
        census={k:dict(count=len(v),sha256=hashlib.sha256(canonical(v)).hexdigest()) for k,v in stages.items()},
        planning_evidence=dict(previous_experiment='EXP-014-refinement',prior_trace_count=425,
            prior_acquisition_seconds=178.036,prior_workers=4,
            prior_summary='docs/EXP-014-refinement-summary.json',
            note='Historical timing is not a measurement of this new event/reference pipeline. Fixed development-stage timings are recorded without response-based tuning; 3600 seconds is a producer ceiling, not a runtime prediction.'))


def validate(plan, *, bindings=True, reconstruct=True):
    if any(canonical(plan.get(k))!=canonical(v) for k,v in CONTRACT.items()):raise ValueError('Frozen contract differs')
    if plan['phases']!=phase_rows():raise ValueError('Prospective phase census differs')
    stages=stage_tasks(plan['phases'])
    if [len(stages[k]) for k in ('references','development','heldout')]!=[96,918,1782]:raise ValueError('Complete job census differs')
    if len({t['id'] for rows in stages.values() for t in rows})!=2796:raise ValueError('Duplicate task identity')
    if plan['census']!={k:dict(count=len(v),sha256=hashlib.sha256(canonical(v)).hexdigest()) for k,v in stages.items()}:raise ValueError('Census hash differs')
    if [x['path'] for x in plan['source_files']]!=SOURCE_PATHS:raise ValueError('Source census differs')
    if bindings:
        if plan['historical_inputs']!=old_inputs():raise ValueError('Historical input bindings differ')
        for entry in plan['source_files']:
            if binding(ROOT/entry['path'])!=entry:raise ValueError('Source changed: '+entry['path'])
    if reconstruct and plan!=build_plan():raise ValueError('Metadata-only plan reconstruction differs')
    return dict(contexts=48,histories=1296,reference_jobs=96,traces=2700,total_jobs=2796,comparisons=1404)


def source_guard(plan,path,expected_head):
    if not re.fullmatch('[0-9a-f]{40}',expected_head):raise ValueError('Exact expected commit required')
    if legacy.git('status','--porcelain','--untracked-files=all'):raise ValueError('Clean checkout required')
    commit=legacy.git('rev-parse','HEAD');branch=legacy.git('branch','--show-current')
    if commit!=expected_head or not branch.startswith('codex/'):raise ValueError('Source HEAD/branch differs')
    if Path(path).resolve()!=ROOT/PLAN_REL:raise ValueError('Canonical absolute-resolved plan path required')
    for relative in [*SOURCE_PATHS,PLAN_REL,REVIEW_REL]:
        content=subprocess.check_output(['git','-C',str(ROOT),'show',f'{commit}:{relative}'])
        if hashlib.sha256(content).hexdigest()!=legacy.sha(ROOT/relative):raise ValueError('Uncommitted source/plan/review')
    review=read(ROOT/REVIEW_REL)
    if review.get('status')!='ready_for_exploratory_execution' or review.get('plan_sha256')!=legacy.sha(path):raise ValueError('Exact-plan local review required')
    remote=legacy.git('ls-remote','--exit-code','origin',f'refs/heads/{branch}').split()[0]
    if remote!=commit:raise ValueError('Exact private source not pushed')
    return dict(source_commit=commit,pushed_branch=branch,remote_verified_at=legacy.now(),review=binding(ROOT/REVIEW_REL))


def external_attempt(data_root,attempt):
    if not isinstance(attempt,str) or re.fullmatch(r'attempt-\d+',attempt) is None:raise ValueError('Named attempt-N required')
    root=Path(data_root).resolve()
    if root==ROOT or root.is_relative_to(ROOT):raise ValueError('Raw evidence must be external')
    if root==OLD or root.is_relative_to(OLD):raise ValueError('Never write into historical evidence')
    root.mkdir(parents=True,exist_ok=True);parent=root/'EXP-016';parent.mkdir(exist_ok=True)
    if parent.resolve()!=parent:raise ValueError('Experiment directory alias forbidden')
    directory=parent/attempt;directory.mkdir()
    return directory


def resource_check(directory,*,launch=False):
    directory=Path(directory);state=codec.resources(directory)
    threshold=CONTRACT['launch_ram_bytes'] if launch else CONTRACT['reserve_ram_bytes']
    if state['available_ram_bytes']<threshold:raise RuntimeError('Shared RAM reserve unavailable')
    owned={};pending=[os.getpid()]
    while pending:
        pid=pending.pop()
        if pid in owned:continue
        identity=work.identity(pid)
        if identity is None:continue
        owned[pid]=identity
        try:pending.extend(int(x) for x in Path(f'/proc/{pid}/task/{pid}/children').read_text().split())
        except FileNotFoundError:pass
    rss=0
    for pid,identity in owned.items():
        try:
            if work.identity(pid)==identity:rss+=int(Path(f'/proc/{pid}/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')
        except FileNotFoundError:pass
    raw_bytes=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
    state.update(owned_rss_bytes=rss,raw_bytes=raw_bytes,owned_identities=list(owned.values()),shared_pages_counted_per_process=True)
    work.append(directory/'resource-monitor.jsonl',state)
    if rss>CONTRACT['owned_rss_bytes'] or raw_bytes>CONTRACT['raw_budget_bytes']:raise RuntimeError('Owned memory/raw cap exceeded')
    if state['disk_free_bytes']<CONTRACT['raw_budget_bytes']:raise RuntimeError('Disk reserve unavailable')
    return state


def event(directory,name,**fields):
    work.append(Path(directory)/'events.jsonl',dict(event=name,at=legacy.now(),monotonic_ns=time.monotonic_ns(),**fields))


def initial_contexts(plan):
    result={}
    for setting in ('core','failed-near-edge'):
        reference=codec.load_packet(OLD/f'references/payloads/{setting}/packet.json')
        if reference.get('qualified') is not True:raise ValueError('Historical reference not qualified')
        for phase in plan['phases']:
            cid=f'{setting}--p{phase["phase_index"]:02d}'
            context=acquisition.prepare_context(reference,phase['phase_fraction'],context_id=cid,phase_index=phase['phase_index'])
            context.update(setting=setting,**phase)
            result[cid]=context
    return result


def worker(task,context,directory,libraries,parent_pid):
    started=time.monotonic();cpu=time.process_time();directory=Path(directory)
    dest=directory/'payloads'/task['id'];dest.mkdir()
    row=dict(id=task['id'],kind=task['kind'],case_id=task['case']['id'],method=task['method']['id'],
        status='failed',qualified=False,pid=os.getpid(),identity=work.identity(os.getpid()),started_at=legacy.now(),rmse=None)
    def cancelled(signum,frame):
        if os.getppid()!=parent_pid:os._exit(125)
        raise TimeoutError('Owned worker interrupted; parent retains forced-kill fallback')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,cancelled)
    packet=None
    try:
        if task['kind']=='reference':packet=acquisition.acquire_tangent(context,task['method']['id'])
        elif task['kind']=='trace':
            flat=dict(task['case'],id=task['id'],history_id=task['case']['id'],method=task['method']['id'])
            fn=legacy.load_native(libraries['trace']['path']) if flat['method'].startswith('rk4-') else None
            packet=acquisition.acquire_trace(flat,context,native_fn=fn)
        else:raise ValueError('Unknown task kind')
        row.update(status=packet['status'],qualified=packet.get('complete',packet['status']=='complete') is True)
    except BaseException as error:
        row['error']=repr(error)
        if hasattr(error,'payload'):packet=error.payload
    if packet is not None:
        try:
            sealed=codec.save_packet(dest,'packet',packet)
            row.update(packet_path=str(Path(sealed['path']).relative_to(directory)),packet_sha256=sealed['sha256'],packet_semantic_sha256=sealed['semantic_sha256'])
        except BaseException as error:row.update(status='serialization_failed',qualified=False,error=repr(error))
    row.update(ended_at=legacy.now(),wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    work.write(directory/'records'/f'{task["id"]}.json',row)


def load_task_packet(directory,task,row):
    if row['id']!=task['id'] or any(row.get(k)!=v for k,v in dict(kind=task['kind'],case_id=task['case']['id'],method=task['method']['id']).items()):raise ValueError('Terminal identity differs')
    if row['status']!='complete':return None
    relative=f'payloads/{task["id"]}/packet.json'
    if row.get('packet_path')!=relative:raise ValueError('Packet path differs')
    path=Path(directory)/relative
    if legacy.sha(path)!=row['packet_sha256']:raise ValueError('Packet metadata hash differs')
    packet=codec.load_packet(path)
    if codec.packet_digest(packet)!=row['packet_semantic_sha256']:raise ValueError('Packet digest differs')
    return packet


def make_reference_seals(directory,stages,rows,contexts,*,persist):
    directory=Path(directory);byid={r['id']:r for r in rows};specs={t['id']:t for t in stages['references']};result={}
    for cid,context in contexts.items():
        try:
            pair=[load_task_packet(directory/'references',specs[f'{cid}--reference--{m}'],byid[f'{cid}--reference--{m}']) for m in CONTRACT['reference_methods']]
            if any(p is None for p in pair):raise ValueError('Incomplete reference pair')
            sealed_context=acquisition.seal_context(context,pair[0])
            evaluated=evaluation.seal_reference(sealed_context,*pair)
            if codec.packet_digest(evaluated['context'])!=codec.packet_digest(sealed_context):raise ValueError('Evaluator changed the physical event context')
            packet=dict(context_id=cid,status='complete',qualified=evaluated['qualified'],acquisition=sealed_context,evaluation=evaluated)
        except Exception as error:packet=dict(context_id=cid,status='failed',qualified=False,error=repr(error))
        if persist:codec.save_packet(directory/'seals',cid,packet)
        result[cid]=packet
    return result


def analyze(plan,directory,rows,seals):
    directory=Path(directory);stages=stage_tasks(plan['phases']);histories=evaluation.history_census(plan['phases'])
    task_index={t['id']:(stage,t) for stage in ('development','heldout') for t in stages[stage]}
    row_index={r['id']:r for stage in ('development','heldout') for r in rows[stage]}
    scores={};comparisons=[];errors=[]
    for history in histories:
        cid=history['context_id'];sealed=seals[cid]
        packets={};local_scores={}
        methods=['rk4-coarse','rk4-fine']+(['dop853-independent'] if history['phase_index'] in (0,8) else [])
        for method in methods:
            ident=f'{history["id"]}--{method}';stage,task=task_index[ident]
            try:
                if sealed.get('status')!='complete' or not sealed.get('evaluation',{}).get('complete'):continue
                packet=load_task_packet(directory/stage,task,row_index[ident])
                if packet is None:continue
                score=evaluation.score_trace(dict(history,id=ident,history_id=history['id'],method=method),packet,sealed['evaluation'])
                packets[method]=packet;local_scores[method]=score;scores[ident]=score
            except Exception as error:errors.append(dict(id=ident,error=repr(error)))
        for left,right in [('rk4-coarse','rk4-fine')]+([('rk4-fine','dop853-independent')] if history['phase_index'] in (0,8) else []):
            try:
                comparisons.append(evaluation.compare_traces(history,packets.get(left),packets.get(right),sealed.get('evaluation'),left_score=local_scores.get(left),right_score=local_scores.get(right),methods=(left,right)))
            except Exception as error:errors.append(dict(id=history['id'],pair=[left,right],error=repr(error)))
    prerequisites={cid:dict(context_id=cid,complete=s.get('evaluation',{}).get('complete') is True,qualified=s.get('qualified') is True,status=s['status'],error=s.get('error')) for cid,s in seals.items()}
    summary=evaluation.summarize(histories,scores,comparisons,prerequisites)
    return dict(scores=scores,comparisons=comparisons,packet_errors=errors,summary=summary)


def fill_rows(specs,rows,reason):
    return codec.fill_terminal(specs,rows,reason)


def run(plan_path,data_root,attempt,expected_head):
    plan_path=Path(plan_path).resolve();plan=read(plan_path);validate(plan);provenance=source_guard(plan,plan_path,expected_head)
    if any(importlib.metadata.version(k)!=v for k,v in CONTRACT['required_packages'].items()):raise ValueError('Frozen numerical versions differ')
    if any(os.environ.get(k)!='1' for k in legacy.THREAD_ENV):raise ValueError('Six one-thread numerical environment settings required')
    directory=external_attempt(data_root,attempt);started=time.monotonic();deadline=started+CONTRACT['producer_seconds']
    stages=stage_tasks(plan['phases']);rows={k:[] for k in stages};seals={};analysis=None;timings={};error=None;stage='setup';postflight_error=None
    manifest=dict(**provenance,plan=plan,plan_sha256=legacy.sha(plan_path),started_at=legacy.now(),producer_pid=os.getpid(),producer_identity=work.identity(os.getpid()),
        executable=sys.executable,argv=list(sys.argv),cwd=str(Path.cwd()),python=sys.version,platform=platform.platform(),
        packages={k:importlib.metadata.version(k) for k in CONTRACT['required_packages']},thread_env={k:os.environ.get(k) for k in legacy.THREAD_ENV})
    work.write(directory/'manifest.json',manifest);work.write(directory/'planned-ledger.json',stages)
    handlers={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    def stop(signum,frame):
        if signum!=signal.SIGALRM or time.monotonic()>=deadline:raise ProducerStop('Whole producer deadline/external interruption')
        try:resource_check(directory)
        except Exception as exc:raise ProducerStop('Resource ceiling: '+repr(exc)) from exc
    for sig in handlers:signal.signal(sig,stop)
    signal.setitimer(signal.ITIMER_REAL,1.,1.)
    try:
        resource_check(directory,launch=True);stage='input_seal';mark=time.monotonic()
        contexts=initial_contexts(plan);sealed=codec.save_packet(directory,'input-seal',contexts);contexts=codec.load_packet(sealed['path'])
        event(directory,'inputs_sealed',sha256=sealed['sha256'],semantic_sha256=sealed['semantic_sha256']);timings[stage]=time.monotonic()-mark
        stage='build';mark=time.monotonic();libraries=work.build_libraries(directory/'build',deadline);timings[stage]=time.monotonic()-mark
        stage='references';mark=time.monotonic();event(directory,'references_started',planned=96)
        rows[stage]=runtime.execute(stages[stage],contexts,directory/stage,libraries,workers=4,deadline=deadline,per_task=90,worker_fn=worker)
        timings[stage]=time.monotonic()-mark;stage='reference_seal';mark=time.monotonic()
        seals=make_reference_seals(directory,stages,rows['references'],contexts,persist=True)
        # Every context seal is persisted and reloaded before any pulse is dispatched.
        seals={cid:codec.load_packet(directory/'seals'/f'{cid}.json') for cid in seals}
        seal_index={cid:dict(binding=binding(directory/'seals'/f'{cid}.json',directory),semantic_sha256=codec.packet_digest(s),qualified=s['qualified']) for cid,s in seals.items()}
        work.write(directory/'reference-seal-index.json',seal_index);event(directory,'reference_seals_complete',index_sha256=legacy.sha(directory/'reference-seal-index.json'))
        timings[stage]=time.monotonic()-mark;contexts=None;gc.collect()
        for stage in ('development','heldout'):
            mark=time.monotonic();resource_check(directory,launch=True)
            eligible=[t for t in stages[stage] if seals[t['case']['context_id']]['qualified'] is True]
            trace_contexts={t['case']['id']:seals[t['case']['context_id']]['acquisition'] for t in eligible}
            event(directory,stage+'_started',planned=len(stages[stage]),eligible=len(eligible))
            if eligible:rows[stage]=runtime.execute(eligible,trace_contexts,directory/stage,libraries,workers=4,deadline=deadline,per_task=60,worker_fn=worker)
            rows[stage]=fill_rows(stages[stage],rows[stage],'not_started_reference_qualification')
            timings[stage]=time.monotonic()-mark
            event(directory,stage+'_finished',statuses=dict(Counter(r['status'] for r in rows[stage])),wall_seconds=timings[stage])
        stage='saved_analysis';mark=time.monotonic();analysis=analyze(plan,directory,rows,seals);timings[stage]=time.monotonic()-mark
    except BaseException as exc:
        error=repr(exc);work.write(directory/'producer-error.json',dict(stage=stage,error=error,at=legacy.now()))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        def sealing_stop(signum,frame):raise ProducerStop('Sealing grace exceeded; existing evidence remains')
        signal.signal(signal.SIGALRM,sealing_stop);signal.setitimer(signal.ITIMER_REAL,CONTRACT['sealing_seconds'])
        try:
            for name,specs in stages.items():
                saved_path=directory/name/'terminal-records.json'
                if not rows[name] and saved_path.exists():rows[name]=read(saved_path)
                rows[name]=fill_rows(specs,rows[name],'not_started_producer_or_prerequisite_failure')
            work.write(directory/'all-terminal-records.json',rows)
            if analysis is None:analysis=dict(status='unavailable',reason=error or 'No analysis',packet_errors=[],summary=None)
            codec.save_packet(directory,'analysis',analysis)
            try:
                validate(plan)
                if legacy.git('rev-parse','HEAD')!=expected_head or legacy.git('status','--porcelain','--untracked-files=all'):raise ValueError('Frozen checkout changed during run')
            except Exception as exc:postflight_error=repr(exc)
            work.write(directory/'postflight.json',dict(source_bindings_unchanged=postflight_error is None,error=postflight_error))
            complete=error is None and postflight_error is None and all(r['status']=='complete' for rs in rows.values() for r in rs) and not analysis.get('packet_errors') and analysis.get('summary',{}).get('complete') is True
            summary=dict(experiment_id='EXP-016',complete=bool(complete),source_valid=postflight_error is None,error=error,postflight_error=postflight_error,
                status_counts={k:dict(Counter(r['status'] for r in v)) for k,v in rows.items()},analysis=analysis.get('summary'),
                stage_wall_seconds=timings,total_wall_seconds=time.monotonic()-started,prior_failures_unchanged=True)
            work.write(directory/'summary.json',summary);event(directory,'producer_finished',complete=complete)
            artifacts=inventory(directory)
            receipt=dict(**provenance,experiment_id='EXP-016',directory=str(directory),plan_sha256=legacy.sha(plan_path),
                status='complete' if complete else 'incomplete_or_invalid',artifacts=artifacts,artifact_bytes=sum(r['bytes'] for r in artifacts),sealed_at=legacy.now())
            work.write(directory/'receipt.json',receipt)
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            for sig,handler in handlers.items():signal.signal(sig,handler)
    return dict(directory=str(directory),receipt_sha256=legacy.sha(directory/'receipt.json'),summary=summary)


def normalized(value):
    arrays={};tree=codec._encode(value,arrays)
    return codec._decode(tree,arrays,set())


def verify_seals(directory,plan,stages,rows,*,validate_trace_prefix=True):
    """Audit chronological and numerical prefixes even if final analysis failed."""
    directory=Path(directory)
    events=[json.loads(s) for s in (directory/'events.jsonl').read_text().splitlines()] if (directory/'events.jsonl').exists() else []
    def matching(name):return [e for e in events if e['event']==name]
    reference_dispatched=any(r.get('pid') is not None or r['status']=='complete' for r in rows['references'])
    trace_dispatched=any(r.get('pid') is not None or r['status']=='complete' for stage in ('development','heldout') for r in rows[stage])
    input_path=directory/'input-seal.json';input_events=matching('inputs_sealed')
    ref_starts=matching('references_started')
    if reference_dispatched and (not input_path.exists() or len(input_events)!=1 or len(ref_starts)!=1):raise ValueError('References lack prior input seal')
    if len(input_events)>1 or len(ref_starts)>1:raise ValueError('Duplicate input/reference stage event')
    if reference_dispatched and input_events[0]['monotonic_ns']>=ref_starts[0]['monotonic_ns']:raise ValueError('Input-before-reference chronology differs')
    contexts=None;seals={}
    if input_path.exists():
        contexts=initial_contexts(plan)
        saved=codec.load_packet(input_path);codec.compare(saved,normalized(contexts),'input-seal')
        if input_events and (input_events[0]['sha256']!=legacy.sha(input_path) or input_events[0]['semantic_sha256']!=codec.packet_digest(saved)):raise ValueError('Input seal event binding differs')
        seals=make_reference_seals(directory,stages,rows['references'],contexts,persist=False)
        # Every readable saved context is checked, including partial-stage prefixes.
        for cid,packet in seals.items():
            path=directory/'seals'/f'{cid}.json'
            if path.exists():codec.compare(codec.load_packet(path),normalized(packet),'reference-seal.'+cid)
        # Independently validate each complete reference, even if its mate failed.
        specs={t['id']:t for t in stages['references']}
        for row in rows['references']:
            if row['status']=='complete':
                task=specs[row['id']];packet=load_task_packet(directory/'references',task,row)
                evaluation._tangent(packet,contexts[task['case']['id']],task['method']['id'])
    index_path=directory/'reference-seal-index.json';sealed_events=matching('reference_seals_complete')
    starts=matching('development_started')+matching('heldout_started')
    if len(sealed_events)>1 or len(matching('development_started'))>1 or len(matching('heldout_started'))>1:raise ValueError('Duplicate reference/pulse stage event')
    if trace_dispatched and (not index_path.exists() or len(sealed_events)!=1 or not contexts):raise ValueError('Pulse dispatch lacks complete reference seal')
    if index_path.exists():
        index=read(index_path)
        if set(index)!=set(seals):raise ValueError('Complete reference seal index census differs')
        for cid,packet in seals.items():
            path=directory/'seals'/f'{cid}.json'
            if not path.exists():raise ValueError('Missing indexed reference seal')
            expected=dict(binding=binding(path,directory),semantic_sha256=codec.packet_digest(packet),qualified=packet['qualified'])
            if index[cid]!=expected:raise ValueError('Reference seal index differs')
    if sealed_events:
        if not index_path.exists() or sealed_events[0]['index_sha256']!=legacy.sha(index_path):raise ValueError('Reference seal event binding differs')
        if any(e['monotonic_ns']<=sealed_events[0]['monotonic_ns'] for e in starts):raise ValueError('Reference-before-pulse chronology differs')
    elif starts:raise ValueError('Pulse stage began without reference seal event')
    for stage in ('development','heldout'):
        if any(r.get('pid') is not None or r['status']=='complete' for r in rows[stage]) and len(matching(stage+'_started'))!=1:raise ValueError('Dispatched stage lacks its event')
    for stage in ('development','heldout'):
        specs={t['id']:t for t in stages[stage]}
        for row in rows[stage]:
            if row.get('pid') is None and row['status']!='complete':continue
            task=specs[row['id']];cid=task['case']['context_id']
            if cid not in seals or seals[cid].get('qualified') is not True:raise ValueError('Pulse dispatch had no qualified prerequisite')
            if validate_trace_prefix and row['status']=='complete':
                packet=load_task_packet(directory/stage,task,row)
                flat=dict(task['case'],id=task['id'],history_id=task['case']['id'],method=task['method']['id'])
                evaluation._trace(flat,packet,seals[cid]['evaluation'])
    return seals


def verify(directory):
    directory=Path(directory).resolve();receipt=read(directory/'receipt.json');manifest=read(directory/'manifest.json');plan=manifest['plan']
    if receipt['artifacts']!=inventory(directory):raise ValueError('Immutable byte inventory differs')
    if receipt['artifact_bytes']!=sum(r['bytes'] for r in receipt['artifacts']):raise ValueError('Artifact byte count differs')
    if receipt['source_commit']!=manifest['source_commit'] or receipt['plan_sha256']!=manifest['plan_sha256'] or receipt['experiment_id']!='EXP-016':raise ValueError('Receipt source/plan identity differs')
    committed=subprocess.check_output(['git','-C',str(ROOT),'show',f'{receipt["source_commit"]}:{PLAN_REL}'])
    if hashlib.sha256(committed).hexdigest()!=receipt['plan_sha256'] or json.loads(committed)!=plan:raise ValueError('Committed plan differs')
    for item in plan['source_files']+[manifest['review']]:
        b=subprocess.check_output(['git','-C',str(ROOT),'show',f'{receipt["source_commit"]}:{item["path"]}'])
        if len(b)!=item['bytes'] or hashlib.sha256(b).hexdigest()!=item['sha256']:raise ValueError('Committed source/review differs')
    validate(plan);stages=stage_tasks(plan['phases']);rows=read(directory/'all-terminal-records.json')
    if read(directory/'planned-ledger.json')!=stages or set(rows)!=set(stages):raise ValueError('Planned denominator differs')
    processes=[]
    for stage,specs in stages.items():
        if len(rows[stage])!=len(specs) or len({r['id'] for r in rows[stage]})!=len(specs) or {r['id'] for r in rows[stage]}!={t['id'] for t in specs}:raise ValueError('Complete terminal denominator differs')
        index={t['id']:t for t in specs}
        for row in rows[stage]:
            task=index[row['id']]
            if any(row.get(k)!=v for k,v in dict(case_id=task['case']['id'],kind=task['kind'],method=task['method']['id']).items()):raise ValueError('Terminal identity differs')
        stage_path=directory/stage
        if stage_path.exists():
            cleanup=read(stage_path/'cleanup.json')
            if cleanup['all_registered_workers_joined'] is not True or cleanup['all_original_identities_absent'] is not True:raise ValueError('Owned workers not cleaned up')
            owned=[json.loads(s) for s in (stage_path/'processes.jsonl').read_text().splitlines()] if (stage_path/'processes.jsonl').exists() else []
            if len(owned)!=len(cleanup['workers']) or len({r['task_id'] for r in owned})!=len(owned):raise ValueError('Owned process census differs')
            for a,b in zip(owned,cleanup['workers'],strict=True):
                if any(b.get(k)!=v for k,v in a.items()) or b['original_identity_absent'] is not True:raise ValueError('Owned identity differs')
            terminal=read(stage_path/'terminal-records.json')
            journal=[json.loads(s) for s in (stage_path/'journal.jsonl').read_text().splitlines()] if (stage_path/'journal.jsonl').exists() else []
            if terminal!=journal or len({r['id'] for r in terminal})!=len(terminal):raise ValueError('Stage journal differs or repeats a task')
            outer={r['id']:r for r in rows[stage]};registered={r['task_id']:r for r in owned}
            inner={r['id']:r for r in terminal}
            if not set(inner).issubset(outer) or not set(registered).issubset(inner):raise ValueError('Stage dispatch census differs')
            for row in rows[stage]:
                if (row['status']=='complete' or row.get('pid') is not None) and (inner.get(row['id'])!=row or row['id'] not in registered):raise ValueError('Parent completion lacks journal/ownership')
            for row in terminal:
                if outer.get(row['id'])!=row:raise ValueError('Stage/parent terminal differs')
                if row.get('pid') is not None and (row['id'] not in registered or registered[row['id']]['pid']!=row['pid']):raise ValueError('Missing ownership')
                if row['status']=='complete' and read(stage_path/'records'/f'{row["id"]}.json')!=row:raise ValueError('Worker record differs')
            processes.extend(owned)
        elif any(r['status']=='complete' or r.get('pid') for r in rows[stage]):raise ValueError('Missing lifecycle stage')
    saved=codec.load_packet(directory/'analysis.json');summary=read(directory/'summary.json');post=read(directory/'postflight.json')
    if summary['status_counts']!={k:dict(Counter(r['status'] for r in v)) for k,v in rows.items()}:raise ValueError('Status summary differs')
    if summary['postflight_error']!=post['error'] or summary['source_valid'] is not (post['error'] is None) or post['source_bindings_unchanged'] is not (post['error'] is None):raise ValueError('Source validity differs')
    seals=verify_seals(directory,plan,stages,rows,validate_trace_prefix=saved.get('status')=='unavailable')
    replayed=False
    if saved.get('status')!='unavailable':
        if len(seals)!=48:raise ValueError('Analysis lacks complete context seal census')
        replay=analyze(plan,directory,rows,seals);codec.compare(saved,normalized(replay),'analysis');replayed=True
    elif not (directory/'producer-error.json').exists():raise ValueError('Missing analysis needs retained producer failure')
    codec.compare(summary['analysis'],work.baseline_engine.safe(saved.get('summary')),'summary.analysis')
    error=read(directory/'producer-error.json')['error'] if (directory/'producer-error.json').exists() else None
    if summary['error']!=error:raise ValueError('Producer failure differs')
    complete=error is None and post['error'] is None and all(r['status']=='complete' for rs in rows.values() for r in rs) and not saved.get('packet_errors') and saved.get('summary',{}).get('complete') is True
    if summary['complete']!=complete or receipt['status']!=('complete' if complete else 'incomplete_or_invalid'):raise ValueError('Completion conclusion differs')
    return dict(complete=bool(complete),saved_array_replay=replayed,total_jobs=2796,receipt_sha256=legacy.sha(directory/'receipt.json'),
        all_original_owned_identities_absent_now=all(work.identity(r['pid']) is None or work.identity(r['pid'])!=r['identity'] for r in processes))


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',default=str(ROOT/PLAN_REL))
    p=sub.add_parser('validate');p.add_argument('--plan',default=str(ROOT/PLAN_REL))
    p=sub.add_parser('run');p.add_argument('--plan',default=str(ROOT/PLAN_REL));p.add_argument('--data-root',required=True);p.add_argument('--attempt',required=True);p.add_argument('--expected-head',required=True)
    p=sub.add_parser('verify');p.add_argument('--attempt-dir',required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        plan=build_plan();work.write(Path(args.output),plan);out=validate(plan)
    elif args.command=='validate':out=validate(read(args.plan))
    elif args.command=='run':
        result=run(args.plan,args.data_root,args.attempt,args.expected_head)
        summary=result['summary'];scientific=summary.get('analysis') or {}
        out=dict(directory=result['directory'],receipt_sha256=result['receipt_sha256'],
            complete=summary['complete'],status_counts=summary['status_counts'],error=summary['error'],
            scientific_counts=scientific.get('counts'),qualified=scientific.get('qualified'),
            primary=scientific.get('primary'),primary_failure_bounds=scientific.get('primary_failure_bounds'))
    else:out=verify(args.attempt_dir)
    print(json.dumps(work.baseline_engine.safe(out),sort_keys=True,allow_nan=False))

if __name__=='__main__':main()
