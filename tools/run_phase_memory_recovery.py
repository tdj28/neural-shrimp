#!/usr/bin/env python3
"""EXP-014 recovery: measured memory-cap amendment, unchanged science.

prepare/validate read historical seed metadata and hashes only. run alone
acquires dynamics, after an exact privately pushed source/plan/review guard.
verify reconstructs saved-array diagnostics and readouts; never solves an ODE.
"""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import signal
import shutil
import subprocess
import sys
import time

import fresh_preview_runtime as work  # sets one numerical thread before NumPy
import run_hopfield_refinement as legacy
import numpy as np
import phase_memory_acquisition as acquire
import phase_memory_evaluation as evaluation

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT/'experiments/hopfield/EXP-014-recovery-plan.json'
REVIEW = ROOT/'docs/EXP-014-recovery-review.json'
GIB = 1024**3
CONTRACT = dict(experiment_id='EXP-014',schema_version='EXP-014-phase-memory-v1',exploratory=True,
    reference_jobs=2,phase_propagators=16,pulse_sham_traces=912,total_tasks=930,
    workers=4,thread_limit=1,producer_seconds=1800,reference_task_seconds=120,trace_task_seconds=60,
    sealing_grace_seconds=120,launch_available_ram_bytes=16*GIB,reserve_ram_bytes=8*GIB,
    owned_rss_bytes=12*GIB,raw_budget_bytes=2*GIB,sample_dt=.025,duration=320.,samples=12801,
    phase_nodes=4096,methods=list(acquire.METHODS),reference_solver=dict(acquire.FINE),
    reference_gates=dict(acquire.REFERENCE_GATES),paid_compute=False,backend='native_cpu',
    required_packages={'numpy':'2.5.3','scipy':'1.18.1'},
    planning_estimate=dict(historical_task_ledger='/data2/neural-shrimp-data/raw/EXP-013/attempt-002/all-terminal-records.json',
        historical_task_ledger_sha256='49a8753b5bbc80754897cb2aa2af500f2de1e23159586d879fcfdbafd5a0f7ec',
        historical_task_ledger_bytes=48998353,historical_duration=75.,new_duration=320.,
        historical_integration_medians_seconds={'rk4-0.0025':.0068950070,'rk4-0.00125':.0120649740,'dop853-1e-12':.2403591090},
        duration_only_compute_task_seconds=336.35,ideal_four_worker_compute_seconds=84.09,
        conservative_producer_estimate_seconds=[600,900],synthetic_infrastructure_tests=142,synthetic_suite_seconds=24.80,
        scope='Historical duration extrapolation only; new dense sampling, prerequisite solves, sealed predictions, serial scoring/replay/storage add cost. Conservative estimate is not measured target throughput. Hard producer limit remains1800s.'))
SOURCE_PATHS = list(dict.fromkeys([
    'tools/run_phase_memory_recovery.py','tests/test_phase_memory_recovery.py','docs/EXP-014-operational-amendment.md',
    'tools/run_phase_memory.py','tools/phase_memory_acquisition.py','tools/phase_memory.py','tools/phase_memory_evaluation.py',
    'tests/test_phase_memory_acquisition.py','tests/test_phase_memory.py','tests/test_phase_memory_evaluation.py','tests/test_phase_memory_runner.py',
    'docs/EXP-014-protocol.md','docs/phase-memory-design.md','tools/fresh_preview_runtime.py',
    'tools/run_hopfield.py','tools/run_hopfield_drift.py',*legacy.SOURCE_PATHS]))


class ProducerStop(BaseException):
    """Parent deadline/resource cancellation must bypass per-case diagnostics."""


def read(path):return json.loads(Path(path).read_text())
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(path):return legacy.sha(path)
def binding(path,root=ROOT):
    path=Path(path);return dict(path=str(path.relative_to(root)),bytes=path.stat().st_size,sha256=sha(path))


def _encode(value, arrays):
    if isinstance(value,np.ndarray):
        if value.dtype.hasobject:raise ValueError('Object arrays forbidden')
        key=f'a{len(arrays):06d}';arrays[key]=np.array(value,copy=True)
        return {'__array__':key,'dtype':value.dtype.str,'shape':list(value.shape)}
    if isinstance(value,np.generic):return _encode(value.item(),arrays)
    if isinstance(value,float) and not np.isfinite(value):return {'__nonfinite__':repr(value)}
    if isinstance(value,dict):
        if any(type(k) is not str for k in value):raise ValueError('Packet dictionary keys must be strings')
        if '__array__' in value or '__nonfinite__' in value:raise ValueError('Reserved codec key')
        return {k:_encode(v,arrays) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_encode(v,arrays) for v in value]
    if value is None or type(value) in (str,int,float,bool):return value
    raise ValueError('Unsupported packet value: '+type(value).__name__)


def _decode(value, arrays, used):
    if isinstance(value,list):return [_decode(v,arrays,used) for v in value]
    if isinstance(value,dict):
        if '__array__' in value:
            if set(value)!={'__array__','dtype','shape'}:raise ValueError('Malformed array marker')
            key=value['__array__']
            if key in used or key not in arrays:raise ValueError('Duplicate/missing array binding')
            used.add(key);array=arrays[key]
            if array.dtype.str!=value['dtype'] or list(array.shape)!=value['shape'] or array.dtype.hasobject:raise ValueError('Array dtype/shape mismatch')
            return array.copy()
        if '__nonfinite__' in value:
            if set(value)!={'__nonfinite__'} or value['__nonfinite__'] not in ('nan','inf','-inf'):raise ValueError('Malformed nonfinite marker')
            return float(value['__nonfinite__'])
        return {k:_decode(v,arrays,used) for k,v in value.items()}
    return value


def packet_digest(value):
    arrays={};tree=_encode(value,arrays)
    return hashlib.sha256(canonical(dict(tree=tree,arrays={k:hashlib.sha256(v.tobytes(order='C')).hexdigest() for k,v in arrays.items()}))).hexdigest()


def save_packet(directory,name,value):
    """Exclusive lossless numeric NPZ plus hash-bound JSON; no pickle/object data."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    if not name or Path(name).name!=name:raise ValueError('Simple packet name required')
    arrays={};tree=_encode(value,arrays);raw=directory/f'{name}.npz';meta=directory/f'{name}.json'
    if raw.exists() or meta.exists():raise FileExistsError('Packet already exists')
    with raw.open('xb') as stream:
        np.savez(stream,**arrays);stream.flush();os.fsync(stream.fileno())
    work.write(meta,dict(schema='numeric-packet-v1',tree=tree,arrays=dict(path=raw.name,bytes=raw.stat().st_size,sha256=sha(raw)),semantic_sha256=packet_digest(value)))
    restored=load_packet(meta)
    if packet_digest(restored)!=packet_digest(value):raise ValueError('Packet write/read identity mismatch')
    return dict(path=str(meta),sha256=sha(meta),bytes=meta.stat().st_size,semantic_sha256=packet_digest(value))


def load_packet(path):
    path=Path(path);meta=read(path)
    if set(meta)!={'schema','tree','arrays','semantic_sha256'} or meta['schema']!='numeric-packet-v1':raise ValueError('Unknown packet schema')
    entry=meta['arrays'];raw=path.parent/entry['path']
    if Path(entry['path']).name!=entry['path'] or not raw.resolve().is_relative_to(path.parent.resolve()):raise ValueError('Packet path escape')
    if raw.stat().st_size!=entry['bytes'] or sha(raw)!=entry['sha256']:raise ValueError('Packet array hash mismatch')
    with np.load(raw,allow_pickle=False) as data:
        arrays={k:data[k].copy() for k in data.files}
    used=set();value=_decode(meta['tree'],arrays,used)
    if set(arrays)!=used or packet_digest(value)!=meta['semantic_sha256']:raise ValueError('Packet semantic/array census mismatch')
    return value


def compare(saved,replay,path='packet'):
    """Frozen replay tolerance, with failure nonfinite identities preserved."""
    if isinstance(saved,np.ndarray):
        if not isinstance(replay,np.ndarray) or saved.dtype!=replay.dtype or saved.shape!=replay.shape:raise ValueError('Array identity differs: '+path)
        if saved.dtype.kind in 'fc':
            same=np.allclose(saved,replay,rtol=1e-12,atol=1e-14,equal_nan=True)
        else:same=np.array_equal(saved,replay)
        if not same:raise ValueError('Array values differ: '+path)
    elif isinstance(saved,dict):
        if not isinstance(replay,dict) or set(saved)!=set(replay):raise ValueError('Mapping differs: '+path)
        for key in saved:compare(saved[key],replay[key],path+'.'+key)
    elif isinstance(saved,list):
        if not isinstance(replay,list) or len(saved)!=len(replay):raise ValueError('Sequence differs: '+path)
        for i,(a,b) in enumerate(zip(saved,replay,strict=True)):compare(a,b,f'{path}[{i}]')
    elif type(saved) is float:
        if type(replay) is not float or (np.isfinite(saved) and not np.isclose(saved,replay,rtol=1e-12,atol=1e-14)) or (not np.isfinite(saved) and repr(saved)!=repr(replay)):
            raise ValueError('Number differs: '+path)
    elif type(saved) is not type(replay) or saved!=replay:raise ValueError('Category differs: '+path)


def stages(plan):
    def task(ident,kind,case,method):return dict(id=ident,kind=kind,case=case,method=dict(id=method))
    return dict(references=[task(setting,'reference',dict(id=setting,setting=setting),'certificate-and-adjoint') for setting in acquire.SETTINGS],
        propagators=[task(f'{setting}--p{phase}','propagator',dict(id=f'{setting}--p{phase}',setting=setting,phase_index=phase),'fine-dop853')
                     for setting in acquire.SETTINGS for phase in range(8)],
        traces=[task(row['id'],'trace',row,row['method']) for row in plan['tasks']])


def build_plan():
    seeds=acquire.load_seed_records(ROOT)
    return dict(**CONTRACT,date='2026-09-17',seeds=[dict(setting=s['setting'],alpha=s['alpha'],gamma=s['gamma'],period=s['period'],section_state=s['section_state'].tolist()) for s in seeds],
        historical_inputs=[binding(ROOT/rel,ROOT) for rel in acquire.HISTORICAL],
        source_files=[binding(ROOT/rel,ROOT) for rel in SOURCE_PATHS],tasks=acquire.build_tasks(),
        claim_boundary='Two fixed cycles, known external phase clock, synthetic observation noise; no geometry-causal, capacity or learned-generalization claim.')


def validate(plan,*,bindings=False,reconstruct=False):
    if any(canonical(plan.get(k))!=canonical(v) for k,v in CONTRACT.items()):raise ValueError('Frozen phase-memory contract differs')
    if canonical(plan['tasks'])!=canonical(acquire.build_tasks()) or canonical(plan['tasks'])!=canonical(evaluation.task_census()):raise ValueError('912-task identity census differs')
    if plan['seeds']!=[dict(setting=k,**v) for k,v in acquire.SETTINGS.items()]:raise ValueError('Historical seed identity differs')
    if [b['path'] for b in plan['source_files']]!=SOURCE_PATHS or [b['path'] for b in plan['historical_inputs']]!=list(acquire.HISTORICAL):raise ValueError('Binding census differs')
    if bindings:
        for entry in plan['source_files']+plan['historical_inputs']:
            if binding(ROOT/entry['path'],ROOT)!=entry:raise ValueError('Source/input changed: '+entry['path'])
    if reconstruct and plan!=build_plan():raise ValueError('Metadata-only plan reconstruction differs')
    return dict(references=2,propagators=16,traces=912,total=930)


def source_guard(plan,path):
    if legacy.git('status','--porcelain','--untracked-files=all'):raise ValueError('Clean source checkout required')
    commit=legacy.git('rev-parse','HEAD');branch=legacy.git('branch','--show-current')
    if not branch or not branch.startswith('codex/'):raise ValueError('Dedicated codex branch required')
    for relative in [*SOURCE_PATHS,str(Path(path).resolve().relative_to(ROOT)),str(REVIEW.relative_to(ROOT))]:
        data=subprocess.check_output(['git','-C',str(ROOT),'show',f'{commit}:{relative}'])
        if hashlib.sha256(data).hexdigest()!=sha(ROOT/relative):raise ValueError('Source/plan/review not committed')
    if legacy.git('ls-remote','--exit-code','origin',f'refs/heads/{branch}').split()[0]!=commit:raise ValueError('Exact private source not pushed')
    review=read(REVIEW)
    if review.get('status')!='ready_for_exploratory_execution' or review.get('plan_sha256')!=sha(path):raise ValueError('Exact-plan local review required')
    return dict(source_commit=commit,pushed_branch=branch,remote_verified_at=legacy.now(),review=binding(REVIEW,ROOT))


def external_attempt(data_root,attempt):
    if not isinstance(attempt,str) or not attempt.startswith('attempt-') or not attempt[8:].isdigit():raise ValueError('Named attempt-N required')
    root=Path(data_root).resolve()
    if root==ROOT.resolve() or root.is_relative_to(ROOT.resolve()):raise ValueError('All outcome data must be external')
    root.mkdir(parents=True,exist_ok=True);parent=root/'EXP-014';parent.mkdir(exist_ok=True)
    directory=parent/attempt
    if parent.resolve()!=parent or directory.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Output symlink escape')
    directory.mkdir()
    return directory


def resources(directory=None):
    mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith(('MemAvailable:','MemTotal:'))}
    return dict(at=legacy.now(),available_ram_bytes=mem['MemAvailable'],total_ram_bytes=mem['MemTotal'],load_average=list(os.getloadavg()),
        logical_cpus=os.cpu_count(),disk_free_bytes=shutil.disk_usage(directory or ROOT.parent).free)


def resource_check(directory,*,launch=False):
    resource_state=resources(directory);threshold=CONTRACT['launch_available_ram_bytes'] if launch else CONTRACT['reserve_ram_bytes']
    if resource_state['available_ram_bytes']<threshold:raise RuntimeError('Insufficient shared RAM headroom')
    directory=Path(directory)
    raw_bytes=sum(p.stat().st_size for p in directory.rglob('*') if p.is_file())
    # Include compiler descendants and just-started workers, not merely rows
    # that have reached the durable process log. Never signal this inventory.
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
    resource_state.update(owned_rss_bytes=rss,raw_bytes=raw_bytes,owned_identities=list(owned.values()),shared_pages_counted_per_process=True)
    work.append(directory/'resource-monitor.jsonl',resource_state)
    if rss>CONTRACT['owned_rss_bytes'] or raw_bytes>CONTRACT['raw_budget_bytes']:raise RuntimeError('Owned memory or raw-output hard cap exceeded')
    if resource_state['disk_free_bytes']<CONTRACT['raw_budget_bytes']:raise RuntimeError('Insufficient remaining disk reserve')
    return resource_state


def worker(task,context,directory,libraries,parent_pid):
    started=time.monotonic();cpu=time.process_time();directory=Path(directory);dest=directory/'payloads'/task['id'];dest.mkdir()
    row=dict(id=task['id'],kind=task['kind'],case_id=task['case']['id'],method=task['method']['id'],status='failed',
        pid=os.getpid(),identity=work.identity(os.getpid()),started_at=legacy.now(),started_monotonic_ns=time.monotonic_ns(),qualified=False,rmse=None)
    packet=None
    def cancelled(signum,frame):
        if os.getppid()!=parent_pid:os._exit(125)
        raise TimeoutError('Owned child cancellation; scheduler retains forced-kill fallback')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,cancelled)
    try:
        if task['kind']=='reference':packet=acquire.acquire_reference(context['seed'])
        elif task['kind']=='propagator':packet=acquire.acquire_phase_propagator(context['reference'],task['case']['phase_index'])
        elif task['kind']=='trace':
            fn=legacy.load_native(libraries['trace']['path']) if task['case']['method'].startswith('rk4-') else None
            packet=acquire.acquire_case(task['case'],context['reference'],native_fn=fn)
        else:raise ValueError('Unplanned phase-memory worker kind')
        row.update(status='complete',qualified=packet.get('qualified',packet.get('complete',False)) is True)
    except BaseException as error:
        row['error']=repr(error)
        if hasattr(error,'payload'):packet=error.payload
    if packet is not None:
        try:
            saved=save_packet(dest,'packet',packet)
            row.update(packet_path=str(Path(saved['path']).relative_to(directory)),packet_sha256=saved['sha256'],packet_semantic_sha256=saved['semantic_sha256'])
        except BaseException as error:row.update(status='serialization_failed',qualified=False,serialization_error=repr(error))
    row.update(ended_at=legacy.now(),wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    work.write(directory/'records'/f'{task["id"]}.json',row)


def fill_terminal(specs,rows,reason='not_started_dependency_failure'):
    index={r['id']:r for r in rows}
    if len(index)!=len(rows) or set(index)-{t['id'] for t in specs}:raise ValueError('Terminal task census mismatch')
    return [index.get(t['id'],dict(id=t['id'],kind=t['kind'],case_id=t['case']['id'],method=t['method']['id'],
        status=reason,qualified=False,rmse=None)) for t in specs]


def _packets(directory,records):
    result={}
    for row in records:
        metadata=row
        if 'packet_path' not in metadata:
            # A cooperative timed-out child can seal evidence during the owned
            # scheduler's one-second grace; timeout remains the terminal status.
            child=Path(directory)/'records'/f'{row["id"]}.json'
            if not child.exists():continue
            metadata=read(child)
            if any(metadata.get(k)!=row.get(k) for k in ('id','kind','case_id','method','pid')):raise ValueError('Partial worker identity differs')
            if 'packet_path' not in metadata:continue
        path=Path(directory)/metadata['packet_path']
        if not path.resolve().is_relative_to(Path(directory).resolve()) or sha(path)!=metadata['packet_sha256']:raise ValueError('Worker packet path/hash mismatch')
        packet=load_packet(path)
        if packet_digest(packet)!=metadata['packet_semantic_sha256']:raise ValueError('Worker packet semantic hash mismatch')
        if row['status']!='complete':
            packet=dict(packet,status=row['status'],qualified=False,complete=False,operational_status=row['status'])
        result[row['id']]=packet
    return result


def analyze(references,propagators,traces,prepared):
    scored=[];errors=[];comparisons=[]
    for task in acquire.build_tasks():
        trace=traces.get(task['id'])
        if trace is None or trace.get('complete') is not True:continue
        try:
            score=evaluation.score_trace(task,trace,references[task['setting']],propagators[(task['setting'],task['phase_index'])],
                state_derivatives=acquire.trace_derivatives(trace),prepared=prepared)
            scored.append(score)
        except Exception as error:errors.append(dict(id=task['id'],stage='score',error=repr(error)))
    indexed={r['task']['id']:r for r in scored}
    for task in acquire.build_tasks():
        for left,right in evaluation.METHOD_PAIRS:
            if task['method']!=left:continue
            peer=dict(task,method=right,id=task['id'].rsplit('--',1)[0]+'--'+right)
            if task['id'] not in indexed or peer['id'] not in indexed:continue
            try:comparisons.append(evaluation.compare_trace_pair(task,traces[task['id']],indexed[task['id']],peer,traces[peer['id']],indexed[peer['id']]))
            except Exception as error:errors.append(dict(ids=[task['id'],peer['id']],stage='compare',error=repr(error)))
    paired_dynamics=[]
    for minus in acquire.build_tasks():
        if minus['kind']!='pulse' or minus['sign']!=-1:continue
        plus=dict(minus,sign=1,id=minus['id'].replace('--s-1--','--s+1--'))
        paired=evaluation.paired_response_diagnostics(plus,traces.get(plus['id'],{}),indexed.get(plus['id'],{}),
            minus,traces.get(minus['id'],{}),indexed.get(minus['id'],{}),prepared=prepared)
        paired_dynamics.append(paired)
        if paired['status']!='complete' and plus['id'] in indexed and minus['id'] in indexed:
            errors.append(dict(ids=paired['ids'],stage='signed_response',error=paired.get('error')))
    prerequisites={setting:dict(reference=references.get(setting,{}).get('qualified') is True,
        propagators=[propagators.get((setting,j),{}).get('qualified') is True for j in range(8)]) for setting in acquire.SETTINGS}
    summary=evaluation.summarize(scored,comparisons,prerequisites=prerequisites)
    if errors:summary['qualified']=False
    return dict(scored_records=scored,trace_comparisons=comparisons,paired_dynamics=paired_dynamics,analysis_errors=errors,summary=summary)


def event(directory,stage,**fields):
    work.append(Path(directory)/'events.jsonl',dict(stage=stage,at=legacy.now(),monotonic_ns=time.monotonic_ns(),**fields))


def run(plan_path,data_root,attempt):
    plan=read(plan_path);validate(plan,bindings=True,reconstruct=True);provenance=source_guard(plan,plan_path)
    for package,version in CONTRACT['required_packages'].items():
        if importlib.metadata.version(package)!=version:raise ValueError('Unqualified numerical package version')
    directory=external_attempt(data_root,attempt);started=time.monotonic();deadline=started+CONTRACT['producer_seconds']
    specs=stages(plan);terminal={};stage_times={};error=None;prepared=None;analysis=None;stage='preflight';libraries={}
    reference_packets={};propagator_packets={};trace_packets={};postflight_error=None
    manifest=dict(**provenance,plan=plan,plan_sha256=sha(plan_path),started_at=legacy.now(),producer_pid=os.getpid(),
        producer_identity=work.identity(os.getpid()),python=sys.version,platform=platform.platform(),executable=sys.executable,
        argv=sys.argv,working_directory=str(Path.cwd()),thread_env={k:os.environ.get(k) for k in legacy.THREAD_ENV},
        packages={k:importlib.metadata.version(k) for k in CONTRACT['required_packages']},resources=resources(directory))
    work.write(directory/'manifest.json',manifest)
    old_handlers={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGINT,signal.SIGTERM)}
    def stop(signum,frame):
        if signum!=signal.SIGALRM or time.monotonic()>=deadline:raise ProducerStop('Whole producer deadline/external interruption')
        try:resource_check(directory)
        except Exception as error:raise ProducerStop('Owned resource ceiling: '+repr(error)) from error
    for s in old_handlers:signal.signal(s,stop)
    signal.setitimer(signal.ITIMER_REAL,.5,.5)
    def execute(name,contexts,seconds):
        nonlocal stage
        stage=name;resource_check(directory,launch=True);event(directory,'stage_started',name=name)
        eligible=[t for t in specs[name] if t['case']['id'] in contexts]
        mark=time.monotonic()
        try:
            if eligible:terminal[name]=work.execute(eligible,contexts,directory/name,libraries,workers=CONTRACT['workers'],deadline=deadline,per_task=seconds,worker_fn=worker)
            else:terminal[name]=[]
        finally:stage_times[name]=time.monotonic()-mark
        terminal[name]=fill_terminal(specs[name],terminal[name]);event(directory,'stage_finished',name=name,status_counts=dict(Counter(r['status'] for r in terminal[name])))
        return _packets(directory/name,terminal[name])
    try:
        resource_check(directory,launch=True)
        seeds=acquire.load_seed_records(ROOT);mark=time.monotonic();libraries=work.build_libraries(directory/'build',deadline);stage_times['build']=time.monotonic()-mark
        reference_packets=execute('references',{s['setting']:dict(seed=s) for s in seeds},CONTRACT['reference_task_seconds'])
        contexts={t['id']:dict(reference=reference_packets[t['case']['setting']]) for t in specs['propagators']
            if reference_packets.get(t['case']['setting'],{}).get('qualified') is True}
        raw=execute('propagators',contexts,CONTRACT['reference_task_seconds'])
        propagator_packets={(p['setting'],p['phase_index']):p for p in raw.values()}
        stage='reference_preparation';mark=time.monotonic()
        prepared=evaluation.prepare_reference_outputs(reference_packets,propagator_packets)
        saved=save_packet(directory,'reference-seal',prepared)
        prepared=load_packet(saved['path'])
        prepared_replay=evaluation.replay_prepared_reference_outputs(prepared,reference_packets,propagator_packets)
        work.write(directory/'reference-seal-replay.json',prepared_replay)
        if not prepared_replay['match']:raise ValueError('Saved reference seal exact replay failed')
        event(directory,'reference_stage_sealed',path=Path(saved['path']).name,sha256=saved['sha256'],semantic_sha256=saved['semantic_sha256'])
        stage_times['reference_preparation_and_seal']=time.monotonic()-mark
        contexts={t['id']:dict(reference=reference_packets[t['case']['setting']]) for t in specs['traces']
            if reference_packets.get(t['case']['setting'],{}).get('qualified') is True
            and propagator_packets.get((t['case']['setting'],t['case']['phase_index']),{}).get('qualified') is True
            and prepared.get('contexts',{}).get(f"{t['case']['setting']}--p{t['case']['phase_index']}",{}).get('status')=='complete'
            and prepared.get('cases',{}).get(t['id'].rsplit('--',1)[0],{}).get('status')=='complete'}
        event(directory,'traces_started',reference_seal_sha256=saved['sha256'])
        # Do not inherit the ~264MB sealed null packet in four forked workers.
        prepared=None;gc.collect()
        trace_packets=execute('traces',contexts,CONTRACT['trace_task_seconds'])
        stage='saved_analysis';mark=time.monotonic();prepared=load_packet(directory/'reference-seal.json')
        prepared_replay=evaluation.replay_prepared_reference_outputs(prepared,reference_packets,propagator_packets)
        if not prepared_replay['match']:raise ValueError('Reference seal changed before scoring')
        analysis=analyze(reference_packets,propagator_packets,trace_packets,prepared)
        stage_times['saved_analysis']=time.monotonic()-mark
    except BaseException as exc:
        error=repr(exc);work.write(directory/'producer-error.json',dict(stage=stage,error=error,at=legacy.now(),all_existing_evidence_retained=True))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        # No target work remains. Failure sealing has its own fixed ceiling.
        def sealing_stop(signum,frame):raise ProducerStop('Failure-sealing grace exceeded; existing partial files retained')
        signal.signal(signal.SIGALRM,sealing_stop);signal.setitimer(signal.ITIMER_REAL,CONTRACT['sealing_grace_seconds'])
        try:
            for name,expected in specs.items():
                path=directory/name/'terminal-records.json'
                if name not in terminal and path.exists():terminal[name]=read(path)
                terminal[name]=fill_terminal(expected,terminal.get(name,[]))
            work.write(directory/'all-terminal-records.json',terminal)
            finalization_errors=[]
            if analysis is None:
                # Reload all completed worker evidence after an interrupted stage;
                # never discard earlier jobs merely because execute raised.
                recovered={}
                for name in specs:
                    try:recovered[name]=_packets(directory/name,terminal[name])
                    except Exception as exc:
                        recovered[name]={};finalization_errors.append(dict(stage=name,error=repr(exc)))
                reference_packets=recovered['references']
                propagator_packets={(p['setting'],p['phase_index']):p for p in recovered['propagators'].values()}
                trace_packets=recovered['traces']
                try:
                    if prepared is None:
                        seal_path=directory/'reference-seal.json'
                        prepared=load_packet(seal_path) if seal_path.exists() else evaluation.prepare_reference_outputs(reference_packets,propagator_packets)
                    checked=evaluation.replay_prepared_reference_outputs(prepared,reference_packets,propagator_packets)
                    if not checked['match']:raise ValueError('Saved reference seal mismatch during failure scoring')
                    analysis=analyze(reference_packets,propagator_packets,trace_packets,prepared)
                except Exception as exc:
                    finalization_errors.append(dict(stage='saved_analysis',error=repr(exc)))
                    prerequisites={setting:dict(reference=False,propagators=[False]*8) for setting in acquire.SETTINGS}
                    analysis=dict(scored_records=[],trace_comparisons=[],paired_dynamics=[dict(ids=[t['id'].replace('--s-1--','--s+1--'),t['id']],status='failed',error='analysis_unavailable') for t in acquire.build_tasks() if t['kind']=='pulse' and t['sign']==-1],analysis_errors=finalization_errors,
                        summary=evaluation.summarize([],[],prerequisites=prerequisites))
            if finalization_errors:
                work.write(directory/'finalization-errors.json',finalization_errors)
                error=error or 'Saved-evidence finalization failure'
            save_packet(directory,'analysis',analysis)
            try:validate(plan,bindings=True,reconstruct=False)
            except Exception as exc:postflight_error=repr(exc)
            work.write(directory/'postflight.json',dict(resources=resources(directory),error=postflight_error))
            complete=all(r['status']=='complete' for rows in terminal.values() for r in rows)
            summary=dict(experiment_id='EXP-014',complete=complete,qualified=complete and bool(analysis['summary']['qualified']) and error is None and postflight_error is None,
                error=error,postflight_error=postflight_error,status_counts={name:dict(Counter(r['status'] for r in rows)) for name,rows in terminal.items()},
                scientific=analysis['summary'],stage_wall_seconds=stage_times,producer_and_analysis_seconds=time.monotonic()-started,
                ended_at=legacy.now(),scope=plan['claim_boundary'])
            work.write(directory/'summary.json',summary)
            artifacts=work.inventory(directory)
            work.write(directory/'receipt.json',dict(**provenance,plan_sha256=sha(plan_path),directory=str(directory),
                experiment_id='EXP-014',status='complete_qualified' if summary['qualified'] else 'complete_unqualified' if complete else 'incomplete',
                artifacts=artifacts,artifact_bytes=sum(a['bytes'] for a in artifacts),sealed_at=legacy.now()))
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            for s,handler in old_handlers.items():signal.signal(s,handler)
    return summary



def bind_packet(task,row,packet):
    """Bind successful and failed raw payloads back to the immutable task row."""
    case=task['case'];setting=case['setting'];expected=acquire.SETTINGS[setting]
    if task['kind']=='trace':
        evaluation._task(packet.get('task'))
        if packet['task']!=case:raise ValueError('Trace payload task differs')
    else:
        if packet.get('setting')!=setting:raise ValueError('Reference/propagator setting differs')
    if any(packet.get(key)!=expected[key] for key in ('alpha','gamma')):raise ValueError('Payload parameter identity differs')
    if task['kind']=='reference':
        acquire._validate_seed(packet['seed'])
        if packet['seed']['setting']!=setting:raise ValueError('Reference seed setting differs')
    elif task['kind']=='propagator':
        if type(packet.get('phase_index')) is not int or packet['phase_index']!=case['phase_index']:raise ValueError('Propagator phase differs')
        if packet['phase_epoch']!=case['phase_index']*packet['period']/8:raise ValueError('Propagator epoch differs')
    if row['status']=='complete':
        if packet.get('status')!='complete' or row.get('qualified') is not True:raise ValueError('Completed raw packet qualification differs')
        if task['kind']=='trace':
            if packet.get('complete') is not True or packet.get('duration')!=320. or packet.get('sample_dt')!=.025:raise ValueError('Trace completion/grid declaration differs')
        elif packet.get('qualified') is not True:raise ValueError('Prerequisite completion differs')
        if task['kind']=='reference':
            if packet['orbit']['solver']!=acquire.FINE or packet['adjoint']['solver']!=acquire.FINE:raise ValueError('Reference solver declaration differs')
        elif task['kind']=='propagator' and packet['solver']!=acquire.FINE:raise ValueError('Propagator solver declaration differs')


def verify(directory):
    """Saved packet, chronology and complete readout replay; no integration."""
    directory=Path(directory).resolve();receipt=read(directory/'receipt.json');manifest=read(directory/'manifest.json');plan=manifest['plan']
    if receipt['plan_sha256']!=manifest['plan_sha256'] or receipt['source_commit']!=manifest['source_commit']:raise ValueError('Manifest/source binding differs')
    original=subprocess.check_output(['git','-C',str(ROOT),'show',f'{receipt["source_commit"]}:experiments/hopfield/EXP-014-recovery-plan.json'])
    if hashlib.sha256(original).hexdigest()!=receipt['plan_sha256'] or json.loads(original)!=plan:raise ValueError('Committed original plan differs')
    validate(plan,bindings=True,reconstruct=False)
    if work.inventory(directory)!=receipt['artifacts']:raise ValueError('Full artifact inventory differs')
    terminal=read(directory/'all-terminal-records.json');expected=stages(plan)
    if set(terminal)!=set(expected):raise ValueError('Stage census differs')
    packets={}
    for stage,spec in expected.items():
        rows=terminal[stage]
        if [r['id'] for r in rows]!=[t['id'] for t in spec]:raise ValueError('Full ordered task census differs')
        for task,row in zip(spec,rows,strict=True):
            if any(row[k]!=task[k] for k in ('id','kind')) or row['case_id']!=task['case']['id'] or row['method']!=task['method']['id']:raise ValueError('Task metadata differs')
            if row['status']=='complete':
                path=directory/stage/'records'/f'{row["id"]}.json'
                if not path.exists() or read(path)!=row or not row.get('packet_path'):raise ValueError('Completed worker packet/record missing')
        stage_path=directory/stage
        if stage_path.exists():
            cleanup=read(stage_path/'cleanup.json')
            if cleanup['all_registered_workers_joined'] is not True or cleanup['all_original_identities_absent'] is not True:raise ValueError('Owned cleanup did not pass')
            events=[json.loads(x) for x in (stage_path/'processes.jsonl').read_text().splitlines()] if (stage_path/'processes.jsonl').exists() else []
            if len(events)!=len(cleanup['workers']) or len({e['task_id'] for e in events})!=len(events):raise ValueError('Cleanup census differs')
            if set(e['task_id'] for e in events)-set(t['id'] for t in spec):raise ValueError('Unplanned owned worker')
            saved_rows=read(stage_path/'terminal-records.json');row_index={r['id']:r for r in rows}
            if len({r['id'] for r in saved_rows})!=len(saved_rows) or any(row_index.get(r['id'])!=r for r in saved_rows):raise ValueError('Stage terminal ledger differs')
            for owner,check in zip(events,cleanup['workers'],strict=True):
                if any(check[k]!=v for k,v in owner.items()) or check['original_identity_absent'] is not True:raise ValueError('Cleanup identity differs')
            for row in rows:
                if row['status']=='complete' and not any(e['task_id']==row['id'] and e['pid']==row['pid'] for e in events):raise ValueError('Completed worker ownership missing')
        elif any(r['status']!='not_started_dependency_failure' for r in rows):raise ValueError('Executed stage missing owned cleanup')
        packets[stage]=_packets(directory/stage,rows)
        for task,row in zip(spec,rows,strict=True):
            if row['id'] in packets[stage]:bind_packet(task,row,packets[stage][row['id']])
    references=packets['references'];propagators={(p['setting'],p['phase_index']):p for p in packets['propagators'].values()}
    for reference in references.values():acquire.replay_saved_reference(reference)
    for (setting,phase),propagator in propagators.items():
        if propagator.get('qualified') is True:
            reference=references[setting];initial=reference['phases'][phase]['initial']
            residuals=acquire.replay_saved_variational(propagator,reference['alpha'],reference['gamma'],reference['period'],initial)
            if residuals['relative_closure']>acquire.REFERENCE_GATES['relative_closure']:raise ValueError('Phase propagator closure gate failed')
    prepared=evaluation.prepare_reference_outputs(references,propagators)
    seal_path=directory/'reference-seal.json'
    trace_processes=directory/'traces'/'processes.jsonl'
    dispatched=trace_processes.exists() and bool(trace_processes.read_text().strip())
    if seal_path.exists():
        prepared=load_packet(seal_path)
        checked=evaluation.replay_prepared_reference_outputs(prepared,references,propagators)
        if not checked['match']:raise ValueError('Saved reference seal exact replay failed')
        replay_path=directory/'reference-seal-replay.json'
        if replay_path.exists():compare(read(replay_path),checked,'reference-seal-replay')
        elif dispatched:raise ValueError('Trace dispatch missing saved reference replay')
        event_path=directory/'events.jsonl'
        events=[json.loads(x) for x in event_path.read_text().splitlines()] if event_path.exists() else []
        seals=[e for e in events if e['stage']=='reference_stage_sealed'];starts=[e for e in events if e['stage']=='traces_started']
        if len(seals)>1 or len(starts)>1 or (dispatched and (len(seals)!=1 or len(starts)!=1)):raise ValueError('Reference seal chronology census differs')
        if seals and (seals[0]['sha256']!=sha(seal_path) or seals[0]['semantic_sha256']!=packet_digest(prepared)):raise ValueError('Reference seal binding differs')
        if starts and (not seals or seals[0]['monotonic_ns']>=starts[0]['monotonic_ns'] or starts[0]['reference_seal_sha256']!=sha(seal_path)):raise ValueError('Reference seal chronology/binding differs')
        for row in terminal['traces']:
            if 'packet_path' in row and (not starts or type(row.get('started_monotonic_ns')) is not int or row['started_monotonic_ns']<=starts[0]['monotonic_ns']):raise ValueError('Trace worker preceded reference seal')
    elif dispatched or any(r['status']=='complete' for r in terminal['traces']):raise ValueError('Trace acquired before reference seal')
    analysis=analyze(references,propagators,packets['traces'],prepared);compare(load_packet(directory/'analysis.json'),analysis,'analysis')
    summary=read(directory/'summary.json')
    compare(summary['scientific'],work.baseline_engine.safe(analysis['summary']),'summary.scientific')
    if summary['status_counts']!={k:dict(Counter(r['status'] for r in v)) for k,v in terminal.items()}:raise ValueError('Summary census differs')
    postflight=read(directory/'postflight.json')
    if summary['postflight_error']!=postflight['error']:raise ValueError('Postflight conclusion differs')
    producer_error=read(directory/'producer-error.json')['error'] if (directory/'producer-error.json').exists() else None
    finalization_failed=(directory/'finalization-errors.json').exists()
    expected_error=producer_error or ('Saved-evidence finalization failure' if finalization_failed else None)
    if summary['error']!=expected_error:raise ValueError('Producer error conclusion differs')
    complete=all(r['status']=='complete' for rows in terminal.values() for r in rows)
    qualified=complete and bool(analysis['summary']['qualified']) and expected_error is None and postflight['error'] is None
    if type(summary['complete']) is not bool or type(summary['qualified']) is not bool or summary['complete']!=complete or summary['qualified']!=qualified:raise ValueError('Top-level completion/qualification differs')
    expected_status='complete_qualified' if qualified else 'complete_unqualified' if complete else 'incomplete'
    if receipt['status']!=expected_status or receipt['artifact_bytes']!=sum(x['bytes'] for x in receipt['artifacts']):raise ValueError('Receipt conclusion/count differs')
    return dict(passed=True,tasks=sum(map(len,terminal.values())),artifact_count=len(receipt['artifacts']),saved_array_replay=True,new_dynamics=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','validate','run'):
        p=commands.add_parser(name);p.add_argument('--plan',type=Path,default=PLAN)
        if name=='run':p.add_argument('--data-root',type=Path,default=legacy.DEFAULT_DATA_ROOT);p.add_argument('--attempt',required=True)
    p=commands.add_parser('verify');p.add_argument('directory',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':work.write(args.plan,build_plan());result=dict(plan=str(args.plan),sha256=sha(args.plan))
    elif args.command=='validate':result=validate(read(args.plan),bindings=True,reconstruct=True)
    elif args.command=='verify':result=verify(args.directory)
    else:result=run(args.plan,args.data_root,args.attempt)
    print(json.dumps(work.baseline_engine.safe(result),indent=2,allow_nan=False))
    if args.command=='run' and not result['qualified']:raise SystemExit(1)


if __name__=='__main__':main()
