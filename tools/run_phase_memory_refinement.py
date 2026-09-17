#!/usr/bin/env python3
"""Outcome-selected EXP014 refinement, entirely separate from original evidence.

prepare/validate read only saved metadata/arrays; run alone acquires new traces.
No original prerequisite, fit, classifier, seal, tolerance or label is rewritten.
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
import signal
import subprocess
import sys
import time

import fresh_preview_runtime as work
import run_hopfield_refinement as legacy
import run_phase_memory_recovery as codec
import phase_memory_acquisition as acquisition
import phase_memory as math_helpers
import phase_memory_refinement_evaluation as evaluate
import numpy as np
from scipy.integrate import solve_ivp
from hopfield_diagnostics import hopfield_rhs
from hopfield_drift_probe import integrate_trace,ProbeFailure

ROOT=work.ROOT
OLD=Path('/data2/neural-shrimp-data/raw/EXP-014/attempt-002')
OLD_SOURCE='406b89553b956bac1b97ef05413c799e383a9103'
OLD_RECEIPT_SHA='2494966914cb68d8c5286a5dca6710212ade171ead42c9cfaed82cebd44c12a8'
PLAN_REL='experiments/hopfield/EXP-014-refinement-plan.json'
REVIEW_REL='docs/EXP-014-refinement-review.json'
NEW_SOURCE_PATHS=['tools/run_phase_memory_refinement.py','tools/phase_memory_refinement_evaluation.py',
    'tests/test_phase_memory_refinement.py','tests/test_phase_memory_refinement_evaluation.py',
    'docs/EXP-014-refinement-protocol.md','docs/EXP-014-refinement-design-amendment.md']
NEW_METHODS=[dict(id=f'rk4-{dt:g}',kind='RK4',dt=dt) for dt in (.000625,.0003125,.00015625)]+[
    dict(id='dop853-1e-13-h0.0125',kind='DOP853',rtol=1e-13,atol=1e-15,max_step=.0125),
    dict(id='dop853-1e-13-h0.00625',kind='DOP853',rtol=1e-13,atol=1e-15,max_step=.00625)]
CONTRACT=dict(experiment_id='EXP-014-refinement',schema_version='EXP014-refinement-v1',outcome_selected=True,
    physical_cases=85,failed_signed_cases=18,binary_contexts=13,phase_contexts=7,new_traces=425,old_traces=255,
    method_pairs=2380,pairs_new_new=850,pairs_old_new=1275,pairs_old_old=255,paired_ba_cells=3744,
    duration=320.,sample_dt=.025,samples=12801,new_methods=NEW_METHODS,old_methods=list(acquisition.METHODS),
    workers=4,thread_limit=1,producer_seconds=900,per_trace_seconds=60,sealing_seconds=120,
    launch_available_ram_bytes=16*1024**3,reserve_ram_bytes=8*1024**3,owned_rss_bytes=12*1024**3,raw_budget_bytes=2*1024**3,
    potential_max_threshold=1e-6,activation_max_threshold=1e-7,primary_standardized_readout_threshold=.01,
    late_interval=[240.,320.],readout_delays=[1,2,4,8],noise_levels=[1e-5,1e-4,1e-3],primary_noise=1e-4,
    required_packages=dict(numpy='2.5.3',scipy='1.18.1'),paid_compute=False,paid_review=False)
SCOPE=('Outcome-selected numerical diagnosis of 85 matched physical cases, not a new population study. '
       'All original numerical failures, labels, readouts and unqualified setting conclusions remain unchanged. '
       'No setting/cohort means, convergence-order claim, policy fitting, replacement cases or prerequisite reruns. '
       'Same-method paired BA is diagnostic and cannot establish reliable waveforms.')

class ProducerStop(BaseException):
    """Deadline/resource cancellation bypasses per-case Exception handlers."""


def read(path):return json.loads(Path(path).read_text())
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def binding(path,root):return dict(path=str(Path(path).relative_to(root)),bytes=Path(path).stat().st_size,sha256=legacy.sha(path))

def inventory(directory):
    directory=Path(directory);result=[]
    for path in sorted(directory.rglob('*')):
        if path.is_symlink():raise ValueError('Owned evidence cannot contain aliases')
        if path.is_file() and path!=directory/'receipt.json':result.append(binding(path,directory))
    return result


def source_file(relative):
    return ROOT/relative


def select_cases(summary,old_tasks):
    """Exact final85-case rule, deterministic original-census order."""
    if old_tasks!=acquisition.build_tasks():raise ValueError('Original912 task census differs')
    tasks={t['id']:t for t in old_tasks};failed=summary['scientific']['trace_comparisons']
    failed=[r for r in failed if r['qualified'] is not True]
    if len(failed)!=30 or any(r.get('missing') is not False for r in failed):raise ValueError('Original failure census differs')
    selected_signed={ident.rsplit('--',1)[0] for row in failed for ident in row['ids']}
    if len(selected_signed)!=18:raise ValueError('Exact18 failed signed identities required')
    failed_tasks=[tasks[ident] for row in failed for ident in row['ids']]
    if any(t['kind']!='pulse' or t['dose_index']!=2 or t['epsilon']!=.01 for t in failed_tasks):raise ValueError('Selected failures differ from strong-pulse rule')
    binary={(t['setting'],t['phase_index'],t['direction']) for t in failed_tasks}
    phases={(s,p) for s,p,_ in binary}
    if len(binary)!=13 or len(phases)!=7:raise ValueError('Exact13binary/7phase contexts required')
    cases=[];categories={}
    for task in old_tasks:
        if task['method']!=acquisition.METHODS[0]:continue
        key=(task['setting'],task['phase_index'])
        if not ((task['kind']=='sham' and key in phases) or (task['kind']=='pulse' and (*key,task['direction']) in binary)):continue
        case={k:v for k,v in task.items() if k!='method'};case['id']=task['id'].rsplit('--',1)[0];cases.append(case)
        categories[case['id']]='matched-sham' if case['kind']=='sham' else 'original-failed-strong-sign' if case['id'] in selected_signed else 'matched-strong-opposite-sign' if case['dose_index']==2 else 'matched-lower-dose-both-signs'
    if len(cases)!=85 or Counter(c['setting'] for c in cases)!=dict(core=26,**{'failed-near-edge':59}):raise ValueError('Exact85 physical case census differs')
    return dict(cases=cases,selection_categories=categories,failed_signed_ids=sorted(selected_signed),
        binary_contexts=[list(v) for v in sorted(binary)],phase_contexts=[list(v) for v in sorted(phases)],
        original_failed_comparisons=failed,case_census_sha256=hashlib.sha256(canonical(cases)).hexdigest())


def old_metadata():
    if legacy.sha(OLD/'receipt.json')!=OLD_RECEIPT_SHA:raise ValueError('Original receipt differs')
    receipt=read(OLD/'receipt.json');index={r['path']:r for r in receipt['artifacts']}
    if len(index)!=len(receipt['artifacts']) or receipt['source_commit']!=OLD_SOURCE:raise ValueError('Original receipt source/census differs')
    def checked(name):
        if binding(OLD/name,OLD)!=index[name]:raise ValueError('Original artifact differs: '+name)
        return read(OLD/name)
    manifest=checked('manifest.json');summary=checked('summary.json');ledger=checked('all-terminal-records.json')
    if manifest['source_commit']!=OLD_SOURCE or manifest['plan_sha256']!=receipt['plan_sha256']:raise ValueError('Original plan/source differs')
    return manifest,summary,ledger,index


def input_paths(selection,ledger):
    paths=['receipt.json','manifest.json','summary.json','all-terminal-records.json','postflight.json','events.jsonl',
           'reference-seal.json','reference-seal.npz','reference-seal-replay.json','analysis.json','analysis.npz']
    ids={case['id'] for case in selection['cases']};contexts={tuple(x) for x in selection['phase_contexts']}
    for stage,rows in ledger.items():
        for row in rows:
            wanted=(stage=='traces' and row['id'].rsplit('--',1)[0] in ids)
            if stage=='references':wanted=True
            if stage=='propagators':
                # The original task's phase identity is saved in the packet metadata.
                tree=read(OLD/stage/row['packet_path'])['tree']
                wanted=(tree.get('setting'),tree.get('phase_index')) in contexts
            if not wanted:continue
            if row['status']!='complete' or row.get('qualified') is not True:raise ValueError('Required original raw task incomplete')
            meta=Path(stage)/row['packet_path']
            if meta.is_absolute() or '..' in meta.parts:raise ValueError('Original packet path escape')
            descriptor=read(OLD/meta)
            raw=meta.parent/descriptor['arrays']['path']
            if Path(descriptor['arrays']['path']).name!=descriptor['arrays']['path']:raise ValueError('Original array path escape')
            paths.extend([str(Path(stage)/'records'/f'{row["id"]}.json'),str(meta),str(raw)])
    return list(dict.fromkeys(paths))


def build_plan():
    manifest,summary,ledger,index=old_metadata();selection=select_cases(summary,manifest['plan']['tasks'])
    inputs=[]
    for name in input_paths(selection,ledger):
        item=binding(OLD/name,OLD)
        if name!='receipt.json' and index.get(name)!=item:raise ValueError('Receipt does not bind required original input: '+name)
        inputs.append(item)
    source_paths=list(dict.fromkeys(codec.SOURCE_PATHS+NEW_SOURCE_PATHS))
    sources=[dict(path=rel,bytes=source_file(rel).stat().st_size,sha256=legacy.sha(source_file(rel))) for rel in source_paths]
    return dict(**CONTRACT,scope=SCOPE,original_directory=str(OLD),original_source=OLD_SOURCE,original_receipt_sha256=OLD_RECEIPT_SHA,
        selection=selection,source_files=sources,input_files=inputs,original_prerequisites=summary['scientific']['prerequisites'],
        original_setting_qualification={k:v['qualified'] for k,v in summary['scientific']['settings'].items()},
        selection_rule='Both signs at all3doses for13binarycontexts containing the18originalfailedstrongsigns;7phase-matchedshams.',
        first_crossing_scope='First exceeding saved time for potentials/activations; first among4declared completed readout windows. Never truncates acquisition.',
        failure_policy='Preserve425terminaltasks,680old/newmethodscore rows,2380pair comparisons and3744same-methodBA cells including unavailable entries.')


def tasks(plan):
    return [dict(id=case['id']+'--'+method['id'],kind='refinement_trace',case=case,method=method)
            for case in plan['selection']['cases'] for method in NEW_METHODS]


def validate(plan,*,bindings=True,reconstruct=False):
    if any(plan.get(k)!=v for k,v in CONTRACT.items()) or plan.get('scope')!=SCOPE:raise ValueError('Frozen refinement contract differs')
    if plan.get('original_directory')!=str(OLD) or plan.get('original_source')!=OLD_SOURCE or plan.get('original_receipt_sha256')!=OLD_RECEIPT_SHA:
        raise ValueError('Original attempt binding differs')
    manifest,summary,ledger,index=old_metadata();selection=select_cases(summary,manifest['plan']['tasks'])
    if plan['selection']!=selection or len(tasks(plan))!=425:raise ValueError('Exact matched85-case rule differs')
    if [r['path'] for r in plan['input_files']]!=input_paths(selection,ledger):raise ValueError('Input path census differs')
    if [r['path'] for r in plan['source_files']]!=list(dict.fromkeys(codec.SOURCE_PATHS+NEW_SOURCE_PATHS)):
        raise ValueError('Source path census differs')
    if bindings:
        for item in plan['input_files']:
            if binding(OLD/item['path'],OLD)!=item:raise ValueError('Original evidence drift: '+item['path'])
        for item in plan['source_files']:
            p=source_file(item['path'])
            if p.stat().st_size!=item['bytes'] or legacy.sha(p)!=item['sha256']:raise ValueError('Source drift: '+item['path'])
    if reconstruct and plan!=build_plan():raise ValueError('Plan reconstruction differs')
    return dict(physical_cases=85,new_traces=425,old_traces=255,comparisons=2380,paired_ba_cells=3744)


def source_guard(plan,path,expected_head):
    if not re.fullmatch('[0-9a-f]{40}',expected_head):raise ValueError('Exact expected source commit required')
    if legacy.git('status','--porcelain','--untracked-files=all'):raise ValueError('Clean checkout required')
    commit=legacy.git('rev-parse','HEAD');branch=legacy.git('branch','--show-current')
    if commit!=expected_head or not branch.startswith('codex/'):raise ValueError('Expected private source HEAD/branch differs')
    relative=str(Path(path).resolve().relative_to(ROOT))
    if relative!=PLAN_REL:raise ValueError('Canonical refinement plan path required')
    for rel in [r['path'] for r in plan['source_files']]+[relative,REVIEW_REL]:
        content=subprocess.check_output(['git','-C',str(ROOT),'show',f'{commit}:{rel}'])
        if hashlib.sha256(content).hexdigest()!=legacy.sha(ROOT/rel):raise ValueError('Source/plan/review not committed')
        declared=next((item for item in plan['source_files'] if item['path']==rel),None)
        if declared is not None and (declared['bytes']!=len(content) or declared['sha256']!=hashlib.sha256(content).hexdigest()):raise ValueError('Committed source differs from plan binding')
    review=read(ROOT/REVIEW_REL)
    if review.get('status')!='ready_for_exploratory_execution' or review.get('plan_sha256')!=legacy.sha(path):raise ValueError('Exact-plan local review required')
    if legacy.git('ls-remote','--exit-code','origin',f'refs/heads/{branch}').split()[0]!=commit:raise ValueError('Exact private source freeze not pushed')
    return dict(source_commit=commit,pushed_branch=branch,remote_verified_at=legacy.now(),review=binding(ROOT/REVIEW_REL,ROOT))


def make_contexts(plan,prepared):
    contexts={}
    for case in plan['selection']['cases']:
        cid=f'{case["setting"]}--p{case["phase_index"]}';saved=prepared['cases'][case['id']];clock=prepared['contexts'][cid]
        if (saved['status']!='complete' or clock['status']!='complete' or saved['case']!={k:v for k,v in case.items() if k!='id'}
                or saved['context_id']!=cid):raise ValueError('Original sealed case/context identity differs')
        setting=acquisition.SETTINGS[case['setting']]
        contexts[case['id']]=dict(setting=case['setting'],alpha=setting['alpha'],gamma=setting['gamma'],
            phase_initial=clock['initial'],phase_epoch=clock['phase_epoch'],period=clock['period'],
            pulse=saved['pulse'],orientation=saved['orientation'],readouts=clock['readouts'],predictions=saved['readouts'],
            original_reference_semantic_sha256=read(OLD/'reference-seal.json')['semantic_sha256'],original_case_id=case['id'])
    return contexts


def original_packets(plan,contexts,*,replay=True):
    ledger=read(OLD/'all-terminal-records.json')['traces'];index={r['id']:r for r in ledger};packets={};scores={}
    saved=codec.load_packet(OLD/'analysis.json');saved_scores={r['task']['id']:r for r in saved['scored_records']}
    for case in plan['selection']['cases']:
        for method in acquisition.METHODS:
            ident=case['id']+'--'+method;row=index[ident];path=OLD/'traces'/row['packet_path']
            if legacy.sha(path)!=row['packet_sha256']:raise ValueError('Original selected packet hash differs')
            packet=codec.load_packet(path)
            if codec.packet_digest(packet)!=row['packet_semantic_sha256']:raise ValueError('Original selected packet semantic binding differs')
            score=evaluate.score_packet(packet,case,contexts[case['id']])
            if score['status']!='complete':raise ValueError('Original selected clock scoring incomplete: '+ident)
            if replay:
                old=saved_scores[ident]
                if old['status']!='complete' or old['orientation']!=score['orientation']:raise ValueError('Original score identity differs')
                for delay in ('1','2','4','8'):
                    for name in ('q','noise_gain','phase_only_q','full_linear_q','zero_response_q'):
                        a,b=old['readouts'][delay][name],score['readouts'][delay][name]
                        if not np.isfinite(a) or not np.isfinite(b) or not np.isclose(a,b,rtol=1e-12,atol=1e-14):
                            raise ValueError('Original sealed-clock score replay differs: '+ident+'/'+delay+'/'+name)
            packets[ident]=packet;scores[ident]=score
    return packets,scores


def acquire_new(task,context,native_fn=None):
    """Same RHS and native integrator, truthful newmethod metadata/partial evidence."""
    case=task['case'];method=task['method'];flat=dict(case,id=task['id'],method=method['id'])
    packet=dict(task=flat,alpha=context['alpha'],gamma=context['gamma'],phase_initial=context['phase_initial'],
        phase_epoch=context['phase_epoch'],period=context['period'],**context['pulse'],duration=320.,sample_dt=.025,
        initial_activation_displacement=np.tanh(context['pulse']['initial'])-np.tanh(context['phase_initial']),
        status='started',complete=False,solver=dict(method))
    last_rhs={};started=time.monotonic()
    try:
        if method['kind']=='RK4':
            if native_fn is None:raise ValueError('Owned native library required')
            packet['solver'].update(method='RK4',steps=round(320/method['dt']),samples=12801)
            times,states=integrate_trace(native_fn,context['alpha'],context['gamma'],context['pulse']['initial'],method['dt'],duration=320.,sample_dt=.025)
            packet.update(times=times,states=states)
        else:
            def rhs(t,x):
                last_rhs.update(time=float(t),values=np.array(x,dtype=np.float64,copy=True),accepted_solver_state=False)
                return hopfield_rhs(t,x,context['alpha'],context['gamma'])
            solution=solve_ivp(rhs,(0.,320.),context['pulse']['initial'],method='DOP853',t_eval=acquisition.TIMES,
                rtol=method['rtol'],atol=method['atol'],max_step=method['max_step'])
            packet.update(times=np.asarray(solution.t,dtype=np.float64),states=np.asarray(solution.y.T,dtype=np.float64))
            packet['solver'].update(method='DOP853',success=bool(solution.success),nfev=int(solution.nfev),njev=int(solution.njev),nlu=int(solution.nlu),message=str(solution.message))
            if not solution.success:raise ValueError('Independent solver returned failure')
        if (packet['times'].dtype!=np.float64 or packet['states'].dtype!=np.float64
                or not np.array_equal(packet['times'],acquisition.TIMES) or packet['states'].shape!=(12801,3)
                or not np.isfinite(packet['states']).all() or not np.array_equal(packet['states'][0],context['pulse']['initial'])):
            raise ValueError('Complete unchanged grid/finite states/exact initial required')
        packet.update(status='complete',complete=True)
    except BaseException as error:
        packet.update(status='failed',complete=False,error=repr(error))
        if isinstance(error,ProbeFailure):packet.update(times=error.payload['times'],states=error.payload['states'],native_failure=error.payload)
        if last_rhs:packet['last_rhs']=last_rhs
    packet['integration_seconds']=time.monotonic()-started
    return packet


def worker(task,context,directory,libraries,parent_pid):
    start=time.monotonic();cpu=time.process_time();directory=Path(directory);dest=directory/'payloads'/task['id'];dest.mkdir()
    row=dict(id=task['id'],kind=task['kind'],case_id=task['case']['id'],method=task['method']['id'],status='failed',qualified=False,
        pid=os.getpid(),identity=work.identity(os.getpid()),started_at=legacy.now(),rmse=None)
    def cancelled(signum,frame):
        if os.getppid()!=parent_pid:os._exit(125)
        raise TimeoutError('Owned child cancellation; parent has forced-kill fallback')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,cancelled)
    packet=None
    try:
        fn=legacy.load_native(libraries['trace']['path']) if task['method']['kind']=='RK4' else None
        packet=acquire_new(task,context,fn)
        row.update(status=packet['status'],qualified=packet['complete'] is True)
    except BaseException as error:row['error']=repr(error)
    if packet is not None:
        try:
            sealed=codec.save_packet(dest,'packet',packet)
            row.update(packet_path=str(Path(sealed['path']).relative_to(directory)),packet_sha256=sealed['sha256'],packet_semantic_sha256=sealed['semantic_sha256'])
        except BaseException as error:row.update(status='serialization_failed',qualified=False,error=repr(error))
    row.update(ended_at=legacy.now(),wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    work.write(directory/'records'/f'{task["id"]}.json',row)


def load_new_packets(directory,rows,specs):
    directory=Path(directory);expected={t['id']:t for t in specs};packets={};errors=[]
    for row in rows:
        try:
            task=expected[row['id']]
            if any(row.get(k)!=v for k,v in dict(kind='refinement_trace',case_id=task['case']['id'],method=task['method']['id']).items()):raise ValueError('Terminal identity differs')
            metadata=row
            if 'packet_path' not in row:
                child=directory/'records'/f'{row["id"]}.json'
                if not child.exists():continue
                metadata=read(child)
                if any(metadata.get(k)!=row.get(k) for k in ('id','kind','case_id','method','pid')):raise ValueError('Partial child identity differs')
                if 'packet_path' not in metadata:continue
            expected_path=f'payloads/{row["id"]}/packet.json'
            if metadata['packet_path']!=expected_path:raise ValueError('Packet path differs')
            path=directory/expected_path
            if legacy.sha(path)!=metadata['packet_sha256']:raise ValueError('Packet file hash differs')
            packet=codec.load_packet(path)
            if codec.packet_digest(packet)!=metadata['packet_semantic_sha256']:raise ValueError('Packet semantic hash differs')
            if row['status']!='complete':packet=dict(packet,status=row['status'],complete=False,operational_status=row['status'])
            packets[row['id']]=packet
        except (ValueError,KeyError,TypeError,OSError) as error:errors.append(dict(id=row.get('id'),error=repr(error)))
    return packets,errors


def analyze(plan,directory,rows,contexts):
    packets,scores=original_packets(plan,contexts,replay=True)
    new,errors=load_new_packets(Path(directory)/'traces',rows,tasks(plan));packets.update(new)
    bycase={case['id']:case for case in plan['selection']['cases']}
    for task in tasks(plan):
        if task['id'] not in new:continue
        try:scores[task['id']]=evaluate.score_packet(new[task['id']],bycase[task['case']['id']],contexts[task['case']['id']])
        except (ValueError,KeyError,TypeError,FloatingPointError) as error:errors.append(dict(id=task['id'],error=repr(error)))
    result=evaluate.summarize(plan['selection']['cases'],packets,scores)
    return dict(summary=result,packet_errors=errors,original_failed_comparisons=plan['selection']['original_failed_comparisons'],
        original_setting_qualification=plan['original_setting_qualification'],scope=SCOPE)


def placeholder(plan,reason):
    return dict(summary=evaluate.summarize(plan['selection']['cases'],{},{}),packet_errors=[dict(stage='unavailable_analysis',error=reason)],
        original_failed_comparisons=plan['selection']['original_failed_comparisons'],original_setting_qualification=plan['original_setting_qualification'],scope=SCOPE)


def external_attempt(data_root,attempt):
    if not re.fullmatch('attempt-[0-9]{3}',attempt):raise ValueError('Immutable attempt-NNN required')
    path=(Path(data_root)/CONTRACT['experiment_id']/attempt).resolve()
    if any(path==base.resolve() or base.resolve() in path.parents for base in (ROOT,OLD)):
        raise ValueError('Output must be outside repository and original immutable evidence')
    path.parent.mkdir(parents=True,exist_ok=True);path.mkdir();return path


def event(directory,name,**fields):work.append(Path(directory)/'events.jsonl',dict(event=name,at=legacy.now(),monotonic_ns=time.monotonic_ns(),**fields))


def run(plan_path,data_root,attempt,expected_head):
    plan=read(plan_path);validate(plan,reconstruct=True);provenance=source_guard(plan,plan_path,expected_head)
    if any(importlib.metadata.version(k)!=v for k,v in CONTRACT['required_packages'].items()):raise ValueError('Frozen numerical versions differ')
    directory=external_attempt(data_root,attempt);start=time.monotonic();deadline=start+900;specs=tasks(plan)
    manifest=dict(**provenance,plan=plan,plan_sha256=legacy.sha(plan_path),started_at=legacy.now(),producer_pid=os.getpid(),producer_identity=work.identity(os.getpid()),
        executable=sys.executable,argv=list(sys.argv),cwd=str(Path.cwd()),python=sys.version,platform=platform.platform(),
        packages={k:importlib.metadata.version(k) for k in CONTRACT['required_packages']},thread_env={k:os.environ.get(k) for k in legacy.THREAD_ENV})
    work.write(directory/'manifest.json',manifest)
    # The complete planned failure denominator exists before any target work.
    work.write(directory/'planned-ledger.json',dict(tasks=specs,comparison_specs=evaluate.comparison_specs(plan['selection']['cases']),required_paired_ba_cells=3744))
    rows=[];analysis=None;contexts=None;error=None;stage='preflight';postflight_error=None;timings={}
    handlers={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)}
    def stop(signum,frame):
        if signum!=signal.SIGALRM or time.monotonic()>=deadline:raise ProducerStop('Whole900-second deadline/external interruption')
        try:codec.resource_check(directory)
        except Exception as exc:raise ProducerStop('Owned resource ceiling: '+repr(exc)) from exc
    for sig in handlers:signal.signal(sig,stop)
    signal.setitimer(signal.ITIMER_REAL,.5,.5)
    try:
        codec.resource_check(directory,launch=True);mark=time.monotonic();stage='saved_input_replay'
        prepared=codec.load_packet(OLD/'reference-seal.json');contexts=make_contexts(plan,prepared);prepared=None;gc.collect()
        old_packets,old_scores=original_packets(plan,contexts,replay=True)
        work.write(directory/'old-score-replay.json',dict(complete=True,scores_replayed=len(old_scores),required=255,source_receipt_sha256=OLD_RECEIPT_SHA))
        old_packets=None;old_scores=None;gc.collect()
        sealed=codec.save_packet(directory,'input-seal',contexts);contexts=codec.load_packet(sealed['path'])
        event(directory,'original_inputs_sealed',sha256=sealed['sha256'],semantic_sha256=sealed['semantic_sha256']);timings[stage]=time.monotonic()-mark
        stage='build';mark=time.monotonic();libraries=work.build_libraries(directory/'build',deadline);timings[stage]=time.monotonic()-mark
        stage='traces';codec.resource_check(directory,launch=True);mark=time.monotonic();event(directory,'traces_started',planned=425)
        rows=work.execute(specs,contexts,directory/'traces',libraries,workers=4,deadline=deadline,per_task=60,worker_fn=worker)
        timings[stage]=time.monotonic()-mark;event(directory,'traces_finished',statuses=dict(Counter(r['status'] for r in rows)))
        stage='saved_analysis';mark=time.monotonic();analysis=analyze(plan,directory,rows,contexts);timings[stage]=time.monotonic()-mark
    except BaseException as exc:
        error=repr(exc);work.write(directory/'producer-error.json',dict(stage=stage,error=error,at=legacy.now()))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        def sealing_stop(signum,frame):raise ProducerStop('120-second sealing grace exceeded; all existing evidence retained')
        signal.signal(signal.SIGALRM,sealing_stop);signal.setitimer(signal.ITIMER_REAL,120.)
        try:
            path=directory/'traces/terminal-records.json'
            if not rows and path.exists():rows=read(path)
            try:rows=codec.fill_terminal(specs,rows,'not_started_prerequisite_or_producer_failure')
            except Exception as exc:
                work.write(directory/'invalid-terminal-records.json',dict(records=rows,error=repr(exc)));error=error or repr(exc)
                rows=codec.fill_terminal(specs,[],'invalid_terminal_census');analysis=None
            work.write(directory/'all-terminal-records.json',rows)
            if analysis is None:analysis=placeholder(plan,error or 'Analysis unavailable; saved partial traces retained for separate replay')
            codec.save_packet(directory,'analysis',analysis)
            try:
                validate(plan,reconstruct=False)
                if legacy.git('rev-parse','HEAD')!=expected_head or legacy.git('status','--porcelain','--untracked-files=all'):raise ValueError('Source checkout changed')
            except Exception as exc:postflight_error=repr(exc)
            work.write(directory/'postflight.json',dict(error=postflight_error,source_bindings_unchanged=postflight_error is None))
            complete=all(r['status']=='complete' for r in rows) and error is None and not analysis['packet_errors'] and analysis['summary']['complete'] is True
            summary=dict(experiment_id=CONTRACT['experiment_id'],complete=complete,source_valid=postflight_error is None,error=error,postflight_error=postflight_error,
                status_counts=dict(Counter(r['status'] for r in rows)),analysis=analysis['summary'],original_unchanged=True,
                stage_wall_seconds=timings,producer_and_analysis_seconds=time.monotonic()-start,ended_at=legacy.now(),scope=SCOPE)
            work.write(directory/'summary.json',summary);items=inventory(directory)
            work.write(directory/'receipt.json',dict(**provenance,experiment_id=CONTRACT['experiment_id'],plan_sha256=legacy.sha(plan_path),directory=str(directory),
                status='complete' if complete and postflight_error is None else 'incomplete_or_invalid',artifacts=items,artifact_bytes=sum(x['bytes'] for x in items),sealed_at=legacy.now()))
        except BaseException as exc:
            try:work.write(directory/'sealing-error.json',dict(error=repr(exc),at=legacy.now(),planned_ledger='planned-ledger.json'))
            finally:raise
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            for sig,handler in handlers.items():signal.signal(sig,handler)
    return dict(directory=str(directory),complete=summary['complete'],source_valid=summary['source_valid'],error=error,
        receipt_sha256=legacy.sha(directory/'receipt.json'),status_counts=summary['status_counts'])


def verify(directory):
    """Read-only complete/partial source, ownership, packets and metric replay."""
    directory=Path(directory).resolve();receipt=read(directory/'receipt.json');manifest=read(directory/'manifest.json');plan=manifest['plan']
    if receipt['artifacts']!=inventory(directory):raise ValueError('Exact immutable artifact inventory differs')
    if (receipt['source_commit']!=manifest['source_commit'] or receipt['plan_sha256']!=manifest['plan_sha256']
            or receipt['experiment_id']!=CONTRACT['experiment_id']):raise ValueError('Receipt/manifest identity differs')
    commit=receipt['source_commit'];committed=subprocess.check_output(['git','-C',str(ROOT),'show',f'{commit}:{PLAN_REL}'])
    if hashlib.sha256(committed).hexdigest()!=receipt['plan_sha256'] or json.loads(committed)!=plan:raise ValueError('Committed original plan differs')
    for item in plan['source_files']+[manifest['review']]:
        content=subprocess.check_output(['git','-C',str(ROOT),'show',f'{commit}:{item["path"]}'])
        if len(content)!=item['bytes'] or hashlib.sha256(content).hexdigest()!=item['sha256']:raise ValueError('Committed source/review bytes differ')
    validate(plan,reconstruct=True)
    specs=tasks(plan);index={t['id']:t for t in specs};rows=read(directory/'all-terminal-records.json')
    if len(rows)!=425 or {r.get('id') for r in rows}!=set(index):raise ValueError('Exact425 terminal census differs')
    for row in rows:
        task=index[row['id']]
        required=dict(id=task['id'],kind='refinement_trace',case_id=task['case']['id'],method=task['method']['id'])
        if any(type(row.get(k)) is not type(v) or row.get(k)!=v for k,v in required.items()):raise ValueError('Terminal task identity differs')
    planned=read(directory/'planned-ledger.json')
    if planned!=dict(tasks=specs,comparison_specs=evaluate.comparison_specs(plan['selection']['cases']),required_paired_ba_cells=3744):raise ValueError('Planned full-denominator ledger differs')
    saved=codec.load_packet(directory/'analysis.json');summary=read(directory/'summary.json');post=read(directory/'postflight.json')
    error_packet=read(directory/'producer-error.json') if (directory/'producer-error.json').exists() else None
    invalid=read(directory/'invalid-terminal-records.json') if (directory/'invalid-terminal-records.json').exists() else None
    expected_error=error_packet['error'] if error_packet else invalid['error'] if invalid else None
    if summary['error']!=expected_error or summary['postflight_error']!=post['error'] or summary['source_valid'] is not (post['error'] is None):raise ValueError('Error/source conclusion differs')
    if post['source_bindings_unchanged'] is not (post['error'] is None):raise ValueError('Postflight flag differs')
    if summary['status_counts']!=dict(Counter(r['status'] for r in rows)):raise ValueError('Terminal count summary differs')
    processes=[];stage=directory/'traces'
    if stage.exists():
        cleanup=read(stage/'cleanup.json')
        if cleanup['all_registered_workers_joined'] is not True or cleanup['all_original_identities_absent'] is not True:raise ValueError('Owned cleanup incomplete')
        processes=[json.loads(line) for line in (stage/'processes.jsonl').read_text().splitlines()] if (stage/'processes.jsonl').exists() else []
        if len({r['task_id'] for r in processes})!=len(processes) or len(cleanup['workers'])!=len(processes) or any(r['task_id'] not in index for r in processes):raise ValueError('Owned process census differs')
        for owned,check in zip(processes,cleanup['workers'],strict=True):
            if any(check[k]!=v for k,v in owned.items()) or check['original_identity_absent'] is not True:raise ValueError('Owned process identity differs')
        terminal=read(stage/'terminal-records.json');journal=[json.loads(line) for line in (stage/'journal.jsonl').read_text().splitlines()] if (stage/'journal.jsonl').exists() else []
        if terminal!=journal or len(terminal)!=425 or {r['id'] for r in terminal}!=set(index):raise ValueError('Stage terminal/journal census differs')
        if invalid is None and {r['id']:r for r in terminal}!={r['id']:r for r in rows}:raise ValueError('Stage/parent terminal ledger differs')
        owned={r['task_id']:r for r in processes}
        for row in rows:
            if row.get('pid') is not None or row['status']=='complete':
                if row['id'] not in owned or row.get('pid')!=owned[row['id']]['pid']:raise ValueError('Dispatched task lacks ownership')
            if row['status']=='complete' and read(stage/'records'/f'{row["id"]}.json')!=row:raise ValueError('Worker record differs')
    elif any(r.get('pid') is not None or r['status']=='complete' for r in rows):raise ValueError('Dispatched task lacks cleanup stage')
    fallback=placeholder(plan,expected_error or 'Analysis unavailable; saved partial traces retained for separate replay')
    placeholder_saved=expected_error is not None and not codec_compare_differences(saved,fallback)
    if (directory/'input-seal.json').exists():
        prepared=codec.load_packet(OLD/'reference-seal.json');contexts=make_contexts(plan,prepared);prepared=None;gc.collect()
        sealed=codec.load_packet(directory/'input-seal.json');codec.compare(sealed,replay_normalized(contexts),'original-input-seal')
        events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()] if (directory/'events.jsonl').exists() else []
        seal_events=[e for e in events if e['event']=='original_inputs_sealed'];starts=[e for e in events if e['event']=='traces_started']
        if len(seal_events)>1 or len(starts)>1:raise ValueError('Duplicate input seal/start event')
        if stage.exists() and (len(seal_events)!=1 or len(starts)!=1 or seal_events[0]['monotonic_ns']>=starts[0]['monotonic_ns']):raise ValueError('Original-input seal chronology differs')
        if not stage.exists() and expected_error is None:raise ValueError('Absent trace stage needs an explicit producer failure')
        if seal_events and (seal_events[0]['sha256']!=legacy.sha(directory/'input-seal.json') or seal_events[0]['semantic_sha256']!=codec.packet_digest(sealed)):raise ValueError('Input seal event identity differs')
        replay=analyze(plan,directory,rows,contexts)
        if not placeholder_saved:codec.compare(saved,replay_normalized(replay),'analysis')
    elif stage.exists() or not placeholder_saved:raise ValueError('Trace stage requires prior bound input seal')
    codec.compare(summary['analysis'],work.baseline_engine.safe(saved['summary']),'summary.analysis')
    expected_complete=all(r['status']=='complete' for r in rows) and expected_error is None and not saved['packet_errors'] and saved['summary']['complete'] is True
    if type(summary['complete']) is not bool or summary['complete']!=expected_complete:raise ValueError('Completion conclusion differs')
    status='complete' if expected_complete and post['error'] is None else 'incomplete_or_invalid'
    if receipt['status']!=status or receipt['artifact_bytes']!=sum(r['bytes'] for r in receipt['artifacts']):raise ValueError('Receipt conclusion differs')
    return dict(complete=expected_complete and post['error'] is None,source_commit=commit,planned_new_traces=425,
        planned_comparisons=2380,planned_paired_ba_cells=3744,receipt_sha256=legacy.sha(directory/'receipt.json'),
        saved_array_replay=not placeholder_saved,partial_placeholder_verified=placeholder_saved,
        all_original_owned_identities_absent_now=all(work.identity(p['pid']) is None or work.identity(p['pid'])!=p['identity'] for p in processes))


def replay_normalized(value):
    """Normalize serialized scalar metadata; retain strict raw array identities."""
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:replay_normalized(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [replay_normalized(v) for v in value]
    return value


def codec_compare_differences(a,b):
    try:codec.compare(a,replay_normalized(b));return []
    except (ValueError,TypeError,KeyError) as error:return [str(error)]


def main():
    parser=argparse.ArgumentParser();subs=parser.add_subparsers(dest='command',required=True)
    p=subs.add_parser('prepare');p.add_argument('--out',required=True)
    p=subs.add_parser('validate');p.add_argument('--plan',required=True)
    p=subs.add_parser('run');p.add_argument('--plan',required=True);p.add_argument('--data-root',required=True);p.add_argument('--attempt',required=True);p.add_argument('--expected-head',required=True)
    p=subs.add_parser('verify');p.add_argument('--directory',required=True)
    args=parser.parse_args()
    if args.command=='prepare':plan=build_plan();work.write(args.out,plan);result=dict(plan=args.out,sha256=legacy.sha(args.out))
    elif args.command=='validate':result=validate(read(args.plan),reconstruct=True)
    elif args.command=='run':result=run(args.plan,args.data_root,args.attempt,args.expected_head)
    else:result=verify(args.directory)
    print(json.dumps(work.baseline_engine.safe(result),indent=2,allow_nan=False),flush=True)

if __name__=='__main__':main()
