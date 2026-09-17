#!/usr/bin/env python3
"""EXP-017 guarded reference-only reconstruction and immutable saved replay."""
from __future__ import annotations
import argparse
from collections import Counter
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

import run_two_write_memory as prior
from run_two_write_memory import work, legacy, codec, runtime
import phase_reference_diagnostic as acquisition
import phase_reference_diagnostic_evaluation as evaluation

ROOT=prior.ROOT
PREVIOUS=Path('/data2/neural-shrimp-data/raw/EXP-016/attempt-001')
PREVIOUS_SHA='5bb1ce4aa6e3934af04049853b621628ab9cf86e36a6b4091db75b01211f47ef'
PLAN_REL='experiments/hopfield/EXP-017-plan.json'
REVIEW_REL='docs/EXP-017-local-review.json'
METHODS=('baseline-dop853','fine-dop853','fine-rk45')
FINE=METHODS[1:]
SETTINGS=('core','failed-near-edge')
SOURCE_PATHS=list(dict.fromkeys([
    'tools/run_phase_reference_diagnostic.py','tools/phase_reference_diagnostic.py',
    'tools/phase_reference_diagnostic_evaluation.py',
    'tests/test_phase_reference_diagnostic.py','tests/test_phase_reference_diagnostic_evaluation.py',
    'tests/test_phase_reference_diagnostic_runner.py','docs/EXP-017-protocol.md',
    'experiments/hopfield/EXP-016-plan.json',*prior.SOURCE_PATHS]))
CONTRACT=dict(experiment_id='EXP-017',exploratory=True,base_jobs=6,event_jobs=96,contexts=48,
    total_jobs=102,methods=list(METHODS),event_methods=list(FINE),rtol=1e-12,atol=1e-14,
    baseline_max_step=.025,fine_max_step_period_divisor=8192,grids=[4096,8192],
    terminal_adjoint='unchanged inherited vector; no normalization',workers=4,
    producer_seconds=1800,per_job_seconds=180,sealing_seconds=180,
    normalization_threshold=1e-7,state_threshold=1e-6,activation_threshold=1e-7,
    epsilon=1e-4,relative_closure_threshold=1e-8,adjoint_periodicity_threshold=1e-7,
    source_reference_receipt_sha256=prior.OLD_RECEIPT_SHA,previous_receipt_sha256=PREVIOUS_SHA,
    launch_ram_bytes=16*prior.GIB,reserve_ram_bytes=8*prior.GIB,
    owned_rss_bytes=12*prior.GIB,raw_budget_bytes=8*prior.GIB,
    required_packages=dict(numpy='2.5.3',scipy='1.18.1'),paid_compute=False,
    new_pulses=False,new_memory_scores=False,original_study_requalified=False)
read=prior.read
binding=prior.binding


def stages(phases):
    bases=[];events=[]
    for setting in SETTINGS:
        for method in METHODS:
            ident=f'{setting}--{method}'
            bases.append(dict(id=ident,kind='base',case=dict(id=ident,setting=setting),method=dict(id=method)))
        for phase in phases:
            for method in FINE:
                ident=f'{setting}--p{phase["phase_index"]:02d}--{method}'
                events.append(dict(id=ident,kind='event',case=dict(id=ident,setting=setting,
                    base_id=f'{setting}--{method}',**phase),method=dict(id=method)))
    return dict(bases=bases,events=events)


def historical_inputs():
    refs=[dict(root=str(prior.OLD),**row) for row in prior.old_inputs()]
    if legacy.sha(PREVIOUS/'receipt.json')!=PREVIOUS_SHA:raise ValueError('Previous receipt differs')
    receipt=read(PREVIOUS/'receipt.json')
    if receipt['source_commit']!='9904368b7b13721292efdc84dcf31efd7add30c7':raise ValueError('Previous source differs')
    index={r['path']:r for r in receipt['artifacts']}
    if len(index)!=len(receipt['artifacts']):raise ValueError('Duplicate previous paths')
    refs.append(dict(root=str(PREVIOUS),**binding(PREVIOUS/'receipt.json',PREVIOUS)))
    for setting in SETTINGS:
        for phase in range(24):
            for suffix in ('json','npz'):
                rel=f'seals/{setting}--p{phase:02d}.{suffix}'
                item=binding(PREVIOUS/rel,PREVIOUS)
                if item!=index[rel]:raise ValueError('Previous seal differs: '+rel)
                refs.append(dict(root=str(PREVIOUS),**item))
    return refs


def build_plan():
    phases=prior.phase_rows();ledger=stages(phases)
    return dict(**CONTRACT,date='2026-09-17',phases=phases,
        acquisition_methods=acquisition.METHODS,evaluation_contract=evaluation.contract(),
        source_files=[binding(ROOT/p) for p in SOURCE_PATHS],historical_inputs=historical_inputs(),
        census={k:dict(count=len(v),sha256=hashlib.sha256(prior.canonical(v)).hexdigest()) for k,v in ledger.items()},
        planning=dict(prior_reference_jobs=96,prior_reference_summed_worker_seconds=476.75,
            estimated_event_summed_worker_seconds=60,scope='duration scaling proxy, not a new measured pilot'))


def validate(plan):
    if plan!=build_plan():raise ValueError('Frozen plan reconstruction/bindings differ')
    ledger=stages(plan['phases'])
    if [len(ledger[k]) for k in ('bases','events')]!=[6,96]:raise ValueError('Planned census differs')
    return dict(total_jobs=102,contexts=48)


def guard(plan,path,head):
    if not re.fullmatch('[0-9a-f]{40}',head):raise ValueError('Exact expected HEAD required')
    if legacy.git('status','--porcelain','--untracked-files=all'):raise ValueError('Clean checkout required')
    branch=legacy.git('branch','--show-current')
    if legacy.git('rev-parse','HEAD')!=head or not branch.startswith('codex/'):raise ValueError('HEAD/branch differs')
    if Path(path).resolve()!=ROOT/PLAN_REL:raise ValueError('Canonical plan path required')
    for rel in [*SOURCE_PATHS,PLAN_REL,REVIEW_REL]:
        b=subprocess.check_output(['git','-C',str(ROOT),'show',f'{head}:{rel}'])
        if b!=(ROOT/rel).read_bytes():raise ValueError('Committed source differs')
    review=read(ROOT/REVIEW_REL)
    if review.get('status')!='ready_for_exploratory_execution' or review.get('plan_sha256')!=legacy.sha(path):raise ValueError('Exact-plan local review required')
    if legacy.git('remote','get-url','origin')!='git@github-personal:tdj28/neural-shrimp.git':raise ValueError('Expected private origin required')
    if legacy.git('ls-remote','--exit-code','origin',f'refs/heads/{branch}').split()[0]!=head:raise ValueError('Private push not verified')
    return dict(source_commit=head,pushed_branch=branch,remote_verified_at=legacy.now(),review=binding(ROOT/REVIEW_REL))


def external_attempt(data_root,attempt):
    if re.fullmatch(r'attempt-\d+',attempt) is None:raise ValueError('Named attempt required')
    root=Path(data_root).resolve()
    for protected in (ROOT,prior.OLD,PREVIOUS):
        if root==protected or root.is_relative_to(protected):raise ValueError('Evidence output must be outside repository and old attempts')
    parent=root/'EXP-017';parent.mkdir(parents=True,exist_ok=True)
    if parent.resolve()!=parent:raise ValueError('Aliased output forbidden')
    dest=parent/attempt;dest.mkdir();return dest


def inputs():
    return dict(references={s:codec.load_packet(prior.OLD/f'references/payloads/{s}/packet.json') for s in SETTINGS},
        oldseals={f'{s}--p{p:02d}':codec.load_packet(PREVIOUS/f'seals/{s}--p{p:02d}.json') for s in SETTINGS for p in range(24)})


def worker(task,context,directory,libraries,parent_pid):
    started=time.monotonic();cpu=time.process_time();directory=Path(directory)
    dest=directory/'payloads'/task['id'];dest.mkdir()
    row=dict(id=task['id'],kind=task['kind'],case_id=task['case']['id'],method=task['method']['id'],
        status='failed',qualified=False,pid=os.getpid(),identity=work.identity(os.getpid()),started_at=legacy.now(),rmse=None)
    def cancelled(signum,frame):
        if os.getppid()!=parent_pid:os._exit(125)
        raise TimeoutError('Owned diagnostic worker interrupted')
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,cancelled)
    packet=None
    try:
        if task['kind']=='base':packet=acquisition.acquire_base(context['reference'],context['phases'],task['method']['id'])
        elif task['kind']=='event':packet=acquisition.acquire_event(context['base'],task['case']['phase_index'])
        else:raise ValueError('Unknown diagnostic task')
        row.update(status=packet['status'],qualified=False)
    except BaseException as exc:
        row['error']=repr(exc)
        if hasattr(exc,'payload'):packet=exc.payload
    if packet is not None:
        try:
            seal=codec.save_packet(dest,'packet',packet)
            row.update(packet_path=str(Path(seal['path']).relative_to(directory)),packet_sha256=seal['sha256'],packet_semantic_sha256=seal['semantic_sha256'])
        except BaseException as exc:row.update(status='serialization_failed',error=repr(exc))
    row.update(ended_at=legacy.now(),wall_seconds=time.monotonic()-started,cpu_seconds=time.process_time()-cpu,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    work.write(directory/'records'/f'{task["id"]}.json',row)


def packets(directory,stage,specs,rows):
    index={r['id']:r for r in rows};result={}
    for task in specs:
        packet=prior.load_task_packet(Path(directory)/stage,task,index[task['id']])
        if packet is not None:result[task['id']]=packet
    return result


def check_packet_bindings(plan,sealed,bases,events):
    for ident,base in bases.items():
        setting=ident.rsplit('--',1)[0];reference=sealed['references'][setting]
        if base['id']!=ident or base['setting']!=setting:raise ValueError('Base packet identity differs')
        for name in ('alpha','gamma','period'):
            if base[name]!=reference[name]:raise ValueError('Inherited parameter differs')
        for name,wanted in [('section_state',reference['certificate']['section_state']),
                            ('terminal_adjoint',reference['adjoint']['terminal_normalized_once'])]:
            if not prior.np.array_equal(base['inherited'][name],wanted):raise ValueError('Inherited boundary differs')
        if len(base['phases'])!=24:raise ValueError('Complete base phase census required')
        for index,point in enumerate(base['phases']):
            phase=plan['phases'][index]
            if any(point[k]!=phase[k] for k in ('phase_index','phase_fraction','split')):raise ValueError('Base phase census differs')
            old=sealed['oldseals'][f'{setting}--p{index:02d}']['acquisition']
            if not prior.np.array_equal(point['old_hermite']['state'],old['initial']) or not prior.np.array_equal(point['old_hermite']['z'],old['phase_gradient']):raise ValueError('Old Hermite sample differs')
    for ident,packet in events.items():
        setting,phase_name,method=ident.split('--');index=int(phase_name[1:]);base_id=f'{setting}--{method}'
        if packet['id']!=ident or packet['base_id']!=base_id or packet['method']!=method or packet['phase_index']!=index:raise ValueError('Event packet identity differs')
        if base_id not in bases:raise ValueError('Event lacks complete base')
        base=bases[base_id];point=base['phases'][index]
        for k in ('setting','alpha','gamma','period'):
            if packet[k]!=base[k]:raise ValueError('Event parameters differ')
        if not prior.np.array_equal(packet['initial'],point['state']) or not prior.np.array_equal(packet['phase_gradient'],point['z']):raise ValueError('Event initial/gradient differs from base')


def analyze(directory,plan,rows,sealed):
    ledger=stages(plan['phases'])
    bases=packets(directory,'bases',ledger['bases'],rows['bases'])
    events=packets(directory,'events',ledger['events'],rows['events'])
    check_packet_bindings(plan,sealed,bases,events)
    return evaluation.summarize(bases,events,sealed['oldseals'])


def run(plan_path,data_root,attempt,head):
    path=Path(plan_path).resolve();plan=read(path);validate(plan);provenance=guard(plan,path,head)
    if any(importlib.metadata.version(k)!=v for k,v in CONTRACT['required_packages'].items()):raise ValueError('Numerical versions differ')
    if any(os.environ.get(k)!='1' for k in legacy.THREAD_ENV):raise ValueError('One-thread numerical settings required')
    directory=external_attempt(data_root,attempt);started=time.monotonic();deadline=started+CONTRACT['producer_seconds']
    ledger=stages(plan['phases']);rows={k:[] for k in ledger};analysis=None;error=None;post_error=None;stage='setup';timings={}
    work.write(directory/'manifest.json',dict(**provenance,plan=plan,plan_sha256=legacy.sha(path),started_at=legacy.now(),
        producer_pid=os.getpid(),producer_identity=work.identity(os.getpid()),python=sys.version,platform=platform.platform(),
        executable=sys.executable,argv=sys.argv,cwd=str(Path.cwd()),thread_env={k:os.environ[k] for k in legacy.THREAD_ENV},
        packages={k:importlib.metadata.version(k) for k in CONTRACT['required_packages']}))
    work.write(directory/'planned-ledger.json',ledger)
    handlers={s:signal.getsignal(s) for s in (signal.SIGALRM,signal.SIGINT,signal.SIGTERM)}
    def stop(signum,frame):
        if signum!=signal.SIGALRM or time.monotonic()>=deadline:raise prior.ProducerStop('Diagnostic deadline/interruption')
        try:prior.resource_check(directory)
        except Exception as exc:raise prior.ProducerStop(repr(exc)) from exc
    for sig in handlers:signal.signal(sig,stop)
    signal.setitimer(signal.ITIMER_REAL,1.,1.)
    try:
        prior.resource_check(directory,launch=True);stage='input_seal'
        seal=codec.save_packet(directory,'input-seal',inputs());sealed=codec.load_packet(seal['path'])
        prior.event(directory,'inputs_sealed',sha256=seal['sha256'],semantic_sha256=seal['semantic_sha256'])
        stage='bases';mark=time.monotonic();prior.event(directory,'bases_started',planned=6)
        contexts={t['case']['id']:dict(reference=sealed['references'][t['case']['setting']],phases=plan['phases']) for t in ledger['bases']}
        rows['bases']=runtime.execute(ledger['bases'],contexts,directory/'bases',{},workers=4,deadline=deadline,per_task=180,worker_fn=worker)
        timings['bases']=time.monotonic()-mark
        bases=packets(directory,'bases',ledger['bases'],rows['bases'])
        base_index={ident:dict(metadata=binding(directory/'bases'/'payloads'/ident/'packet.json',directory),semantic_sha256=codec.packet_digest(p)) for ident,p in bases.items()}
        work.write(directory/'base-seal-index.json',base_index)
        prior.event(directory,'bases_sealed',sha256=legacy.sha(directory/'base-seal-index.json'))
        stage='events';mark=time.monotonic()
        eligible=[t for t in ledger['events'] if t['case']['base_id'] in bases]
        contexts={t['case']['id']:dict(base=bases[t['case']['base_id']]) for t in eligible}
        prior.event(directory,'events_started',planned=96,eligible=len(eligible))
        if eligible:rows['events']=runtime.execute(eligible,contexts,directory/'events',{},workers=4,deadline=deadline,per_task=180,worker_fn=worker)
        rows['events']=prior.fill_rows(ledger['events'],rows['events'],'not_started_base_unavailable')
        timings['events']=time.monotonic()-mark
        stage='analysis';analysis=analyze(directory,plan,rows,sealed)
    except BaseException as exc:
        error=repr(exc);work.write(directory/'producer-error.json',dict(stage=stage,error=error,at=legacy.now()))
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        def sealing_stop(signum,frame):raise prior.ProducerStop('Diagnostic sealing deadline')
        signal.signal(signal.SIGALRM,sealing_stop);signal.setitimer(signal.ITIMER_REAL,180)
        try:
            for name,specs in ledger.items():
                p=directory/name/'terminal-records.json'
                if not rows[name] and p.exists():rows[name]=read(p)
                rows[name]=prior.fill_rows(specs,rows[name],'not_started_producer_failure')
            work.write(directory/'all-terminal-records.json',rows)
            if analysis is None:analysis=dict(status='unavailable',error=error)
            codec.save_packet(directory,'analysis',analysis)
            try:
                validate(plan)
                if legacy.git('rev-parse','HEAD')!=head or legacy.git('status','--porcelain','--untracked-files=all'):raise ValueError('Checkout changed during diagnostic')
            except Exception as exc:post_error=repr(exc)
            work.write(directory/'postflight.json',dict(source_bindings_unchanged=post_error is None,error=post_error))
            complete=error is None and post_error is None and all(r['status']=='complete' for rs in rows.values() for r in rs) and analysis.get('diagnostic_complete') is True
            summary=dict(experiment_id='EXP-017',complete=complete,error=error,postflight_error=post_error,source_valid=post_error is None,
                status_counts={k:dict(Counter(r['status'] for r in rs)) for k,rs in rows.items()},analysis=analysis,
                stage_wall_seconds=timings,total_wall_seconds=time.monotonic()-started,original_study_requalified=False)
            work.write(directory/'summary.json',summary);prior.event(directory,'producer_finished',complete=complete)
            artifacts=prior.inventory(directory)
            work.write(directory/'receipt.json',dict(**provenance,experiment_id='EXP-017',directory=str(directory),plan_sha256=legacy.sha(path),
                status='complete' if complete else 'incomplete_or_invalid',artifacts=artifacts,artifact_bytes=sum(r['bytes'] for r in artifacts),sealed_at=legacy.now()))
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            for sig,handler in handlers.items():signal.signal(sig,handler)
    return dict(directory=str(directory),receipt_sha256=legacy.sha(directory/'receipt.json'),complete=complete,error=error,status_counts=summary['status_counts'])


def verify(directory):
    directory=Path(directory).resolve();receipt=read(directory/'receipt.json');manifest=read(directory/'manifest.json');plan=manifest['plan']
    if receipt['artifacts']!=prior.inventory(directory):raise ValueError('Byte inventory differs')
    if receipt['artifact_bytes']!=sum(r['bytes'] for r in receipt['artifacts']):raise ValueError('Artifact bytes differ')
    if receipt['source_commit']!=manifest['source_commit'] or receipt['plan_sha256']!=manifest['plan_sha256'] or receipt['experiment_id']!='EXP-017':raise ValueError('Receipt identity differs')
    b=subprocess.check_output(['git','-C',str(ROOT),'show',f'{receipt["source_commit"]}:{PLAN_REL}'])
    if hashlib.sha256(b).hexdigest()!=receipt['plan_sha256'] or json.loads(b)!=plan:raise ValueError('Committed plan differs')
    for item in [*plan['source_files'],manifest['review']]:
        b=subprocess.check_output(['git','-C',str(ROOT),'show',f'{receipt["source_commit"]}:{item["path"]}'])
        if len(b)!=item['bytes'] or hashlib.sha256(b).hexdigest()!=item['sha256']:raise ValueError('Committed source differs')
    validate(plan);ledger=stages(plan['phases']);rows=read(directory/'all-terminal-records.json');processes=[]
    if read(directory/'planned-ledger.json')!=ledger or set(rows)!=set(ledger):raise ValueError('Ledger differs')
    for stage,specs in ledger.items():
        outer={r['id']:r for r in rows[stage]};index={t['id']:t for t in specs}
        if len(rows[stage])!=len(specs) or set(outer)!=set(index):raise ValueError('Terminal census differs')
        for ident,row in outer.items():
            t=index[ident]
            if any(row.get(k)!=v for k,v in dict(kind=t['kind'],case_id=t['case']['id'],method=t['method']['id']).items()):raise ValueError('Terminal task identity differs')
        p=directory/stage
        if not p.exists():
            if any(r.get('pid') or r['status']=='complete' for r in outer.values()):raise ValueError('Missing lifecycle')
            continue
        cleanup=read(p/'cleanup.json');owned=[json.loads(x) for x in (p/'processes.jsonl').read_text().splitlines()] if (p/'processes.jsonl').exists() else []
        terminal=read(p/'terminal-records.json');journal=[json.loads(x) for x in (p/'journal.jsonl').read_text().splitlines()] if (p/'journal.jsonl').exists() else []
        if terminal!=journal or len({r['id'] for r in terminal})!=len(terminal):raise ValueError('Terminal journal differs')
        if cleanup['all_registered_workers_joined'] is not True or cleanup['all_original_identities_absent'] is not True or len(owned)!=len(cleanup['workers']):raise ValueError('Cleanup differs')
        registered={r['task_id']:r for r in owned};inner={r['id']:r for r in terminal}
        if len(registered)!=len(owned) or not set(registered).issubset(inner) or not set(inner).issubset(outer):raise ValueError('Dispatch census differs')
        for a,c in zip(owned,cleanup['workers'],strict=True):
            if any(c.get(k)!=v for k,v in a.items()) or c['original_identity_absent'] is not True:raise ValueError('Cleanup identity differs')
        for row in outer.values():
            if row.get('pid') is not None or row['status']=='complete':
                if inner.get(row['id'])!=row or row['id'] not in registered:raise ValueError('Missing registered result')
        for row in terminal:
            if outer.get(row['id'])!=row:raise ValueError('Outer terminal differs')
            if row.get('pid') and registered[row['id']]['pid']!=row['pid']:raise ValueError('Worker ownership differs')
            if row['status']=='complete' and read(p/'records'/f'{row["id"]}.json')!=row:raise ValueError('Worker record differs')
        processes.extend([dict(r,stage=stage) for r in owned])
    events=[json.loads(s) for s in (directory/'events.jsonl').read_text().splitlines()] if (directory/'events.jsonl').exists() else []
    ev={e['event']:e for e in events}
    if len(ev)!=len(events):raise ValueError('Repeated stage event')
    sealed=None
    if (directory/'input-seal.json').exists():
        sealed=codec.load_packet(directory/'input-seal.json');codec.compare(sealed,prior.normalized(inputs()),'input-seal')
        if 'inputs_sealed' in ev:
            if ev['inputs_sealed']['sha256']!=legacy.sha(directory/'input-seal.json') or ev['inputs_sealed']['semantic_sha256']!=codec.packet_digest(sealed):raise ValueError('Input chronology/hash differs')
        elif processes or any(r['status']=='complete' for rs in rows.values() for r in rs):raise ValueError('Dispatch preceded input seal event')
    elif processes or any(r['status']=='complete' for rs in rows.values() for r in rs):raise ValueError('Dispatch without input seal')
    bases=packets(directory,'bases',ledger['bases'],rows['bases'])
    event_packets=packets(directory,'events',ledger['events'],rows['events'])
    if sealed is not None:check_packet_bindings(plan,sealed,bases,event_packets)
    for owned in processes:
        boundary='inputs_sealed' if owned['stage']=='bases' else 'bases_sealed'
        if owned['started_at']<ev[boundary]['at']:raise ValueError('Worker started before prerequisite seal')
    if any(r.get('pid') is not None for r in rows['events']):
        if not (ev['inputs_sealed']['monotonic_ns']<ev['bases_started']['monotonic_ns']<ev['bases_sealed']['monotonic_ns']<ev['events_started']['monotonic_ns']):raise ValueError('Stage chronology differs')
        expected={ident:dict(metadata=binding(directory/'bases'/'payloads'/ident/'packet.json',directory),semantic_sha256=codec.packet_digest(p)) for ident,p in bases.items()}
        if read(directory/'base-seal-index.json')!=expected or ev['bases_sealed']['sha256']!=legacy.sha(directory/'base-seal-index.json'):raise ValueError('Base seal differs')
        for t in ledger['events']:
            row=next(r for r in rows['events'] if r['id']==t['id'])
            if row.get('pid') is not None and t['case']['base_id'] not in bases:raise ValueError('Event had no base prerequisite')
    saved=codec.load_packet(directory/'analysis.json');summary=read(directory/'summary.json');post=read(directory/'postflight.json')
    replayed=False
    if saved.get('status')!='unavailable':
        if sealed is None:raise ValueError('Analysis lacks input seal')
        replay=analyze(directory,plan,rows,sealed);codec.compare(saved,prior.normalized(replay),'analysis');replayed=True
    elif not (directory/'producer-error.json').exists():raise ValueError('Unavailable analysis lacks failure')
    else:
        for ident,packet in bases.items():evaluation.validate_base_packet(packet,expected_id=ident)
        for ident,packet in event_packets.items():evaluation.validate_event_packet(packet,bases[packet['base_id']],expected_id=ident)
    codec.compare(summary['analysis'],work.baseline_engine.safe(saved),'summary.analysis')
    error=read(directory/'producer-error.json')['error'] if (directory/'producer-error.json').exists() else None
    complete=error is None and post['error'] is None and all(r['status']=='complete' for rs in rows.values() for r in rs) and saved.get('diagnostic_complete') is True
    if summary['error']!=error or summary['postflight_error']!=post['error'] or summary['source_valid'] is not (post['error'] is None) or post['source_bindings_unchanged'] is not (post['error'] is None):raise ValueError('Failure/source status differs')
    if summary['complete']!=complete or receipt['status']!=('complete' if complete else 'incomplete_or_invalid'):raise ValueError('Completion differs')
    if summary['status_counts']!={k:dict(Counter(r['status'] for r in rs)) for k,rs in rows.items()}:raise ValueError('Counts differ')
    all_owned=processes+[dict(pid=manifest['producer_pid'],identity=manifest['producer_identity'])]
    return dict(complete=complete,saved_array_replay=replayed,total_jobs=102,receipt_sha256=legacy.sha(directory/'receipt.json'),
        all_original_owned_identities_absent_now=all(work.identity(r['pid']) is None or work.identity(r['pid'])!=r['identity'] for r in all_owned))


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',default=str(ROOT/PLAN_REL))
    p=sub.add_parser('validate');p.add_argument('--plan',default=str(ROOT/PLAN_REL))
    p=sub.add_parser('run');p.add_argument('--plan',default=str(ROOT/PLAN_REL));p.add_argument('--data-root',required=True);p.add_argument('--attempt',required=True);p.add_argument('--expected-head',required=True)
    p=sub.add_parser('verify');p.add_argument('--attempt-dir',required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        plan=build_plan();work.write(args.output,plan);result=dict(plan=args.output,sha256=legacy.sha(args.output))
    elif args.command=='validate':result=validate(read(args.plan))
    elif args.command=='run':result=run(args.plan,args.data_root,args.attempt,args.expected_head)
    else:result=verify(args.attempt_dir)
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)


if __name__=='__main__':main()
