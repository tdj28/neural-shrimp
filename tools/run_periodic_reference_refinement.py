#!/usr/bin/env python3
"""EXP-018 guarded reference-only reconstruction and immutable saved replay."""
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
import periodic_reference_refinement as acquisition
import periodic_reference_refinement_evaluation as evaluation

ROOT=prior.ROOT
PREVIOUS=Path('/data2/neural-shrimp-data/raw/EXP-017/attempt-001')
PREVIOUS_SHA='7d8daa8d4bfe816247c568149b040758785c82d4fd4bb94a97b9c71b625b40f7'
PREVIOUS_SOURCE='8fbd8ddcff5758526817e0f800a07f08c903a7c9'
PREVIOUS_SUMMARY='docs/EXP-017-summary.json'
PLAN_REL='experiments/hopfield/EXP-018-plan.json'
REVIEW_REL='docs/EXP-018-local-review.json'
PROTOCOL_REL='docs/EXP-018-protocol.md'
METHODS=('tight-dop853','tight-rk45')
SETTINGS=('core','failed-near-edge')
SOURCE_PATHS=list(dict.fromkeys([
    'tools/run_periodic_reference_refinement.py','tools/periodic_reference_refinement.py',
    'tools/periodic_reference_refinement_evaluation.py',
    'tools/phase_reference_diagnostic.py','tools/phase_reference_diagnostic_evaluation.py',
    'tests/test_periodic_reference_refinement.py','tests/test_periodic_reference_refinement_evaluation.py',
    'tests/test_periodic_reference_refinement_runner.py','tests/test_periodic_reference_refinement_integration.py',
    PROTOCOL_REL,PREVIOUS_SUMMARY,*prior.SOURCE_PATHS]))
CONTRACT=dict(experiment_id='EXP-018',exploratory=True,shooting_jobs=4,reference_jobs=2,event_jobs=96,contexts=48,
    total_jobs=102,nested_slots=208,shooting_trial_slots=100,cross_forward_slots=8,adjoint_slots=4,
    methods=list(METHODS),canonical_method='tight-dop853',workers=4,thread_limit=1,
    producer_seconds=1800,per_job_seconds=180,sealing_seconds=180,
    source_reference_receipt_sha256=prior.OLD_RECEIPT_SHA,previous_receipt_sha256=PREVIOUS_SHA,
    launch_ram_bytes=16*prior.GIB,reserve_ram_bytes=8*prior.GIB,
    owned_rss_bytes=12*prior.GIB,raw_budget_bytes=8*prior.GIB,
    required_packages=dict(numpy='2.5.3',scipy='1.18.1'),paid_compute=False,
    new_pulses=False,new_memory_scores=False,original_study_requalified=False,
    eight_period_memory_qualified=False)
read=prior.read
binding=prior.binding


def stages(phases):
    shoots=[];references=[];events=[]
    for setting in SETTINGS:
        for method in METHODS:
            ident=f'{setting}--{method}'
            shoots.append(dict(id=ident,kind='shoot',case=dict(id=ident,setting=setting),method=dict(id=method)))
        references.append(dict(id=f'{setting}--reference-pair',kind='reference',
            case=dict(id=setting,setting=setting),method=dict(id='paired')))
        for phase in phases:
            for method in METHODS:
                ident=f'{setting}--p{phase["phase_index"]:02d}--{method}'
                events.append(dict(id=ident,kind='event',case=dict(id=ident,setting=setting,
                    reference_id=f'{setting}--reference-pair',base_id=f'{setting}--{method}',**phase),method=dict(id=method)))
    return dict(shooting=shoots,references=references,events=events)


def nested_ledger(phases):
    rows=[]
    for setting in SETTINGS:
        for method in METHODS:
            for index in range(25):
                rows.append(dict(id=f'{setting}--{method}--trial-{index:02d}',kind='shooting_trial',
                    stage='shooting',worker_id=f'{setting}--{method}',index=index))
        for candidate in METHODS:
            for method in METHODS:
                rows.append(dict(id=f'{setting}--verify--{candidate}--{method}',kind='cross_forward',
                    stage='references',worker_id=f'{setting}--reference-pair',key=f'{candidate}--{method}'))
        for method in METHODS:
            rows.append(dict(id=f'{setting}--adjoint--{method}',kind='adjoint',stage='references',
                worker_id=f'{setting}--reference-pair',key=method))
        for phase in phases:
            for method in METHODS:
                ident=f'{setting}--p{phase["phase_index"]:02d}--{method}'
                rows.append(dict(id=ident,kind='event',stage='events',worker_id=ident))
    return rows


def historical_inputs():
    refs=[dict(root=str(prior.OLD),**row) for row in prior.old_inputs()]
    if legacy.sha(PREVIOUS/'receipt.json')!=PREVIOUS_SHA:raise ValueError('Previous receipt differs')
    receipt=read(PREVIOUS/'receipt.json')
    if receipt['source_commit']!=PREVIOUS_SOURCE:raise ValueError('Previous source differs')
    refs.append(dict(root=str(PREVIOUS),**binding(PREVIOUS/'receipt.json',PREVIOUS)))
    summary=read(ROOT/PREVIOUS_SUMMARY)
    if summary.get('source_commit')!=PREVIOUS_SOURCE or summary.get('raw_receipt',{}).get('sha256')!=PREVIOUS_SHA:
        raise ValueError('Previous compact summary provenance differs')
    refs.append(dict(root=str(ROOT),**binding(ROOT/PREVIOUS_SUMMARY,ROOT)))
    return refs


def build_plan():
    phases=prior.phase_rows();ledger=stages(phases);nested=nested_ledger(phases)
    sources=[]
    for p in SOURCE_PATHS:
        if p==PROTOCOL_REL and not (ROOT/p).exists():sources.append(dict(path=p,status='pending_protocol'))
        else:sources.append(binding(ROOT/p))
    return dict(**CONTRACT,date='2026-09-17',phases=phases,
        acquisition_methods=acquisition.METHODS,evaluation_contract=evaluation.contract(),
        source_files=sources,historical_inputs=historical_inputs(),
        ledger=ledger,nested_ledger=nested,
        census={k:dict(count=len(v),sha256=hashlib.sha256(prior.canonical(v)).hexdigest()) for k,v in ledger.items()},
        nested_census=dict(count=len(nested),sha256=hashlib.sha256(prior.canonical(nested)).hexdigest()),
        planning=dict(previous_experiment='EXP-017',event_jobs=96,event_summed_worker_seconds=157.198,
            base_jobs=6,base_summed_worker_seconds=17.032,maximum_ivp_slots=208,
            estimated_summed_worker_seconds=680,scope='historical step-density scaling proxy; no new target pilot'))


def validate(plan):
    if any(x.get('status')=='pending_protocol' for x in plan['source_files']):raise ValueError('Protocol must be bound before validation/execution')
    if plan!=build_plan():raise ValueError('Frozen plan reconstruction/bindings differ')
    ledger=stages(plan['phases']);nested=nested_ledger(plan['phases'])
    if [len(ledger[k]) for k in ('shooting','references','events')]!=[4,2,96] or len(nested)!=208:raise ValueError('Planned census differs')
    if len({t['id'] for v in ledger.values() for t in v})!=102 or len({t['id'] for t in nested})!=208:raise ValueError('Duplicate planned identity')
    return dict(total_jobs=102,contexts=48,nested_slots=208,historical_inputs=7)


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
    parent=root/'EXP-018';parent.mkdir(parents=True,exist_ok=True)
    if parent.resolve()!=parent:raise ValueError('Aliased output forbidden')
    dest=parent/attempt;dest.mkdir();return dest


def inputs():
    return dict(references={s:codec.load_packet(prior.OLD/f'references/payloads/{s}/packet.json') for s in SETTINGS})


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
        if task['kind']=='shoot':packet=acquisition.refine_cycle(context['reference'],task['method']['id'])
        elif task['kind']=='reference':packet=acquisition.crossverify_reference(context['reference'],context['candidates'],context['phases'])
        elif task['kind']=='event':packet=acquisition.acquire_refined_event(context['base'],task['case']['phase_index'])
        else:raise ValueError('Unknown diagnostic task')
        row.update(status='complete' if packet.get('complete') is True else packet['status'],qualified=packet.get('qualified') is True)
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


def packet_record(directory,task,row):
    """Parent termination stays authoritative; a saved child prefix is retained."""
    expected=dict(id=task['id'],kind=task['kind'],case_id=task['case']['id'],method=task['method']['id'])
    if any(row.get(k)!=v for k,v in expected.items()):raise ValueError('Terminal identity differs')
    source=row
    if source.get('packet_path') is None:
        p=Path(directory)/'records'/f'{task["id"]}.json'
        if not p.exists():
            if row['status']=='complete':raise ValueError('Complete task lacks packet')
            return None
        source=read(p)
        if any(source.get(k)!=v for k,v in expected.items()) or source.get('pid')!=row.get('pid'):raise ValueError('Partial worker identity differs')
        if source.get('packet_path') is None:return None
    relative=f'payloads/{task["id"]}/packet.json';path=Path(directory)/relative
    if source['packet_path']!=relative or path.resolve()!=path:raise ValueError('Packet path differs')
    if legacy.sha(path)!=source['packet_sha256']:raise ValueError('Packet metadata hash differs')
    packet=codec.load_packet(path)
    if codec.packet_digest(packet)!=source['packet_semantic_sha256']:raise ValueError('Packet semantic hash differs')
    if packet.get('id')!=task['id'] or packet.get('kind')!=task['kind']:raise ValueError('Packet identity differs')
    if row['status']=='complete' and (packet.get('complete') is not True or packet.get('status')!='complete'):raise ValueError('Complete row/packet differs')
    return packet


def packets(directory,stage,specs,rows):
    index={r['id']:r for r in rows};result={}
    for task in specs:
        packet=packet_record(Path(directory)/stage,task,index[task['id']])
        if packet is not None:result[task['id']]=packet
    return result


def all_packets(directory,plan,rows):
    return {stage:packets(directory,stage,specs,rows[stage]) for stage,specs in stages(plan['phases']).items()}


def packet_index(directory,stage,packets):
    return {ident:dict(metadata=binding(Path(directory)/stage/'payloads'/ident/'packet.json',Path(directory)),
        semantic_sha256=codec.packet_digest(p)) for ident,p in packets.items()}


def nested_records(plan,rows,packets_by_stage):
    result=[];outer={stage:{r['id']:r for r in values} for stage,values in rows.items()}
    for spec in nested_ledger(plan['phases']):
        packet=packets_by_stage[spec['stage']].get(spec['worker_id']);parent=outer[spec['stage']][spec['worker_id']];saved=None
        if packet is not None:
            if spec['kind']=='shooting_trial':
                trials=packet.get('trials',[])
                if len(trials)!=25 or len({r['id'] for r in trials})!=25:raise ValueError('Shooting trial census differs')
                saved=trials[spec['index']]
            elif spec['kind']=='cross_forward':saved=packet.get('verifications',{}).get(spec['key'])
            elif spec['kind']=='adjoint':saved=packet.get('adjoints',{}).get(spec['key'])
            else:saved=packet
        if saved is not None and saved.get('id')!=(spec['worker_id'] if spec['kind']=='event' else spec['id']):raise ValueError('Nested identity differs')
        row=dict(spec,owner_status=parent['status'],status=saved.get('status','unknown') if saved else 'unavailable',
            packet_available=saved is not None,ivp_started=saved.get('ivp_started') if saved else None,
            ivp_completed=saved.get('ivp_completed') if saved else None)
        if saved is None:row['reason']=parent['status']
        if saved is not None:
            for key in ('complete','qualified','attempted','accepted','reason','ivp_attempted','error','failed_stage'):
                if key in saved:row[key]=saved[key]
        result.append(row)
    return result


def check_packet_bindings(plan,sealed,loaded):
    shoots=loaded['shooting'];refs=loaded['references'];events=loaded['events']
    for ident,p in shoots.items():
        setting=p.get('setting')
        if setting not in SETTINGS or ident!=f'{setting}--{p.get("method")}':raise ValueError('Shooting identity differs')
        old=sealed['references'][setting]
        if p.get('alpha')!=old['alpha'] or p.get('gamma')!=old['gamma'] or p.get('original_period')!=old['period'] or not prior.np.array_equal(p.get('original_state'),old['certificate']['section_state']):raise ValueError('Shooting seed/parameter binding differs')
        if p.get('complete') is True:evaluation.validate_shoot_packet(p,old,expected_id=ident)
        else:
            diagnostic=evaluation.validate_partial_packet(p,old,plan['phases'])
            if not diagnostic['partial_replay_pass']:raise ValueError('Partial shooting arithmetic differs: '+repr(diagnostic['errors']))
    for ident,p in refs.items():
        setting=p.get('setting')
        if setting not in SETTINGS or ident!=f'{setting}--reference-pair':raise ValueError('Reference identity differs')
        old=sealed['references'][setting]
        if p.get('alpha')!=old['alpha'] or p.get('gamma')!=old['gamma'] or p.get('original_period')!=old['period'] or not prior.np.array_equal(p.get('original_state'),old['certificate']['section_state']):raise ValueError('Reference seed/parameter binding differs')
        candidates={f'{setting}--{m}':shoots[f'{setting}--{m}'] for m in METHODS}
        if any(c.get('converged') is not True for c in candidates.values()):raise ValueError('Reference used unconverged shooting')
        for method in METHODS:
            if codec.packet_digest(p['candidates'][method])!=codec.packet_digest(candidates[f'{setting}--{method}']['candidate']):raise ValueError('Reference candidate copy differs')
        if p.get('complete') is True or p.get('status')=='blocked_crossverification':
            evaluation.validate_reference_packet(p,candidates,old,plan['phases'],expected_id=ident)
        else:
            diagnostic=evaluation.validate_partial_packet(p,old,plan['phases'],shoot_packets=candidates)
            if not diagnostic['partial_replay_pass']:raise ValueError('Partial reference arithmetic differs: '+repr(diagnostic['errors']))
    for ident,p in events.items():
        setting=p.get('setting');method=p.get('method');ref=refs[f'{setting}--reference-pair']
        if ref.get('qualified') is not True:raise ValueError('Event lacks qualified reference')
        base=ref['bases'][method];index=p.get('phase_index')
        if type(index) is not int or index not in range(24) or ident!=f'{setting}--p{index:02d}--{method}':raise ValueError('Event phase identity differs')
        point=base['phases'][index]
        for key in ('alpha','gamma','period'):
            if p.get(key)!=base[key]:raise ValueError('Event parameter differs')
        for key in ('phase_fraction','phase_epoch','split'):
            if p.get(key)!=point[key]:raise ValueError('Event phase metadata differs')
        if not prior.np.array_equal(p.get('initial'),point['state']) or not prior.np.array_equal(p.get('phase_gradient'),point['z']):raise ValueError('Event initial/global gradient differs')
        if p.get('complete') is True:evaluation.validate_event_packet(p,base,expected_id=ident)
        else:
            diagnostic=evaluation.validate_partial_packet(p,sealed['references'][setting],plan['phases'],reference_packet=ref)
            if not diagnostic['partial_replay_pass']:raise ValueError('Partial event arithmetic differs: '+repr(diagnostic['errors']))


def analyze(directory,plan,rows,sealed):
    loaded=all_packets(directory,plan,rows);check_packet_bindings(plan,sealed,loaded)
    return evaluation.summarize(loaded['shooting'],loaded['references'],loaded['events'],sealed['references'],plan['phases'])


def eligible_references(ledger,rows,shoots):
    terminal={r['id']:r for r in rows}
    return [t for t in ledger if all(terminal[f'{t["case"]["setting"]}--{m}']['status']=='complete' and
        shoots.get(f'{t["case"]["setting"]}--{m}',{}).get('converged') is True for m in METHODS)]


def eligible_events(ledger,rows,references):
    terminal={r['id']:r for r in rows}
    return [t for t in ledger if terminal[t['case']['reference_id']]['status']=='complete' and
        references.get(t['case']['reference_id'],{}).get('qualified') is True]


def unavailable_analysis(plan,error):
    return dict(status='unavailable',error=error,complete=False,qualified=False,
        expected_workers=102,expected_nested_slots=208,expected_contexts=48,
        contexts=[dict(context_id=f'{setting}--p{phase["phase_index"]:02d}',setting=setting,**phase,
            status='unavailable',complete=False,qualified=False) for setting in SETTINGS for phase in plan['phases']])


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
    work.write(directory/'planned-nested-ledger.json',nested_ledger(plan['phases']))
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
        stage='shooting';mark=time.monotonic();prior.event(directory,'shooting_started',planned=4)
        contexts={t['case']['id']:dict(reference=sealed['references'][t['case']['setting']]) for t in ledger['shooting']}
        rows['shooting']=runtime.execute(ledger['shooting'],contexts,directory/'shooting',{},workers=4,deadline=deadline,per_task=180,worker_fn=worker)
        timings['shooting']=time.monotonic()-mark
        shoots=packets(directory,'shooting',ledger['shooting'],rows['shooting'])
        check_packet_bindings(plan,sealed,dict(shooting=shoots,references={},events={}))
        work.write(directory/'shooting-seal-index.json',packet_index(directory,'shooting',shoots))
        prior.event(directory,'shooting_sealed',sha256=legacy.sha(directory/'shooting-seal-index.json'))
        stage='references';mark=time.monotonic()
        eligible=eligible_references(ledger['references'],rows['shooting'],shoots)
        contexts={t['case']['id']:dict(reference=sealed['references'][t['case']['setting']],phases=plan['phases'],
            candidates={m:shoots[f'{t["case"]["setting"]}--{m}'] for m in METHODS}) for t in eligible}
        prior.event(directory,'references_started',planned=2,eligible=len(eligible))
        if eligible:rows['references']=runtime.execute(eligible,contexts,directory/'references',{},workers=4,deadline=deadline,per_task=180,worker_fn=worker)
        rows['references']=prior.fill_rows(ledger['references'],rows['references'],'blocked_shooting_prerequisite')
        timings['references']=time.monotonic()-mark
        references=packets(directory,'references',ledger['references'],rows['references'])
        check_packet_bindings(plan,sealed,dict(shooting=shoots,references=references,events={}))
        work.write(directory/'reference-seal-index.json',packet_index(directory,'references',references))
        prior.event(directory,'references_sealed',sha256=legacy.sha(directory/'reference-seal-index.json'))
        stage='events';mark=time.monotonic()
        eligible=eligible_events(ledger['events'],rows['references'],references)
        contexts={t['case']['id']:dict(base=references[t['case']['reference_id']]['bases'][t['method']['id']]) for t in eligible}
        prior.event(directory,'events_started',planned=96,eligible=len(eligible))
        if eligible:rows['events']=runtime.execute(eligible,contexts,directory/'events',{},workers=4,deadline=deadline,per_task=180,worker_fn=worker)
        rows['events']=prior.fill_rows(ledger['events'],rows['events'],'blocked_reference_prerequisite')
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
            try:nested=nested_records(plan,rows,all_packets(directory,plan,rows))
            except Exception as exc:
                work.write(directory/'nested-replay-error.json',dict(error=repr(exc)))
                nested=nested_records(plan,rows,{name:{} for name in ledger})
                if error is None:
                    error=repr(exc);work.write(directory/'producer-error.json',dict(stage='nested_finalization',error=error,at=legacy.now()))
            work.write(directory/'nested-terminal-records.json',nested)
            if analysis is None:analysis=unavailable_analysis(plan,error)
            codec.save_packet(directory,'analysis',analysis)
            try:
                validate(plan)
                if legacy.git('rev-parse','HEAD')!=head or legacy.git('status','--porcelain','--untracked-files=all'):raise ValueError('Checkout changed during diagnostic')
            except Exception as exc:post_error=repr(exc)
            work.write(directory/'postflight.json',dict(source_bindings_unchanged=post_error is None,error=post_error))
            complete=error is None and post_error is None and all(r['status']=='complete' for rs in rows.values() for r in rs) and analysis.get('complete') is True
            summary=dict(experiment_id='EXP-018',complete=complete,error=error,postflight_error=post_error,source_valid=post_error is None,
                status_counts={k:dict(Counter(r['status'] for r in rs)) for k,rs in rows.items()},analysis=analysis,
                qualified=complete and analysis.get('qualified') is True,nested_status_counts=dict(Counter(r['status'] for r in nested)),
                stage_wall_seconds=timings,total_wall_seconds=time.monotonic()-started,original_study_requalified=False)
            work.write(directory/'summary.json',summary);prior.event(directory,'producer_finished',complete=complete)
            artifacts=prior.inventory(directory)
            work.write(directory/'receipt.json',dict(**provenance,experiment_id='EXP-018',directory=str(directory),plan_sha256=legacy.sha(path),
                status='complete' if complete else 'incomplete_or_invalid',artifacts=artifacts,artifact_bytes=sum(r['bytes'] for r in artifacts),sealed_at=legacy.now()))
        finally:
            signal.setitimer(signal.ITIMER_REAL,0)
            for sig,handler in handlers.items():signal.signal(sig,handler)
    return dict(directory=str(directory),receipt_sha256=legacy.sha(directory/'receipt.json'),complete=complete,error=error,status_counts=summary['status_counts'])


def verify(directory):
    directory=Path(directory).resolve();receipt=read(directory/'receipt.json');manifest=read(directory/'manifest.json');plan=manifest['plan']
    if receipt['artifacts']!=prior.inventory(directory):raise ValueError('Byte inventory differs')
    if receipt['artifact_bytes']!=sum(r['bytes'] for r in receipt['artifacts']):raise ValueError('Artifact bytes differ')
    if receipt['source_commit']!=manifest['source_commit'] or receipt['plan_sha256']!=manifest['plan_sha256'] or receipt['experiment_id']!='EXP-018':raise ValueError('Receipt identity differs')
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
            if (row['status']=='complete' or row.get('packet_path') is not None) and read(p/'records'/f'{row["id"]}.json')!=row:raise ValueError('Worker record differs')
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
    loaded=all_packets(directory,plan,rows)
    if sealed is not None:check_packet_bindings(plan,sealed,loaded)
    nested=nested_records(plan,rows,loaded)
    if read(directory/'planned-nested-ledger.json')!=nested_ledger(plan['phases']) or read(directory/'nested-terminal-records.json')!=nested:raise ValueError('Nested ledger differs')
    for owned in processes:
        boundary={'shooting':'inputs_sealed','references':'shooting_sealed','events':'references_sealed'}[owned['stage']]
        if boundary not in ev or owned['started_at']<ev[boundary]['at']:raise ValueError('Worker started before prerequisite seal')
    ordered=['inputs_sealed','shooting_started','shooting_sealed','references_started','references_sealed','events_started','producer_finished']
    observed=[e['event'] for e in events]
    if any(x not in ordered for x in observed) or observed!=[x for x in ordered if x in ev]:raise ValueError('Stage chronology order differs')
    if any(a['monotonic_ns']>=b['monotonic_ns'] for a,b in zip(events,events[1:])):raise ValueError('Nonmonotone chronology')
    for stage,filename,event_name in [('shooting','shooting-seal-index.json','shooting_sealed'),('references','reference-seal-index.json','references_sealed')]:
        path=directory/filename
        if path.exists():
            if read(path)!=packet_index(directory,stage,loaded[stage]):raise ValueError('Prerequisite seal index differs')
            if event_name in ev and ev[event_name]['sha256']!=legacy.sha(path):raise ValueError('Prerequisite seal hash differs')
        elif event_name in ev:raise ValueError('Missing prerequisite seal')
    allowed_refs={t['id'] for t in eligible_references(ledger['references'],rows['shooting'],loaded['shooting'])}
    allowed_events={t['id'] for t in eligible_events(ledger['events'],rows['references'],loaded['references'])}
    for owned in processes:
        if owned['stage']=='references' and owned['task_id'] not in allowed_refs:raise ValueError('Reference dispatched without both converged shoots')
        if owned['stage']=='events' and owned['task_id'] not in allowed_events:raise ValueError('Event dispatched without qualified reference')
    saved=codec.load_packet(directory/'analysis.json');summary=read(directory/'summary.json');post=read(directory/'postflight.json')
    replayed=False
    if saved.get('status')!='unavailable':
        if sealed is None:raise ValueError('Analysis lacks input seal')
        replay=analyze(directory,plan,rows,sealed);codec.compare(saved,prior.normalized(replay),'analysis');replayed=True
    elif not (directory/'producer-error.json').exists():raise ValueError('Unavailable analysis lacks failure')
    else:codec.compare(saved,unavailable_analysis(plan,read(directory/'producer-error.json')['error']),'unavailable-analysis')
    codec.compare(summary['analysis'],work.baseline_engine.safe(saved),'summary.analysis')
    error=read(directory/'producer-error.json')['error'] if (directory/'producer-error.json').exists() else None
    complete=error is None and post['error'] is None and all(r['status']=='complete' for rs in rows.values() for r in rs) and saved.get('complete') is True
    if summary['error']!=error or summary['postflight_error']!=post['error'] or summary['source_valid'] is not (post['error'] is None) or post['source_bindings_unchanged'] is not (post['error'] is None):raise ValueError('Failure/source status differs')
    if summary['complete']!=complete or receipt['status']!=('complete' if complete else 'incomplete_or_invalid'):raise ValueError('Completion differs')
    if summary.get('qualified') is not (complete and saved.get('qualified') is True):raise ValueError('Qualification differs')
    if summary.get('nested_status_counts')!=dict(Counter(r['status'] for r in nested)):raise ValueError('Nested counts differ')
    if summary['status_counts']!={k:dict(Counter(r['status'] for r in rs)) for k,rs in rows.items()}:raise ValueError('Counts differ')
    all_owned=processes+[dict(pid=manifest['producer_pid'],identity=manifest['producer_identity'])]
    return dict(complete=complete,saved_array_replay=replayed,total_jobs=102,receipt_sha256=legacy.sha(directory/'receipt.json'),
        nested_slots=208,qualified=summary['qualified'],all_original_owned_identities_absent_now=all(work.identity(r['pid']) is None or work.identity(r['pid'])!=r['identity'] for r in all_owned))


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
