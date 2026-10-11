#!/usr/bin/env python3
"""Explicit numerical amendment for the one failed EXP-024 held-out history group.

Keep original contexts, observers, scientific endpoints and all passing histories.
Reintegrate four signs with two stricter solvers; preserve the original attempt.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

import run_two_write_interaction_heldout as h
import numpy as np
from scipy.integrate import solve_ivp

ROOT=h.ROOT
OLD=Path('/data2/neural-shrimp-data/raw/EXP-024/heldout-001')
OLD_SHA='3625e1d1cf8f78d4848ad088b6414223555246654bcbfbc9417d3aba823940bd'
PLAN=ROOT/'experiments/hopfield/EXP-024-refinement-plan.json'
REVIEW=ROOT/'docs/EXP-024-refinement-review.json'
METHODS={
 'primary':dict(method='DOP853',rtol=5e-14,atol=5e-16,steps_per_period=8192),
 'independent':dict(method='RK45',rtol=5e-14,atol=5e-16,steps_per_period=32768)}


def tasks():
    return [h.task('finite','core--p21',m,dose=2,sign1=a,sign2=b)
        for a,b in h.scoring.SIGN_PAIRS for m in METHODS]


def source_paths():
    return sorted(set(h.source_paths()+[ROOT/'docs/EXP-024-refinement-protocol.md',
        ROOT/'tests/test_refine_two_write_interaction.py']))


def old_guard():
    if h.sha(OLD/'receipt.json')!=OLD_SHA:raise ValueError('Original receipt changed')
    if h.read(OLD/'receipt.json')['artifacts']!=h.work.inventory(OLD):raise ValueError('Original evidence inventory changed')
    summary=h.read(OLD/'summary.json')
    if summary['complete'] is not True or summary['qualified'] is not False:raise ValueError('Expected complete but unqualified original')
    bad=[r for r in h.read(OLD/'analysis.json') if not r['qualified']]
    if len(bad)!=4 or {(r['context'],r['amplitude']) for r in bad}!={('core--p21',.01)}:
        raise ValueError('Refinement selection differs from all original failed cells')


def build_plan():
    old_guard()
    return dict(experiment='EXP-024',stage='explicit-numerical-refinement',original_root=str(OLD),
        original_receipt_sha256=OLD_SHA,tasks=tasks(),methods=METHODS,job_count=8,maximum_ivp_calls=16,
        scientific_contract=h.CONTRACT,amplitudes=list(h.AMPLITUDES),windows=h.WINDOWS,
        workers=4,per_task_seconds=180,wall_seconds=1800,automatic_retries=0,
        source_files=[h.binding(p) for p in source_paths()])


def validate(plan):
    if h.canonical(plan)!=h.canonical(build_plan()):raise ValueError('Refinement plan differs')
    return dict(status='valid_metadata_only',jobs=8,maximum_ivp_calls=16)


def source_guard(plan,head):
    validate(plan)
    git=lambda *args:subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()
    if git('status','--porcelain','--untracked-files=all') or git('rev-parse','HEAD')!=head:raise ValueError('Clean exact commit required')
    branch=git('branch','--show-current')
    if not branch.startswith('codex/') or git('remote','get-url','origin')!='git@github-personal:tdj28/neural-shrimp.git':raise ValueError('Private research branch required')
    if git('ls-remote','origin','refs/heads/'+branch).split()[0]!=head:raise ValueError('Exact private push required')
    review=h.read(REVIEW)
    if review['status']!='ready_for_numerical_refinement' or review['plan_sha256']!=h.sha(PLAN):raise ValueError('Exact plan review required')
    for p in [*source_paths(),PLAN,REVIEW]:
        raw=subprocess.check_output(['git','-C',str(ROOT),'show',head+':'+str(p.relative_to(ROOT))])
        if hashlib.sha256(raw).hexdigest()!=h.sha(p):raise ValueError('Uncommitted source')
    return dict(source_commit=head,plan_sha256=h.sha(PLAN),review=h.binding(REVIEW),branch=branch)


def strict_finite(rhs,initial,q,T,times,a1,a2,method):
    config=dict(METHODS[method]);config['max_step']=T/config.pop('steps_per_period')
    before=solve_ivp(rhs,(0.,T),initial+a1*q,dense_output=True,**config)
    if not before.success or before.t[-1]!=T or not np.isfinite(before.y).all():raise RuntimeError('Strict first segment failed')
    after=solve_ivp(rhs,(T,float(times[-1])),before.y[:,-1]+a2*q,dense_output=True,**config)
    if not after.success or after.t[-1]!=times[-1] or not np.isfinite(after.y).all():raise RuntimeError('Strict second segment failed')
    return dict(samples=after.sol(times).T,nfev=before.nfev+after.nfev,
        method=method,configuration=METHODS[method],event_before=before.y[:,-1],event_after=before.y[:,-1]+a2*q)


def worker(t,context,directory,libraries,parent_pid):
    start=time.monotonic();cpu=time.process_time();directory=Path(directory)
    row=dict(id=t['id'],case_id=t['case']['id'],method=t['method']['id'],kind=t['kind'],pid=os.getpid(),status='failed',rmse=None)
    try:
        rhs,_,_=h.sensitivity.tanh_flow(context['W']);a=h.AMPLITUDES[t['dose']]
        result=strict_finite(rhs,context['initial'],context['direction'],context['period'],
            h.sensitivity.sample_times(context['period']),a*t['sign1'],a*t['sign2'],t['method']['id'])
        row.update(status='complete',packet=h.codec.save_packet(directory/'payloads'/t['id'],'packet',result))
    except Exception as exc:row['error']=repr(exc)
    row.update(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    h.work.write(directory/'records'/(t['id']+'.json'),row)


def original_finite():
    plan=h.read(OLD/'manifest.json')['plan'];all_tasks=plan['tasks']['finite']
    benchmark=h.benchmark_tasks(all_tasks);rest=[t for t in all_tasks if t not in benchmark]
    packets=h.packets(OLD/'benchmark',benchmark);packets.update(h.packets(OLD/'finite',rest))
    paths={t['id']:dict(path=str(OLD/stage/'payloads'/t['id']/'packet.json'),
        sha256=h.sha(OLD/stage/'payloads'/t['id']/'packet.json'))
        for stage,rows in [('benchmark',benchmark),('finite',rest)] for t in rows}
    return packets,paths


def composite(directory):
    finite,paths=original_finite()
    replacements=h.packets(directory/'strict',tasks());finite.update(replacements)
    for t in tasks():
        p=directory/'strict'/'payloads'/t['id']/'packet.json'
        paths[t['id']]=dict(path=str(p),sha256=h.sha(p))
    contexts=h.codec.load_packet(OLD/'contexts.json');seals=h.codec.load_packet(OLD/'observer-seal.json')
    rows=h.analyze(seals,finite,contexts)
    return rows,h.aggregate(rows),paths


def run(head):
    plan=h.read(PLAN);guard=source_guard(plan,head)
    directory=OLD.parent/'heldout-refinement-001';directory.mkdir()
    h.work.write(directory/'manifest.json',dict(**guard,reference_root=str(OLD),plan=plan,parent=h.work.identity(os.getpid())))
    start=time.monotonic();error=None;qualified=False
    def interrupted(sig,frame):raise KeyboardInterrupt(f'Signal {sig}')
    handlers={s:signal.signal(s,interrupted) for s in (signal.SIGINT,signal.SIGTERM)}
    try:
        h.inherited.resource_check(directory,launch=True)
        contexts=h.codec.load_packet(OLD/'contexts.json')
        h.runtime.execute(tasks(),contexts,directory/'strict',{},workers=4,
            deadline=time.monotonic()+1800,per_task=180,worker_fn=worker)
        rows,agg,paths=composite(directory)
        h.work.write(directory/'analysis.json',rows);h.work.write(directory/'aggregate.json',agg)
        h.work.write(directory/'finite-provenance.json',paths)
        qualified=agg['status']=='qualified_fixed_panel'
    except BaseException as exc:error=repr(exc)
    finally:
        terminal=h.read(directory/'strict'/'terminal-records.json') if (directory/'strict'/'terminal-records.json').exists() else []
        ledger=[dict(id=t['id'],status=next((r['status'] for r in terminal if r['id']==t['id']),'not_started')) for t in tasks()]
        summary=dict(stage='explicit-numerical-refinement',qualified=qualified,error=error,jobs=ledger,
            wall_seconds=time.monotonic()-start,original_receipt_sha256=OLD_SHA,
            scientific_contract_changed=False,original_attempt_requalified=False)
        h.work.write(directory/'summary.json',summary)
        h.work.write(directory/'receipt.json',dict(**guard,original_receipt_sha256=OLD_SHA,artifacts=h.work.inventory(directory)))
        print(json.dumps(summary),flush=True)
        for sig,handler in handlers.items():signal.signal(sig,handler)
    return 0 if qualified and error is None else 1


def verify(directory):
    directory=Path(directory);validate(h.read(PLAN));receipt=h.read(directory/'receipt.json')
    if receipt['artifacts']!=h.work.inventory(directory):raise ValueError('Refinement artifacts changed')
    rows,agg,paths=composite(directory)
    if rows!=h.read(directory/'analysis.json') or agg!=h.read(directory/'aggregate.json') or paths!=h.read(directory/'finite-provenance.json'):raise ValueError('Composite replay differs')
    return dict(status='composite_saved_replay_pass',qualified=agg['status']=='qualified_fixed_panel',rows=len(rows),
        replacement_jobs=8,original_receipt_sha256=OLD_SHA,refinement_receipt_sha256=h.sha(directory/'receipt.json'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','validate','run','verify']);parser.add_argument('--head')
    args=parser.parse_args()
    if args.command=='prepare':h.work.write(PLAN,build_plan());print(json.dumps(validate(h.read(PLAN))))
    elif args.command=='validate':print(json.dumps(validate(h.read(PLAN))))
    elif args.command=='run':return run(args.head)
    else:print(json.dumps(verify(OLD.parent/'heldout-refinement-001')))
    return 0


if __name__=='__main__':raise SystemExit(main())
