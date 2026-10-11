#!/usr/bin/env python3
"""EXP-024 fixed held-out lifecycle, forked from development freeze 635a0b9e.

Same scientific solvers and readouts; 32 reserved phases, no new finite-difference
tuning, and a predeclared fixed-sample phase bootstrap. No target IO on import.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

import fresh_preview_runtime as work  # numerical threads fixed before NumPy
import numpy as np
import run_refined_two_write_memory as inherited
import run_phase_memory_recovery as codec
import two_write_runtime as runtime
import two_write_sensitivity as sensitivity
import two_write_interaction as scoring
from hopfield_diagnostics import weight_matrix

ROOT=work.ROOT
PLAN=ROOT/'experiments/hopfield/EXP-024-heldout-plan.json'
REVIEW=ROOT/'docs/EXP-024-heldout-review.json'
SETTINGS=('core','failed-near-edge')
AMPLITUDES=(1e-4,1e-3,1e-2)
WINDOWS={'early':(0,32),'later':(32,64)}
CONTRACT=dict(experiment='EXP-024',stage='heldout',phase_seed=2026092201,
    development_phases=8,reserved_heldout_phases=32,sigma=1e-4,workers=4,
    deadline_seconds=28800,per_task_seconds=180,finite_difference_phases=[],
    finite_difference_steps=[1e-3,5e-4],automatic_retries=0,
    state_tolerance=2e-7,coefficient_tolerance=2e-7,contrast_tolerance=1e-7,
    risk_tolerance=.005,phase_normalization_tolerance=1e-7,
    phase_adjoint_tolerance=1e-6,finite_difference_first_tolerance=.001,
    finite_difference_second_tolerance=.02,benchmark_primary_count=32,
    heldout_execution_enabled=True,paid_compute=False)
CONTRACT.update(bootstrap_seed=2026092301,bootstrap_draws=20000,
    primary=dict(setting='failed-near-edge',amplitude=.01,observation='state',window='early'),
    engineering_continuation_threshold=.05)
DEVELOPMENT_ROOT=Path('/data2/neural-shrimp-data/raw/EXP-024/attempt-001')
DEVELOPMENT_RECEIPT_SHA='6f1488818683757a90181e0e99fcc48f1dd716d20ec1f33b0648d77f50b4bdef'


def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def binding(p):return dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p))


def phases():
    values=np.random.Generator(np.random.PCG64(CONTRACT['phase_seed'])).random(40)
    return [dict(index=i,fraction=float(x),split='development' if i<8 else 'heldout_reserved') for i,x in enumerate(values)]


def task(kind,cid,method,**fields):
    suffix='--'.join(f'{k}-{v}' for k,v in fields.items())
    return dict(id=f'{cid}--{kind}--{method}'+('--'+suffix if suffix else ''),
        kind=kind,case={'id':cid},method={'id':method},**fields)


def census():
    contexts=[f'{s}--p{i:02d}' for s in SETTINGS for i in range(8,40)]
    phase=[task('phase',s,m) for s in SETTINGS for m in sensitivity.METHODS]
    refs=[task('reference',c,m) for c in contexts for m in sensitivity.METHODS]
    # Entire finite census is declared in advance. First32 primary histories
    # are a timing batch, not an outcome-selected expansion decision.
    finite=[task('finite',c,m,dose=d,sign1=a,sign2=b) for c in contexts
        for d in range(3) for a,b in scoring.SIGN_PAIRS for m in sensitivity.METHODS]
    fd=[]
    for setting in SETTINGS:
        for i in CONTRACT['finite_difference_phases']:
            cid=f'{setting}--p{i:02d}'
            fd.append(task('difference',cid,'primary',step=-1,sign1=0,sign2=0))
            for h in range(2):
                for a,b in [(1,0),(-1,0),(0,1),(0,-1),*scoring.SIGN_PAIRS]:
                    fd.append(task('difference',cid,'primary',step=h,sign1=a,sign2=b))
    return dict(phases=phase,references=refs,differences=fd,finite=finite)


def benchmark_tasks(tasks):
    return [t for t in tasks if t['method']['id']=='primary' and t['dose']==2
        and int(t['case']['id'].rsplit('--p',1)[1]) in range(8,12)]


def source_paths():
    # Bind the transitive Python code of the reused codec/owned-worker runtime.
    return sorted(set([*ROOT.glob('tools/**/*.py'),*ROOT.glob('src/**/*.py'),
        ROOT/'tests/test_two_write_interaction.py',ROOT/'tests/test_two_write_sensitivity.py',
        ROOT/'tests/test_two_write_interaction_runner.py',
        ROOT/'docs/EXP-024-protocol.md',ROOT/'docs/EXP-024-development-protocol.md',
        ROOT/'docs/EXP-024-heldout-protocol.md',ROOT/'docs/EXP-024-development-results.json',
        ROOT/'tests/test_two_write_interaction_heldout.py',
        ROOT/inherited.OLD_SUMMARY]))


def build_plan():
    tasks=census()
    if sha(DEVELOPMENT_ROOT/'receipt.json')!=DEVELOPMENT_RECEIPT_SHA:raise ValueError('Development receipt changed')
    summary=read(DEVELOPMENT_ROOT/'summary.json')
    if summary['complete'] is not True or summary['qualified'] is not True:raise ValueError('Qualified development required')
    return dict(contract=CONTRACT,methods=sensitivity.METHODS,amplitudes=list(AMPLITUDES),
        windows=WINDOWS,phases=phases(),tasks=tasks,
        counts={k:len(v) for k,v in tasks.items()},
        maximum_ivp_calls=2*sum(map(len,tasks.values())),
        historical_inputs=inherited.old_inputs(),
        development_receipt=dict(path=str(DEVELOPMENT_ROOT/'receipt.json'),sha256=DEVELOPMENT_RECEIPT_SHA),
        source_files=[binding(p) for p in source_paths()],
        packages={n:importlib.metadata.version(n) for n in ('numpy','scipy')})


def validate(plan):
    if canonical(plan)!=canonical(build_plan()):raise ValueError('Plan reconstruction or source binding mismatch')
    if plan['counts']!=dict(phases=4,references=128,differences=0,finite=1536):raise ValueError('Census mismatch')
    ids=[t['id'] for rows in plan['tasks'].values() for t in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate job id')
    if len(benchmark_tasks(plan['tasks']['finite']))!=32:raise ValueError('Benchmark census mismatch')
    if any(int(t['case']['id'].rsplit('--p',1)[1]) not in range(8,40) for name,rows in plan['tasks'].items() if name!='phases' for t in rows):raise ValueError('Reserved held-out phases only')
    return dict(status='valid_metadata_only',counts=plan['counts'],ivp_calls=plan['maximum_ivp_calls'])


def source_guard(plan,head):
    validate(plan)
    git=lambda *a:subprocess.check_output(['git','-C',str(ROOT),*a],text=True).strip()
    if git('status','--porcelain','--untracked-files=all'):raise ValueError('Clean checkout required')
    if head!=git('rev-parse','HEAD'):raise ValueError('Exact source HEAD required')
    branch=git('branch','--show-current')
    if not branch.startswith('codex/'):raise ValueError('Research branch required')
    if git('remote','get-url','origin')!='git@github-personal:tdj28/neural-shrimp.git':raise ValueError('Private origin mismatch')
    if git('ls-remote','origin','refs/heads/'+branch).split()[0]!=head:raise ValueError('Source is not privately pushed')
    review=read(REVIEW)
    if review.get('status')!='ready_for_heldout_execution' or review.get('plan_sha256')!=sha(PLAN):raise ValueError('Exact-plan review missing')
    for p in [PLAN,REVIEW,*source_paths()]:
        raw=subprocess.check_output(['git','-C',str(ROOT),'show',head+':'+str(p.relative_to(ROOT))])
        if hashlib.sha256(raw).hexdigest()!=sha(p):raise ValueError('Source file is not committed')
    return dict(source_commit=head,branch=branch,plan_sha256=sha(PLAN),review=binding(REVIEW))


def execute_payload(t,context):
    method=t['method']['id'];kind=t['kind']
    rhs,jac,hess=sensitivity.tanh_flow(context['W'])
    T=context['period']
    if kind=='phase':
        return sensitivity.phase_contexts(rhs,jac,context['section'],context['terminal_adjoint'],
            T,np.array([p['fraction'] for p in phases()[8:]]),method)
    x,q=context['initial'],context['direction'];times=sensitivity.sample_times(T)
    if kind=='reference':return sensitivity.reference(rhs,jac,hess,x,q,q,T,times,method)
    if kind=='finite':amplitude=AMPLITUDES[t['dose']]
    elif kind=='difference':amplitude=0. if t['step']==-1 else CONTRACT['finite_difference_steps'][t['step']]
    else:raise ValueError('Unknown task kind')
    return sensitivity.finite(rhs,x,q,q,T,times,amplitude*t['sign1'],amplitude*t['sign2'],method)


def load_anchors():
    inherited.old_inputs()
    anchors={}
    for setting in SETTINGS:
        pair=codec.load_packet(inherited.OLD/f'references/payloads/{setting}--reference-pair/packet.json')
        if pair.get('qualified') is not True:raise ValueError('Qualified anchor required')
        anchors[setting]=dict(W=weight_matrix(pair['alpha'],pair['gamma']),section=pair['canonical']['state'],
            terminal_adjoint=pair['canonical']['terminal_adjoint'],period=pair['canonical']['period'])
    return anchors


def worker(t,context,directory,libraries,parent_pid):
    start=time.monotonic();cpu=time.process_time();directory=Path(directory)
    row=dict(id=t['id'],case_id=t['case']['id'],kind=t['kind'],method=t['method']['id'],
        pid=os.getpid(),status='failed',rmse=None)
    try:
        memory=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
        if memory<8*1024**3:raise RuntimeError('Shared RAM reserve unavailable')
        result=execute_payload(t,context)
        path=directory/'payloads'/t['id']
        receipt=codec.save_packet(path,'packet',result)
        row.update(status='complete',packet=receipt)
    except Exception as exc:row['error']=repr(exc)
    row.update(wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    work.write(directory/'records'/(t['id']+'.json'),row)


def packets(directory,tasks):
    directory=Path(directory);rows=read(directory/'terminal-records.json')
    expected={t['id']:t for t in tasks}
    if len(rows)!=len(expected) or set(r['id'] for r in rows)!=set(expected):raise ValueError('Terminal census differs')
    result={}
    for row in rows:
        t=expected[row['id']]
        if row['status']!='complete':raise ValueError('Unqualified task '+row['id']+': '+row['status'])
        path=directory/'payloads'/t['id']/'packet.json'
        if str(path)!=row['packet']['path'] or sha(path)!=row['packet']['sha256']:raise ValueError('Packet identity differs')
        result[t['id']]=codec.load_packet(path)
    if read(directory/'cleanup.json')['all_original_identities_absent'] is not True:raise ValueError('Owned cleanup failed')
    return result


def phase_seal(anchors,ps):
    contexts={};diagnostics=[]
    for setting,anchor in anchors.items():
        primary=ps[task('phase',setting,'primary')['id']]
        independent=ps[task('phase',setting,'independent')['id']]
        if len(primary['contexts'])!=32 or len(independent['contexts'])!=32:raise ValueError('Thirty-two held-out phases required')
        closure=max(primary['closure'],independent['closure'])
        if closure>CONTRACT['state_tolerance']:raise ValueError('New phase base closure failed')
        for i,(p,q) in enumerate(zip(primary['contexts'],independent['contexts']),8):
            metrics=dict(id=f'{setting}--p{i:02d}',state_error=float(np.max(np.abs(p['initial']-q['initial']))),
                adjoint_error=float(np.max(np.abs(p['phase_gradient']-q['phase_gradient']))),
                normalization=max(abs(p['normalization_residual']),abs(q['normalization_residual'])),
                direction_leakage=max(abs(p['direction_leakage']),abs(q['direction_leakage'])),closure=closure)
            metrics['qualified']=bool(metrics['state_error']<=CONTRACT['state_tolerance'] and
                metrics['adjoint_error']<=CONTRACT['phase_adjoint_tolerance'] and
                metrics['normalization']<=CONTRACT['phase_normalization_tolerance'] and metrics['direction_leakage']<=1e-12)
            diagnostics.append(metrics)
            contexts[metrics['id']]=dict(**p,W=anchor['W'],setting=setting,index=i)
    return contexts,diagnostics


def reference_seal(contexts,refs):
    seals={};diagnostics=[]
    for cid in contexts:
        p=refs[task('reference',cid,'primary')['id']]['samples']
        q=refs[task('reference',cid,'independent')['id']]['samples']
        coefficients={};errors=[]
        for dose,a in enumerate(AMPLITUDES):
            for observation in ('state','activation'):
                for window,(start,stop) in WINDOWS.items():
                    for order in ('first','second','ablated'):
                        key=f'{dose}--{observation}--{window}--{order}'
                        c=sensitivity.coefficients(p[start:stop],a,observation,order)
                        other=sensitivity.coefficients(q[start:stop],a,observation,order)
                        err=max(float(np.max(np.abs(c[k]-other[k]))) for k in ('m','A','B','C'))
                        errors.append(err)
                        w=scoring.reference_observer(c['A'],c['B'],c['C'],CONTRACT['sigma'])['w']
                        coefficients[key]=dict(**c,w=w)
                    if observation=='activation':
                        # An exact static tanh of additive reference states;
                        # all sign means here are reference predictions, not data.
                        static=[]
                        for samples in (p,q):
                            linear=sensitivity.coefficients(samples[start:stop],a,'state','first')
                            means=scoring.reference_means(**{k:linear[k] for k in ('m','A','B','C')})
                            static.append(scoring.factorial_components(np.tanh(means)))
                        c,other=static
                        errors.append(max(float(np.max(np.abs(c[k]-other[k]))) for k in ('m','A','B','C')))
                        w=scoring.reference_observer(c['A'],c['B'],c['C'],CONTRACT['sigma'])['w']
                        coefficients[f'{dose}--activation--{window}--static']=dict(**c,w=w)
        metric=dict(id=cid,maximum_coefficient_error=max(errors),qualified=max(errors)<=CONTRACT['coefficient_tolerance'])
        diagnostics.append(metric);seals[cid]=coefficients
    return seals,diagnostics


def difference_checks(contexts,refs,packets_by_id):
    rows=[]
    for cid in contexts:
        if contexts[cid]['index'] not in CONTRACT['finite_difference_phases']:continue
        ref=refs[task('reference',cid,'primary')['id']]['samples']
        get=lambda step,a,b:packets_by_id[task('difference',cid,'primary',step=step,sign1=a,sign2=b)['id']]['samples']
        zero=get(-1,0,0)
        for hi,h in enumerate(CONTRACT['finite_difference_steps']):
            u1=(get(hi,1,0)-get(hi,-1,0))/(2*h);u2=(get(hi,0,1)-get(hi,0,-1))/(2*h)
            v11=(get(hi,1,0)+get(hi,-1,0)-2*zero)/h**2
            v22=(get(hi,0,1)+get(hi,0,-1)-2*zero)/h**2
            v12=(get(hi,1,1)-get(hi,1,-1)-get(hi,-1,1)+get(hi,-1,-1))/(4*h**2)
            errors={name:float(np.max(np.abs(value-ref[:,idx]))/max(1.,np.max(np.abs(ref[:,idx]))))
                for idx,(name,value) in enumerate(zip(('u1','u2','v11','v22','v12'),(u1,u2,v11,v22,v12)),1)}
            qualified=all(errors[k]<=CONTRACT['finite_difference_first_tolerance'] for k in ('u1','u2')) and all(
                errors[k]<=CONTRACT['finite_difference_second_tolerance'] for k in ('v11','v22','v12'))
            rows.append(dict(id=cid,step=h,errors=errors,qualified=qualified))
    return rows


def analyze(seals,finite_packets,contexts):
    rows=[]
    for cid in contexts:
        for dose,a in enumerate(AMPLITUDES):
            means={m:np.stack([finite_packets[task('finite',cid,m,dose=dose,sign1=s1,sign2=s2)['id']]['samples']
                for s1,s2 in scoring.SIGN_PAIRS]) for m in sensitivity.METHODS}
            state_error=float(np.max(np.abs(means['primary']-means['independent'])))
            for observation in ('state','activation'):
                transform=(lambda x:x) if observation=='state' else np.tanh
                for window,(start,stop) in WINDOWS.items():
                    observed={m:transform(v[:,start:stop]).reshape(4,-1) for m,v in means.items()}
                    direct={m:scoring.factorial_components(v) for m,v in observed.items()}
                    scores={};risk_error=0.
                    orders=('first','second','ablated','static') if observation=='activation' else ('first','second','ablated')
                    for order in orders:
                        c=seals[cid][f'{dose}--{observation}--{window}--{order}']
                        risks={m:scoring.parity_risk(c['w'],c['m'],v,CONTRACT['sigma'])['r_squared'] for m,v in observed.items()}
                        risk_error=max(risk_error,abs(risks['primary']-risks['independent']))
                        scores[order]=risks['primary']
                    c=direct['primary'];w=scoring.reference_observer(c['A'],c['B'],c['C'],CONTRACT['sigma'])['w']
                    oracle=scoring.parity_risk(w,c['m'],observed['primary'],CONTRACT['sigma'])['r_squared']
                    pred=seals[cid][f'{dose}--{observation}--{window}--second']['C']
                    contrast_error=float(np.max(np.abs(c['C']-direct['independent']['C'])))
                    rows.append(dict(context=cid,setting=contexts[cid]['setting'],phase_index=contexts[cid]['index'],
                        amplitude=a,observation=observation,window=window,scores=scores,oracle_r_squared=oracle,
                        direct_contrast_norm=float(np.linalg.norm(c['C'])),predicted_contrast_norm=float(np.linalg.norm(pred)),
                        prediction_error_norm=float(np.linalg.norm(c['C']-pred)),
                        contrast_numerical_error_norm=float(np.linalg.norm(c['C']-direct['independent']['C'])),
                        state_error=state_error,contrast_error=contrast_error,risk_error=risk_error,
                        qualified=state_error<=CONTRACT['state_tolerance'] and contrast_error<=CONTRACT['contrast_tolerance'] and risk_error<=CONTRACT['risk_tolerance']))
    return rows


def aggregate(rows):
    """Fixed 32-phase paired bootstrap; no adaptive phase expansion."""
    expected={(setting,index,a,obs,w) for setting in SETTINGS for index in range(8,40)
        for a in AMPLITUDES for obs in ('state','activation') for w in WINDOWS}
    keys=[(r['setting'],r['phase_index'],r['amplitude'],r['observation'],r['window']) for r in rows]
    if len(keys)!=len(expected) or set(keys)!=expected:raise ValueError('Complete 768-cell held-out census required')
    if not all(r['qualified'] for r in rows):
        return dict(status='unqualified',primary_inference=None,failed_rows=sum(not r['qualified'] for r in rows))
    rng=np.random.Generator(np.random.PCG64(CONTRACT['bootstrap_seed']))
    draws=rng.integers(0,32,size=(CONTRACT['bootstrap_draws'],32))
    groups=[];series={}
    for setting in SETTINGS:
        for a in AMPLITUDES:
            for obs in ('state','activation'):
                for window in WINDOWS:
                    subset=sorted([r for r in rows if (r['setting'],r['amplitude'],r['observation'],r['window'])==(setting,a,obs,window)],key=lambda r:r['phase_index'])
                    values=np.array([r['scores']['second'] for r in subset])
                    interval=np.quantile(values[draws].mean(axis=1),[.025,.975])
                    series[(setting,a,obs,window)]=values
                    groups.append(dict(setting=setting,amplitude=a,observation=obs,window=window,phases=32,
                        mean_r_squared=float(values.mean()),phase_bootstrap_95=interval.tolist(),
                        min_r_squared=float(values.min()),max_r_squared=float(values.max()),
                        mean_oracle_r_squared=float(np.mean([r['oracle_r_squared'] for r in subset])),
                        mean_static_r_squared=float(np.mean([r['scores']['static'] for r in subset])) if obs=='activation' else None,
                        resolved_contrast_phases=sum(r['direct_contrast_norm']>10*r['contrast_numerical_error_norm'] for r in subset)))
    primary=next(r for r in groups if all(r[k]==v for k,v in CONTRACT['primary'].items()))
    difference=series[('failed-near-edge',.01,'state','early')]-series[('core',.01,'state','early')]
    paired=dict(mean=float(difference.mean()),phase_bootstrap_95=np.quantile(difference[draws].mean(axis=1),[.025,.975]).tolist())
    return dict(status='qualified_fixed_panel',groups=groups,primary=primary,paired_primary_edge_minus_core=paired,
        engineering_continuation_trigger=primary['mean_r_squared']>.05 and primary['phase_bootstrap_95'][0]>0 and primary['resolved_contrast_phases']>0,
        interpretation='Phase-sampling uncertainty at two fixed selected networks; not a population-of-networks or shrimp-specific effect; nonprimary intervals descriptive, not multiplicity-adjusted.')


def run(head,attempt):
    plan=read(PLAN);guard=source_guard(plan,head)
    if not attempt.startswith('heldout-') or not attempt[8:].isdigit():raise ValueError('Named heldout-N required')
    parent=Path('/data2/neural-shrimp-data/raw/EXP-024');parent.mkdir(parents=True,exist_ok=True)
    directory=parent/attempt;directory.mkdir()
    started=time.monotonic();deadline=started+CONTRACT['deadline_seconds']
    work.write(directory/'manifest.json',dict(**guard,created_at=inherited.legacy.now(),plan=plan,parent=work.identity(os.getpid())))
    work.write(directory/'job-census.json',plan['tasks'])
    stage_rows={};error=None;qualified=False;analysis=[];timing={}
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Parent received signal {signum}')
    previous_handlers={s:signal.signal(s,interrupted) for s in (signal.SIGINT,signal.SIGTERM)}
    def stage(name,tasks,contexts):
        inherited.resource_check(directory,launch=False)
        mark=time.monotonic()
        stage_rows[name]=runtime.execute(tasks,contexts,directory/name,{},workers=4,deadline=deadline,
            per_task=CONTRACT['per_task_seconds'],worker_fn=worker)
        timing[name]=time.monotonic()-mark
        print(json.dumps(dict(stage=name,seconds=timing[name],complete=sum(r['status']=='complete' for r in stage_rows[name]),total=len(tasks))),flush=True)
        return packets(directory/name,tasks)
    try:
        inherited.resource_check(directory,launch=True)
        anchors=load_anchors()
        ps=stage('phases',plan['tasks']['phases'],anchors)
        contexts,phase_checks=phase_seal(anchors,ps)
        codec.save_packet(directory,'contexts',contexts);work.write(directory/'phase-checks.json',phase_checks)
        if not all(r['qualified'] for r in phase_checks):raise ValueError('Phase qualification failed')
        contexts=codec.load_packet(directory/'contexts.json')
        refs=stage('references',plan['tasks']['references'],contexts)
        seals,reference_checks=reference_seal(contexts,refs)
        codec.save_packet(directory,'observer-seal',seals);work.write(directory/'reference-checks.json',reference_checks)
        if not all(r['qualified'] for r in reference_checks):raise ValueError('Sensitivity comparison failed')
        seals=codec.load_packet(directory/'observer-seal.json')
        work.write(directory/'observer-barrier.json',dict(time=inherited.legacy.now(),seal_sha256=sha(directory/'observer-seal.json'),finite_jobs_started=0))
        fd=stage('differences',plan['tasks']['differences'],contexts)
        checks=difference_checks(contexts,refs,fd);work.write(directory/'difference-checks.json',checks)
        if not all(r['qualified'] for r in checks):raise ValueError('Development finite-difference qualification failed')
        benchmark=benchmark_tasks(plan['tasks']['finite'])
        remaining=[t for t in plan['tasks']['finite'] if t not in benchmark]
        finite=stage('benchmark',benchmark,contexts)
        finite.update(stage('finite',remaining,contexts))
        analysis=analyze(seals,finite,contexts);work.write(directory/'analysis.json',analysis)
        work.write(directory/'aggregate.json',aggregate(analysis))
        qualified=all(r['qualified'] for r in analysis)
    except BaseException as exc:
        error=repr(exc)
    finally:
        executed={row['id']:row for rows in stage_rows.values() for row in rows}
        # Read terminal records even if a stage raised during interruption.
        for path in directory.glob('*/terminal-records.json'):
            for row in read(path):executed[row['id']]=row
        ledger=[dict(id=t['id'],kind=t['kind'],status=executed.get(t['id'],{}).get('status','not_started_prerequisite'),
            completed_ivp_calls=2 if executed.get(t['id'],{}).get('status')=='complete' else None)
            for tasks in plan['tasks'].values() for t in tasks]
        work.write(directory/'complete-ledger.json',ledger)
        summary=dict(experiment='EXP-024',stage='heldout',source_commit=head,plan_sha256=sha(PLAN),
            complete=all(r['status']=='complete' for r in ledger),qualified=qualified,error=error,
            rows=len(analysis),planned_jobs=len(ledger),complete_jobs=sum(r['status']=='complete' for r in ledger),
            timings=timing,wall_seconds=time.monotonic()-started,heldout_trajectories=sum(r['status']=='complete' for r in ledger if r['kind']=='finite'))
        work.write(directory/'summary.json',summary)
        work.write(directory/'receipt.json',dict(**guard,artifacts=work.inventory(directory)))
        print(json.dumps(summary),flush=True)
        for sig,handler in previous_handlers.items():signal.signal(sig,handler)
    return 0 if qualified and error is None else 1


def verify(directory):
    directory=Path(directory);receipt=read(directory/'receipt.json')
    if receipt['artifacts']!=work.inventory(directory):raise ValueError('Artifact hashes differ')
    plan=read(directory/'manifest.json')['plan'];tasks=plan['tasks']
    if canonical(plan)!=canonical(read(PLAN)):raise ValueError('Replay requires exact frozen plan')
    validate(plan)
    if read(directory/'summary.json')['complete'] is not True:raise ValueError('Attempt incomplete; preserved without qualification')
    contexts=codec.load_packet(directory/'contexts.json');seals=codec.load_packet(directory/'observer-seal.json')
    ps=packets(directory/'phases',tasks['phases'])
    rebuilt_contexts,phase_checks=phase_seal(load_anchors(),ps)
    codec.compare(contexts,rebuilt_contexts)
    if phase_checks!=read(directory/'phase-checks.json'):raise ValueError('Phase-context replay differs')
    refs=packets(directory/'references',tasks['references'])
    rebuilt,checks=reference_seal(contexts,refs);codec.compare(seals,rebuilt)
    if checks!=read(directory/'reference-checks.json'):raise ValueError('Reference replay differs')
    fd=packets(directory/'differences',tasks['differences'])
    if difference_checks(contexts,refs,fd)!=read(directory/'difference-checks.json'):raise ValueError('Difference replay differs')
    benchmark=benchmark_tasks(tasks['finite'])
    remaining=[t for t in tasks['finite'] if t not in benchmark]
    finite=packets(directory/'benchmark',benchmark);finite.update(packets(directory/'finite',remaining))
    replay=analyze(seals,finite,contexts)
    if replay!=read(directory/'analysis.json'):raise ValueError('Saved arithmetic replay differs')
    if aggregate(replay)!=read(directory/'aggregate.json'):raise ValueError('Fixed-panel aggregate replay differs')
    return dict(status='saved_array_replay_pass',rows=len(replay),qualified=all(r['qualified'] for r in replay),receipt_sha256=sha(directory/'receipt.json'))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','validate','run','verify'])
    parser.add_argument('--head');parser.add_argument('--attempt',default='heldout-001');parser.add_argument('--directory')
    args=parser.parse_args()
    if args.command=='prepare':work.write(PLAN,build_plan());print(json.dumps(validate(read(PLAN))))
    elif args.command=='validate':print(json.dumps(validate(read(PLAN))))
    elif args.command=='run':return run(args.head,args.attempt)
    else:print(json.dumps(verify(args.directory)))
    return 0


if __name__=='__main__':raise SystemExit(main())
