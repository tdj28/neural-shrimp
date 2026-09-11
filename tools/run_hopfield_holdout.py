#!/usr/bin/env python3
"""EXP007: frozen new-start waveform interventions with explicit simulator preview."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import json
from numbers import Real
import os
from pathlib import Path
import platform
import shutil
import sys
import threading
import time
import numpy as np
import run_hopfield as base
from run_hopfield_drift import atomic_npz, baseline, assert_portable_equal, verify_wave_metrics
from hopfield_drift_metrics import waveform_errors
from hopfield_probe_selector import experiment_candidates, score_probe
from hopfield_drift_probe import build_trace, integrate_trace, local_parameter_sensitivity, phase_projected_gram

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'experiments/hopfield/EXP-007-plan.json'
ARMS=['atlas','local','tangent','axes','uniform','search','hybrid']
RADII=[.00025,.001,.004,.016]
PHASES=[337,1012,1687,2362]
SUBSETS={'local':[16,17],'tangent':[18,19],'axes':[0,4,8,12],'uniform':list(range(16)),'search':list(range(21))}


def validate(p):
    assert p['experiment_id']=='EXP-007' and len(p['anchors'])==12
    assert len({a['id'] for a in p['anchors']})==12
    assert p['arms']==ARMS and p['radii']==RADII and p['phases']==PHASES
    assert p['dt']==[.01,.005] and p['primary_dt']==.005 and p['decision_dt']==.005
    assert p['preview_duration']==75 and p['probe_duration']==300 and p['sample_dt']==.1
    assert p['phase_grid']==128 and p['workers']==4 and p['wall_clock_ceiling_seconds']==1800
    assert p['initial_state']==[.1,.1,.1] and p['burn_time']==10000 and p['sample_time']==2000
    assert p['event_capacity']==128 and p['max_peak_period']==32 and p['minimum_repeats']==4
    assert p['recurrence_atol']==2e-6 and p['recurrence_rtol']==2e-5
    assert p['hybrid_local_through_radius']==.004 and p['bootstrap_samples']==10000
    assert p['bootstrap_seed']==20260912 and not p['paid_compute'] and not p['paid_review']
    return 12*4*(4*7+1)*2


def specs(p, anchors=None):
    for a in p['anchors'] if anchors is None else anchors:
        for phase in p['phases']:
            for dt in p['dt']:
                yield dict(anchor=a['id'],phase=phase,radius=0.,arm='zero',dt=dt)
                for radius in p['radii']:
                    for arm in p['arms']:
                        yield dict(anchor=a['id'],phase=phase,radius=radius,arm=arm,dt=dt)


def trial_id(s):
    return f"{s['anchor']}-p{s['phase']}-r{s['radius']:g}-{s['arm']}-dt{s['dt']:g}"


def condition_id(anchor,phase,radius):
    return f'{anchor}-p{phase}-r{radius:g}'


def candidate_directions(anchor,local,radius):
    rows=experiment_candidates(local,anchor['tangent'],anchor['atlas_margin'][str(radius)]['direction'])
    return np.asarray([r['direction'] for r in rows])


def valid_score(value):
    return isinstance(value,Real) and not isinstance(value,(bool,np.bool_)) and np.isfinite(value) and 0<=value<=2


def choose_policies(previews,radius):
    """No target data enter selection. Failure invalidates affected subset only."""
    by_index={r['candidate']:r for r in previews}
    assert len(previews)==21 and set(by_index)==set(range(21))
    result={'atlas':{'status':'complete','candidate':20,'preview_calls':0}}
    for arm,indices in SUBSETS.items():
        rows=[by_index[i] for i in indices]
        if any(r['status']!='complete' or not valid_score(r.get('score')) for r in rows):
            result[arm]={'status':'selector_failed','candidate':None,'preview_calls':len(indices)}
        else:
            winner=min(rows,key=lambda r:(r['score'],r['candidate']))
            result[arm]={'status':'complete','candidate':winner['candidate'],'preview_calls':len(indices)}
    source='local' if radius<=.004 else 'atlas'
    result['hybrid']={**result[source],'source_policy':source}
    return result


def summarize(records,p,eligible):
    """Equal phase, dose and anchor weights; no success-only deletion."""
    lookup={(r['anchor'],r['phase'],r['radius'],r['arm'],r['dt']):r for r in records}
    assert len(lookup)==len(records)==validate(p)
    assert set(lookup)=={(s['anchor'],s['phase'],s['radius'],s['arm'],s['dt']) for s in specs(p)}
    rng=np.random.default_rng(p['bootstrap_seed'])
    draws=(rng.integers(len(eligible),size=(p['bootstrap_samples'],len(eligible))) if eligible else None)
    result=dict(planned=len(records),eligible_anchors=eligible,status_counts=dict(Counter(r['status'] for r in records)),steps={})
    for dt in p['dt']:
        blocks={}
        for radius in p['radii']:
            errors={}
            for arm in p['arms']:
                per=[]
                for a in eligible:
                    rows=[lookup[(a,phase,radius,arm,dt)] for phase in p['phases']]
                    per.append(float(np.mean([r['tail_metrics']['aligned_rms']['aggregate'] for r in rows]))
                               if all(r['status']=='complete' for r in rows) else None)
                errors[arm]=per
            blocks[str(radius)]=errors
        blocks['pooled']={arm:[float(np.mean([blocks[str(r)][arm][i] for r in p['radii']]))
                         if all(blocks[str(r)][arm][i] is not None for r in p['radii']) else None
                         for i in range(len(eligible))] for arm in p['arms']}
        panels={}
        for radius,errors in blocks.items():
            means={arm:float(np.mean(v)) if v and all(x is not None for x in v) else None for arm,v in errors.items()}
            contrasts={}
            for comparator in ['local','tangent','axes','uniform','search','atlas']:
                if means['hybrid'] is None or means[comparator] is None:
                    contrasts[comparator]=dict(status='undefined_incomplete');continue
                difference=np.asarray(errors[comparator])-np.asarray(errors['hybrid'])
                ci=np.percentile(difference[draws].mean(axis=1),[2.5,97.5])
                contrasts[comparator]=dict(status='complete',benefit=float(difference.mean()),ci95=ci.tolist(),
                                          positive_anchors=int((difference>0).sum()),per_anchor=difference.tolist())
            panels[radius]=dict(means=means,per_anchor=errors,comparator_minus_hybrid=contrasts)
        zero=[lookup[(a,phase,0.,'zero',dt)] for a in eligible for phase in p['phases']]
        errors=[r['tail_metrics']['aligned_rms']['aggregate'] for r in zero if r['status']=='complete']
        result['steps'][str(dt)]={'panels':panels,'zero':dict(complete=len(errors),planned=len(zero),
               mean=float(np.mean(errors)) if errors and len(errors)==len(zero) else None,maximum=max(errors) if errors and len(errors)==len(zero) else None)}
    return base.safe(result)


def run_preview(index,direction,anchor,initial,reference,T,epoch,radius,trace_fn,p,directory,condition):
    dest=directory/'previews'/f'{condition}-c{index:02d}.npz'
    delta=radius*np.asarray(direction);alpha,gamma=np.array([anchor['alpha'],anchor['gamma']])+delta
    row=dict(candidate=index,direction=direction,raw=str(dest.relative_to(directory)))
    started=time.perf_counter()
    try:
        times,states=integrate_trace(trace_fn,alpha,gamma,initial,p['decision_dt'],duration=p['preview_duration'],
                                     sample_dt=p['sample_dt'],failure_path=dest.with_name(dest.stem+'-failure.npz'))
        atomic_npz(dest,times=times,states=states,initial=initial,delta=delta,alpha=alpha,gamma=gamma,dt=p['decision_dt'])
        metrics=score_probe(reference,times,np.tanh(states),T,phase_epoch=epoch,phase_grid=p['phase_grid'])
        row.update(status='complete',score=metrics['aligned_rms']['aggregate'],metrics=metrics)
    except Exception as error:row.update(status='failed',score=None,error=repr(error))
    row['wall_seconds']=time.perf_counter()-started
    return base.safe(row)


def run_target(spec,anchor,initial,reference,T,epoch,direction,trace_fn,p,directory):
    dest=directory/'trials'/f'{trial_id(spec)}.npz'
    delta=spec['radius']*np.asarray(direction);alpha,gamma=np.array([anchor['alpha'],anchor['gamma']])+delta
    row={**spec,'id':trial_id(spec),'raw':str(dest.relative_to(directory)),'direction':direction,'delta':delta,
         'alpha':alpha,'gamma':gamma,'phase_epoch':epoch}
    started=time.perf_counter()
    try:
        times,states=integrate_trace(trace_fn,alpha,gamma,initial,spec['dt'],duration=p['probe_duration'],
                                     sample_dt=p['sample_dt'],failure_path=dest.with_name(dest.stem+'-failure.npz'))
        atomic_npz(dest,times=times,states=states,initial=initial,delta=delta,alpha=alpha,gamma=gamma,dt=spec['dt'])
        tail=times>p['preview_duration']
        row.update(status='complete',tail_metrics=waveform_errors(reference,times[tail]+epoch,np.tanh(states[tail]),T,phase_grid=p['phase_grid']),
                   full_metrics=waveform_errors(reference,times+epoch,np.tanh(states),T,phase_grid=p['phase_grid']))
    except Exception as error:row.update(status='failed',error=repr(error))
    row['wall_seconds']=time.perf_counter()-started
    return base.safe(row)


def run(directory):
    p=json.loads(PLAN.read_text());validate(p)
    assert not base.git('status','--porcelain'),'Source must be clean'
    source=base.git('rev-parse','HEAD');assert base.git('branch','-r','--contains',source),'Push before targets'
    for item in p['source_artifacts']:assert base.sha(ROOT/item['path'])==item['sha256'],item['path']
    directory.mkdir(parents=True,exist_ok=False);shutil.copyfile(PLAN,directory/'manifest.json')
    start=time.monotonic()
    started=dict(experiment_id='EXP-007',source_commit=source,manifest_sha256=base.sha(PLAN),started_at=base.now(),
                 status='running',planned_target_records=validate(p),planned_preview_records=12*4*4*21,
                 platform=platform.platform(),python=sys.version,paid_compute=0,paid_review=0)
    base.write(directory/'started.json',started)
    def hard_stop():
        base.write(directory/'timeout.json',dict(status='hard_timeout',ended_at=base.now(),source_commit=source));os._exit(124)
    watchdog=threading.Timer(p['wall_clock_ceiling_seconds'],hard_stop);watchdog.daemon=True;watchdog.start()
    try:
        fn,info=base.build_engine(directory/'build'/'long');base.write(directory/'long-build.json',info)
        trace_fn,info=build_trace(directory/'build'/'trace');base.write(directory/'trace-build.json',info)
        for name in ['trials','previews','decisions','sensitivity']:(directory/name).mkdir()
        baselines={}
        for a in p['anchors']:
            try:r=baseline(fn,a,p,directory)
            except Exception as error:r=dict(anchor=a['id'],eligible=False,status='failed',error=repr(error))
            if r['eligible'] and len(r['certificate']['orbit_states'])!=2701:
                r.update(eligible=False,status='unexpected_orbit_length')
            baselines[a['id']]=r;base.write(directory/'baselines.json',baselines)
            print(json.dumps(dict(anchor=a['id'],eligible=r['eligible'])),flush=True)
        eligible=[a['id'] for a in p['anchors'] if baselines[a['id']]['eligible']]
        records=[];preview_status=Counter();preview_seconds=0.
        with ThreadPoolExecutor(max_workers=4) as pool,(directory/'records.jsonl').open('x') as journal:
            def append(row):
                records.append(row);journal.write(json.dumps(base.safe(row),allow_nan=False)+'\n');journal.flush()
            for a in p['anchors']:
                if a['id'] not in eligible:
                    for s in specs(p,[a]):append({**s,'id':trial_id(s),'status':'baseline_not_evaluable'})
                    continue
                cert=baselines[a['id']]['certificate'];T=cert['period']
                orbit=np.asarray(cert['orbit_states'])[:-1];reference=np.asarray(cert['orbit_activations'])[:-1]
                for phase in p['phases']:
                    initial=orbit[phase].copy();epoch=phase*T/len(orbit)
                    times=np.linspace(0,300,3001)
                    sens=local_parameter_sensitivity(a['alpha'],a['gamma'],initial,times)
                    tail=times>75
                    gram=phase_projected_gram(sens['activation_sensitivity'][tail],sens['activation_velocity'][tail])
                    local=np.asarray(gram['minimum_eigenvector'])
                    atomic_npz(directory/'sensitivity'/f"{a['id']}-p{phase}.npz",**{k:v for k,v in sens.items() if isinstance(v,np.ndarray)},tail_direction=local,tail_gram=gram['gram'])
                    base.write(directory/'sensitivity'/f"{a['id']}-p{phase}.json",dict(wall_seconds=sens['wall_seconds'],nfev=sens['nfev'],solver=sens['solver'],tail_gram=gram))
                    zero=[s for s in specs(p,[a]) if s['phase']==phase and s['radius']==0]
                    for row in pool.map(lambda s:run_target(s,a,initial,reference,T,epoch,np.zeros(2),trace_fn,p,directory),zero):append(row)
                    for radius in p['radii']:
                        condition=condition_id(a['id'],phase,radius);directions=candidate_directions(a,local,radius)
                        previews=list(pool.map(lambda i:run_preview(i,directions[i],a,initial,reference,T,epoch,radius,trace_fn,p,directory,condition),range(21)))
                        preview_status.update(r['status'] for r in previews)
                        preview_seconds+=sum(r['wall_seconds'] for r in previews)
                        policies=choose_policies(previews,radius)
                        base.write(directory/'decisions'/f'{condition}.json',dict(anchor=a['id'],phase=phase,radius=radius,previews=previews,policies=policies))
                        selected=[s for s in specs(p,[a]) if s['phase']==phase and s['radius']==radius]
                        def execute(s):
                            choice=policies[s['arm']]
                            if choice['status']!='complete':return {**s,'id':trial_id(s),'status':'selector_failed'}
                            return run_target(s,a,initial,reference,T,epoch,directions[choice['candidate']],trace_fn,p,directory)
                        for row in pool.map(execute,selected):append(row)
                    base.write(directory/'progress.json',dict(anchor=a['id'],phase=phase,records=len(records),wall_seconds=time.monotonic()-start))
                    print(json.dumps(dict(anchor=a['id'],phase=phase,records=len(records),seconds=round(time.monotonic()-start,1))),flush=True)
        base.write(directory/'preview-census.json',dict(planned=4032,eligible=len(eligible)*4*4*21,
            baseline_not_evaluable=(12-len(eligible))*4*4*21,status_counts=dict(preview_status),
            sum_concurrent_call_wall_seconds=preview_seconds))
        base.write(directory/'summary.json',summarize(records,p,eligible))
        artifacts=[f for f in sorted(directory.rglob('*')) if f.is_file() and 'build' not in f.relative_to(directory).parts]
        base.write(directory/'receipt.json',{**started,'status':'complete','ended_at':base.now(),'wall_seconds':time.monotonic()-start,
             'eligible_anchors':eligible,'artifacts':[dict(path=str(f.relative_to(directory)),sha256=base.sha(f),bytes=f.stat().st_size) for f in artifacts]})
        verify(directory)
    except BaseException as error:
        if hasattr(error,'payload'):atomic_npz(directory/'unhandled-numerical-failure.npz',**error.payload)
        base.write(directory/'failure.json',dict(source_commit=source,error=repr(error),ended_at=base.now(),wall_seconds=time.monotonic()-start));raise
    finally:watchdog.cancel()


def verify(directory):
    p=json.loads((directory/'manifest.json').read_text());receipt=json.loads((directory/'receipt.json').read_text())
    assert receipt['status']=='complete' and receipt['planned_target_records']==validate(p)
    assert base.sha(directory/'manifest.json')==receipt['manifest_sha256']
    assert not (directory/'failure.json').exists() and not (directory/'timeout.json').exists()
    paths=[v['path'] for v in receipt['artifacts']];assert len(paths)==len(set(paths))
    for item in receipt['artifacts']:
        path=directory/item['path'];assert base.sha(path)==item['sha256'] and path.stat().st_size==item['bytes'],item['path']
    baselines=json.loads((directory/'baselines.json').read_text())
    anchors={a['id']:a for a in p['anchors']};assert set(baselines)==set(anchors)
    eligible=[a for a in anchors if baselines[a]['eligible']];assert eligible==receipt['eligible_anchors']
    decisions={};contexts={};preview_status=Counter();preview_seconds=0.
    for aid in eligible:
        cert=baselines[aid]['certificate'];assert cert['verified'] and cert['attracting']
        assert [r['period'] for r in baselines[aid]['steps']]==[9,9]
        orbit=np.asarray(cert['orbit_states'])[:-1];reference=np.asarray(cert['orbit_activations'])[:-1];T=cert['period']
        assert orbit.shape==reference.shape==(2700,3)
        for phase in p['phases']:
            initial=orbit[phase];epoch=phase*T/2700
            with np.load(directory/'sensitivity'/f'{aid}-p{phase}.npz',allow_pickle=False) as raw:
                np.testing.assert_array_equal(raw['states'][0],initial)
                selected=raw['times']>75
                gram=phase_projected_gram(raw['activation_sensitivity'][selected],raw['activation_velocity'][selected])
                np.testing.assert_allclose(raw['tail_direction'],gram['minimum_eigenvector'],rtol=1e-12,atol=1e-14)
                local=raw['tail_direction'].copy()
            contexts[(aid,phase)]=(initial,reference,T,epoch,local)
            for radius in p['radii']:
                cid=condition_id(aid,phase,radius);decision=json.loads((directory/'decisions'/f'{cid}.json').read_text())
                assert (decision['anchor'],decision['phase'],decision['radius'])==(aid,phase,radius)
                ds=candidate_directions(anchors[aid],local,radius)
                preview_status.update(r['status'] for r in decision['previews'])
                preview_seconds+=sum(r['wall_seconds'] for r in decision['previews'])
                for row in decision['previews']:
                    index=row['candidate'];np.testing.assert_array_equal(row['direction'],ds[index])
                    if row['status']!='complete':continue
                    with np.load(directory/row['raw'],allow_pickle=False) as raw:
                        check_trace(raw,initial,anchors[aid],radius*ds[index],p['decision_dt'],75)
                        verify_wave_metrics(row['metrics'],reference,raw['times']+epoch,np.tanh(raw['states']),T,p['phase_grid'])
                    assert row['score']==row['metrics']['aligned_rms']['aggregate']
                assert_portable_equal(decision['policies'],choose_policies(decision['previews'],radius),'policies')
                decisions[(aid,phase,radius)]=(decision,ds)
    assert_portable_equal(json.loads((directory/'preview-census.json').read_text()),dict(planned=4032,
       eligible=len(eligible)*4*4*21,baseline_not_evaluable=(12-len(eligible))*4*4*21,
       status_counts=dict(preview_status),sum_concurrent_call_wall_seconds=preview_seconds),'preview census')
    records=[json.loads(line) for line in (directory/'records.jsonl').read_text().splitlines()]
    for row in records:
        assert row['id']==trial_id(row)
        if row['anchor'] not in eligible:assert row['status']=='baseline_not_evaluable';continue
        initial,reference,T,epoch,local=contexts[(row['anchor'],row['phase'])]
        if row['radius']==0:direction=np.zeros(2)
        else:
            decision,ds=decisions[(row['anchor'],row['phase'],row['radius'])]
            choice=decision['policies'][row['arm']]
            if choice['status']!='complete':assert row['status']=='selector_failed';continue
            direction=ds[choice['candidate']]
        assert row['status'] in ['complete','failed']
        if row['status']!='complete':continue
        assert row['phase_epoch']==epoch and row['raw']==f"trials/{row['id']}.npz"
        np.testing.assert_array_equal(row['direction'],direction)
        np.testing.assert_array_equal(row['delta'],row['radius']*direction)
        np.testing.assert_array_equal([row['alpha'],row['gamma']],np.array([anchors[row['anchor']]['alpha'],anchors[row['anchor']]['gamma']])+row['delta'])
        with np.load(directory/row['raw'],allow_pickle=False) as raw:
            check_trace(raw,initial,anchors[row['anchor']],row['radius']*direction,row['dt'],300)
            tail=raw['times']>75
            verify_wave_metrics(row['tail_metrics'],reference,raw['times'][tail]+epoch,np.tanh(raw['states'][tail]),T,p['phase_grid'])
            verify_wave_metrics(row['full_metrics'],reference,raw['times']+epoch,np.tanh(raw['states']),T,p['phase_grid'])
    assert_portable_equal(json.loads((directory/'summary.json').read_text()),summarize(records,p,eligible),'summary')
    print(json.dumps(dict(verified=True,targets=len(records),eligible=len(eligible),source=receipt['source_commit'])))


def check_trace(raw,initial,anchor,delta,dt,duration):
    np.testing.assert_array_equal(raw['initial'],initial);np.testing.assert_array_equal(raw['states'][0],initial)
    np.testing.assert_array_equal(raw['delta'],delta)
    np.testing.assert_array_equal([raw['alpha'],raw['gamma']],np.array([anchor['alpha'],anchor['gamma']])+delta)
    assert float(raw['dt'])==dt
    expected=np.arange(round(duration/.1)+1)*.1;expected[-1]=duration
    np.testing.assert_array_equal(raw['times'],expected)
    assert raw['states'].shape==(len(expected),3) and np.isfinite(raw['states']).all()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['validate','run','verify']);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.action=='validate':print(json.dumps(dict(valid=True,targets=validate(json.loads(PLAN.read_text())))))
    elif args.output is None:parser.error('--output is required')
    elif args.action=='run':run(args.output)
    else:verify(args.output)
if __name__=='__main__':main()
