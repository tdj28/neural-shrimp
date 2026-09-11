#!/usr/bin/env python3
"""Frozen matched weight-drift assay and separate periodic-orbit continuation."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import threading
import time
import numpy as np
import run_hopfield as base
from hopfield_drift_metrics import waveform_errors, periodic_interpolate

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'experiments/hopfield/EXP-006-plan.json'
ARMS=['outward','inward','tangent_plus','tangent_minus','local_plus','local_minus',
      'random_plus','random_minus','shuffled_outward','shuffled_inward','atlas_margin']
RADII=[.00025,.001,.004,.016]


def atomic_npz(path, **payload):
    """Keep a previous completed payload intact until its replacement is closed.

    An interrupted/failed write leaves the distinct partial file as evidence.
    Exclusive partial-file creation prevents accidentally resuming over it.
    """
    path=Path(path);partial=path.with_name(path.stem+'.partial.npz')
    with partial.open('xb') as stream:
        np.savez_compressed(stream,**payload)
        stream.flush();os.fsync(stream.fileno())
    os.replace(partial,path)


def validate(p):
    assert p['experiment_id']=='EXP-006' and len(p['anchors'])==12
    assert p['arms']==ARMS and p['radii']==RADII and p['phases']==[0,1,2,3]
    assert p['dt']==[.01,.005] and p['primary_dt']==.005
    assert p['initial_state']==[.1,.1,.1]
    assert p['burn_time']==10000 and p['sample_time']==2000
    assert p['probe_duration']==300 and p['sample_dt']==.1
    assert p['event_capacity']==128 and p['max_peak_period']==32 and p['minimum_repeats']==4
    assert p['recurrence_atol']==2e-6 and p['recurrence_rtol']==2e-5
    assert p['workers']==4 and p['wall_clock_ceiling_seconds']==1800
    assert p['paid_compute'] is False and p['paid_review'] is False
    assert len({a['id'] for a in p['anchors']})==12
    return 12*4*(4*11+1)*2


def trial_specs(p, anchors=None):
    for a in p['anchors'] if anchors is None else anchors:
        for phase in p['phases']:
            for dt in p['dt']:
                yield dict(anchor=a['id'],phase=phase,radius=0.,arm='zero',dt=dt)
                for radius in p['radii']:
                    for arm in p['arms']:
                        yield dict(anchor=a['id'],phase=phase,radius=radius,arm=arm,dt=dt)


def trial_id(s):
    return f"{s['anchor']}-p{s['phase']}-r{s['radius']:g}-{s['arm']}-dt{s['dt']:g}"


def directions(anchor, local, radius):
    outward=np.asarray(anchor['outward'],float)
    tangent=np.array([-outward[1],outward[0]])
    random=np.asarray(anchor['random_unit'],float)
    shuffled=np.asarray(anchor['shuffled_outward'],float)
    atlas=np.asarray(anchor['atlas_margin'][str(radius)]['direction'],float)
    return dict(outward=outward,inward=-outward,tangent_plus=tangent,tangent_minus=-tangent,
                local_plus=local,local_minus=-local,random_plus=random,random_minus=-random,
                shuffled_outward=shuffled,shuffled_inward=-shuffled,atlas_margin=atlas)


def baseline(fn, anchor, p, directory):
    dest=directory/'baselines'/anchor['id'];dest.mkdir(parents=True)
    steps=[]
    for dt in p['dt']:
        result=base.cell(fn,anchor['alpha'],anchor['gamma'],p['initial_state'],p,dt=dt,
                         failure_path=dest/f'failed-dt-{dt:g}.npz')
        final,events,stats,n,q,residual,T=result
        atomic_npz(dest/f'dt-{dt:g}.npz',final=final,events=events,stats=stats,
                            n_events=n,period=q,residual=residual,period_time=T,initial=p['initial_state'])
        steps.append(dict(dt=dt,period=q,residual=residual,period_time=T,stats=stats))
    cert=(base.certify(anchor['alpha'],anchor['gamma'],events[n-9,1:],T,9)
          if all(s['period']==9 for s in steps) else
          dict(verified=False,attracting=False,reason='Both independent baseline steps did not retain nine peaks'))
    result=dict(anchor=anchor['id'],steps=steps,certificate=cert,
                eligible=bool(cert.get('verified') and cert.get('attracting')))
    base.write(dest/'result.json',result)
    return result


def execute_trial(spec, anchor, initial, local, reference, period, phase_epoch, fn, trace_fn, p, directory):
    from hopfield_drift_probe import integrate_trace
    dest=directory/'trials'/f'{trial_id(spec)}.npz'
    direction=(np.zeros(2) if spec['radius']==0 else directions(anchor,local,spec['radius'])[spec['arm']])
    if spec['radius']:
        np.testing.assert_allclose(np.linalg.norm(direction),1.,rtol=0,atol=1e-12)
    delta=spec['radius']*direction
    alpha,gamma=np.asarray([anchor['alpha'],anchor['gamma']])+delta
    record={**spec,'id':trial_id(spec),'alpha':float(alpha),'gamma':float(gamma),
            'weight_delta':delta,'raw_weight_norm':float(np.linalg.norm(delta)),
            'baseline_phase_epoch':phase_epoch,'raw':str(dest.relative_to(directory))}
    try:
        started=time.monotonic()
        times,states=integrate_trace(trace_fn,alpha,gamma,initial,spec['dt'],
                                     duration=p['probe_duration'],sample_dt=p['sample_dt'],
                                     failure_path=dest.with_name(dest.stem+'-trace-failure.npz'))
        payload=dict(times=times,states=states,initial=initial,alpha=alpha,gamma=gamma,delta=delta,dt=spec['dt'])
        # Persist the short trajectory before long integration or analysis can fail.
        atomic_npz(dest,**payload)
        short_seconds=time.monotonic()-started
        long_started=time.monotonic()
        final,events,stats,n,q,residual,T=base.cell(fn,alpha,gamma,initial,p,dt=spec['dt'],
                      failure_path=dest.with_name(dest.stem+'-native-failure.npz'))
        payload.update(final=final,events=events,stats=stats,n_events=n,period=q,residual=residual,period_time=T)
        atomic_npz(dest,**payload)
        long_seconds=time.monotonic()-long_started
        analysis_started=time.monotonic()
        metric=waveform_errors(reference,times+phase_epoch,np.tanh(states),period,phase_grid=p['phase_grid'])
        record.update(status='complete',wave_metrics=metric,period=q,period_time=T,
                      recurrence_residual=residual,ftle=float(stats[1]),stats=stats,
                      nine_peak_retained=bool(q==9),short_seconds=short_seconds,
                      long_seconds=long_seconds,analysis_seconds=time.monotonic()-analysis_started)
    except Exception as error:
        record.update(status='failed',error=repr(error))
    return base.safe(record)


def run_continuations(p,directory):
    from hopfield_continuation import continue_hopfield_branch
    source=json.loads((ROOT/p['continuation']['source_summary']).read_text())
    out=directory/'continuation';out.mkdir()
    for name in p['continuation']['seeds']:
        item=source[name];cert=item['certificate']
        for direction in [-1,1]:
            stem=f'{name}-direction-{direction:+d}'
            def append(suffix, payload):
                with (out/f'{stem}-{suffix}.jsonl').open('a') as stream:
                    stream.write(json.dumps(base.safe(payload),allow_nan=False)+'\n');stream.flush()
            base.write(directory/'progress.json',dict(phase='continuation',branch=stem))
            result=continue_hopfield_branch(item['alpha'],item['gamma'],cert['section_state'],cert['period'],direction,
                    config=p['continuation']['config'],on_attempt=lambda row:append('attempts',row),
                    on_node=lambda row:append('nodes',row))
            base.write(out/f'{stem}.json',result)
            print(json.dumps(dict(branch=stem,stop_reason=result.get('stop_reason'),nodes=len(result.get('nodes',[])))),flush=True)


def run(directory):
    from hopfield_drift_probe import build_trace, local_parameter_sensitivity
    from hopfield_drift_design import summarize_trials
    p=json.loads(PLAN.read_text());count=validate(p)
    assert not base.git('status','--porcelain'),'Source must be clean'
    source=base.git('rev-parse','HEAD');assert base.git('branch','-r','--contains',source),'Push before targets'
    for artifact in p['source_artifacts']:
        assert base.sha(ROOT/artifact['path'])==artifact['sha256'],artifact['path']
    directory.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(PLAN,directory/'manifest.json')
    started=dict(experiment_id='EXP-006',source_commit=source,manifest_sha256=base.sha(PLAN),
                 started_at=base.now(),status='running',planned_trial_step_calls=count,
                 platform=platform.platform(),python=sys.version,paid_compute=0,paid_review=0)
    base.write(directory/'started.json',started);start=time.monotonic()
    def hard_stop():
        base.write(directory/'timeout.json',dict(source_commit=source,status='hard_timeout',ended_at=base.now(),
                       ceiling_seconds=p['wall_clock_ceiling_seconds'],original_attempt_preserved=True))
        os._exit(124)
    watchdog=threading.Timer(p['wall_clock_ceiling_seconds'],hard_stop);watchdog.daemon=True;watchdog.start()
    try:
        fn,build=base.build_engine(directory/'build'/'long');base.write(directory/'long-build.json',build)
        trace_fn,trace_build=build_trace(directory/'build'/'trace');base.write(directory/'trace-build.json',trace_build)
        baselines={}
        for anchor in p['anchors']:
            base.write(directory/'progress.json',dict(phase='baseline',anchor=anchor['id']))
            try: result=baseline(fn,anchor,p,directory)
            except Exception as error:result=dict(anchor=anchor['id'],eligible=False,status='failed',error=repr(error))
            baselines[anchor['id']]=result;base.write(directory/'baselines.json',baselines)
            print(json.dumps(dict(anchor=anchor['id'],eligible=result['eligible'])),flush=True)
        run_continuations(p,directory)
        (directory/'trials').mkdir();(directory/'sensitivity').mkdir()
        records=[];eligible=[a['id'] for a in p['anchors'] if baselines[a['id']]['eligible']]
        with ThreadPoolExecutor(max_workers=p['workers']) as pool, (directory/'records.jsonl').open('x') as journal:
            for anchor in p['anchors']:
                if anchor['id'] not in eligible:
                    for spec in trial_specs(p,[anchor]):
                        row={**spec,'id':trial_id(spec),'status':'baseline_not_evaluable'}
                        journal.write(json.dumps(row)+'\n');records.append(row)
                    journal.flush();continue
                cert=baselines[anchor['id']]['certificate'];T=cert['period']
                reference=np.asarray(cert['orbit_activations'])[:-1]
                orbit=np.asarray(cert['orbit_states'])[:-1]
                for phase in p['phases']:
                    initial=orbit[phase*len(orbit)//4].copy();epoch=phase*T/4
                    times=np.linspace(0,p['probe_duration'],round(p['probe_duration']/p['sample_dt'])+1)
                    base.write(directory/'progress.json',dict(phase='local_sensitivity',anchor=anchor['id'],orbit_phase=phase))
                    try:
                        sensitivity=local_parameter_sensitivity(anchor['alpha'],anchor['gamma'],initial,times)
                    except Exception as error:
                        if hasattr(error,'payload'):
                            np.savez_compressed(directory/'sensitivity'/f"{anchor['id']}-p{phase}-failure.npz",**error.payload)
                        raise
                    arrays={k:v for k,v in sensitivity.items() if isinstance(v,np.ndarray)}
                    atomic_npz(directory/'sensitivity'/f"{anchor['id']}-p{phase}.npz",**arrays)
                    base.write(directory/'sensitivity'/f"{anchor['id']}-p{phase}.json",{k:v for k,v in sensitivity.items() if not isinstance(v,np.ndarray)})
                    local=np.asarray(sensitivity['minimum_eigenvector'])
                    specs=[s for s in trial_specs(p,[anchor]) if s['phase']==phase]
                    futures=[pool.submit(execute_trial,s,anchor,initial,local,reference,T,epoch,fn,trace_fn,p,directory) for s in specs]
                    for future in as_completed(futures):
                        row=future.result();records.append(row)
                        journal.write(json.dumps(row,allow_nan=False)+'\n');journal.flush()
                    print(json.dumps(dict(anchor=anchor['id'],phase=phase,records=len(records),elapsed_seconds=round(time.monotonic()-start,1))),flush=True)
        summary=summarize_trials([r for r in records if r['anchor'] in eligible],[a['id'] for a in p['anchors']],eligible,
                                 phases=tuple(p['phases']),radii=tuple(p['radii']),dt_values=tuple(p['dt']),
                                 bootstrap_samples=p['bootstrap_samples'],seed=p['bootstrap_seed'])
        base.write(directory/'summary.json',summary)
        base.write(directory/'progress.json',dict(phase='complete'))
        artifacts=[f for f in sorted(directory.rglob('*')) if f.is_file() and 'build' not in f.relative_to(directory).parts]
        base.write(directory/'receipt.json',{**started,'status':'complete','ended_at':base.now(),
                    'wall_seconds':time.monotonic()-start,'eligible_anchors':eligible,
                    'artifacts':[dict(path=str(f.relative_to(directory)),sha256=base.sha(f),bytes=f.stat().st_size) for f in artifacts]})
        verify(directory)
    except BaseException as error:
        base.write(directory/'failure.json',dict(source_commit=source,error=repr(error),ended_at=base.now(),
                  wall_seconds=time.monotonic()-start,original_attempt_preserved=True));raise
    finally:watchdog.cancel()


def assert_portable_equal(actual, expected, path='value'):
    """Compare derived numerical fields portably; artifact hashes stay exact."""
    if isinstance(expected,dict):
        assert isinstance(actual,dict) and set(actual)==set(expected),path
        for key in expected:assert_portable_equal(actual[key],expected[key],path+'.'+key)
    elif isinstance(expected,list):
        assert isinstance(actual,list) and len(actual)==len(expected),path
        for index,(a,b) in enumerate(zip(actual,expected)):assert_portable_equal(a,b,f'{path}[{index}]')
    elif isinstance(expected,int) and not isinstance(expected,bool):
        assert isinstance(actual,int) and not isinstance(actual,bool) and actual==expected,path
    elif isinstance(expected,float):
        assert isinstance(actual,(int,float)) and not isinstance(actual,bool),path
        np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-14,err_msg=path)
    else:assert actual==expected,path


def verify_wave_metrics(saved, reference, times, activations, period, phase_grid):
    """Recompute scores, accepting equivalent numerical phase minimizers.

    Scalar-optimizer evaluation counts and the last digits of an argmin are
    platform-dependent. Re-evaluate the saved phase directly and tightly bind
    every saved RMS. The fresh optimum has a separate interpolation-slope bound
    derived from bounded scalar minimization's phase stopping tolerance. No
    phase refit per window occurs and no reported endpoint is recomputed with
    a different fitted phase.
    """
    fresh=base.safe(waveform_errors(reference,times,activations,period,phase_grid=phase_grid))
    phase=saved['phase_offset_cycles']
    assert isinstance(phase,(int,float)) and not isinstance(phase,bool) and np.isfinite(phase) and -.5<=phase<.5
    errors=activations-periodic_interpolate(reference,times/period+phase)
    def rms(values):
        return dict(aggregate=min(2.,float(np.sqrt(np.mean(values**2)))),
                    per_component=np.minimum(2.,np.sqrt(np.mean(values**2,axis=0))).tolist())
    fixed=rms(errors)
    # scipy bounded minimization uses sqrt(2.2e-16)*abs(x) + xatol/3.
    # Brackets lie within [-1/grid,1+1/grid]. Four terminal tolerances cover
    # two independent terminal brackets. The interpolant's maximum RMS slope
    # converts this phase allowance into an objective allowance. This does not
    # relax the 1e-12/1e-14 direct re-evaluation of the saved scientific scores.
    phase_allowance=4*(np.sqrt(2.2e-16)*(1+1/phase_grid)+1e-10/3)
    slopes=np.diff(np.vstack((reference,reference[:1])),axis=0)*len(reference)
    rms_lipschitz=float(np.sqrt(np.mean(slopes**2,axis=1)).max())
    optimizer_allowance=rms_lipschitz*phase_allowance+1e-14+1e-12*abs(fresh['aligned_rms']['aggregate'])
    assert abs(fixed['aggregate']-fresh['aligned_rms']['aggregate'])<=optimizer_allowance,'saved phase versus fresh numerical optimum'
    fresh['phase_offset_cycles']=phase;fresh['aligned_rms']=fixed
    for window in fresh['windows']:
        selected=(times>=window['first_sample_time'])&(times<=window['last_sample_time'])
        assert int(selected.sum())==window['sample_count']
        window['aligned_rms']=rms(errors[selected])
    actual=json.loads(json.dumps(saved))
    for result in (fresh,actual):
        evaluations=result['phase_search'].pop('refinement_evaluations')
        assert isinstance(evaluations,int) and evaluations>0
    assert_portable_equal(actual,fresh,'wave_metrics')


def verify_complete_trial(row, raw, anchor, cert, local, p):
    """Re-derive a completed trial's action, phase state, classifier and readout."""
    assert row['id']==trial_id(row)
    assert row['raw']==f"trials/{row['id']}.npz"
    assert row['phase'] in p['phases'] and row['dt'] in p['dt']
    assert (row['radius']==0. and row['arm']=='zero') or (row['radius'] in p['radii'] and row['arm'] in p['arms'])
    orbit=np.asarray(cert['orbit_states'],float)[:-1]
    reference=np.asarray(cert['orbit_activations'],float)[:-1]
    assert orbit.ndim==2 and orbit.shape[1]==3 and len(orbit)%4==0 and len(orbit)>0
    assert reference.shape==orbit.shape and np.isfinite(orbit).all() and np.isfinite(reference).all()
    initial=orbit[row['phase']*len(orbit)//4];period=cert['period'];epoch=row['phase']*period/4
    np.testing.assert_array_equal(raw['initial'],initial)
    np.testing.assert_array_equal(raw['states'][0],initial)
    assert row['baseline_phase_epoch']==epoch and float(raw['dt'])==row['dt']
    direction=np.zeros(2) if row['radius']==0 else directions(anchor,local,row['radius'])[row['arm']]
    delta=row['radius']*direction
    np.testing.assert_array_equal(raw['delta'],delta)
    np.testing.assert_array_equal(raw['delta'],row['weight_delta'])
    np.testing.assert_allclose(np.linalg.norm(delta),row['radius'],rtol=1e-12,atol=1e-14)
    assert_portable_equal(row['raw_weight_norm'],float(np.linalg.norm(delta)),'raw_weight_norm')
    expected=np.array([anchor['alpha'],anchor['gamma']])+delta
    np.testing.assert_array_equal([raw['alpha'],raw['gamma']],expected)
    np.testing.assert_array_equal([row['alpha'],row['gamma']],expected)
    times=np.asarray(raw['times']);states=np.asarray(raw['states'])
    expected_times=np.linspace(0,p['probe_duration'],round(p['probe_duration']/p['sample_dt'])+1)
    np.testing.assert_allclose(times,expected_times,rtol=1e-12,atol=1e-14)
    assert states.shape==(len(expected_times),3) and np.isfinite(states).all()
    assert times[0]==0 and times[-1]==p['probe_duration']
    n=int(raw['n_events']);assert 0<=n<=p['event_capacity']
    q,residual,duration=base.classify(raw['events'][:n],raw['stats'],p)
    assert q==int(raw['period'])==row['period'] and row['nine_peak_retained']==(q==9)
    assert_portable_equal(base.safe(float(raw['residual'])),base.safe(residual),'raw recurrence residual')
    assert_portable_equal(base.safe(float(raw['period_time'])),base.safe(duration),'raw period time')
    assert_portable_equal(row['recurrence_residual'],base.safe(residual),'record recurrence residual')
    assert_portable_equal(row['period_time'],base.safe(duration),'record period time')
    assert_portable_equal(row['stats'],base.safe(raw['stats']),'record stats')
    assert_portable_equal(row['ftle'],base.safe(float(raw['stats'][1])),'record FTLE')
    assert raw['stats'][18]==round(p['burn_time']/row['dt']) and raw['stats'][19]==round(p['sample_time']/row['dt'])
    verify_wave_metrics(row['wave_metrics'],reference,times+epoch,np.tanh(states),period,p['phase_grid'])


def verify(directory):
    from hopfield_drift_design import summarize_trials
    receipt=json.loads((directory/'receipt.json').read_text());p=json.loads((directory/'manifest.json').read_text())
    assert receipt['status']=='complete' and validate(p)==receipt['planned_trial_step_calls']
    assert base.sha(directory/'manifest.json')==receipt['manifest_sha256']
    assert not (directory/'failure.json').exists() and not (directory/'timeout.json').exists()
    for item in receipt['artifacts']:
        assert base.sha(directory/item['path'])==item['sha256'],item['path']
        assert (directory/item['path']).stat().st_size==item['bytes']
    records=[json.loads(line) for line in (directory/'records.jsonl').read_text().splitlines()]
    assert len(records)==validate(p)
    assert {r['id'] for r in records}=={trial_id(s) for s in trial_specs(p)}
    anchors={a['id']:a for a in p['anchors']}
    baselines=json.loads((directory/'baselines.json').read_text())
    assert set(baselines)==set(anchors)
    eligible=[a for a in anchors if baselines[a]['eligible']]
    assert eligible==receipt['eligible_anchors']
    contexts={}
    for row in records:
        assert row['id']==trial_id(row)
        if row['anchor'] not in eligible:
            assert row['status']=='baseline_not_evaluable';continue
        assert row['status'] in ['complete','failed']
        if row['status']!='complete':continue
        key=(row['anchor'],row['phase'])
        if key not in contexts:
            cert=baselines[row['anchor']]['certificate']
            assert cert['verified'] and cert['attracting']
            with np.load(directory/'sensitivity'/f"{row['anchor']}-p{row['phase']}.npz",allow_pickle=False) as sensitivity:
                local=sensitivity['minimum_eigenvector'].copy()
                assert local.shape==(2,) and np.isfinite(local).all()
                np.testing.assert_allclose(np.linalg.norm(local),1.,rtol=1e-12,atol=1e-14)
                orbit=np.asarray(cert['orbit_states'])[:-1]
                np.testing.assert_array_equal(sensitivity['states'][0],orbit[row['phase']*len(orbit)//4])
                np.testing.assert_array_equal(sensitivity['activation_sensitivity'][0],np.zeros((3,2)))
                assert sensitivity['times'][0]==0 and sensitivity['times'][-1]==p['probe_duration']
            contexts[key]=(cert,local)
        cert,local=contexts[key]
        with np.load(directory/row['raw'],allow_pickle=False) as raw:
            verify_complete_trial(row,raw,anchors[row['anchor']],cert,local,p)
    expected=summarize_trials([r for r in records if r['anchor'] in receipt['eligible_anchors']],list(anchors),receipt['eligible_anchors'],phases=tuple(p['phases']),
                   radii=tuple(p['radii']),dt_values=tuple(p['dt']),bootstrap_samples=p['bootstrap_samples'],seed=p['bootstrap_seed'])
    assert_portable_equal(json.loads((directory/'summary.json').read_text()),base.safe(expected),'summary')
    print(json.dumps(dict(verified=True,planned_trial_step_calls=len(records),eligible_anchors=len(receipt['eligible_anchors']),source=receipt['source_commit'])))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['validate','run','verify']);parser.add_argument('--output',type=Path)
    a=parser.parse_args()
    if a.action=='validate':print(json.dumps(dict(valid=True,planned_trial_step_calls=validate(json.loads(PLAN.read_text())))))
    elif a.output is None:parser.error('--output required')
    elif a.action=='run':run(a.output)
    else:verify(a.output)
if __name__=='__main__':main()
