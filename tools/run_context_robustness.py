"""PAUSED DRAFT: EXP-027 requires a control-complete protocol before execution.

See docs/EXP-027-status.md. No target acquisition has been run.
"""
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import argparse
import json
import os
import resource
import shutil
import time
import traceback
import numpy as np

import run_hopfield as base
import run_boundary_behavior as original
from hopfield_drift_probe import build_trace
from context_robustness import success_counts,failed_contexts,score_counts

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'experiments/hopfield/EXP-027-plan.json'
TEST=np.r_[32:96,128:192]
DIRECTION=np.ones(3)/np.sqrt(3)
FILES=['tools/context_robustness.py','tools/run_context_robustness.py',
       'tests/test_context_robustness.py','docs/EXP-027-protocol.md','experiments/hopfield/EXP-027-plan.json',
       *original.SOURCE]


def validate(p):
    assert p['experiment_id']=='EXP-027'
    assert p['points']==[2,3,10,11,16,17,20,21]
    assert p['pairs']==[[3,2],[10,11],[17,16],[20,21]]
    assert p['epsilons']==[1e-6,1e-4] and p['primary_epsilon']==1e-4
    assert p['steps']==[[.00125,.000625],[.0003125,.00015625],[.000078125,.0000390625]]
    assert p['noise_draws']==2048 and p['sigma']==.001 and p['workers']==6
    assert p['activation_gate']==1e-4 and p['accuracy_gate']==.002 and p['mcse_gate']==.001
    assert p['wall_seconds']==1800 and p['paid_compute'] is False


def point_run(index,p,raw,fn,deadline):
    folder=raw/f'point-{index:02d}'
    folder.mkdir()
    parent=Path(p['parent'])/f'point-{index:02d}'
    context=np.load(parent/'contexts.npz')['contexts'][TEST]
    geo=json.loads((parent/'geometry.json').read_text())
    alpha,gamma=geo['alpha'],geo['gamma']
    active=list(range(128));selected=[];previous=None
    try:
        for level,steps in enumerate(p['steps']):
            selected.append(dict(level=level,contexts=active.copy(),steps=steps))
            arrays=[]
            for k,dt in enumerate(steps):
                data=np.lib.format.open_memmap(folder/f'traces-level-{level}-{k}.npy',mode='w+',dtype='float64',shape=(2,2,128,4,61,3))
                if previous is None: data[:]=np.nan
                else: data[:]=previous[k]
                for c in active:
                    if time.monotonic()>deadline:raise TimeoutError('Frozen acquisition ceiling')
                    for ei,epsilon in enumerate(p['epsilons']):
                        for ji,jitter in enumerate([-1,1]):
                            initial=context[c]+jitter*epsilon*DIRECTION
                            for si,signs in enumerate(original.assay.SIGNS):
                                data[ei,ji,c,si]=original.two_pulse(fn,alpha,gamma,initial,.1,signs,dt)
                    data.flush()
                    base.write(folder/'progress.json',dict(level=level,method=k,context=c))
                arrays.append(data)
            active,error=failed_contexts(arrays[0],arrays[1],p['activation_gate'])
            selected[-1].update(remaining_failed_contexts=active.copy(),max_activation_difference=error)
            base.write(folder/'precision.json',selected)
            previous=arrays
            if not active:break
        for k in range(2):
            np.save(folder/f'final-{k}.npy',np.asarray(previous[k]),allow_pickle=False)
        independent=np.array([[original.independent_pulse(alpha,gamma,context[c]+1e-4*DIRECTION,.1,signs)
                              for signs in original.assay.SIGNS] for c in [0,64]])
        np.save(folder/'independent.npy',independent,allow_pickle=False)
        independent_error=float(np.max(np.abs(np.tanh(independent)-np.tanh(previous[1][1,1,[0,64]]))))
        base.write(folder/'acquisition.json',dict(index=index,status=geo['status'],alpha=alpha,gamma=gamma,
                    qualified=not active and independent_error<=p['activation_gate'],max_activation_difference=error,
                    independent_activation_difference=independent_error,precision=selected))
        print(json.dumps(dict(point=index,acquisition_complete=True,precision_levels=len(selected))),flush=True)
    except BaseException as exc:
        if hasattr(exc,'payload'):np.savez_compressed(folder/'native-failure-payload.npz',**exc.payload)
        base.write(folder/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise


def evaluate_point(index,p,raw):
    folder=raw/f'point-{index:02d}'
    parent=Path(p['parent'])/f'point-{index:02d}'
    counts=np.zeros((2,3,128,p['noise_draws']),dtype=np.uint8)
    for k in range(2):
        templates=np.tanh(np.load(parent/f'traces-{k}-1.npy',mmap_mode='r')[TEST,:,55:61]).reshape(128,4,18)
        actual=np.load(folder/f'final-{k}.npy',mmap_mode='r')
        for c in range(128):
            # Same draws across numerical methods, interventions and weight settings.
            noise=np.random.default_rng(p['noise_seed']+c).standard_normal((p['noise_draws'],18))
            counts[k,0,c]=success_counts(templates[c],templates[c],p['sigma'],noise)
            for ei in range(2):
                means=np.tanh(actual[ei,:,c,:,55:61]).reshape(8,18)
                counts[k,ei+1,c]=success_counts(templates[c],means,p['sigma'],noise)
    np.save(folder/'success-counts.npy',counts,allow_pickle=False)
    context,accuracy,se=score_counts(counts)
    np.save(folder/'context-accuracy.npy',context,allow_pickle=False)
    acq=json.loads((folder/'acquisition.json').read_text())
    delta=float(np.max(abs(accuracy[0]-accuracy[1])))
    result=dict(**acq,accuracy=accuracy,mcse=se,accuracy_step_difference=delta)
    result['qualified']=bool(acq['qualified'] and delta<=p['accuracy_gate'] and np.max(se)<=p['mcse_gate'])
    base.write(folder/'evaluation.json',result)
    return result


def summarize(p,raw,points):
    methods=[]
    for k in range(2):
        differences=[]
        for periodic,irregular in p['pairs']:
            cp=np.load(raw/f'point-{periodic:02d}/context-accuracy.npy')[k]
            ci=np.load(raw/f'point-{irregular:02d}/context-accuracy.npy')[k]
            differences.append((ci[0]-ci[2])-(cp[0]-cp[2]))
        d=np.array(differences);per_context=d.mean(axis=0)
        rng=np.random.default_rng(p['bootstrap_seed'])
        bootstrap=per_context[rng.integers(0,128,(4000,128))].mean(axis=1)
        interval=np.quantile(bootstrap,[.025,.975])
        mean=float(per_context.mean());pairs=d.mean(axis=1)
        if mean>.05 and interval[0]>0 and np.mean(pairs>0)>=.75:
            decision='periodic regimes protect calibrated memory better at the frozen uncertainty'
        elif mean<-.05 and interval[1]<0 and np.mean(pairs<0)>=.75:
            decision='irregular regimes protect calibrated memory better at the frozen uncertainty'
        else:decision='inconclusive or heterogeneous calibrated-memory robustness'
        methods.append(dict(mean_periodic_robustness_advantage=mean,conditional_interval=interval,
                            pair_advantages=pairs,decision=decision))
    qualified=all(x['qualified'] for x in points)
    decision=methods[1]['decision'] if qualified and methods[0]['decision']==methods[1]['decision'] else 'unqualified'
    result=dict(experiment_id='EXP-027',qualified=qualified,decision=decision,methods=methods,points=points)
    base.write(raw/'analysis.json',result)
    return result


def run(raw):
    raise RuntimeError('Paused draft: complete the distant-regime controls, protocol, tests and private freeze before enabling execution.')
    p=json.loads(PLAN.read_text());validate(p)
    parent=Path(p['parent'])
    assert base.sha(parent/'receipt.json')==p['parent_receipt_sha256']
    original.verify(parent)
    assert not base.git('status','--porcelain')
    commit=base.git('rev-parse','HEAD');assert base.git('branch','-r','--contains',commit)
    assert all(os.environ.get(k)=='1' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'])
    raw.mkdir(parents=True,exist_ok=False);shutil.copyfile(PLAN,raw/'manifest.json')
    started=time.monotonic();status='failed'
    meta=dict(source_commit=commit,plan_sha256=base.sha(PLAN),pid=os.getpid(),started_at=base.now(),
              parent=str(parent),parent_receipt_sha256=p['parent_receipt_sha256'],workers=6,
              source_files={f:base.sha(ROOT/f) for f in FILES})
    base.write(raw/'start.json',meta)
    try:
        fn,build=build_trace(raw/'build');base.write(raw/'build.json',build)
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures=[pool.submit(point_run,i,p,raw,fn,started+p['wall_seconds']) for i in p['points']]
            for f in as_completed(futures):f.result()
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures=[pool.submit(evaluate_point,i,p,raw) for i in p['points']]
            points=[f.result() for f in futures]
        result=summarize(p,raw,points);status='complete'
        print(json.dumps(dict(qualified=result['qualified'],decision=result['decision'])),flush=True)
    except BaseException as exc:
        base.write(raw/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
    finally:
        base.write(raw/'receipt.json',dict(**meta,status=status,wall_seconds=time.monotonic()-started,
              maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,artifacts=original.inventory(raw)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path)
    args=parser.parse_args();run(args.directory)
