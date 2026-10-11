"""EXP-026 saved-trajectory information diagnostic; no new neural integrations."""
from pathlib import Path
import argparse
import json
import os
import shutil
import time
import numpy as np

import run_hopfield as base
import run_boundary_behavior as parent_runner
from context_memory_bounds import batch

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'experiments/hopfield/EXP-026-plan.json'
SOURCES=['tools/context_memory_bounds.py','tools/run_context_memory.py',
         'tests/test_context_memory_bounds.py','experiments/hopfield/EXP-026-plan.json',
         'docs/EXP-026-protocol.md','tools/run_hopfield.py','tools/run_boundary_behavior.py',
         'tools/boundary_behavior.py','tools/hopfield_diagnostics.py','tools/hopfield_drift_probe.py']


def validate(p):
    assert p['experiment_id']=='EXP-026' and p['sigmas']==[.0001,.001,.01]
    assert p['primary_sigma']==.001 and p['high_threshold']==.95 and p['low_threshold']==.55
    assert p['amplitudes']==[.01,.1] and p['windows']==[[20,25],[55,60]]
    assert p['mean_bound_gate']==.002 and p['wall_seconds']==300
    assert p['paid_compute'] is False


def analyze(source,destination,p):
    prior=json.loads((source/'analysis.json').read_text())
    test=np.r_[32:96,128:192]
    points=[]
    for i,point in enumerate(prior['points']):
        records=[]
        for ai,amplitude in enumerate(p['amplitudes']):
            traces=[np.load(source/f'point-{i:02d}/traces-{k}-{ai}.npy',mmap_mode='r') for k in range(2)]
            for wi,(lo,hi) in enumerate(p['windows']):
                means=[np.tanh(x[test,:,lo:hi+1]).reshape(128,4,-1) for x in traces]
                for si,sigma in enumerate(p['sigmas']):
                    b=np.array([batch(m,sigma) for m in means])
                    np.save(destination/f'point-{i:02d}-{ai}-{wi}-{si}.npy',b,allow_pickle=False)
                    lower=np.min(b[:,:,:,0],axis=0)
                    upper=np.max(b[:,:,:,1],axis=0)
                    record=dict(amplitude=amplitude,window=[lo,hi],sigma=sigma,
                        lower_mean=lower.mean(axis=0),upper_mean=upper.mean(axis=0),
                        linear_accuracy_mean=np.min(b[:,:,:,2],axis=0).mean(axis=0),
                        high_fraction=np.mean(lower>=p['high_threshold'],axis=0),
                        low_fraction=np.mean(upper<=p['low_threshold'],axis=0),
                        mean_step_discrepancy=float(np.max(np.abs(b[0,:,:,:2].mean(axis=0)-b[1,:,:,:2].mean(axis=0)))),
                        max_context_step_discrepancy=float(np.max(np.abs(b[0,:,:,:2]-b[1,:,:,:2]))))
                    records.append(record)
        points.append(dict(index=i,alpha=point['alpha'],gamma=point['gamma'],status=point['status'],records=records))
        print(json.dumps(dict(point=i,analysis_complete=True)),flush=True)
    primary=lambda point:next(x for x in point['records'] if x['amplitude']==.1 and x['window']==[55,60] and x['sigma']==.001)
    qualified=all(r['mean_step_discrepancy']<=p['mean_bound_gate'] for x in points for r in x['records'])
    group={}
    for status in ['periodic','irregular-supported','mixed-or-unresolved']:
        rows=[primary(x) for x in points if x['status']==status]
        group[status]=dict(points=len(rows),lower_mean=float(np.mean([r['lower_mean'][0] for r in rows])),
                          upper_mean=float(np.mean([r['upper_mean'][0] for r in rows])),
                          high_fraction=float(np.mean([r['high_fraction'][0] for r in rows])),
                          low_fraction=float(np.mean([r['low_fraction'][0] for r in rows])))
    pairs=[]
    for pi,qi in prior['method_decisions'][1]['pairs']:
        a,b=primary(points[pi]),primary(points[qi])
        pairs.append(dict(periodic=pi,irregular=qi,periodic_minus_irregular_high_fraction=float(a['high_fraction'][0]-b['high_fraction'][0]),
                          periodic_lower=float(a['lower_mean'][0]),irregular_lower=float(b['lower_mean'][0])))
    result=dict(experiment_id='EXP-026',qualified=qualified,groups=group,pairs=pairs,points=points)
    base.write(destination/'analysis.json',result)
    return result


def run(destination):
    p=json.loads(PLAN.read_text());validate(p)
    source=Path(p['parent'])
    assert base.sha(source/'receipt.json')==p['parent_receipt_sha256']
    assert json.loads((source/'analysis.json').read_text())['qualified']
    parent_runner.verify(source)
    assert not base.git('status','--porcelain')
    commit=base.git('rev-parse','HEAD')
    assert base.git('branch','-r','--contains',commit)
    assert os.environ.get('OPENBLAS_NUM_THREADS')=='1'
    destination.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(PLAN,destination/'manifest.json')
    started=time.monotonic()
    meta=dict(source_commit=commit,plan_sha256=base.sha(PLAN),pid=os.getpid(),started_at=base.now(),
              parent=str(source),parent_receipt_sha256=p['parent_receipt_sha256'],
              source_files={f:base.sha(ROOT/f) for f in SOURCES})
    base.write(destination/'start.json',meta)
    status='failed'
    try:
        result=analyze(source,destination,p)
        assert time.monotonic()-started<p['wall_seconds']
        status='complete'
        print(json.dumps(base.safe(dict(qualified=result['qualified'],groups=result['groups']))),flush=True)
    except BaseException as exc:
        base.write(destination/'failure.json',dict(error=repr(exc)))
        raise
    finally:
        base.write(destination/'receipt.json',dict(**meta,status=status,wall_seconds=time.monotonic()-started,
                    artifacts=parent_runner.inventory(destination)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    run(args.directory)
