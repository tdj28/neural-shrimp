"""Explicit EXP-025 numerical refinement; parent raw evidence is never edited."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import shutil
import time
import traceback
import numpy as np

import run_boundary_behavior as original
import run_hopfield as base
from hopfield_drift_probe import build_trace

PLAN=original.ROOT/'experiments/hopfield/EXP-025-refinement-plan.json'


def selected_contexts(parent):
    cases=[]
    for i in range(26):
        bad=set()
        for a in range(2):
            x=np.load(parent/f'point-{i:02d}/traces-0-{a}.npy')
            y=np.load(parent/f'point-{i:02d}/traces-1-{a}.npy')
            error=np.max(np.abs(np.tanh(x)-np.tanh(y)),axis=(1,2,3))
            bad.update(np.flatnonzero(error>1e-4).tolist())
        cases.extend(dict(point=i,context=c) for c in sorted(bad))
    return cases


def validate(plan):
    assert plan['experiment_id']=='EXP-025-refinement'
    assert plan['dt']==[.000625,.0003125] and plan['workers']==6
    assert len(plan['cases'])==73 and plan['planned_trajectories']==1168
    assert len({(x['point'],x['context']) for x in plan['cases']})==73
    assert plan['wall_seconds']==600
    return plan


def refine_point(i, contexts, p, scientific, destination, fn, deadline):
    folder=destination/f'point-{i:02d}'
    point=scientific['points'][i]
    saved=np.load(folder/'contexts.npz')['contexts']
    try:
        for k,dt in enumerate(p['dt']):
            for ai,amplitude in enumerate(scientific['amplitudes']):
                data=np.load(folder/f'traces-{k}-{ai}.npy',mmap_mode='r+')
                for c in contexts:
                    if time.monotonic()>deadline: raise TimeoutError('Refinement deadline')
                    for s,signs in enumerate(original.assay.SIGNS):
                        data[c,s]=original.two_pulse(fn,point['alpha'],point['gamma'],saved[c],amplitude,signs,dt)
                    data.flush()
                del data
        base.write(folder/'refinement-progress.json',dict(complete=True,contexts=contexts))
    except BaseException as exc:
        if hasattr(exc,'payload'): np.savez_compressed(folder/'failure-payload.npz',**exc.payload)
        base.write(folder/'refinement-failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise


def run(destination):
    p=validate(json.loads(PLAN.read_text()))
    parent=Path(p['parent'])
    assert base.sha(parent/'receipt.json')==p['parent_receipt_sha256']
    original.verify(parent)
    assert selected_contexts(parent)==p['cases']
    assert not base.git('status','--porcelain')
    source=base.git('rev-parse','HEAD')
    assert base.git('branch','-r','--contains',source)
    assert all(os.environ.get(k)=='1' for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'])
    destination.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    scientific=json.loads((parent/'manifest.json').read_text())
    shutil.copyfile(parent/'manifest.json',destination/'manifest.json')
    shutil.copyfile(PLAN,destination/'refinement-manifest.json')
    files=original.SOURCE+['tools/refine_boundary_behavior.py','experiments/hopfield/EXP-025-refinement-plan.json',
                           'docs/EXP-025-refinement.md']
    meta=dict(source_commit=source,started_at=base.now(),pid=os.getpid(),workers=6,
        parent=str(parent),parent_receipt_sha256=p['parent_receipt_sha256'],
        plan_sha256=base.sha(parent/'manifest.json'),refinement_plan_sha256=base.sha(PLAN),
        source_files={f:base.sha(original.ROOT/f) for f in files},
        provenance='Composite: unchanged contexts use original paired RK4 steps; 73 explicitly mapped contexts use refined paired steps.')
    base.write(destination/'start.json',meta)
    status='failed'
    try:
        # Physical copies: later readout generation or memmap writes cannot alter parent inodes.
        for i in range(26):
            folder=destination/f'point-{i:02d}'
            folder.mkdir()
            for f in (parent/f'point-{i:02d}').iterdir():
                if f.name.startswith(('geometry','traces-','repeat-')) or f.name in ['contexts.npz','terminal_states.npy','independent.npy']:
                    shutil.copyfile(f,folder/f.name)
        fn,build=build_trace(destination/'build')
        base.write(destination/'build.json',build)
        groups={i:[c['context'] for c in p['cases'] if c['point']==i] for i in range(26)}
        with ThreadPoolExecutor(max_workers=6) as pool:
            futures=[pool.submit(refine_point,i,contexts,p,scientific,destination,fn,started+p['wall_seconds'])
                     for i,contexts in groups.items() if contexts]
            for f in futures: f.result()
        base.write(destination/'method-provenance.json',dict(parent=meta['parent'],
            parent_receipt_sha256=p['parent_receipt_sha256'],original_dt=scientific['task_dt'],refined_dt=p['dt'],
            refined_contexts=p['cases'],scope='Only two-pulse arrays change; autonomous tapes, geometry, independent solver checks and unforced repeatability arrays remain byte-identical.'))
        result=original.analyze(destination,scientific)
        status='complete'
        print(json.dumps(dict(qualified=result['qualified'],decision=result['decision'])),flush=True)
    except BaseException as exc:
        base.write(destination/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        base.write(destination/'receipt.json',dict(**meta,status=status,finished_at=base.now(),
            wall_seconds=time.monotonic()-started,artifacts=original.inventory(destination)))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    run(args.directory)
