"""Execute only frozen full-model passages; preserve every trace and failed row (Task 5b / PT2, adapted from
confirm-fold): the primary verdict uses the continuous passage integral (arctan), the secondary the discrete count."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from codex.crosscheck.qwen_map import ManualQwenSoftMap
from build_design import sha

OWNED_OUTPUT=Path('/data2/neural-shrimp-data/codex/task5b-fresh-prediction')


def validate_plan(path,target,threads):
    plan=json.loads(Path(path).read_text())
    if plan['schema']!=1 or plan['execution']['threads']!=threads or plan['execution']['device']!='cpu' or plan['execution']['dtype']!='float64':
        raise ValueError('frozen execution contract mismatch')
    for f in plan['source']:
        if sha(REPO/f['path'])!=f['sha256']:
            raise ValueError('frozen source changed: '+f['path'])
    for name,version in plan['software'].items():
        if importlib.metadata.version(name)!=version:
            raise ValueError('library version mismatch: '+name)
    found=[x for x in plan['targets'] if x['label']==target]
    if len(found)!=1:
        raise ValueError('unknown or repeated target')
    t=found[0]
    if t['status']!='ready':
        raise ValueError('target failed its fold setup; no passage authorized')
    for k in ('setup','setup_arrays','inputs'):
        f=t[k]
        if sha(f['path'])!=f['sha256'] or Path(f['path']).stat().st_size!=f['bytes']:
            raise ValueError('bound input changed: '+k)
    if len(t['rows'])!=12 or len({r['id'] for r in t['rows']})!=12:
        raise ValueError('complete unique trial table required')
    return plan,t


def run_group(model, t, eps_index, arrays, save, tick=lambda:None):
    rows=[r for r in t['rows'] if r['group']==eps_index]
    if len(rows)!=3 or [r['entrance_index'] for r in rows]!=[0,1,2]:
        raise ValueError('exact frozen three-entrance group required')
    p=rows[0]['parameter']
    alpha=p if t['coordinate']=='alpha' else t['alpha']
    logT=p if t['coordinate']=='log10T' else t['log10T']
    x=torch.tensor(arrays['x_c'],dtype=torch.float64)
    r=torch.tensor(arrays['r'],dtype=torch.float64)
    l=torch.tensor(arrays['l'],dtype=torch.float64)
    states=torch.tensor(arrays['starts'],dtype=torch.float64)
    starts=states.numpy().copy()
    counts=[None]*3
    pre=np.full_like(starts,np.nan)
    crossed=np.full_like(starts,np.nan)
    u_trace=[]; transverse=[]; state_trace=[]
    cache=model.prefix_cache(alpha)
    with torch.no_grad():
        for g in range(1,t['max_g']+1):
            tick()
            previous=states.clone()
            states=model.period_map(states,alpha,logT,t['period'],cache=cache)
            if states.shape!=x.expand(3,-1).shape or states.dtype!=torch.float64 or states.device.type!='cpu' or not torch.isfinite(states).all():
                raise ValueError('invalid full-state map output')
            diff=states-x
            u=diff@l
            transverse_norm=torch.linalg.vector_norm(diff-u[:,None]*r,dim=1)
            ur=np.full(3,np.nan); tr=np.full(3,np.nan); sr=np.full_like(starts,np.nan)
            for j in range(3):
                if counts[j] is None:
                    ur[j]=float(u[j]);tr[j]=float(transverse_norm[j]);sr[j]=states[j].numpy()
                    if ur[j]>t['U']:
                        counts[j]=g
                        pre[j]=previous[j].numpy()
                        crossed[j]=states[j].numpy()
            u_trace.append(ur);transverse.append(tr);state_trace.append(sr)
            if True:  # Keep every generated prefix, including interrupted groups.
                save(dict(u=np.asarray(u_trace),transverse=np.asarray(transverse),state_trace=np.asarray(state_trace),initial_states=starts,
                          previous_states=pre,crossing_states=crossed,states_at_stop=np.where(np.isfinite(crossed),crossed,states.numpy()).copy()))
            # Re-evaluate safe initial inputs in inactive columns; their outputs are masked.
            # Columns are independent and this avoids unobserved post-exit divergence.
            for j,n in enumerate(counts):
                if n is not None:
                    states[j]=torch.tensor(starts[j],dtype=torch.float64)
            if all(v is not None for v in counts):
                break
    result=[]
    for j,row in enumerate(rows):
        n=counts[j]
        error=None if n is None else n-row['quadratic_G']
        arcerr=None if n is None else n-row['arctan_G']
        result.append(dict(**row,status='censored' if n is None else 'observed',measured_G=n,
                           measured_F=None if n is None else t['period']*n,recorded_G=len(u_trace) if n is None else n,
                           quadratic_error=error,arctan_error=arcerr,primary_pass=n is not None and abs(arcerr)<=1,
                           secondary_pass=n is not None and abs(error)<=1))
    final=dict(u=np.asarray(u_trace),transverse=np.asarray(transverse),state_trace=np.asarray(state_trace),initial_states=starts,previous_states=pre,
               crossing_states=crossed,states_at_stop=np.where(np.isfinite(crossed),crossed,states.numpy()).copy())
    save(final)
    return result,dict(alpha=alpha,log10T=logT,G_evaluations=len(u_trace),batch_model_steps=t['period']*len(u_trace))


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--design',default=str(REPO/'codex/confirm-fold-2/design.json'))
    p.add_argument('--target',choices=['C','D'],required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--threads',type=int,required=True)
    p.add_argument('--freeze',required=True)
    p.add_argument('--pushed-at',required=True)
    args=p.parse_args()
    if len(args.freeze)!=40 or any(c not in '0123456789abcdef' for c in args.freeze):
        raise ValueError('full freeze hash required')
    if args.freeze != os.environ.get('TASK5_SOURCE_COMMIT') or args.pushed_at != os.environ.get('TASK5_PUSH_VERIFIED_AT'):
        raise ValueError('launcher provenance mismatch')
    if Path(args.design).resolve() != REPO/'codex/confirm-fold-2/design.json' or sha(args.design) != os.environ.get('TASK5_PLAN_SHA256'):
        raise ValueError('plan must be the pushed source snapshot plan')
    plan,t=validate_plan(args.design,args.target,args.threads)
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='' or os.environ.get('OMP_NUM_THREADS')!=str(args.threads) or os.environ.get('MKL_NUM_THREADS')!=str(args.threads):
        raise ValueError('CPU/thread launcher contract missing')
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    output=Path(args.output).resolve()
    if not output.is_relative_to(OWNED_OUTPUT) or output==OWNED_OUTPUT:
        raise ValueError('output outside task ownership')
    output.mkdir(parents=True,exist_ok=False)
    meta=dict(freeze_commit=args.freeze,plan_sha256=sha(args.design),freeze_pushed_at_utc=args.pushed_at,
              target=args.target,started_utc=datetime.now(timezone.utc).isoformat(),threads=torch.get_num_threads(),
              numpy=np.__version__,torch=torch.__version__,status='starting',rows=[dict(**r,status='not_run',primary_pass=False,secondary_pass=False) for r in t['rows']])
    def report():
        (output/'results.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n')
    report();start=time.monotonic()
    try:
        model=ManualQwenSoftMap.from_pinned_files()
        arrays=np.load(t['inputs']['path'],allow_pickle=False)
        meta['first_target_outcome_at_utc']=datetime.now(timezone.utc).isoformat()
        if datetime.fromisoformat(meta['first_target_outcome_at_utc'])<=datetime.fromisoformat(args.pushed_at):
            raise ValueError('freeze must precede first target evaluation')
        report()
        for i in range(4):
            for row in meta['rows'][3*i:3*i+3]:
                row['status']='running'
            report()
            def save(data):
                temporary=output/f'group-{i}.tmp.npz'
                np.savez(temporary,**data)
                temporary.replace(output/f'group-{i}.npz')
            rows,group=run_group(model,t,i,arrays,save)
            meta['rows'][3*i:3*i+3]=rows
            meta.setdefault('groups',[]).append(dict(index=i,**group,trace_sha256=sha(output/f'group-{i}.npz')))
            report()
            print(json.dumps(dict(target=args.target,group=i,counts=[r['measured_G'] for r in rows],elapsed_seconds=time.monotonic()-start)),flush=True)
        meta['status']='completed'
        meta['primary_all_pass']=all(r['primary_pass'] for r in meta['rows'])
        meta['secondary_all_pass']=all(r['secondary_pass'] for r in meta['rows'])
    except BaseException as exc:
        meta['status']='failed';meta['exception']=repr(exc);meta['traceback']=traceback.format_exc()
        raise
    finally:
        meta['elapsed_seconds']=time.monotonic()-start
        report()

if __name__=='__main__':
    main()
