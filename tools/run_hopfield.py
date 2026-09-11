#!/usr/bin/env python3
"""Frozen CPU reproduction: native integration, immutable row evidence, independent orbit checks."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import numpy as np
from hopfield_diagnostics import certify, behavior_probe

ROOT=Path(__file__).resolve().parents[1]
PLAN=ROOT/'experiments/hopfield/EXP-004-plan.json'
STATS=['total_peaks','ftle','ftle_time','sample_time','mean_x1','mean_x2','mean_x3','std_x1','std_x2','std_x3','mean_tanh_x1','mean_tanh_x2','mean_tanh_x3','std_tanh_x1','std_tanh_x2','std_tanh_x3','mean_squared_speed','final_speed','burn_steps','sample_steps']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def safe(x):
    if isinstance(x,np.ndarray):return safe(x.tolist())
    if isinstance(x,np.generic):return safe(x.item())
    if isinstance(x,dict):return {str(k):safe(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [safe(v) for v in x]
    if isinstance(x,float) and not np.isfinite(x):return None
    return x
def write(path,data):Path(path).write_text(json.dumps(safe(data),indent=2,allow_nan=False)+'\n')
def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()

def validate(p):
    assert p['experiment_id']=='EXP-004' and p['kind']=='hopfield_reproduction'
    assert p['workers']==4 and p['event_capacity']>=p['max_peak_period']*p['minimum_repeats']
    for g in p['grids']:
        for axis in ['alpha','gamma']:
            lo,hi,n=g[axis];assert lo<hi and isinstance(n,int) and n>2
    assert p['dt']>0 and p['sample_time']>100 and p['burn_time']>0
    assert p['paid_compute'] is False and p['paid_review'] is False
    return sum(g['alpha'][2]*g['gamma'][2] for g in p['grids'])+p['transect']['gamma'][2]

def build_engine(directory):
    directory.mkdir(parents=True,exist_ok=True)
    lib=directory/('hopfield.dylib' if sys.platform=='darwin' else 'hopfield.so')
    compiler=os.environ.get('CXX','c++')
    flags=['-O3','-std=c++17','-fPIC','-dynamiclib' if sys.platform=='darwin' else '-shared']
    subprocess.run([compiler,*flags,str(ROOT/'native/hopfield.cpp'),'-o',str(lib)],check=True)
    info={'compiler':subprocess.check_output([compiler,'--version'],text=True).splitlines()[0], 'flags':flags,'library_sha256':sha(lib),'source_sha256':sha(ROOT/'native/hopfield.cpp')}
    dll=ctypes.CDLL(str(lib));fn=dll.hopfield_integrate
    ptr=ctypes.POINTER(ctypes.c_double)
    fn.argtypes=[ctypes.c_double,ctypes.c_double,ptr,ctypes.c_double,ctypes.c_int,ctypes.c_int,ctypes.c_int,ptr,ptr,ptr]
    fn.restype=ctypes.c_int
    return fn,info

def classify(events,stats,p):
    """Require four repeats of the full peak-section state, retaining unresolved tails."""
    if len(events)<4 or stats[16]<1e-12:return 0,float('inf'),float('nan')
    best=float('inf')
    for period in range(1,p['max_peak_period']+1):
        if len(events)<p['minimum_repeats']*period:continue
        tail=events[-p['minimum_repeats']*period:]
        a,b=tail[period:,1:],tail[:-period,1:]
        scaled=np.abs(a-b)/(p['recurrence_atol']+p['recurrence_rtol']*np.maximum(np.abs(a),np.abs(b)))
        residual=float(scaled.max());best=min(best,residual)
        if residual<=1:
            return period,residual,float(np.mean(tail[period:,0]-tail[:-period,0]))
    return 0,best,float('nan')

def cell(fn,alpha,gamma,initial,p,dt=None,burn=None,sample=None,failure_path=None):
    dt=p['dt'] if dt is None else dt;burn=p['burn_time'] if burn is None else burn;sample=p['sample_time'] if sample is None else sample
    initial=np.asarray(initial,dtype=np.float64).copy();final=np.full(3,np.nan);events=np.full((p['event_capacity'],4),np.nan);stats=np.full(20,np.nan)
    ptr=lambda x:x.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
    n=fn(float(alpha),float(gamma),ptr(initial),float(dt),round(burn/dt),round(sample/dt),len(events),ptr(final),ptr(events),ptr(stats))
    if n<0:
        if failure_path is not None:np.savez_compressed(failure_path,alpha=alpha,gamma=gamma,initial=initial,final=final,events=events,stats=stats,native_status=n)
        raise RuntimeError(f'Native cell failure {n} at {alpha},{gamma}')
    assert 0<=n<=len(events) and np.isfinite(final).all()
    period,residual,duration=classify(events[:n],stats,p)
    return final,events,stats,n,period,residual,duration

def row(fn,alpha,gamma,p,path,deadline,settings=None):
    if time.monotonic()>deadline:raise TimeoutError('Frozen wall ceiling reached')
    out={k:[] for k in ['states','events','stats','n_events','period','residual','period_time']}
    initial=np.array(p['initial_state'],dtype=float)
    journal=path.with_suffix('.journal');progress=path.with_suffix('.progress.json')
    try:
        with journal.open('xb') as stream:
            for j,a in enumerate(alpha):
                if time.monotonic()>deadline:raise TimeoutError('Frozen wall ceiling reached')
                write(progress,{'next_column':j,'alpha':float(a),'gamma':float(gamma),'completed_cells':j})
                result=cell(fn,a,gamma,initial,p,failure_path=path.with_suffix('.failed-cell.npz'),**(settings or {}))
                # Append a full exact numeric checkpoint after every completed cell.
                checkpoint=np.concatenate([[a,gamma],initial,*[np.asarray(v).reshape(-1) for v in result]])
                np.save(stream,checkpoint,allow_pickle=False);stream.flush()
                for key,value in zip(out,result):out[key].append(value)
                initial=result[0]
        np.savez_compressed(path,alpha=np.asarray(alpha),gamma=float(gamma),**{k:np.asarray(v) for k,v in out.items()})
        with np.load(path,allow_pickle=False) as saved:
            assert all(np.array_equal(saved[k],np.asarray(v),equal_nan=True) for k,v in out.items())
        journal.unlink();progress.unlink()
    except BaseException as error:
        write(path.with_suffix('.failure.json'),{'gamma':gamma,'completed_cells':len(out['states']),'error':repr(error),'journal':'Append-only numpy arrays: alpha,gamma,initial3,final3,events_flat,stats20,n_events,period,residual,period_time'})
        raise
    return {'path':str(path),'cells':len(alpha),'periodic_candidates':sum(v>0 for v in out['period'])}

def load_grid(directory):
    files=sorted(directory.glob('row-*.npz')); rows=[]
    for f in files:
        with np.load(f,allow_pickle=False) as z:rows.append({k:z[k].copy() for k in z.files})
    return {k:np.asarray([r[k] for r in rows]) for k in rows[0]}

def select_points(g,p):
    aa=g['alpha'];gg=np.broadcast_to(g['gamma'][:,None],aa.shape)
    distance=lambda a,b:((aa-a)/.024)**2+((gg-b)/.41)**2
    def pick(mask,d,farthest=False):
        if not mask.any():return None
        scores=np.where(mask,d,-np.inf if farthest else np.inf)
        return tuple(int(x) for x in np.unravel_index(np.argmax(scores) if farthest else np.argmin(scores),scores.shape))
    core=pick(g['period']==9,distance(1.064,10.26))
    result={'core':core,'outer':None,'doubled':None,'exterior':None}
    if core is None:return result
    ca,cg=aa[core],gg[core];d=distance(ca,cg)
    from scipy.ndimage import label
    labels,_=label(g['period']==9,structure=np.ones((3,3)))
    result['outer']=pick(labels==labels[core],d,True)
    result['doubled']=pick(g['period']==18,d)
    result['exterior']=pick((g['period']==0)&(g['stats'][:,:,1]>.005),d)
    if result['exterior'] is None:result['exterior']=pick(g['period']==0,d)
    return result

def details(fn,directory,p,deadline):
    g=load_grid(directory/'focus');selected={};(directory/'selected').mkdir()
    for name,index in select_points(g,p).items():
        if index is None:selected[name]={'status':'missing'};continue
        if time.monotonic()>deadline:raise TimeoutError('Frozen wall ceiling reached')
        i,j=index;alpha=float(g['alpha'][i,j]);gamma=float(g['gamma'][i]);q=int(g['period'][i,j]);state=g['states'][i,j]
        item={'status':'selected','row':i,'column':j,'alpha':alpha,'gamma':gamma,'candidate_period':q,'grid_stats':g['stats'][i,j],'refinements':[]}
        for dt in p['refinement']['dt']:
            z=cell(fn,alpha,gamma,state,p,dt=dt,burn=p['refinement']['burn_time'],sample=p['refinement']['sample_time'],failure_path=directory/'selected'/f'{name}-dt-{dt:g}-failure.npz')
            final,events,stats,n,period,residual,duration=z
            np.savez_compressed(directory/'selected'/f'{name}-dt-{dt:g}.npz',final=final,events=events,stats=stats,n_events=n,period=period,residual=residual,period_time=duration)
            item['refinements'].append({'dt':dt,'period':period,'residual':residual,'period_time':duration,'stats':stats})
        if q and all(r['period']==q for r in item['refinements']):
            item['certificate']=certify(alpha,gamma,events[n-q,1:],duration,q)
        else:item['certificate']={'verified':False,'attracting':False,'reason':'No shared refined periodic candidate'}
        probe=behavior_probe(alpha,gamma,final,duration=p['behavior']['duration'],sample_dt=p['behavior']['sample_dt'],certificate=item['certificate'] if item['certificate'].get('verified') else None)
        item['behavior']=probe;selected[name]=item
        write(directory/'selected.json',selected)
        print(json.dumps({'selected':name,'alpha':alpha,'gamma':gamma,'period':q,'certified':item['certificate'].get('verified'),'attracting':item['certificate'].get('attracting')}),flush=True)
    write(directory/'selected.json',selected)

def verify(directory):
    assert not (directory/'failure.json').exists() and not (directory/'timeout.json').exists(),'Failed attempt cannot be released as complete'
    receipt=json.loads((directory/'receipt.json').read_text())
    for item in receipt['artifacts']:
        assert sha(directory/item['path'])==item['sha256'],item['path']
    p=json.loads((directory/'manifest.json').read_text());validate(p)
    assert receipt['status']=='complete'
    for g in p['grids']:
        rows=load_grid(directory/g['name']);assert rows['period'].shape==(g['gamma'][2],g['alpha'][2])
        assert np.array_equal(rows['gamma'],np.linspace(*g['gamma']))
        assert np.array_equal(rows['alpha'],np.tile(np.linspace(*g['alpha']),(g['gamma'][2],1)))
        for i in range(g['gamma'][2]):
            for j in range(g['alpha'][2]):
                n=rows['n_events'][i,j];q,r,t=classify(rows['events'][i,j,:n],rows['stats'][i,j],p)
                assert q==rows['period'][i,j]
    assert len(list((directory/'transect').glob('row-*.npz')))==p['transect']['gamma'][2]
    for i,f in enumerate(sorted((directory/'transect').glob('row-*.npz'))):
        with np.load(f,allow_pickle=False) as z:
            assert z['alpha'][0]==p['transect']['alpha'] and z['gamma']==np.linspace(*p['transect']['gamma'])[i]
            assert classify(z['events'][0,:z['n_events'][0]],z['stats'][0],p)[0]==z['period'][0]
    for f in (directory/'selected').glob('*-dt-*.npz'):
        with np.load(f,allow_pickle=False) as z:assert classify(z['events'][:int(z['n_events'])],z['stats'],p)[0]==z['period']
    print(json.dumps({'verified':True,'source':receipt['source_commit'],'planned_grid_and_transect_cells':validate(p)}))

def run(directory):
    p=json.loads(PLAN.read_text());total=validate(p)
    assert not git('status','--porcelain'), 'Source must be clean before target outcomes'
    source=git('rev-parse','HEAD');assert git('branch','-r','--contains',source),'Push source before target outcomes'
    directory.mkdir(parents=True,exist_ok=False);write(directory/'manifest.json',p)
    started={'source_commit':source,'manifest_sha256':sha(PLAN),'started_at':now(),'status':'running','platform':platform.platform(),'python':sys.version,'workers':p['workers'],'planned_grid_and_transect_cells':total,'stats_columns':STATS,'paid_compute':0,'paid_review':0}
    write(directory/'started.json',started);start=time.monotonic();deadline=start+p['wall_clock_ceiling_seconds']
    def hard_stop():
        write(directory/'timeout.json',{'source_commit':source,'ended_at':now(),'status':'hard_timeout','ceiling_seconds':p['wall_clock_ceiling_seconds'],'original_attempt_preserved':True})
        os._exit(124)
    watchdog=threading.Timer(p['wall_clock_ceiling_seconds'],hard_stop);watchdog.daemon=True;watchdog.start()
    try:
        fn,build=build_engine(directory/'build');write(directory/'build.json',build)
        for grid in p['grids']:
            dest=directory/grid['name'];dest.mkdir();alpha=np.linspace(*grid['alpha']);gamma=np.linspace(*grid['gamma'])
            with ThreadPoolExecutor(max_workers=p['workers']) as pool:
                futures={pool.submit(row,fn,alpha,float(g),p,dest/f'row-{i:03}.npz',deadline):i for i,g in enumerate(gamma)}
                for completed,f in enumerate(as_completed(futures),1):
                    result=f.result();print(json.dumps({'grid':grid['name'],'rows_complete':completed,'rows_total':len(gamma),'elapsed_seconds':round(time.monotonic()-start,1),**result}),flush=True)
        t=p['transect'];dest=directory/'transect';dest.mkdir()
        settings={'dt':t['dt'],'burn':t['burn_time'],'sample':t['sample_time']}
        with ThreadPoolExecutor(max_workers=p['workers']) as pool:
            futures=[pool.submit(row,fn,[t['alpha']],float(g),p,dest/f'row-{i:03}.npz',deadline,settings) for i,g in enumerate(np.linspace(*t['gamma']))]
            for done,f in enumerate(as_completed(futures),1):
                f.result()
                if done%20==0:print(json.dumps({'transect_complete':done,'elapsed_seconds':round(time.monotonic()-start,1)}),flush=True)
        details(fn,directory,p,deadline)
        files=[f for f in sorted(directory.rglob('*')) if f.is_file() and 'build' not in f.relative_to(directory).parts]
        write(directory/'receipt.json',{**started,'status':'complete','ended_at':now(),'wall_seconds':time.monotonic()-start,'artifacts':[{'path':str(f.relative_to(directory)),'sha256':sha(f),'bytes':f.stat().st_size} for f in files]})
        verify(directory)
    except BaseException as error:
        write(directory/'failure.json',{'source_commit':source,'ended_at':now(),'error':repr(error),'wall_seconds':time.monotonic()-start,'original_attempt_preserved':True})
        raise
    finally:
        watchdog.cancel()

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['validate','run','verify']);parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.action=='validate':print(json.dumps({'valid':True,'planned_cells':validate(json.loads(PLAN.read_text()))}));return
    if args.output is None:parser.error('--output required')
    if args.action=='run':run(args.output)
    else:verify(args.output)
if __name__=='__main__':main()
