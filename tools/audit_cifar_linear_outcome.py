"""Independent saved-array arithmetic audit; no model, HVP, GMRES or trajectory."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse
from datetime import datetime,timezone
import hashlib
import inspect
import json
import math
from pathlib import Path
import resource
import subprocess
import time
import numpy as np
import scipy
from scipy.sparse.linalg import gmres
ROOT=Path(__file__).resolve().parents[1]
SOURCE='a9830e714d259597a53cee6ef7be41a9f07bb205'
REVIEW='dc7b4fea57c26c0a9e6db1b255c0e1d44efc1799'
BUNDLE=Path('/data2/neural-shrimp-data/explore/cifar/device-benchmark/reference-001')
OLD=Path('/data2/neural-shrimp-data/explore/cifar/corrector-only/attempt-001')
PINS={'bundle':'414e8f2b1491a7f28cd1e1522376f7b008d7d3ac57bd29f69f0d5f4b953add38',
      'old':'d02393831f0d0b207b06eec11f114fb798e459bfac6679f19dc5cd68ddf50dc8',
      'state':'36066b4b1d657924bd6efdce83dc3fd2d06a7ab135ef78afb03fb87649dd9674'}
SOURCES=('src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py',
 'src/neural_shrimp/cifar.py','src/neural_shrimp/cifar_linear_diagnostic.py','tools/diagnose_cifar_linear_solve.py',
 'docs/cifar-orbit-compute-plan.md','docs/research-log.md')
EXPECTED=dict(restart=50,cycles=4,rtol=1e-10,forcing_limit=.5,max_matvec=205,
 alphas=[1.,.5,.25,.125],armijo=1e-4,closure_tolerance=1e-8,primitive_separation=1e-7,
 internal_seconds=100.,new_file_bytes_cap=32*1024**2)


def sha(data):return hashlib.sha256(data).hexdigest()
def entry(path):return dict(bytes=path.stat().st_size,sha256=sha(path.read_bytes()))
def load(path):return json.loads(path.read_text())
def canonical(x):return (json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
def require(condition,message):
    if not condition:raise ValueError(message)
def close(a,b,label):
    require(math.isfinite(float(a)) and math.isfinite(float(b)) and math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-20),label)
def norm(x):return math.sqrt(math.fsum(float(v)*float(v) for v in x))
def dot(x,y):return math.fsum(float(a)*float(b) for a,b in zip(x,y))


def inventory(root,pin=None):
    manifest=root/'manifest.json'
    if pin:require(sha(manifest.read_bytes())==pin,'pinned manifest mismatch')
    items=load(manifest)
    require(set(p.name for p in root.iterdir())==set(items)|{'manifest.json'},'inventory mismatch')
    for name,expected in items.items():
        p=root/name
        require(Path(name).name==name and p.is_file() and not p.is_symlink() and entry(p)==expected,'file hash '+name)
    return items


def validate_linear(residual,direction,applied,saved_residual,record):
    require(np.array_equal(saved_residual,applied+residual),'true linear residual array')
    rhs=norm(residual);true=norm(saved_residual);q=true/rhs;slope=dot(residual,applied)
    for key,value in dict(rhs_norm=rhs,true_linear_residual_norm=true,true_forcing_ratio=q,merit_directional_derivative=slope).items():close(record[key],value,'linear '+key)
    decisions=dict(forcing_test_pass=q<=.5,descent_direction=slope<0,requested_tolerance_met=true<=1e-10*rhs)
    for key,value in decisions.items():require(record[key] is value,'linear decision '+key)
    return dict(rhs_norm=rhs,true_linear_residual_norm=true,true_forcing_ratio=q,merit_directional_derivative=slope,**decisions)


def validate_trial(theta,initial,step,applied,alpha,trial,one,two,residual,row,linear):
    require(np.array_equal(trial,theta+alpha*step),'trial base-state/damping mismatch')
    require(np.array_equal(residual,two-trial),'trial residual array mismatch')
    phi0=.5*dot(initial,initial);phi=.5*dot(residual,residual)
    residual_norm=norm(residual);relative=residual_norm/max(1.,norm(trial));separation=norm(one-trial)/max(1.,norm(trial))
    upper=phi0+1e-4*alpha*linear['merit_directional_derivative']
    values=dict(relative_step_norm=alpha*norm(step)/max(1.,norm(theta)),armijo_upper=upper,
       residual_norm=residual_norm,relative_residual=relative,merit=phi,merit_ratio=phi/phi0,relative_separation=separation)
    for key,value in values.items():close(row[key],value,'trial '+key)
    decisions=dict(closure_screen=relative<1e-8,primitive_screen=separation>=1e-7,
       merit_decreased=phi<phi0,armijo_pass=phi<=upper,finite=True,
       inexact_step_candidate=bool(linear['forcing_test_pass'] and linear['descent_direction'] and phi<=upper),
       accepted_orbit=False,applied_to_continuation=False)
    for key,value in decisions.items():require(row[key] is value,'trial decision '+key)
    close(row['alpha'],alpha,'alpha')
    return dict(alpha=alpha,**values,**decisions)


def audit(raw):
    items=inventory(raw)
    require(sum(p.stat().st_size for p in raw.iterdir())<=32*1024**2,'output byte cap')
    runtime=load(raw/'runtime-start.json');result=load(raw/'result.json');identity=load(raw/'input-identity.json')
    require(runtime['source_commit']==SOURCE and runtime['independent_review_commit']==REVIEW,'source/review commit')
    require(set(runtime['source_sha256'])==set(SOURCES),'source inventory')
    for name in SOURCES:
        blob=subprocess.check_output(['git','show',SOURCE+':'+name],cwd=ROOT)
        require(sha(blob)==runtime['source_sha256'][name],'source bytes '+name)
    for key,value in runtime.items():
        if key!='status':require(result[key]==value,'runtime/result binding '+key)
    require(runtime['settings']==EXPECTED,'changed settings')
    require(runtime['cpu_threads']==runtime['torch_interop_threads']==1 and runtime['nice']>=10 and runtime['device']=='cpu' and not runtime['gpu'] and not runtime['paid'],'resource scope')
    require(runtime['gmres_python_source_sha256']==sha(inspect.getsource(gmres).encode()),'GMRES implementation hash')
    require(runtime['numpy']==np.__version__ and runtime['scipy']==scipy.__version__,'array/solver library versions')
    require(result['status']=='diagnostic_complete' and result['accepted_orbit'] is False and result['spectrum_requested'] is False,'scope/status')
    inventory(BUNDLE,PINS['bundle']);inventory(OLD,PINS['old'])
    basis=load(BUNDLE/'result.json');prior=load(OLD/'orbit.json')
    require(identity==result['input_identity'],'input identity mirror')
    require(identity['bundle']==str(BUNDLE) and identity['old_corrector']==str(OLD),'input paths')
    require(identity['bundle_manifest_sha256']==PINS['bundle'] and identity['old_manifest_sha256']==PINS['old'] and identity['state_sha256']==PINS['state'],'input pins')
    require(identity['data_identity']==basis['data_identity']==prior['data_identity'],'subset A data identity')
    require(identity['eta']==basis['eta']==.00827753 and identity['lam']==basis['lam']==.01,'model/rate identity')
    require(prior['state_sha256']==PINS['state'] and prior['status']=='corrector_not_accepted','previous unresolved state')
    state=OLD/prior['last_corrector']['state_file'];require(sha(state.read_bytes())==PINS['state'],'state digest')
    arrays={p.stem:np.load(p,allow_pickle=False) for p in raw.glob('*.npy')}
    require(len(arrays)==23 and all(a.shape==(98666,) and a.dtype==np.float64 and np.isfinite(a).all() for a in arrays.values()),'saved array inventory/type/shape')
    theta=arrays['initial-theta'];r=arrays['initial-residual'];p=arrays['direction'];ap=arrays['applied-direction']
    require(np.array_equal(theta,np.load(state,allow_pickle=False)),'initial historical state')
    with np.load(BUNDLE/'reference.npz',allow_pickle=False) as bundle:
        require(np.array_equal(bundle['theta'],theta),'reference theta')
        for key in ('x','y'):
            a=np.ascontiguousarray(bundle[key]);info=identity['data_identity'][key]
            require(info['shape']==list(a.shape) and info['dtype']==a.dtype.str and info['sha256']==sha(a.tobytes()),'normalized '+key)
    require(np.array_equal(r,arrays['initial-map2']-theta),'initial residual relation')
    initial=load(raw/'initial.json');require(initial==result['initial'],'initial mirror')
    close(initial['residual_norm'],norm(r),'initial norm')
    close(initial['relative_residual'],norm(r)/max(1.,norm(theta)),'initial relative residual')
    close(initial['relative_separation'],norm(arrays['initial-map1']-theta)/max(1.,norm(theta)),'initial separation')
    close(initial['relative_residual'],prior['rows'][0]['relative_residual'],'historical closure')
    close(initial['relative_separation'],prior['rows'][0]['separation'],'historical separation')
    linear=load(raw/'linear.json');require(linear==result['linear'],'linear mirror')
    require(linear['gmres_info']==4 and not linear['gmres_reported_converged'] and linear['inner_iterations']==200 and linear['matvec_calls']==205,'actual solve counts/status')
    require(linear['restart']==50 and linear['cycles']==4 and linear['max_matvec']==205 and linear['forcing_limit']==.5 and linear['requested_linear_tolerance']==1e-10,'linear settings')
    recomputed=validate_linear(r,p,ap,arrays['true-linear-residual'],linear)
    events=[json.loads(line) for line in (raw/'events.jsonl').read_text().splitlines()]
    inner=[e for e in events if e['kind']=='gmres_inner'];products=[e for e in events if e['kind']=='matvec']
    require([e['iteration'] for e in inner]==list(range(1,201)) and [e['index'] for e in products]==list(range(1,206)),'operation event sequence')
    require([e['estimated_relative_residual'] for e in inner]==linear['estimated_relative_residual_history'],'inner residual history')
    require(all(math.isfinite(e['seconds']) and e['seconds']>=0 for e in products),'operation timings')
    trial_events=[{k:v for k,v in e.items() if k!='kind'} for e in events if e['kind']=='trial_result']
    require(trial_events==result['trials'] and len(result['trials'])==4,'trial event/result mirrors')
    trials=[]
    for i,alpha in enumerate((1.,.5,.25,.125)):
        row=result['trials'][i];require(row['index']==i,'trial index')
        trials.append(validate_trial(theta,r,p,ap,alpha,arrays[f'trial-{i}-theta'],arrays[f'trial-{i}-map1'],arrays[f'trial-{i}-map2'],arrays[f'trial-{i}-residual'],row,recomputed))
    require(result['candidate_trial_indices']==[i for i,t in enumerate(trials) if t['inexact_step_candidate']],'candidate aggregate')
    parity=load(raw/'parity.json');require(parity==result['reference_parity'] and set(parity)=={'gradient','hvp','j2vp'},'parity record')
    require(all(v['passed'] and 0<=v['max_scaled_error']<=1 for v in parity.values()),'parity gate recorded failure')
    returned=load(raw/'returned-direction.json');require(returned==result['returned_direction'] and returned['gmres_info']==4 and returned['finite'] and returned['shape']==[98666],'saved returned direction')
    return dict(status='passed_saved_array_audit',source_commit=SOURCE,source_review_commit=REVIEW,
        raw_root=str(raw),manifest=entry(raw/'manifest.json'),files_checked=len(items),arrays_checked=len(arrays),
        raw_total_bytes=sum(p.stat().st_size for p in raw.iterdir()),recomputed_linear=recomputed,recomputed_trials=trials,
        actual_inner_iterations=200,actual_operator_calls=205,candidates=[],accepted_orbit=False,
        target_model_evaluations=0,target_krylov_solves=0,
        limitations=['Stored map images and A*p are hash-bound but not independently reevaluated in this audit.',
          'Arithmetic uses independent math.fsum reductions, with scalar agreement relative1e-12/absolute1e-20; not interval proof.',
          'Small-step merit decrease does not satisfy the declared forcing gate and is not orbit convergence.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('raw',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    record=dict(status='running',audit_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      audit_source_sha256=sha(Path(__file__).read_bytes()),started_utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),nice=os.getpriority(os.PRIO_PROCESS,0))
    (args.out/'runtime-start.json').write_bytes(canonical(record))
    try:record.update(audit(args.raw))
    except BaseException as error:record.update(status='failed_or_interrupted',error=type(error).__name__+': '+str(error));raise
    finally:
        record.update(elapsed_seconds=time.monotonic()-start,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (args.out/'review.json').write_bytes(canonical(record))
        (args.out/'manifest.json').write_bytes(canonical({p.name:entry(p) for p in args.out.iterdir()}))
        for p in args.out.iterdir():p.chmod(0o444)
        args.out.chmod(0o555)
    print(canonical(record).decode())


if __name__=='__main__':main()
