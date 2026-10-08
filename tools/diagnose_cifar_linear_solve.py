"""One CPU-only fixed-budget CIFAR linear-direction diagnostic; no spectrum/continuation."""
import os
for _name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_name]='1'
import argparse
from datetime import datetime,timezone
import hashlib
import inspect
import json
from pathlib import Path
import platform
import resource
import re
import signal
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.sparse.linalg import gmres
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp import cifar
from neural_shrimp.cifar_linear_diagnostic import solve_direction,evaluate_trials,check_deadline

BUNDLE=Path('/data2/neural-shrimp-data/explore/cifar/device-benchmark/reference-001')
OLD=Path('/data2/neural-shrimp-data/explore/cifar/corrector-only/attempt-001')
BUNDLE_MANIFEST_SHA='414e8f2b1491a7f28cd1e1522376f7b008d7d3ac57bd29f69f0d5f4b953add38'
OLD_MANIFEST_SHA='d02393831f0d0b207b06eec11f114fb798e459bfac6679f19dc5cd68ddf50dc8'
STATE_SHA='36066b4b1d657924bd6efdce83dc3fd2d06a7ab135ef78afb03fb87649dd9674'
SOURCES=('src/neural_shrimp/__init__.py','src/neural_shrimp/controls.py','src/neural_shrimp/models.py',
    'src/neural_shrimp/cifar.py','src/neural_shrimp/cifar_linear_diagnostic.py','tools/diagnose_cifar_linear_solve.py',
    'docs/cifar-orbit-compute-plan.md','docs/research-log.md')
SETTINGS=dict(restart=50,cycles=4,rtol=1e-10,forcing_limit=.5,max_matvec=205,
    alphas=[1.,.5,.25,.125],armijo=1e-4,closure_tolerance=1e-8,primitive_separation=1e-7,
    internal_seconds=100.,new_file_bytes_cap=32*1024**2)


class Interrupted(BaseException):pass


def stop(signum,frame):raise Interrupted('SIGTERM: preserve diagnostic evidence')


def sha(path):return cifar.sha256_file(path)


def write(path,value):
    # Each phase path is exclusive; partial files are preserved on interruptions.
    with path.open('x') as handle:
        json.dump(value,handle,indent=2,allow_nan=False);handle.write('\n');handle.flush();os.fsync(handle.fileno())


def save_array(out,name,array):
    array=np.asarray(array)
    if sum(p.stat().st_size for p in out.iterdir() if p.is_file())+array.nbytes+1024>SETTINGS['new_file_bytes_cap']:
        raise RuntimeError('diagnostic retained-byte budget exhausted')
    path=out/(name+'.npy')
    with path.open('xb') as handle:
        np.save(handle,array,allow_pickle=False);handle.flush();os.fsync(handle.fileno())
    return dict(path=path.name,bytes=path.stat().st_size,sha256=sha(path))


def verify_folder(folder,expected_sha):
    path=folder/'manifest.json'
    if sha(path)!=expected_sha:raise ValueError('frozen input manifest digest mismatch')
    entries=json.loads(path.read_text())
    actual={p.name for p in folder.iterdir() if p.is_file() and p.name!='manifest.json'}
    if actual!=set(entries):raise ValueError('frozen input inventory mismatch')
    for name,item in entries.items():
        p=folder/name
        if Path(name).name!=name or p.is_symlink() or p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
            raise ValueError('frozen input file mismatch: '+name)
    return entries


def load_inputs(bundle=BUNDLE,old=OLD):
    verify_folder(bundle,BUNDLE_MANIFEST_SHA);verify_folder(old,OLD_MANIFEST_SHA)
    basis=json.loads((bundle/'result.json').read_text());prior=json.loads((old/'orbit.json').read_text())
    if basis['status']!='complete' or basis['eta']!=.00827753 or basis['lam']!=.01:
        raise ValueError('frozen model/rate changed')
    if prior['status']!='corrector_not_accepted' or prior['state_sha256']!=STATE_SHA:
        raise ValueError('expected unresolved historical corrector')
    if prior['data_identity']!=basis['data_identity']:raise ValueError('subset-A identity mismatch')
    state_path=old/prior['last_corrector']['state_file']
    if state_path.name!=prior['last_corrector']['state_file'] or sha(state_path)!=STATE_SHA:
        raise ValueError('historical state binding mismatch')
    with np.load(bundle/'reference.npz',allow_pickle=False) as z:
        expected={'x','y','theta','directions','gradient','hvp','j2vp'}
        if set(z.files)!=expected:raise ValueError('reference array inventory mismatch')
        arrays={key:z[key] for key in z.files}
    if any(a.dtype!=np.float64 or not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('reference must contain finite float64 arrays')
    n=98666
    shapes={'x':(512,3072),'y':(512,10),'theta':(n,),'gradient':(n,),'directions':(3,n),'hvp':(3,n),'j2vp':(3,n)}
    if any(arrays[k].shape!=shape for k,shape in shapes.items()):raise ValueError('reference shape mismatch')
    if not np.array_equal(np.load(state_path,allow_pickle=False),arrays['theta']):raise ValueError('reference state differs')
    for name in ('x','y'):
        identity=basis['data_identity'][name];a=np.ascontiguousarray(arrays[name])
        if list(a.shape)!=identity['shape'] or a.dtype.str!=identity['dtype'] or hashlib.sha256(a.tobytes()).hexdigest()!=identity['sha256']:
            raise ValueError('normalized subset-A array binding mismatch')
    return arrays,basis,prior


def parity(actual,expected):
    actual=np.asarray(actual);expected=np.asarray(expected)
    if actual.shape!=expected.shape or not np.isfinite(actual).all():raise ValueError('reference parity finite/shape failure')
    error=np.abs(actual-expected);scaled=float(np.max(error/(1e-11+1e-9*np.abs(expected))))
    result=dict(max_absolute_error=float(error.max()),max_scaled_error=scaled,passed=bool(scaled<=1.))
    if not result['passed']:raise ValueError('reference derivative parity failed')
    return result


def source_identity():
    result={}
    for name in SOURCES:
        data=(ROOT/name).read_bytes()
        if data!=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT):raise ValueError('uncommitted executable source: '+name)
        result[name]=hashlib.sha256(data).hexdigest()
    return result


def run(out,review_commit=None):
    sources=source_identity()  # Refuse dirty/untracked source before creating output.
    if not isinstance(review_commit,str) or re.fullmatch(r'[0-9a-f]{40}',review_commit) is None:
        raise ValueError('exact independent-review commit is required')
    subprocess.run(['git','cat-file','-e',review_commit+'^{commit}'],cwd=ROOT,check=True)
    out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+SETTINGS['internal_seconds']
    record=dict(schema='cifar-linear-diagnostic-v1',status='running',evidence_rung=0,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),source_sha256=sources,
        settings=SETTINGS,independent_review_commit=review_commit,pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat(),nice=os.getpriority(os.PRIO_PROCESS,0),
        cpu_threads=torch.get_num_threads(),torch_interop_threads=torch.get_num_interop_threads(),
        torch=torch.__version__,numpy=np.__version__,scipy=scipy.__version__,python=platform.python_version(),
        gmres_python_source_sha256=hashlib.sha256(inspect.getsource(gmres).encode()).hexdigest(),
        device='cpu',gpu=False,paid=False,load_average=os.getloadavg(),
        scope='one diagnostic direction and four fixed independent trial states; no orbit acceptance, spectrum or continuation')
    write(out/'runtime-start.json',record)
    def event(row):
        with (out/'events.jsonl').open('a') as handle:
            handle.write(json.dumps(row,allow_nan=False)+'\n');handle.flush();os.fsync(handle.fileno())
    try:
        arrays,basis,prior=load_inputs();check_deadline(deadline)
        record['input_identity']=dict(bundle=str(BUNDLE),bundle_manifest_sha256=BUNDLE_MANIFEST_SHA,
            old_corrector=str(OLD),old_manifest_sha256=OLD_MANIFEST_SHA,state_sha256=STATE_SHA,
            data_identity=basis['data_identity'],eta=basis['eta'],lam=basis['lam'],reference_source_commit=basis['source_commit'])
        write(out/'input-identity.json',record['input_identity'])
        problem=cifar.Problem(cifar.MLP(**basis['model']),arrays['x'],arrays['y'],basis['lam'])
        theta=cifar.as_tensor(arrays['theta']);eta=basis['eta']
        first=problem.update(theta,eta);second=problem.update(first,eta)
        residual=(second-theta).numpy();separation=float((first-theta).norm())/max(1.,float(theta.norm()))
        relative=float(np.linalg.norm(residual))/max(1.,float(theta.norm()))
        if not np.isfinite(residual).all() or not np.isclose(relative,prior['rows'][0]['relative_residual'],rtol=1e-10,atol=1e-16):
            raise ValueError('historical closure replay mismatch')
        if not np.isclose(separation,prior['rows'][0]['separation'],rtol=1e-10,atol=1e-16):
            raise ValueError('historical separation replay mismatch')
        record['initial']=dict(relative_residual=relative,residual_norm=float(np.linalg.norm(residual)),relative_separation=separation)
        for name,a in [('initial-theta',theta.numpy()),('initial-map1',first.numpy()),('initial-map2',second.numpy()),('initial-residual',residual)]:save_array(out,name,a)
        write(out/'initial.json',record['initial'])
        # One existing sampled direction qualifies the unchanged CPU derivative path.
        check_deadline(deadline);v=cifar.as_tensor(arrays['directions'][0])
        record['reference_parity']=dict(gradient=parity(problem.grad(theta).numpy(),arrays['gradient']),
            hvp=parity(problem.hvp(theta,v).numpy(),arrays['hvp'][0]),
            j2vp=parity(problem.jvp_two_step(theta,v,eta,theta_next=first).numpy(),arrays['j2vp'][0]))
        write(out/'parity.json',record['reference_parity'])
        def action(vector):return problem.jvp_two_step(theta,cifar.as_tensor(vector),eta,theta_next=first).numpy()-vector
        def save_returned(step,info):
            save_array(out,'direction',step)
            record['returned_direction']=dict(gmres_info=info,finite=bool(np.isfinite(step).all()),shape=list(step.shape))
            write(out/'returned-direction.json',record['returned_direction'])
        step,applied,linear_residual,linear=solve_direction(action,residual,
            **{k:SETTINGS[k] for k in ('restart','cycles','rtol','forcing_limit','max_matvec')},deadline=deadline,event=event,save_direction=save_returned)
        for name,a in [('applied-direction',applied),('true-linear-residual',linear_residual)]:save_array(out,name,a)
        record['linear']=linear;write(out/'linear.json',linear)
        def evaluate(trial,index):
            check_deadline(deadline)
            t=cifar.as_tensor(trial);one=problem.update(t,eta);two=problem.update(one,eta);r=(two-t).numpy()
            for suffix,a in [('map1',one.numpy()),('map2',two.numpy()),('residual',r)]:save_array(out,f'trial-{index}-{suffix}',a)
            return dict(residual=r,relative_separation=float((one-t).norm())/max(1.,float(t.norm())))
        def save_trial(index,trial):save_array(out,f'trial-{index}-theta',trial)
        record['trials']=evaluate_trials(theta.numpy(),residual,step,linear,evaluate,armijo=SETTINGS['armijo'],
                                        deadline=deadline,save_state=save_trial,event=event)
        record['status']='diagnostic_complete'
        record['candidate_trial_indices']=[r['index'] for r in record['trials'] if r['inexact_step_candidate']]
        record['accepted_orbit']=False;record['spectrum_requested']=False
    except BaseException as error:
        record['status']='failed_or_interrupted';record['error']=type(error).__name__+': '+str(error)
        raise
    finally:
        record['elapsed_seconds']=time.monotonic()-started;record['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        write(out/'result.json',record)
        write(out/'manifest.json',{p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(out.iterdir()) if p.is_file() and p.name!='manifest.json'})
    print(json.dumps(dict(status=record['status'],gmres_info=record['linear']['gmres_info'],
        true_forcing_ratio=record['linear']['true_forcing_ratio'],candidate_trial_indices=record['candidate_trial_indices'],
        accepted_orbit=False,elapsed_seconds=record['elapsed_seconds']),indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--review-commit',required=True)
    args=parser.parse_args();signal.signal(signal.SIGTERM,stop)
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    run(args.out,args.review_commit)


if __name__=='__main__':main()
