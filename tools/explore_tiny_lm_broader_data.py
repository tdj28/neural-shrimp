"""Bounded fixed-data breadth challenge; profile first, four development cases only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.tiny_lm import TinyTransformer, LMProblem, lanczos_extremes
from neural_shrimp.tiny_lm_prediction import CausalFeatures, burst_labels
from neural_shrimp.tiny_lm_progress import (DIRECTION_SEED,PROFILE_SEED,TARGET_SEEDS,
    canonical_hash,digest,load_fixed_data,progress_targets,verify_inventory)
from explore_tiny_lm import direction,metadata,manifest,relative
from explore_tiny_lm_prediction import (CollectionInterrupted,handle_sigterm,save_json,validated_rate_scale)

PLAN='docs/plans/explore-tiny-lm-broader-data.md'
SOURCE_PATHS=['src/neural_shrimp/tiny_lm.py','src/neural_shrimp/tiny_lm_prediction.py',
              'src/neural_shrimp/tiny_lm_progress.py','tools/explore_tiny_lm.py',
              'tools/explore_tiny_lm_prediction.py','tools/explore_tiny_lm_broader_data.py',PLAN]
STORAGE_CAP=512*1024**2


def committed_identity(paths=SOURCE_PATHS):
    hashes={}
    for name in paths:
        raw=(ROOT/name).read_bytes()
        if raw!=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT):
            raise ValueError('source/plan must match committed HEAD: '+name)
        hashes[name]=digest(raw)
    return {'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'sha256':hashes}


def validate_request(mode,seconds):
    if mode not in ('profile','collect'):raise ValueError('profile or fixed collect mode only')
    cap=60 if mode=='profile' else 450
    if not np.isfinite(seconds) or not 0<seconds<=cap:raise ValueError('invalid wall budget')
    return ((PROFILE_SEED,),100) if mode=='profile' else (TARGET_SEEDS,800)


def probe_steps(steps):
    return sorted({0,25,*range(100,steps+1,100)} & set(range(steps+1)))


def profile_precision(problem,theta,eta):
    cpu=LMProblem(problem.model,problem.tokens.cpu(),problem.targets.cpu(),problem.ridge)
    theta64=theta.double().cpu();v=direction(theta,DIRECTION_SEED)
    return {'direction_seed':DIRECTION_SEED,
            'gradient_relative_error':relative(problem.grad(theta).double().cpu(),cpu.grad(theta64)),
            'hvp_relative_error':relative(problem.hvp(theta,v).double().cpu(),cpu.hvp(theta64,v.double().cpu())),
            'update_relative_error':relative(problem.update(theta,eta).double().cpu(),cpu.update(theta64,eta)),
            'scope':'Matched rounded initial state only; no trajectory precision equivalence.'}


def collect_case(problem,probe,initial,eta,seed,arm,steps,folder,deadline,role,data_identity):
    folder.mkdir(exist_ok=False)
    monitor=CausalFeatures();theta=initial.clone();completed=0;chunk_start=0
    rows=[];features=[];chunk=[];probe_rows=[];last_cpu=None;guard_step=None
    outcome='running';started=time.monotonic();feature_seconds=0.;gradient_seconds=0.;probe_seconds=0.
    final_train_ce=None;schedule=probe_steps(steps)
    grad_value=torch.func.grad_and_value(problem.loss)
    def save_states():
        nonlocal chunk_start,chunk
        if not chunk:return
        with (folder/f'states-{chunk_start:06d}-{chunk_start+len(chunk)-1:06d}.npy').open('xb') as f:
            np.save(f,np.asarray(chunk));f.flush();os.fsync(f.fileno())
        chunk_start+=len(chunk);chunk=[]
    def evaluate_probe(t):
        nonlocal probe_seconds
        tick=time.monotonic()
        try:
            with torch.no_grad():value=float(probe.cross_entropy(theta))
            entry={'step':t,'ce':value,'status':'ok' if np.isfinite(value) else 'nonfinite'}
        except Exception as error:
            entry={'step':t,'ce':None,'status':'failed','error':repr(error)}
        probe_seconds+=time.monotonic()-tick;probe_rows.append(entry)
        save_json(folder/f'probe-{t:06d}.json',entry)
    try:
        for t in range(steps):
            if time.monotonic()>=deadline:outcome='wall_budget_stop';break
            theta_cpu=theta.detach().cpu().numpy().copy();chunk.append(theta_cpu)
            tick=time.monotonic();gradient,objective=grad_value(theta)
            objective=float(objective);g_cpu=gradient.detach().cpu().numpy().copy()
            gradient_seconds+=time.monotonic()-tick
            ce=objective-.5*problem.ridge*float(theta@theta)
            if not np.isfinite(objective) or objective>1e4 or not np.isfinite(g_cpu).all() or not np.isfinite(theta_cpu).all():
                guard_step=t;outcome='numerical_guard_stop'
                save_json(folder/'guard.json',{'step':t,'objective':objective,'ce':ce,'reason':'nonfinite or objective above1e4'})
                break
            tick=time.monotonic()
            x=monitor.observe(ce,g_cpu,theta_cpu,None if last_cpu is None else theta_cpu-last_cpu)
            x=np.r_[x,np.log(problem.targets.numel())]
            feature_seconds+=time.monotonic()-tick
            row={'step':t,'objective':objective,'ce':ce,'gradient_norm':float(np.linalg.norm(g_cpu)),
                 'parameter_norm':float(np.linalg.norm(theta_cpu))}
            rows.append(row);features.append(x)
            with (folder/'trace.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
            if t in schedule:evaluate_probe(t)
            theta=(theta-eta*gradient).detach();last_cpu=theta_cpu;completed+=1
            if len(chunk)>=50:
                save_states();save_json(folder/f'progress-{completed:06d}.json',{'completed_updates':completed})
        else:
            outcome='completed'
            if time.monotonic()<deadline:
                if steps in schedule:evaluate_probe(steps)
                with torch.no_grad():final_train_ce=float(problem.cross_entropy(theta))
                if not np.isfinite(final_train_ce) or final_train_ce>1e4 or not bool(torch.isfinite(theta).all()):
                    outcome='numerical_guard_stop';guard_step=steps
                    save_json(folder/'guard.json',{'step':steps,'ce':final_train_ce,'reason':'nonfinite or excessive terminal state/loss'})
            else:outcome='wall_budget_stop'
    except BaseException as error:
        outcome='failed_or_interrupted';save_json(folder/'failure.json',{'error':repr(error),'completed_updates':completed})
        raise
    finally:
        if chunk_start+len(chunk)<=completed:chunk.append(theta.detach().cpu().numpy().copy())
        save_states()
        ce=np.asarray([r['ce'] for r in rows]);X=np.asarray(features)
        np.savez(folder/'features.npz',X=X,ce=ce,progress=progress_targets(ce),burst=burst_labels(ce,guard_step))
        result={'seed':seed,'arm':arm,'role':role,'eta':eta,'rate_factor':1.6,
                'data_identity_sha256':data_identity,'initial_state_sha256':digest(initial.detach().cpu().numpy().tobytes()),
                'training_targets_per_update':problem.targets.numel(),'requested_updates':steps,
                'completed_updates':completed,'outcome':outcome,'guard_step':guard_step,
                'state_rows':chunk_start,'feature_names':None if monitor.names is None else monitor.names+['log_training_targets'],
                'feature_groups':None if monitor.groups is None else monitor.groups+['common'],
                'elapsed_seconds':time.monotonic()-started,'feature_seconds':feature_seconds,
                'gradient_seconds':gradient_seconds,'probe_seconds':probe_seconds,'probe':probe_rows,
                'initial_train_ce':None if not len(ce) else float(ce[0]),
                'last_preupdate_train_ce':None if not len(ce) else float(ce[-1]),'final_train_ce':final_train_ce,
                'state_convention':'theta_t saved before update t, plus unique terminal/latest accepted theta; no orbit claim.'}
        save_json(folder/'result.json',result)
    return result


def profile_gate(results,ritz_seconds,peak_bytes,profile_bytes):
    completed=len(results)==2 and all(r['outcome']=='completed' and r['completed_updates']==100 for r in results)
    projected_seconds=1.5*(16*sum(r['elapsed_seconds'] for r in results)+2*ritz_seconds+20)
    # State arrays are exact-size; 32MiB reserves token metadata/features/models/logs.
    projected_bytes=profile_bytes+4*801*15776*4+32*1024**2
    numeric_valid=all(np.isfinite(v) and v>=0 for v in [ritz_seconds,peak_bytes,profile_bytes,*[r['elapsed_seconds'] for r in results]])
    passed=completed and numeric_valid and projected_seconds<=420 and peak_bytes<=6*1024**3 and projected_bytes<=STORAGE_CAP
    return {'pass':bool(passed),'complete_representative_profile':completed,'projected_collection_seconds':projected_seconds,
            'projected_unit_bytes':projected_bytes,'peak_cuda_allocated_bytes':peak_bytes,
            'rules':{'max_projected_seconds':420,'max_peak_allocated_bytes':6*1024**3,'max_unit_bytes':STORAGE_CAP},
            'scope':'Feasibility only; no CE, burst, probe-quality or predictive-utility gate.'}


def verify_profile(folder,data_identity,device):
    m=json.loads((folder/'manifest.json').read_text());verify_inventory(folder,m)
    p=json.loads((folder/'plan.json').read_text());d=json.loads((folder/'data.json').read_text())
    f=json.loads((folder/'feasibility.json').read_text());summary=json.loads((folder/'summary.json').read_text())
    current=committed_identity()
    if (p['mode']!='profile' or p['device']!=device or not f['pass']
            or d['data_identity_sha256']!=data_identity
            or p['source']['source_identity']['sha256']!=current['sha256']):
        raise ValueError('profile source/device/data identity is not an eligible feasibility pass')
    if sorted((r['seed'],r['arm']) for r in summary['results'])!=[(PROFILE_SEED,8),(PROFILE_SEED,256)]:
        raise ValueError('unexpected profile cases')
    ritz=json.loads((folder/f'initial-ritz-{PROFILE_SEED}.json').read_text())
    recomputed=profile_gate(summary['results'],ritz['elapsed_seconds'],summary['peak_cuda_allocated_bytes'],
                            sum(v['bytes'] for v in m['files']))
    if not recomputed['pass']:raise ValueError('recomputed resource gate failed')
    return {'directory':str(folder),'manifest_sha256':digest((folder/'manifest.json').read_bytes()),
            'feasibility':f,'recomputed_feasibility':recomputed}


def run(args):
    seeds,steps=validate_request(args.mode,args.max_seconds)
    if args.mode=='collect' and args.profile is None:raise ValueError('reviewed feasibility profile required')
    identity=committed_identity();meta={**metadata(),'source_identity':identity,'role':args.mode}
    args.output.mkdir(parents=True,exist_ok=False);began=time.monotonic();deadline=began+args.max_seconds
    save_json(args.output/'plan.json',{'mode':args.mode,'seeds':seeds,'arms':[8,256],'steps':steps,
              'max_seconds':args.max_seconds,'source':meta,'device':args.device,'dtype':'float32','tf32':False,
              'scope':'One fixed nested text pool; initialization replication only; no assessment/policy/inference.'})
    try:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        arrays,data=load_fixed_data(args.corpus);save_json(args.output/'data.json',data)
        np.savez(args.output/'data-tokens.npz',**{name+'_'+xy:arr for name,pair in arrays.items() for xy,arr in zip(('X','Y'),pair)})
        if args.mode=='collect':save_json(args.output/'profile-reference.json',verify_profile(args.profile,data['data_identity_sha256'],args.device))
        model=TinyTransformer(len(data['vocabulary']));assert model.n_parameters==15776
        problems={name:LMProblem(model,torch.tensor(x,device=args.device),torch.tensor(y,device=args.device),.001) for name,(x,y) in arrays.items()}
        if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
        results=[];ritz_seconds=0.
        for seed in seeds:
            initial=model.initialize(seed,device=args.device)
            def bounded_hvp(v):
                if time.monotonic()>=deadline:raise TimeoutError('wall budget during expanded initial Ritz computation')
                return problems['expanded'].hvp(initial,v)
            tick=time.monotonic();ritz=lanczos_extremes(bounded_hvp,direction(initial,DIRECTION_SEED),32)
            ritz['elapsed_seconds']=time.monotonic()-tick;ritz_seconds+=ritz['elapsed_seconds']
            save_json(args.output/f'initial-ritz-{seed}.json',ritz);scale=validated_rate_scale(ritz);eta=1.6*2/scale
            if args.mode=='profile':save_json(args.output/'matched-precision.json',profile_precision(problems['expanded'],initial,eta))
            for arm,name in ((8,'control'),(256,'expanded')):
                result=collect_case(problems[name],problems['probe'],initial,eta,seed,arm,steps,
                     args.output/f'seed-{seed}-arm-{arm}',deadline,args.mode,data['data_identity_sha256'])
                results.append(result);print(json.dumps({k:result[k] for k in ('seed','arm','outcome','completed_updates','elapsed_seconds')}),flush=True)
                if result['outcome']=='wall_budget_stop':raise TimeoutError('shared wall budget exhausted')
        peak=torch.cuda.max_memory_allocated() if args.device=='cuda' else 0
        summary={'results':results,'elapsed_seconds':time.monotonic()-began,'peak_cuda_allocated_bytes':peak,
                 'parameters':model.n_parameters,'gpu_name':torch.cuda.get_device_name() if args.device=='cuda' else None,
                 'cuda_runtime':torch.version.cuda,'old_protected_outcomes_evaluated':False}
        save_json(args.output/'summary.json',summary)
        if args.mode=='profile':
            size=sum(f.stat().st_size for f in args.output.rglob('*') if f.is_file())
            save_json(args.output/'feasibility.json',profile_gate(results,ritz_seconds,peak,size))
        if sum(f.stat().st_size for f in args.output.rglob('*') if f.is_file())>STORAGE_CAP:
            raise RuntimeError('storage cap exceeded; do not continue')
    except BaseException as error:
        save_json(args.output/'failure.json',{'error':repr(error),'elapsed_seconds':time.monotonic()-began});raise
    finally:manifest(args.output,meta)


def main():
    signal.signal(signal.SIGTERM,handle_sigterm)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['profile','collect'])
    p.add_argument('--output',type=Path,required=True);p.add_argument('--max-seconds',type=float,required=True)
    p.add_argument('--corpus',type=Path,default=Path('/data2/neural-shrimp-data/datasets/tinyshakespeare/attempt-001'))
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--profile',type=Path)
    run(p.parse_args())


if __name__=='__main__':main()
