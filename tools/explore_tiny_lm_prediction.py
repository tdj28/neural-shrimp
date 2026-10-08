"""Profile or collect only the six development-calibration prediction trajectories."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from neural_shrimp.tiny_lm import TinyTransformer, LMProblem, lanczos_extremes
from neural_shrimp.tiny_lm_prediction import CausalFeatures, burst_labels
from explore_tiny_lm import batch, direction, metadata, manifest, relative, write

CALIBRATION_SEEDS = (28151, 28152)
FACTORS = (.8, 1.6, 2.5)


def safe(value):
    if isinstance(value, dict):
        return {k:safe(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    return value


def save_json(path, data):
    write(path, safe(data))


def source_identity():
    paths = ['src/neural_shrimp/tiny_lm.py', 'src/neural_shrimp/tiny_lm_prediction.py',
             'tools/explore_tiny_lm.py', 'tools/explore_tiny_lm_prediction.py']
    hashes = {}
    for name in paths:
        actual = (ROOT/name).read_bytes()
        committed = subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)
        if actual != committed:
            raise ValueError('source must match committed HEAD: '+name)
        hashes[name] = hashlib.sha256(actual).hexdigest()
    return hashes


def validate_request(mode, max_seconds):
    if mode not in ('profile', 'calibration'):
        raise ValueError('Only profile or calibration collection is permitted')
    if not np.isfinite(max_seconds) or not 0 < max_seconds <= 285:
        raise ValueError('max_seconds must lie in (0,285]')
    return ((28151,), (.8,), 100, min(max_seconds,60)) if mode == 'profile' else (CALIBRATION_SEEDS,FACTORS,800,max_seconds)


def profile_precision(problem, theta, eta):
    """One matched state only; no claim of trajectory precision equivalence."""
    cpu = LMProblem(problem.model,problem.tokens.cpu(),problem.targets.cpu(),problem.ridge)
    p64 = theta.double().cpu(); v = direction(theta,28155)
    g32 = problem.grad(theta); h32 = problem.hvp(theta,v)
    g64 = cpu.grad(p64); h64 = cpu.hvp(p64,v.double().cpu())
    return {'gradient_relative_error':relative(g32.double().cpu(),g64),
            'hvp_relative_error':relative(h32.double().cpu(),h64),
            'update_relative_error':relative(problem.update(theta,eta).double().cpu(),cpu.update(p64,eta)),
            'direction_seed':28155, 'scope':'Matched rounded initial state; no trajectory equivalence.'}


def collect_case(problem, initial, eta, seed, factor, steps, folder, deadline, role):
    folder.mkdir(exist_ok=False)
    monitor = CausalFeatures(); theta = initial.clone()
    rows, features, chunk = [], [], []
    chunk_start = 0; completed = 0; last_cpu = None; outcome = 'running'; guard_step = None
    started = time.monotonic(); feature_seconds = 0.; gradient_seconds = 0.
    grad_value = torch.func.grad_and_value(problem.loss)
    def save_states():
        nonlocal chunk_start, chunk
        if not chunk:
            return
        path = folder/f'states-{chunk_start:06d}-{chunk_start+len(chunk)-1:06d}.npy'
        with path.open('xb') as handle:
            np.save(handle,np.asarray(chunk)); handle.flush(); os.fsync(handle.fileno())
        chunk_start += len(chunk); chunk = []
    try:
        for t in range(steps):
            if time.monotonic() >= deadline:
                outcome = 'wall_budget_stop'; break
            theta_cpu = theta.detach().cpu().numpy().copy()
            chunk.append(theta_cpu)
            tick = time.monotonic()
            gradient, objective = grad_value(theta)
            objective = float(objective); g_cpu = gradient.detach().cpu().numpy().copy()
            gradient_seconds += time.monotonic()-tick
            ce = objective-.5*problem.ridge*float(theta@theta)
            if (not np.isfinite(objective) or objective > 1e4 or not np.isfinite(g_cpu).all()
                    or not np.isfinite(theta_cpu).all()):
                outcome = 'numerical_guard_stop'; guard_step = t
                save_json(folder/'guard.json',{'step':t,'objective':objective,'ce':ce,'reason':'nonfinite or objective above 1e4'})
                break
            tick = time.monotonic()
            x = monitor.observe(ce,g_cpu,theta_cpu,None if last_cpu is None else theta_cpu-last_cpu)
            feature_seconds += time.monotonic()-tick
            row = {'step':t,'objective':objective,'ce':ce,'gradient_norm':float(np.linalg.norm(g_cpu)),
                   'parameter_norm':float(np.linalg.norm(theta_cpu))}
            rows.append(row); features.append(x)
            with (folder/'trace.jsonl').open('a') as handle:
                handle.write(json.dumps(row)+'\n'); handle.flush()
            theta = (theta-eta*gradient).detach(); last_cpu = theta_cpu; completed += 1
            if len(chunk) >= 50:
                save_states()
                save_json(folder/f'progress-{t+1:06d}.json',{'completed_updates':t+1,'saved_preupdate_states':chunk_start})
        else:
            outcome = 'completed'
        # The final state is post-last-update for a completed/budget stop and already
        # present as the guard state for a numerical stop.
        if outcome != 'numerical_guard_stop':
            chunk.append(theta.detach().cpu().numpy().copy())
    except BaseException as error:
        outcome = 'failed_or_interrupted'
        save_json(folder/'failure.json',{'error':repr(error),'completed_updates':completed})
        raise
    finally:
        save_states()
        loss = np.asarray([row['ce'] for row in rows])
        X = np.asarray(features)
        labels = burst_labels(loss,guard_step)
        np.savez(folder/'features.npz',X=X,ce=loss,burst=labels)
        result = {'role':role,'seed':seed,'factor':factor,'eta':eta,'requested_updates':steps,
                  'completed_updates':completed,'outcome':outcome,'guard_step':guard_step,
                  'elapsed_seconds':time.monotonic()-started,'feature_seconds':feature_seconds,
                  'gradient_seconds':gradient_seconds,'feature_names':monitor.names,'feature_groups':monitor.groups,
                  'initial_ce':None if not len(loss) else float(loss[0]),
                  'last_preupdate_ce':None if not len(loss) else float(loss[-1]),
                  'finite_feature_fraction':None if X.size == 0 else float(np.isfinite(X).mean()),
                  'state_rows':chunk_start,'state_convention':'Saved theta_t before update t; terminal theta included; no full-state recurrence claim.'}
        save_json(folder/'result.json',result)
    return result


def run(args):
    seeds,factors,steps,seconds = validate_request(args.mode,args.max_seconds)
    args.output.mkdir(parents=True,exist_ok=False)
    began = time.monotonic(); deadline = began+seconds
    meta = {**metadata(),'source_sha256':source_identity(),'role':args.mode}
    plan = {'mode':args.mode,'seeds':seeds,'factors':factors,'steps':steps,'internal_wall_seconds':seconds,
            'dtype':'float32','tf32':False,'deterministic_algorithms':True,'device':args.device,
            'state':'theta only; fixed-rate GD; tied embedding once; no scheduler or policy',
            'scope':'Calibration prediction features only; no assessment or control policies.',
            'sealed':'No final-10%-text validation, assessment seeds, or reserved seeds evaluated.',
            'source':meta}
    save_json(args.output/'plan.json',plan)
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        x,y,data = batch(args.corpus)
        assert data['source']['corpus_sha256'] == '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
        assert data['training_character_offsets'] == [4096*j for j in range(8)]
        save_json(args.output/'data.json',data)
        model = TinyTransformer(len(data['vocabulary']))
        assert model.n_parameters == 15776
        problem = LMProblem(model,torch.tensor(x,device=args.device),torch.tensor(y,device=args.device),.001)
        if args.device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        summaries=[]
        for seed in seeds:
            initial=model.initialize(seed,device=args.device)
            if time.monotonic() >= deadline:
                raise TimeoutError('wall budget before initial Ritz computation')
            start = time.monotonic()
            def bounded_hvp(v):
                if time.monotonic() >= deadline:
                    raise TimeoutError('wall budget during initial Ritz computation')
                return problem.hvp(initial,v)
            hessian=lanczos_extremes(bounded_hvp,direction(initial,28155),steps=32)
            hessian['elapsed_seconds']=time.monotonic()-start
            save_json(args.output/f'initial-ritz-{seed}.json',hessian)
            scale=hessian['maximum']['value']
            if scale <= 0 or hessian['maximum']['relative_residual'] > .02:
                raise RuntimeError('initial rate-scale Ritz estimate unresolved; no rate substitution')
            if args.mode == 'profile':
                save_json(args.output/'matched-precision.json',profile_precision(problem,initial,factors[0]*2/scale))
            for factor in factors:
                result=collect_case(problem,initial,factor*2/scale,seed,factor,steps,
                                    args.output/f'seed-{seed}-factor-{factor:g}',deadline,args.mode)
                summaries.append(result)
                print(json.dumps({k:result[k] for k in ('seed','factor','outcome','completed_updates','elapsed_seconds','feature_seconds')}),flush=True)
                if result['outcome'] == 'wall_budget_stop':
                    raise TimeoutError('shared collection wall budget exhausted')
        summary={'results':summaries,'elapsed_seconds':time.monotonic()-began,
                 'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
                 'parameters':model.n_parameters,'gpu_name':torch.cuda.get_device_name() if args.device == 'cuda' else None,
                 'cuda_runtime':torch.version.cuda,'no_validation_or_assessment_evaluated':True}
        save_json(args.output/'summary.json',summary)
    except BaseException as error:
        save_json(args.output/'failure.json',{'error':repr(error),'elapsed_seconds':time.monotonic()-began})
        raise
    finally:
        manifest(args.output,meta)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['profile','calibration'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--corpus',type=Path,default=Path('/data2/neural-shrimp-data/datasets/tinyshakespeare/attempt-001'))
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    parser.add_argument('--max-seconds',type=float,default=285.)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
