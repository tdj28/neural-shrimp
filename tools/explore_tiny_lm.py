"""Bounded text-transformer pilot: local derivatives, never Floquet claims off an orbit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from neural_shrimp.tiny_lm import TinyTransformer, LMProblem, lanczos_extremes, ar_diagnostics

SOURCE = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
README = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/Readme.md'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def metadata():
    return {'source_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
            'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
            'platform': platform.platform(), 'pid': os.getpid(), 'evidence_rung': 0, 'label': 'exploratory',
            'threads': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}}


def manifest(directory, meta):
    files = [{'path': str(p.relative_to(directory)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
             for p in sorted(directory.rglob('*')) if p.is_file() and p.name != 'manifest.json']
    write(directory/'manifest.json', {'metadata': meta, 'files': files})


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=False)
    for url, filename in ((SOURCE, 'input.txt'), (README, 'source-readme.md')):
        with urllib.request.urlopen(url, timeout=30) as response:
            payload = response.read(2_000_000)
        (directory/filename).write_bytes(payload)
    receipt = {'source': SOURCE, 'license_source': README,
               'license_note': 'Source README states MIT. Text is a subset of Shakespeare; no separate dataset license file was found. Retained privately with attribution.',
               'corpus_sha256': sha(directory/'input.txt'), 'bytes': (directory/'input.txt').stat().st_size,
               'source_readme_sha256': sha(directory/'source-readme.md'), **metadata()}
    write(directory/'receipt.json', receipt)
    manifest(directory, receipt)
    print(json.dumps({'bytes':receipt['bytes'],'corpus_sha256':receipt['corpus_sha256']}), flush=True)


def batch(corpus, context=32, batch_size=8):
    source = json.loads((corpus/'receipt.json').read_text())
    if sha(corpus/'input.txt') != source['corpus_sha256']:
        raise ValueError('corpus hash mismatch')
    text = (corpus/'input.txt').read_text()
    split = int(.9*len(text))
    alphabet = sorted(set(text[:split]))
    mapping = {char: index for index,char in enumerate(alphabet)}
    offsets = [i*4096 for i in range(batch_size)]
    if offsets[-1]+context+1 >= split:
        raise ValueError('corpus too small')
    rows = np.array([[mapping[c] for c in text[start:start+context+1]] for start in offsets], dtype=np.int64)
    info = {'source':source, 'training_character_offsets':offsets, 'context':context, 'batch_size':batch_size,
            'tokens_per_update':batch_size*context, 'vocabulary':alphabet+['<UNK>'],
            'vocabulary_rule':'sorted training-prefix characters plus reserved unknown token',
            'target_rule':'X=chars[start:start+context]; Y=chars[start+1:start+context+1]; predict next character at every position',
            'validation':{'character_interval':[split,len(text)],'outcome_evaluated':False,
                          'note':'Disjoint final 10 percent; no validation losses computed in this pilot.'}}
    return rows[:,:-1], rows[:,1:], info


def direction(theta, seed=28142):
    v = torch.tensor(np.random.default_rng(seed).normal(size=theta.numel()), dtype=theta.dtype, device=theta.device)
    return v/v.norm()


def relative(a,b):
    return float((a-b).norm()/max(float(b.norm()),1e-15))


def validate(problem, theta, eta):
    started=time.monotonic()
    v=direction(theta)
    cpu=LMProblem(problem.model,problem.tokens.cpu(),problem.targets.cpu(),problem.ridge)
    p64,v64=theta.double().cpu(),v.double().cpu()
    g64,h64=cpu.grad(p64),cpu.hvp(p64,v64)
    g32,h32=problem.grad(theta),problem.hvp(theta,v)
    matched={'loss_absolute_difference':abs(float(problem.loss(theta))-float(cpu.loss(p64))),
             'gradient_relative_difference':relative(g32.double().cpu(),g64),
             'hvp_relative_difference':relative(h32.double().cpu(),h64),
             'update_relative_difference':relative(problem.update(theta,eta).double().cpu(),cpu.update(p64,eta)),
             'interpretation':'one matched state, not trajectory precision equivalence'}
    finite=[]
    actual=problem.update_jvp(theta,v,eta)
    for h in (.003,.001):
        fd=(problem.update(theta+h*v,eta)-problem.update(theta-h*v,eta))/(2*h)
        finite.append({'h':h,'relative_error':relative(fd,actual)})
    w=direction(theta,28143)
    hw=problem.hvp(theta,w)
    symmetry=float(abs(w@h32-v@hw)/max(float(h32.norm()+hw.norm()),1e-15))
    model=TinyTransformer(vocabulary=5,context=3,width=2,expansion=2)
    reduced=LMProblem(model,torch.tensor([[0,1,2],[2,3,4]]),torch.tensor([[1,2,3],[3,4,0]]),problem.ridge)
    p=model.initialize(dtype=torch.float64)
    z=direction(p)
    J=torch.func.jacrev(lambda x:reduced.update(x,.03))(p)
    J2=torch.func.jacrev(lambda x:reduced.update(reduced.update(x,.03),.03))(p)
    return {'matched_full_architecture':matched,'full_update_finite_difference':finite,
            'full_hessian_bilinear_symmetry_error':symmetry,
            'reduced_dense':{'parameters':len(p),'dtype':'float64',
                'one_step_jvp_relative_error':relative(reduced.update_jvp(p,z,.03),J@z),
                'two_step_jvp_relative_error':relative(reduced.two_step_jvp(p,z,.03),J2@z)},
            'elapsed_seconds':time.monotonic()-started}


def train(problem,theta,eta,steps,directory):
    initial=theta.clone()
    traces,states=[],[]
    started=time.monotonic()
    outcome='completed'
    grad_value=torch.func.grad_and_value(problem.loss)
    for step in range(steps):
        gradient,value=grad_value(theta)
        objective=float(value)
        ce=objective-.5*problem.ridge*float(theta@theta)
        if not np.isfinite(objective) or objective>1e4 or not bool(torch.isfinite(gradient).all()):
            outcome='numerical_guard_stop'
            break
        traces.append([objective,ce,float(gradient.norm()),float(theta.norm())])
        if step>=steps-128:
            states.append(theta.detach().cpu().numpy().copy())
        theta=(theta-eta*gradient).detach()
    if theta.is_cuda:
        torch.cuda.synchronize()
    elapsed=time.monotonic()-started
    trace,tail=np.asarray(traces),np.asarray(states)
    np.savez(directory/'trajectory.npz',trace=trace,tail_states=tail,initial=initial.cpu().numpy(),final=theta.cpu().numpy())
    result={'eta':eta,'requested_steps':steps,'completed_steps':len(traces),'outcome':outcome,
            'elapsed_seconds':elapsed,'seconds_per_update':elapsed/max(1,len(traces)),
            'trace_columns':['objective','cross_entropy','gradient_norm','parameter_norm'],
            'state_convention':'trace[t] and tail are pre-update theta_t; final is post-last-update',
            'initial_cross_entropy':None if not len(trace) else float(trace[0,1]),
            'final_recorded_cross_entropy':None if not len(trace) else float(trace[-1,1]),
            'loss_ar':ar_diagnostics(trace[-min(512,len(trace)):,0]) if len(trace) else {'outcome':'empty'}}
    if len(tail)>=32:
        norms=np.linalg.norm(tail,axis=1)
        recurrence={str(lag):float(np.max(np.linalg.norm(tail[lag:]-tail[:-lag],axis=1)/np.maximum(1,norms[:-lag]))) for lag in (1,2,4)}
        result['tail_max_relative_recurrence']=recurrence
        result['finite_horizon_label']=('fixed_point_candidate' if recurrence['1']<=1e-5 else
            'period_2_candidate' if recurrence['2']<=1e-5 and recurrence['1']>1e-4 else 'nonconverged_finite_horizon')
    if outcome=='completed':
        spectral_started=time.monotonic()
        result['final_hessian_ritz']=lanczos_extremes(lambda v:problem.hvp(theta,v),direction(theta),steps=24)
        result['hessian_diagnostic_seconds']=time.monotonic()-spectral_started
        theta1=problem.update(theta,eta)
        result['final_one_step_relative_displacement']=relative(theta1,theta)
        result['final_two_step_relative_displacement']=relative(problem.update(theta1,eta),theta)
    write(directory/'result.json',result)
    return result


def run(args):
    args.output.mkdir(parents=True,exist_ok=False)
    meta=metadata()
    write(args.output/'metadata.json',meta)
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        X,Y,data=batch(args.corpus)
        model=TinyTransformer(vocabulary=len(data['vocabulary']))
        theta=model.initialize(device=args.device)
        problem=LMProblem(model,torch.tensor(X,device=args.device),torch.tensor(Y,device=args.device),args.ridge)
        write(args.output/'data.json',data)
        np.savez(args.output/'batch.npz',tokens=X,targets=Y)
        if theta.is_cuda:
            torch.cuda.reset_peak_memory_stats()
        started=time.monotonic()
        hessian=lanczos_extremes(lambda v:problem.hvp(theta,v),direction(theta),steps=32)
        seconds=time.monotonic()-started
        sharpness=hessian['maximum']['value']
        if sharpness<=0 or hessian['maximum']['relative_residual']>.02:
            raise RuntimeError('initial positive sharpness Ritz estimate unresolved')
        factors=args.factors or ([.2] if args.profile=='unit' else [.1,.4,.8,1.,1.2,1.6])
        steps=args.steps or (100 if args.profile=='unit' else 1000)
        plan={'model':model.__dict__,'parameters':model.n_parameters,'optimizer':'full-batch GD',
              'optimizer_state':'theta only; no momentum, schedule or bias correction state',
              'dtype':'float32','device':args.device,'ridge':args.ridge,'seed':28141,
              'cuda_runtime':torch.version.cuda,'gpu_name':torch.cuda.get_device_name() if theta.is_cuda else None,
              'gpu_capability':list(torch.cuda.get_device_capability()) if theta.is_cuda else None,
              'deterministic_algorithms':True,'tf32':False,'dropout':False,
              'initial_hessian_ritz':hessian,'initial_hessian_seconds':seconds,
              'rate_rule':'eta=factor*2/initial_largest_Hessian_Ritz_value; initial point is not stationary',
              'factors':factors,'steps':steps,'profile':args.profile,
              'scope':'local derivatives; no Floquet, chaos or invariant-circle claim'}
        write(args.output/'plan.json',plan)
        write(args.output/'validation.json',validate(problem,theta,.2*2/sharpness))
        print(json.dumps({'parameters':model.n_parameters,'initial_hessian':hessian,'hessian_seconds':seconds}),flush=True)
        results=[]
        for index,factor in enumerate(factors):
            directory=args.output/f'case-{index:02d}'
            directory.mkdir()
            result=train(problem,theta.clone(),factor*2/sharpness,steps,directory)
            result['factor']=factor
            results.append(result)
            print(json.dumps({'case':index,'factor':factor,'eta':result['eta'],'outcome':result['outcome'],
                              'initial_ce':result['initial_cross_entropy'],'final_ce':result['final_recorded_cross_entropy'],
                              'label':result.get('finite_horizon_label'),'seconds':result['elapsed_seconds']}),flush=True)
        write(args.output/'results.json',{'plan':plan,'results':results,
              'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if theta.is_cuda else None})
        manifest(args.output,meta)
    except Exception as exc:
        write(args.output/'failure.json',{'error':repr(exc)})
        manifest(args.output,meta)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','run'])
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--corpus',type=Path,default=Path('/data2/neural-shrimp-data/datasets/tinyshakespeare/attempt-001'))
    parser.add_argument('--profile',choices=['unit','pilot'],default='unit')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--ridge',type=float,default=.001)
    parser.add_argument('--steps',type=int)
    parser.add_argument('--factors',type=float,nargs='+')
    args=parser.parse_args()
    prepare(args.output) if args.command=='prepare' else run(args)


if __name__=='__main__':
    main()
