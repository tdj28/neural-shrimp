"""Bounded local operation timing on existing CIFAR subset A and an owned state.

Measures gradient, HVP, two-step JVP and one tiny GMRES cycle. It makes no
orbit/stability claim and does not load or inspect the sealed subset indices.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator, gmres

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from neural_shrimp import cifar
from explore_cifar import subset_a_problem, subset_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--eta', type=float, default=.00827753)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    record = {'status': 'running', 'started_utc': datetime.now(timezone.utc).isoformat(),
              'source_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'state': str(args.state), 'state_sha256': cifar.sha256_file(args.state),
              'subset_a_sha256': cifar.sha256_file(subset_paths()[0]),
              'eta': args.eta, 'lam': .01, 'host': platform.node(), 'pid': os.getpid(),
              'nice': os.getpriority(os.PRIO_PROCESS,0), 'torch': torch.__version__, 'numpy': np.__version__,
              'torch_threads': torch.get_num_threads(), 'load_average': os.getloadavg(),
              'claim_scope': 'operation timing only; no converged orbit or stability conclusion'}
    (args.output/'timing.json').write_text(json.dumps(record,indent=2)+'\n')
    try:
        problem, model = subset_a_problem(.01)
        theta = torch.as_tensor(np.load(args.state))
        record.update(parameters=model.n_parameters, samples=problem.n, dtype=str(theta.dtype))
        v = torch.as_tensor(np.random.default_rng(28205).normal(size=theta.numel()))
        v /= v.norm()
        t1 = problem.update(theta,args.eta)
        def measured(fn):
            elapsed=[]
            for _ in range(3):
                start=time.perf_counter(); value=fn(); elapsed.append(time.perf_counter()-start)
                if not torch.isfinite(value).all():
                    raise ValueError('nonfinite benchmark operation')
            return {'seconds':elapsed,'median_seconds':float(np.median(elapsed))}
        record['gradient']=measured(lambda:problem.grad(theta))
        record['hvp']=measured(lambda:problem.hvp(theta,v))
        record['two_step_jvp']=measured(lambda:problem.jvp_two_step(theta,v,args.eta,theta_next=t1))
        calls=0
        def matvec(vector):
            nonlocal calls
            calls+=1
            return problem.jvp_two_step(theta,cifar.as_tensor(vector),args.eta,theta_next=t1).numpy()-vector
        operator=LinearOperator((theta.numel(),theta.numel()),matvec=matvec,dtype=np.float64)
        residual=(problem.update(t1,args.eta)-theta).numpy()
        history=[]
        start=time.perf_counter()
        step,info=gmres(operator,-residual,rtol=1e-8,restart=5,maxiter=1,
                        callback=history.append,callback_type='pr_norm')
        record['gmres_one_cycle']={'seconds':time.perf_counter()-start,'matvec_calls':calls,'info':int(info),
                                   'restart':5,'maxiter':1,'relative_residual_history':history,
                                   'step_finite':bool(np.all(np.isfinite(step)))}
        record['status']='complete'
    except BaseException as error:
        record['status']='failed';record['error']=str(error);raise
    finally:
        record['max_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        (args.output/'timing.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
