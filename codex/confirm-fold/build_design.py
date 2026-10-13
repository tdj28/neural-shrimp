"""Build a result-free, hash-bound Task 5 plan after fold setup, before passages."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import sys

import numpy as np

EPSILONS = (3e-6, 1e-5, 3e-5, 1e-4)
ENTRANCES = ('full', 'midpoint', 'full+transverse')
ROOT = Path('/data2/neural-shrimp-data/codex/task5-fresh-prediction')
REPO = Path(__file__).resolve().parents[2]

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()

def artifact(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)

def predict(u, a, b, eps, U, cap=100000):
    if not all(math.isfinite(float(z)) for z in (u,a,b,eps,U)) or min(a,b,eps,U)<=0 or u>U:
        raise ValueError('finite positive quadratic coefficients, distance and channel required')
    n = None
    for i in range(1, cap+1):
        u_new = u + a*eps + b*u*u
        if u_new>U:
            n=i
            break
        u=u_new
    return n

def predictions(u,a,b,eps,U):
    n=predict(u,a,b,eps,U)
    scale=math.sqrt(b/(a*eps))
    return dict(quadratic_G=n, arctan_G=(math.atan(U*scale)-math.atan(u*scale))/math.sqrt(a*b*eps))

def entrances(x,r,l,U,seed):
    x,r,l=(np.asarray(v,dtype=np.float64) for v in (x,r,l))
    if x.ndim!=1 or r.shape!=x.shape or l.shape!=x.shape or not all(np.isfinite(v).all() for v in (x,r,l)):
        raise ValueError('finite, matching full-state modes required')
    if abs(np.linalg.norm(r)-1)>1e-8 or abs(l@r-1)>1e-8 or not U>0:
        raise ValueError('normalized modes and positive U required')
    z=np.random.default_rng(seed).normal(size=x.size)
    z-=r*(l@z)
    norm=np.linalg.norm(z)
    if norm<1e-14:
        raise ValueError('unresolved transverse seed')
    w=z/norm
    return w,np.stack([x-U*r,x,x-U*r+U*w])

def build_target(label, setup_path, inputs_dir):
    source=Path(setup_path).resolve()
    info=json.loads(source.read_text())
    if info['target']['label']!=label:
        raise ValueError('target identity mismatch')
    common=dict(label=label,setup=artifact(source),period=2 if label=='A' else 8,
                coordinate='alpha' if label=='A' else 'log10T')
    if not info.get('passed',False):
        return dict(**common,status='setup_gate_failed',reason=info.get('gate_failure',info.get('stage',info.get('gates'))),
                    rows=[dict(id=f'{label}-{i}-{name}',target=label,eps=eps,entrance=name,status='not_run_setup_gate') for i,eps in enumerate(EPSILONS) for name in ENTRANCES])
    array=source.with_name('setup.npz')
    if sha(array)!=info['array_file_sha256']:
        raise ValueError('setup array hash mismatch')
    data=np.load(array,allow_pickle=False)
    x,r,l=(data[k] for k in ['x_c','r','l'])
    a,b=float(info['a_positive']),float(info['b_positive'])
    U=.1/b
    seed=10501 if label=='A' else 10502
    w,starts=entrances(x,r,l,U,seed)
    path=inputs_dir/f'{label}-inputs.npz'
    np.savez(path,x_c=x,r=r,l=l,w=w,starts=starts,matrix=data['matrix'])
    path.chmod(0o444)
    rows=[]
    for i,eps in enumerate(EPSILONS):
        for j,name in enumerate(ENTRANCES):
            u=0. if name=='midpoint' else -U
            pred=predictions(u,a,b,eps,U)
            if pred['quadratic_G'] is None:
                raise ValueError('scalar prediction did not exit')
            rows.append(dict(id=f'{label}-{i}-{name}',target=label,group=i,entrance_index=j,eps=eps,entrance=name,
                             u_in=u,u_out=U,parameter=float(info['parameter_c'])+int(info['ghost_parameter_direction'])*eps,**pred))
    max_g=max(512,4*max(row['quadratic_G'] for row in rows))
    if max_g>4096:
        raise ValueError('pre-outcome scalar cap exceeds resource plan; explicit amendment needed')
    P=np.eye(x.size)-np.outer(r,l)
    M=P@data['matrix']@P
    powers={}
    K=np.eye(x.size)
    for k in range(1,9):
        K=K@M
        if k in (1,2,4,8):
            powers[str(k)]=float(np.linalg.norm(K,'fro'))
    return dict(**common,status='ready',setup_arrays=artifact(array),inputs=artifact(path),seed=seed,
                parameter_c=float(info['parameter_c']),alpha=float(info['alpha']),log10T=float(info['log10T']),
                ghost_direction=int(info['ghost_parameter_direction']),raw_a=float(info['raw_a']),raw_b=float(info['raw_b']),
                a_positive=a,b_positive=b,U=U,max_g=max_g,rows=rows,
                geometry=dict(left_norm=float(np.linalg.norm(l)),projector_frobenius=float(np.linalg.norm(P,'fro')),
                              complementary_power_frobenius=powers,transverse_projection=float(l@w),
                              transverse_kick_norm=float(np.linalg.norm(U*w)),initial_u=((starts-x)@l).tolist()))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--setup-a',required=True)
    p.add_argument('--setup-b',required=True)
    p.add_argument('--inputs-name',default='frozen-inputs-v1')
    p.add_argument('--output',default=str(REPO/'codex/confirm-fold/design.json'))
    args=p.parse_args()
    output=Path(args.output)
    if output.exists():
        raise FileExistsError('never overwrite a frozen design')
    if '/' in args.inputs_name or args.inputs_name.startswith('.'):
        raise ValueError('simple exclusive input directory name required')
    inputs=ROOT/args.inputs_name
    inputs.mkdir(parents=True,exist_ok=False)
    targets=[build_target('A',args.setup_a,inputs),build_target('B',args.setup_b,inputs)]
    paths=list((REPO/'codex/confirm-fold').glob('*.py'))+[REPO/'codex/crosscheck'/n for n in ('qwen_map.py','numerics.py')]
    plan=dict(schema=1,created_utc=datetime.now(timezone.utc).isoformat(),model=dict(name='Qwen/Qwen2.5-0.5B-Instruct',snapshot='7ae557604adf67be50417f59c2c2f167def9a775',map_source='codex/crosscheck/qwen_map.py',assets='Pinned SHA256 and exact prefix IDs embedded in bound map source; model load verifies asset hashes.'),scope='Prospective finite-range fold passage forecast, one pinned checkpoint/prompt/feedback map; no discrete-token or behavioral claim.',
              criterion=dict(primary='all rows absolute measured_G minus quadratic_G <= 1',secondary='all rows absolute measured_G minus unrounded arctan_G <= 1',
                             missing='failed/not-run/censored rows never pass; target-gate failures retained',endpoint='first strict u>U after complete G step; no burn-in'),
              execution=dict(target_order=['A','B'],epsilon_order=list(EPSILONS),entrances=list(ENTRANCES),threads=4,device='cpu',dtype='float64',
                             timeout_seconds=21600,memory_gib=18,max_workers=2,post_outcome_changes='dated amendment; affected rows exploratory'),
              software={n:importlib.metadata.version(n) for n in ('torch','numpy','scipy','safetensors')},
              python=sys.version,source=[dict(path=str(f.relative_to(REPO)),sha256=sha(f)) for f in sorted(paths)],targets=targets)
    output.write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(plan=artifact(output),targets=[dict(label=t['label'],status=t['status'],rows=len(t['rows'])) for t in targets]),indent=2))

if __name__=='__main__':
    main()
