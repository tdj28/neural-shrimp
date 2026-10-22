"""Bounded exploratory GPU qualification; finite-time diagnostics, never a chaos claim."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import numpy as np


def growth_log(distance, radius):
    if not math.isfinite(distance) or not math.isfinite(radius) or distance <= 0 or radius <= 0:
        raise ValueError('Perturbation collapsed or is nonfinite; precision gate failed')
    return math.log(distance / radius)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seconds', type=int, default=1500)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    def emit(kind, **data):
        row = dict(kind=kind, elapsed=time.monotonic()-start, **data)
        with (a.out/'events.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        print(json.dumps(row), flush=True)
    def budget():
        if time.monotonic()-start > a.seconds:
            raise TimeoutError('Pilot internal deadline reached')
    emit('start', pid=os.getpid(), source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    import torch
    from explore_latent_loop import HuginnLoop, REVISIONS
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(.80)
    emit('runtime', torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), revision=REVISIONS['huginn'])
    loop = HuginnLoop('huginn','float32','cuda',4,seed=0)
    emit('loaded', gpu_allocated=torch.cuda.memory_allocated(), gpu_peak=torch.cuda.max_memory_allocated())
    prompt='What is 17 times 23? The answer is'
    h0=loop.setup(prompt)
    emit('prompt', text=prompt, ids=loop.token_ids, shape=list(h0.shape))
    with torch.no_grad():
        native,_=loop.model.core_block_forward(h0,loop.e,loop.freqs,None,None,torch.tensor(0),0)
        ours=loop.F(h0)
        error=float((native-ours).norm()/native.norm())
        repeat=float((ours-loop.F(h0)).abs().max())
    emit('native_gate',relative_error=error,repeat_max_error=repeat)
    if error>1e-6 or repeat!=0:
        raise RuntimeError('Native map equivalence/determinism failed')
    # Two independently evolved nearby states, rescaled each step; same batch shape.
    timings=[]
    for cells in (1,4,16):
        budget()
        h=h0.repeat(2*cells,1,1)
        for _ in range(3): h=loop.F(h)
        torch.cuda.synchronize(); tic=time.monotonic()
        for _ in range(16): h=loop.F(h)
        torch.cuda.synchronize(); seconds=time.monotonic()-tic
        row=dict(cells=cells,state_rows=2*cells,steps=16,seconds=seconds,seconds_per_cell_step=seconds/(16*cells),peak_bytes=torch.cuda.max_memory_allocated())
        timings.append(row); emit('timing',**row)
        del h
    # Test a real two-control execution path using a disclosed random direction.
    direction=np.random.default_rng(0).normal(size=h0.shape[-1]); direction/=np.linalg.norm(direction)
    loop.set_steering(0, np.stack([0*direction, .01*direction]))
    h=h0.repeat(2,1,1); lam=torch.tensor([1.,1.10],device='cuda')[:,None,None]
    for _ in range(8): h=(1-lam)*h+lam*loop.F(h)
    if not torch.isfinite(h).all(): raise RuntimeError('Nonfinite two-control path')
    emit('plane_smoke',lambdas=[1.,1.10],steering=[0.,.01],steps=8)
    loop.set_steering(0,None)
    records=[]
    for lam in (1.,1.10):
        budget(); h=h0.clone()
        for _ in range(128):
            budget(); h=(1-lam)*h+lam*loop.F(h)
        for eps in (1e-3,3e-4):
            budget(); x=h.clone(); d=torch.randn_like(x); radius=float(x.norm())*eps
            y=x+d/d.norm()*radius
            logs=[]; residuals=[]; tic=time.monotonic()
            for k in range(256):
                budget(); pair=torch.cat([x,y]); z=(1-lam)*pair+lam*loop.F(pair)
                xn,yn=z[:1],z[1:]; delta=yn-xn; distance=float(delta.norm())
                logs.append(growth_log(distance,radius))
                residuals.append(float((xn-x).norm()/x.norm()))
                x=xn; y=x+delta*(radius/distance)
                if (k+1)%64==0: emit('diagnostic_progress',lam=lam,eps=eps,steps=k+1,mean_log_growth=float(np.mean(logs)))
            row=dict(lam=lam,eps=eps,steps=len(logs),seconds=time.monotonic()-tic,mean_log_growth=float(np.mean(logs)),half_means=[float(np.mean(v)) for v in np.array_split(logs,2)],last_residual=residuals[-1])
            records.append(row)
            np.savez_compressed(a.out/f'trace-{lam}-{eps}.npz',logs=logs,residuals=residuals,final_state=x.cpu().numpy())
            emit('diagnostic',**row)
    emit('complete',timings=timings,diagnostics=records,qualification='float32 short-prompt engineering pilot; no chaos classification; no float64 or derivative qualification')
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.out.iterdir() if p.is_file()}
    (a.out/'manifest.json').write_text(json.dumps(files,indent=2)+'\n')

if __name__=='__main__': main()
