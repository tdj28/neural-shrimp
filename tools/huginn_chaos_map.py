"""Exploratory native-hidden-loop parameter map; no soft-token readout feedback.

Float32 screening only. Positive finite-window growth and recurrent returns are
candidates, not proofs of chaos or primitive stable cycles. Each chunk is saved.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np


def axes(plan):
    result=[]
    for key, anchor in [('lambda_range',1.),('steering_range',0.)]:
        x=np.linspace(*plan[key],plan['resolution'])
        if x[0] <= anchor <= x[-1]:
            x[np.argmin(abs(x-anchor))]=anchor
        result.append(np.sort(x))
    return result


def recurrence_candidates(tail, tolerance, max_period=64, repeats=8):
    """Require closure at every one of several consecutive full-state positions."""
    out=np.zeros(tail.shape[1],dtype=np.int32)
    norms=np.linalg.norm(tail[-repeats:].reshape(repeats,tail.shape[1],-1),axis=2)
    for q in range(1,min(max_period,len(tail)-repeats)+1):
        delta=tail[-repeats:]-tail[-repeats-q:-q]
        errors=np.linalg.norm(delta.reshape(repeats,tail.shape[1],-1),axis=2)/np.maximum(norms,1e-30)
        out[(out==0)&(errors.max(axis=0)<tolerance)]=q
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--source-commit',required=True)
    p.add_argument('--mode',choices=['scan'],required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args(); plan=json.loads(args.plan.read_text())
    from explore_latent_loop import HuginnLoop, DATA, REVISIONS
    import torch,transformers
    root=DATA/'explore'/args.out;root.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); deadline=started+plan['max_compute_seconds']
    def emit(phase,**kw):
        row=dict(chunk_start=kw.pop('chunk_start',-1),phase=phase,**kw)
        with (root/'progress.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row),flush=True)
    def gate_time():
        if time.monotonic()>deadline:raise TimeoutError('Numerical budget exhausted; partial chunks preserved')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(.8)
    loop=HuginnLoop('huginn','float32','cuda',4,seed=0)
    h0=loop.setup(plan['prompt']);lam_axis,alpha_axis=axes(plan)
    cells=np.array([(l,a) for l in lam_axis for a in alpha_axis]);B=plan['batch']
    v=np.random.default_rng(plan['direction_seed']).normal(size=h0.shape[-1]);v/=np.linalg.norm(v);v*=math.sqrt(len(v))
    np.save(root/'steering-direction.npy',v)
    runtime=dict(source=args.source_commit,plan_sha256=hashlib.sha256(args.plan.read_bytes()).hexdigest(),model_revision=REVISIONS['huginn'],torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),prompt_ids=loop.token_ids,dtype='float32',tf32=False)
    (root/'runtime.json').write_text(json.dumps(runtime,indent=2)+'\n');(root/'plan.json').write_text(args.plan.read_text())
    with torch.no_grad():
        reference,_=loop.model.core_block_forward(h0,loop.e,loop.freqs,None,None,torch.tensor(0),0)
        ours=loop.F(h0);error=float((reference-ours).norm()/reference.norm());repeat=float((ours-loop.F(h0)).abs().max())
    emit('native_gate',error=error,repeat=repeat)
    if error>1e-6 or repeat!=0:raise RuntimeError('Native gate failed')
    # Exact intervention and batch shape for the remote throughput gate.
    def set_map(c):
        l=torch.tensor(c[:,0],dtype=loop.dtype,device='cuda')[:,None,None];l=torch.cat([l,l])
        s=c[:,1,None]*v[None,:];loop.set_steering(0,np.concatenate([s,s]))
        def F(x):return (1-l)*x+l*loop.F(x)
        return F
    F=set_map(cells[:B]);x=h0.repeat(2*B,1,1)
    for _ in range(4):x=F(x)
    torch.cuda.synchronize();tic=time.monotonic()
    for _ in range(32):x=F(x)
    torch.cuda.synchronize();unit=(time.monotonic()-tic)/(32*B)
    projected=unit*len(cells)*(plan['transient']+len(plan['eps'])*plan['measure'])*1.3
    emit('throughput_gate',seconds_per_cell_step=unit,projected_seconds_with_margin=projected,remaining_seconds=deadline-time.monotonic())
    if projected>deadline-time.monotonic()-600:raise RuntimeError('Throughput cannot fit ceiling; no full scan launched')
    del x
    results=[]
    for start in range(0,len(cells),B):
        gate_time();c=cells[start:start+B];n=len(c);F=set_map(c)
        x=h0.repeat(2*n,1,1)
        for k in range(plan['transient']):
            gate_time();x=F(x)
            if (k+1)%128==0:emit('transient',chunk_start=start,step=k+1)
        anchor=x[:n].clone(); del x
        logs_all=[];half_all=[];period_all=[];norm_all=[];obs_all=[];last=None
        for offset,eps in enumerate(plan['eps']):
            h=anchor.clone();torch.manual_seed(10000+start)
            d=torch.randn_like(h);radius=eps*h.flatten(1).norm(dim=1)
            g=h+d*(radius/d.flatten(1).norm(dim=1))[:,None,None]
            logs=[];recent=[];obs=[];norms=[]
            for k in range(plan['measure']):
                gate_time();x=F(torch.cat([h,g]));hn,gn=x[:n],x[n:];delta=gn-hn;dist=delta.flatten(1).norm(dim=1)
                if not torch.isfinite(x).all() or not torch.isfinite(dist).all() or (dist<=0).any():
                    raise RuntimeError(f'Numerical gate failed in chunk {start}, step {k}; no substitute values')
                logs.append(torch.log(dist/radius).cpu().numpy());h=hn;g=h+delta*(radius/dist)[:,None,None]
                norms.append(h.flatten(1).norm(dim=1).cpu().numpy())
                if k>=plan['measure']-128:obs.append(h[:,-1,:3].cpu().numpy())
                if k>=plan['measure']-72:recent.append(h.cpu().numpy())
                if (k+1)%128==0:emit('measure',chunk_start=start,offset=offset,step=k+1)
            log=np.asarray(logs);tail=np.asarray(recent)
            if plan.get('retain_full_tails',False):
                np.savez_compressed(root/f'tail-{start:04d}-{offset}.npz',states=tail,cells=c,eps=eps)
            logs_all.append(log.mean(axis=0));half_all.append(np.stack([z.mean(axis=0) for z in np.array_split(log,2)]))
            period_all.append(recurrence_candidates(tail,plan['recurrence_tolerance']))
            norm_all.append(np.stack([np.min(norms,axis=0),np.max(norms,axis=0)]));obs_all.append(np.asarray(obs));last=tail[-1]
            del recent,tail,x,h,g,delta,hn,gn
        row=dict(cells=c,lyapunov=np.array(logs_all),half_means=np.array(half_all),recurrence_candidate=np.array(period_all),norm_range=np.array(norm_all),observations=np.array(obs_all),final_state=last)
        np.savez_compressed(root/f'chunk-{start:04d}.npz',**row)
        results.append({k:val for k,val in row.items() if k not in ('final_state','observations')})
        emit('chunk_complete',chunk_start=start,completed=start+n,total=len(cells),elapsed=time.monotonic()-started)
    merged={k:np.concatenate([r[k] for r in results],axis=0 if k=='cells' else -1) for k in results[0]}
    np.savez_compressed(root/'map.npz',**merged,lambda_axis=lam_axis,steering_axis=alpha_axis)
    summary=dict(cells=len(cells),seconds=time.monotonic()-started,qualification='exploratory float32 screening; candidate recurrence only; no certified chaos or shrimp',runtime=runtime)
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    emit('complete',cells=len(cells))
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.iterdir() if p.is_file()}
    (root/'manifest.json').write_text(json.dumps(files,indent=2)+'\n')

if __name__=='__main__':main()
