"""HPC2 bounded exploratory continuation with shared-base two-offset growth.

The diagnostic readout never enters the map. Finite-window recurrence/growth
are candidates, not verified primitive cycles, chaos, or Feigenbaum scaling.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from huginn_periodic_readout import atomic_json, _atomic_file, sha256
from huginn_chaos_map import recurrence_candidates


def refinement_interval(rows):
    """First observed period-label change; otherwise largest growth change."""
    if len(rows) < 2:
        raise ValueError('At least two coarse settings required')
    for i in range(len(rows)-1):
        if rows[i]['period'] != rows[i+1]['period']:
            return i, 'first coarse recurrence-label change (zero means unresolved)'
    changes = [np.max(np.abs(np.asarray(b['growth'])-a['growth'])) for a,b in zip(rows,rows[1:])]
    if not np.isfinite(changes).all():
        raise ValueError('Nonfinite refinement diagnostic')
    return int(np.argmax(changes)), 'largest coarse growth change; no period-label change found'


def interior_grid(low, high, count):
    if not np.isfinite([low,high]).all() or low >= high or count < 1:
        raise ValueError('Invalid refinement interval')
    return np.linspace(low,high,count+2)[1:-1]


def growth_step(base, twins, radius, step):
    """One shared unperturbed base and two renormalized twins per initial state."""
    import torch
    n=len(base);x=step(torch.cat([base,*twins]));h=x[:n]
    if not torch.isfinite(x).all():raise ValueError('Nonfinite hidden state')
    ds=x[n:].reshape(len(twins),n,*h.shape[1:])-h[None]
    dist=ds.flatten(2).norm(dim=2)
    if not torch.isfinite(dist).all() or (dist<=0).any():raise ValueError('Invalid twin separation')
    logs=torch.log(dist/radius)
    g=h[None]+ds*(radius/dist)[(...,)+(None,)*(h.ndim-1)]
    return h,g,logs


def execute(args,plan):
    import torch,transformers
    from explore_latent_loop import HuginnLoop, DATA, REVISIONS
    root=DATA/'explore'/args.out;root.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();deadline=start+plan['max_compute_seconds'];published=[]
    def emit(phase,**kw):
        row=dict(phase=phase,chunk_start=kw.pop('chunk_start',-1),**kw)
        with (root/'progress.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n');f.flush()
        print(json.dumps(row),flush=True)
    def save(name,**arrays):
        if any(not np.isfinite(v).all() for v in arrays.values() if np.asarray(v).dtype.kind in 'fci'):raise ValueError('Nonfinite saved array')
        _atomic_file(root/name,lambda f:np.savez(f,**arrays));published.append(name)
        if not name.startswith('chunk-') and name != 'map.npz':emit('artifact_complete',file=name)
    def guard():
        if time.monotonic()>=deadline:raise TimeoutError('Original numerical deadline reached')
    try:
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(.8)
        inputs=DATA/plan['input_path']
        if sha256(inputs)!=plan['input_sha256']:raise ValueError('Anchor input hash mismatch')
        data=np.load(inputs);anchor=data['anchor'];v=data['direction'];reference=data['reference']
        if anchor.shape!=(1,12,5280) or reference.shape!=(2,12,5280):raise ValueError('Anchor shape mismatch')
        loop=HuginnLoop('huginn','float32','cuda',4,seed=0);h0=loop.setup(plan['prompt'])
        if loop.token_ids!=plan['prompt_ids'] or REVISIONS['huginn']!=plan['model_revision'] or loop.model.config.test_time_noise!=0:raise ValueError('Model or prompt binding failed')
        _atomic_file(root/'plan.json',lambda f:f.write(args.plan.read_bytes()))
        atomic_json(root/'runtime.json',dict(source=args.source_commit,plan_sha256=sha256(args.plan),model_revision=REVISIONS['huginn'],prompt_ids=loop.token_ids,torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),dtype='float32',tf32=False,input_sha256=sha256(inputs)))
        _atomic_file(root/'steering-direction.npy',lambda f:np.save(f,v))
        with torch.no_grad():
            ref,_=loop.model.core_block_forward(h0,loop.e,loop.freqs,None,None,torch.tensor(0),0)
            ours=loop.F(h0);error=float((ref-ours).norm()/ref.norm());repeat=float((ours-loop.F(h0)).abs().max())
        emit('native_gate',error=error,repeat=repeat)
        if error>1e-6 or repeat!=0:raise ValueError('Native gate failed')
        loop.set_steering(0,np.repeat((plan['steering']*v)[None],6,axis=0))
        def step(lam):
            def F(x):
                with torch.no_grad():return (1-lam)*x+lam*loop.F(x)
            return F
        replay=step(torch.tensor(plan['anchor_update_size'],device='cuda',dtype=torch.float32))(torch.from_numpy(reference[:1]).to('cuda').repeat(6,1,1))
        rel=np.linalg.norm((replay.cpu().numpy().astype('float64')-reference[1]).reshape(6,-1),axis=1)/np.linalg.norm(reference[1].astype('float64'))
        emit('saved_step_gate',relative_errors=rel.tolist())
        if rel.max()>3e-5:raise ValueError('Remote six-row anchor replay failed')
        F=step(torch.tensor(plan['lambda_range'][0],device='cuda',dtype=torch.float32));x=h0.repeat(6,1,1)
        for _ in range(8):x=F(x)
        torch.cuda.synchronize();tic=time.monotonic()
        for _ in range(128):x=F(x)
        torch.cuda.synchronize();unit=(time.monotonic()-tic)/128
        # Include full-size state checkpoint cost in qualification, then retain it.
        tic=time.monotonic();save('benchmark.npz',states=np.repeat(h0.cpu().numpy()[None],plan['checkpoint_steps']*2,axis=0));io=time.monotonic()-tic
        projected=(unit*(plan['transient']+plan['measure'])+io*((plan['transient']+plan['measure'])/plan['checkpoint_steps']))*plan['max_settings']*1.3+300
        remaining=deadline-time.monotonic()
        emit('throughput_gate',seconds_per_cell_step=unit,checkpoint_seconds=io,projected_seconds_with_margin=projected,remaining_seconds=remaining)
        if projected>remaining-600:raise ValueError('Exact workload cannot fit original ceiling')
        del x,replay
        rows=[];ends=[];coarse=np.linspace(*plan['lambda_range'],plan['coarse_settings'])
        warm=torch.from_numpy(anchor).to('cuda');selection=None
        def setting(index,lam,warm,stage,parent):
            guard()
            if (root/'.stop-after-setting').exists():
                atomic_json(root/'.stopped-after-setting',dict(index=index-1,source=args.source_commit))
                raise RuntimeError('Controller requested stop after verified setting')
            F=step(torch.tensor(lam,device='cuda',dtype=torch.float32));base=torch.cat([warm,h0]);save(f'initial-{index:04d}.npz',states=base.cpu().numpy(),update_size=np.array(lam),parent=np.array(parent))
            buffer=[];logs=[];windows=[];recent=[];project=[];tic=time.monotonic()
            for k in range(plan['transient']):
                guard();base=F(base.repeat(3,1,1))[:2];buffer.append(base.cpu().numpy())
                if (k+1)%plan['checkpoint_steps']==0:
                    save(f'states-{index:04d}-t-{k+1:05d}.npz',states=np.stack(buffer));buffer=[]
                    emit('transient',chunk_start=index,step=k+1)
            torch.manual_seed(10000+index);d=torch.randn_like(base);d/=d.flatten(1).norm(dim=1)[:,None,None]
            radius=torch.tensor(plan['eps'],device='cuda',dtype=torch.float32)[:,None]*base.flatten(1).norm(dim=1)[None]
            twins=base[None]+d[None]*radius[:,:,None,None]
            axis=(reference[0]-reference[1]).reshape(-1).astype(np.float64);axis/=np.linalg.norm(axis)
            for k in range(plan['measure']):
                guard();base,twins,log=growth_step(base,twins,radius,F);a=base.cpu().numpy();buffer.append(a);logs.append(log.cpu().numpy())
                recent.append(a)
                if len(recent)>256:recent.pop(0)
                if (k+1)%plan['checkpoint_steps']==0:
                    saved=np.stack(buffer);save(f'states-{index:04d}-m-{k+1:05d}.npz',states=saved,log_growth=np.array(logs[-len(buffer):]))
                    windows.append(recurrence_candidates(np.asarray(recent,dtype=np.float64),plan['recurrence_tolerance'],64,128).tolist())
                    project.append(np.einsum('trd,d->tr',saved[-256:].astype(np.float64).reshape(256,2,-1),axis));buffer=[]
                    emit('measure',chunk_start=index,step=k+1)
            tail=np.array(recent);growth=np.asarray(logs);period=recurrence_candidates(tail.astype(np.float64),plan['recurrence_tolerance'],64,128).tolist()
            readouts=[]
            for state in tail[-2:]:
                for row in range(2):
                    z=loop.logits(torch.from_numpy(state[row:row+1])).numpy().astype('float64');z-=z.max();p=np.exp(z);p/=p.sum();ids=np.argsort(-p)[:10]
                    readouts.append(dict(ids=ids.tolist(),tokens=[loop.tok.decode([int(i)]) for i in ids],probabilities=p[ids].tolist()))
            save(f'chunk-{index:04d}.npz',update_size=np.array(lam),lyapunov=growth.mean(0),half_means=np.stack([a.mean(0) for a in np.array_split(growth,2)]),recurrence_candidate=np.array(period),window_period=np.array(windows),projection=np.array(project),final_state=tail[-1],readouts_json=np.array(json.dumps(readouts)))
            row=dict(index=index,update_size=float(lam),stage=stage,parent=parent,period=period,growth=growth.mean(0).tolist(),seconds=time.monotonic()-tic)
            emit('chunk_complete',chunk_start=index,completed=index+1,total=plan['max_settings'],**row)
            if plan.get('retrieval_barrier'):
                from huginn_evidence_gate import setting_payload,wait_for_ack
                payload=setting_payload(root,index,args.source_commit,sha256)
                emit('retrieval_wait',chunk_start=index,barrier=payload)
                wait_for_ack(root,payload,deadline)
                emit('retrieval_verified',chunk_start=index)
                if index==0:
                    remaining=deadline-time.monotonic()
                    projected=(time.monotonic()-tic)*(plan['max_settings']-1)*1.1
                    emit('retrieval_throughput_gate',projected_remaining=projected,remaining=remaining)
                    if projected>remaining-60:
                        atomic_json(root/'.stopped-after-setting',dict(index=index,source=args.source_commit))
                        raise RuntimeError('Measured computation plus verified retrieval cannot fit original deadline')
                if (root/'.stop-after-setting').exists():
                    atomic_json(root/'.stopped-after-setting',dict(index=index,source=args.source_commit))
                    raise RuntimeError('Controller requested stop after verified setting')
            return row,base[:1].clone(),tail[-1,0].copy()
        for lam in coarse:
            i=len(rows);row,warm,end=setting(i,float(lam),warm,'coarse',i-1);rows.append(row);ends.append(end)
        if plan.get('schedule') == 'bidirectional':
            selection=dict(reason='fixed reverse grid; no outcome-based interval selection')
            emit('refinement_selected',**selection)
            for lam in coarse[::-1]:
                i=len(rows);row,warm,end=setting(i,float(lam),warm,'descending',i-1);rows.append(row)
        else:
            choice,why=refinement_interval(rows);lower=rows[choice]['update_size'];upper=rows[choice+1]['update_size']
            selection=dict(lower=lower,upper=upper,coarse_parent=choice,reason=why);emit('refinement_selected',**selection)
            warm=torch.from_numpy(ends[choice][None]).to('cuda')
            for lam in interior_grid(lower,upper,plan['refinement_settings']):
                i=len(rows);parent=choice if i==len(coarse) else i-1
                row,warm,end=setting(i,float(lam),warm,'refined',parent);rows.append(row)
        save('map.npz',update_sizes=np.array([r['update_size'] for r in rows]),lyapunov=np.array([r['growth'] for r in rows]),recurrence_candidate=np.array([r['period'] for r in rows]))
        atomic_json(root/'summary.json',dict(rows=rows,selection=selection,seconds=time.monotonic()-start,peak_cuda_bytes=torch.cuda.max_memory_allocated(),qualification='Finite float32 recurrence and growth; not proven chaos or primitive stability. Shared base, two offsets; no token feedback.'))
        emit('complete',cells=len(rows))
        atomic_json(root/'manifest.json',{p.name:sha256(p) for p in root.iterdir() if p.is_file()})
    except BaseException as error:
        import traceback
        atomic_json(root/'failure.json',dict(error=repr(error),traceback=traceback.format_exc()))
        raise
