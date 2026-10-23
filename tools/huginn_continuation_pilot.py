"""Bounded exploratory warm/fresh Huginn continuation; recurrence is not chaos."""
import argparse,json,subprocess,time,traceback
from pathlib import Path
import numpy as np
from huginn_periodic_readout import EvidenceWriter,atomic_json,sha256,hard_deadline,relaxed_step,relative_distance

def classify(history,tol=3e-5,max_period=64):
    a=np.asarray(history,dtype=np.float64)
    errors=[]
    for q in range(1,min(max_period,len(a)-8)+1):
        errors.append(relative_distance(a[-8:].reshape(8*a.shape[1],-1),a[-8-q:-q].reshape(8*a.shape[1],-1)).reshape(8,a.shape[1]).max(0))
    errors=np.array(errors);labels=[]
    for row in errors.T:
        passing=np.flatnonzero(row<tol);labels.append(int(passing[0]+1) if len(passing) else 0)
    return np.array(labels),errors

def execute(plan,source,w,start):
    import torch,transformers
    from explore_latent_loop import HuginnLoop,REVISIONS
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==source
    assert not subprocess.check_output(['git','diff','HEAD','--','tools/huginn_continuation_pilot.py','tools/explore_latent_loop.py','tools/huginn_periodic_readout.py'])
    for p,h in plan['inputs'].items():
        if sha256(p)!=h:raise ValueError('Input hash mismatch: '+p)
    with np.load(plan['initial_states']) as f:anchor=f['states'][0:1].copy()
    direction=np.load(plan['direction'])
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.cuda.set_per_process_memory_fraction(.8)
    loop=HuginnLoop('huginn','float32','cuda',4,seed=0);h0=loop.setup(plan['prompt'])
    assert loop.token_ids==plan['prompt_ids'] and REVISIONS['huginn']==plan['model_revision'] and loop.model.config.test_time_noise==0
    atomic_json(w.root/'runtime.json',dict(source=source,plan_sha256=sha256(args.plan),model_revision=REVISIONS['huginn'],prompt_ids=loop.token_ids,torch=torch.__version__,cuda=torch.version.cuda,transformers=transformers.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(),dtype='float32',tf32=False,threads=4))
    with torch.no_grad():
        ref,_=loop.model.core_block_forward(h0,loop.e,loop.freqs,None,None,torch.tensor(0),0);ours=loop.F(h0)
        err=float((ref-ours).norm()/ref.norm());repeat=float((ours-loop.F(h0)).abs().max())
    w.emit('native_gate',relative_error=err,repeat=repeat)
    if err>1e-6 or repeat!=0:raise ValueError('Native gate failed')
    loop.set_steering(0,np.repeat((plan['steering']*direction)[None],4,axis=0))
    with np.load(plan['reference_states']) as data: refstates=data['states'][:2].copy()
    with torch.no_grad():
        z=torch.from_numpy(refstates[:1]).to('cuda').repeat(4,1,1)
        lam=torch.full((4,1,1),plan['center'],device='cuda',dtype=torch.float32)
        z=relaxed_step(z,loop.F,lam).cpu().numpy()
    replay=relative_distance(np.repeat(refstates[1:2],4,axis=0),z)
    w.emit('saved_step_gate',relative_errors=replay.tolist())
    if replay.max()>3e-5:raise ValueError('Batch-four saved-state replay failed')
    warm=torch.from_numpy(anchor).to('cuda').repeat(2,1,1)
    # Row order: increasing warm, increasing fresh, decreasing warm, decreasing fresh.
    rows=[];chunk_seconds=None
    for rung in range(plan['rungs']):
        if chunk_seconds is not None and time.monotonic()-start+chunk_seconds*plan['steps']/128*1.3+45>plan['seconds']:
            w.emit('bounded_stop',next_rung=rung,reason='Next full rung does not fit original deadline with margin');break
        vals=[plan['center']+rung*plan['spacing'],plan['center']-rung*plan['spacing']]
        lam=torch.tensor([vals[0],vals[0],vals[1],vals[1]],device='cuda',dtype=torch.float32)[:,None,None]
        h=torch.cat([warm[:1],h0,warm[1:],h0]);recent=[];buffer=[];tic=time.monotonic();timings=[]
        w.checkpoint(f'rung-{rung:02d}-initial',{'states':h.cpu().numpy(),'lambdas':lam.cpu().numpy().ravel()}, {'rung':rung,'row_order':plan['row_order'],'fresh_reset':True})
        for step in range(1,plan['steps']+1):
            with torch.no_grad():h=relaxed_step(h,loop.F,lam)
            state=h.cpu().numpy()
            if not np.isfinite(state).all():raise ValueError('Nonfinite trajectory')
            buffer.append(state)
            if step>plan['steps']-256:recent.append(state)
            if step%128==0:
                w.checkpoint(f'rung-{rung:02d}-steps-{step-127:04d}-{step:04d}',{'states':np.stack(buffer)}, {'rung':rung,'first_step':step-127,'last_step':step,'lambdas':vals,'row_order':plan['row_order']})
                buffer=[];timings.append(time.monotonic()-tic);tic=time.monotonic()
                if rung==0 and step==128:
                    chunk_seconds=timings[-1];remain=plan['seconds']-(time.monotonic()-start)
                    projected=chunk_seconds*(plan['steps']-128)/128*1.3+45
                    w.emit('throughput_gate',chunk_seconds=chunk_seconds,first_rung_remaining_with_margin=projected,remaining_seconds=remain)
                    if projected>remain:raise TimeoutError('Even first rung does not fit')
        tail=np.stack(recent);labels,errors=classify(tail,plan['tolerance'])
        # CPU native readout only at final two states, never fed back.
        readouts=[]
        for phase in [-2,-1]:
            for row in range(4):
                z=loop.logits(torch.from_numpy(tail[phase,row:row+1])).numpy();z-=z.max();p=np.exp(z);p/=p.sum();ids=np.argsort(-p)[:10]
                readouts.append(dict(tail_index=phase,row=row,ids=ids.tolist(),tokens=[loop.tok.decode([int(i)]) for i in ids],probabilities=p[ids].tolist()))
        w.checkpoint(f'rung-{rung:02d}-result',{'tail':tail,'candidate_period':labels,'lag_errors':errors,'lambdas':np.array(vals)}, {'readouts':readouts,'classification':'Smallest passing lag1..64 at final eight full-state positions; not verified primitive/stable orbit or chaos.'})
        row=dict(rung=rung,lambdas=vals,candidate_period=labels.tolist(),seconds_per_128_steps=float(np.median(timings)),elapsed=time.monotonic()-start)
        rows.append(row);w.emit('rung_complete',**row);chunk_seconds=max(timings);warm=h[[0,2]].clone()
    atomic_json(w.root/'summary.json',dict(completed_rungs=rows,planned_rungs=plan['rungs'],bounded_pilot_complete=True,full_ladder_complete=len(rows)==plan['rungs'],peak_cuda_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-start,qualification='Exploratory float32 recurrence screening; no Lyapunov or stability multiplier calculation, no chaos claim.'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--source-commit',required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();plan=json.loads(args.plan.read_text())
    if Path.cwd().resolve() in args.out.resolve().parents:raise ValueError('Evidence must be outside repo')
    w=EvidenceWriter(args.out);atomic_json(w.root/'plan.json',plan);start=time.monotonic()
    try:
        with hard_deadline(plan['seconds']):execute(plan,args.source_commit,w,start)
        w.manifest('manifest.json',True)
    except BaseException as e:
        atomic_json(w.root/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),partial_shards_preserved=True));w.manifest('partial-manifest.json',False);raise
