"""FGR2D: continue paired return states to the next natural departure."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from fold_global_return import backup_file,sha,write_json
from fold_panel_data import coordinates,channel_mask


def audit_departure(v, transverse, width):
    """Independent vectorized exit check, with three initial channel readings known."""
    v,t=np.asarray(v),np.asarray(transverse)
    mask=(abs(v)<width)&(t<.05)
    if not mask[0]:raise ValueError('Continuation must start in the confirmed channel')
    phase=np.arange(0,len(v),2)
    leave=np.flatnonzero(~mask[phase])
    if not len(leave):return None
    k=int(leave[0]);readings=2+k
    if readings<10:return {'kind':'short_passage','readings':readings}
    if v[phase[k]]<0:return {'kind':'negative_side_departure'}
    for step in phase[k:]:
        if mask[step]:return {'kind':'channel_recrossing'}
        if v[step]>=3*width and t[step]<.05:
            return {'kind':'next_natural_exit','update':int(step),'v':float(v[step]),'transverse':float(t[step]),'channel_readings':readings}
    return None


def run(args):
    from explore_llm_soft_loop import SoftLoop
    import explore_llm_phase_map as phase
    parent,out,backup=map(Path,(args.parent,args.out,args.backup))
    out.mkdir(parents=True,exist_ok=False);backup.mkdir(parents=True,exist_ok=False)
    assert out.stat().st_dev!=backup.stat().st_dev
    old=json.loads((parent/'plan.json').read_text());result=json.loads((parent/'result.json').read_text())
    manifest=json.loads((parent/'manifest.json').read_text())
    for r in manifest:assert sha(parent/r['path'])==r['sha256']
    with np.load(parent/'inputs.npz') as z:d={k:z[k] for k in z.files}
    states=[d['seeds'][None]]
    for f in sorted(parent.glob('updates-*.npz')):
        with np.load(f) as z:states.append(z['states'])
    states=np.concatenate(states);seeds=np.array([states[n,i] for i,n in enumerate(result['valid_through'])])
    active=np.array([r['kind']=='return' for r in result['outcomes']])
    assert active.sum()==58 and np.array_equal(active[::2],active[1::2])
    vv,tt=coordinates(seeds,d['centre'],d['left'],d['right'],1.)
    assert channel_mask(vv,tt,old['width'])[active].all()
    plan={k:old[k] for k in ('source','epsilon','width','transverse_limit','rows')}
    plan.update(name='FGR2D',parent=str(parent),parent_manifest_sha256=sha(parent/'manifest.json'),
                max_updates=512,chunk_updates=128,threads=8,
                rule='Continue each returned pair from its third channel reading; require at least10 consecutive same-phase channel readings in total and first subsequentv>=3U with transverse<0.05. P3 pairs excluded; never count padded states.')
    np.savez(out/'inputs.npz',seeds=seeds,active=active,centre=d['centre'],left=d['left'],right=d['right'],
             replay_states=d['replay_states'],replay_F=d['replay_F'],replay_G=d['replay_G'])
    plan['inputs_sha256']=sha(out/'inputs.npz');write_json(out/'plan.json',plan)
    files=[backup_file(out/n,out,backup) for n in ('inputs.npz','plan.json')]
    tic=time.monotonic();so=plan['source']
    loop=SoftLoop(so['model'],so['layer'],8,so['prompt'],so['opener'],so['window'],'float64','cpu')
    torch=loop.torch;direction=phase.load_vectors(so['vectors'])[so['concept']]
    loop.prepare(np.tile((so['alpha']+plan['epsilon'])*direction,(64,1)));T=np.full(64,10**so['log10_temperature'])
    norms=[m for m in loop.model.modules() if type(m).__name__.endswith('RMSNorm')]
    rotaries=[m for m in loop.model.modules() if type(m).__name__.endswith('RotaryEmbedding')]
    assert norms and rotaries and all(m.forward.__func__.__qualname__.startswith('keep_float64_norms.') for m in norms)
    assert all(m._inv_freq64.dtype==torch.float64 for m in rotaries)
    meta={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'source_hashes':{n:sha(ROOT/'tools'/n) for n in ('fold_return_cycle_map.py','fold_global_return.py','fold_panel_data.py','explore_llm_soft_loop.py','explore_llm_attractors.py','explore_llm_phase_map.py')},
          'plan_sha256':sha(out/'plan.json'),'backup':str(backup),'pid':os.getpid(),'model_revision':loop.sm.engine.revision,
          'prefix_ids':loop.prefix_ids.tolist(),'direction_sha256':sha(phase.DATA/'explore'/so['vectors']/'vectors.npz'),
          'torch':torch.__version__,'numpy':np.__version__,'dtype':str(loop.dtype),'threads':8,'device':'cpu'}
    write_json(out/'metadata.json',meta);files.append(backup_file(out/'metadata.json',out,backup))
    with torch.no_grad():
        y,_=loop.step(torch.tensor(np.tile(d['replay_states'],(8,1))[:,None],dtype=torch.float64),T)
        z,_=loop.step(y,T)
    gate={'one_step':float(np.max(abs(y.numpy()[:,0]-np.tile(d['replay_F'],(8,1))))),
          'two_step':float(np.max(abs(z.numpy()[:,0]-np.tile(d['replay_G'],(8,1)))))}
    assert max(gate.values())<1e-11
    write_json(out/'replay-gate.json',gate);files.append(backup_file(out/'replay-gate.json',out,backup))
    X=torch.tensor(seeds[:,None],dtype=torch.float64);valid=np.zeros(64,dtype=int)
    outcomes=[None if a else {'kind':'excluded_P3'} for a in active]
    readings=np.where(active,3,0);departed=np.zeros(64,dtype=bool);chunk=[]
    for step in range(1,513):
        active=np.array([r is None for r in outcomes])
        with torch.no_grad():
            y,_=loop.step(X,T);X=torch.where(torch.as_tensor(active)[:,None,None],y,X)
        arr=X.numpy()[:,0].copy();assert np.isfinite(arr).all()
        valid[active]=step;chunk.append(arr)
        if step%2==0:
            v,t=coordinates(arr,d['centre'],d['left'],d['right'],1.);inside=channel_mask(v,t,plan['width'])
            for i in np.flatnonzero(active):
                if not departed[i]:
                    if inside[i]:readings[i]+=1
                    else:
                        departed[i]=True
                        if readings[i]<10:outcomes[i]={'kind':'short_passage','readings':int(readings[i])};continue
                        if v[i]<0:outcomes[i]={'kind':'negative_side_departure'};continue
                elif inside[i]:
                    outcomes[i]={'kind':'channel_recrossing'};continue
                if departed[i] and v[i]>=3*plan['width'] and t[i]<.05:
                    outcomes[i]={'kind':'next_natural_exit','update':step,'v':float(v[i]),'transverse':float(t[i]),'channel_readings':int(readings[i])}
        finished=all(r is not None for r in outcomes)
        if step%128==0 or finished or step==512:
            name=f'updates-{step-len(chunk)+1:05d}-{step:05d}.npz';p=out/name
            with open(str(p)+'.partial','wb') as f:np.savez(f,states=np.array(chunk),valid_through=valid.copy());f.flush();os.fsync(f.fileno())
            os.replace(str(p)+'.partial',p);files.append(backup_file(p,out,backup));chunk=[]
            write_json(out/'retrieval.json',files)
            status={'step':step,'exited':sum(r is not None and r['kind']=='next_natural_exit' for r in outcomes),'active':sum(r is None for r in outcomes),'seconds':time.monotonic()-tic}
            write_json(out/'status.json',status);print(json.dumps(status),flush=True)
        if finished:break
    outcomes=[r if r is not None else {'kind':'unresolved_at_horizon'} for r in outcomes]
    write_json(out/'result.json',{'outcomes':outcomes,'valid_through':valid.tolist(),'wall_seconds':time.monotonic()-tic})
    files.append(backup_file(out/'result.json',out,backup));write_json(out/'manifest.json',files);backup_file(out/'manifest.json',out,backup)
    write_json(out/'complete.json',{'manifest_sha256':sha(out/'manifest.json'),'backup_manifest_sha256':sha(backup/'manifest.json')});backup_file(out/'complete.json',out,backup)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('parent','out','backup'):p.add_argument('--'+n,required=True)
    run(p.parse_args())
