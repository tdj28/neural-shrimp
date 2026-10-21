"""Exploratory continuation of the child's trace saddle under feedback memory.

No rentals. Every attempted orbit and matrix-free spectral gate is retained.
The beta=0 operator is compared with the previously audited dense operator.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):
    os.environ[key]="8"
import numpy as np
from baby_shrimp_memory import MemoryOrbit, qualified_ritz
from baby_shrimp_orbits import ROOT, DATA, save, primitive_gate
from baby_shrimp_geometry import canonical
import llm_feigenbaum as lfb


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Evaluator:
    def __init__(self, args, plan, out, emit):
        import torch
        import explore_llm_soft_loop as soft
        self.torch, self.soft, self.args, self.plan = torch, soft, args, plan
        self.out, self.emit = out, emit
        self.started=time.monotonic()
        torch.set_num_threads(8)
        torch.set_num_interop_threads(1)
        if args.device=='cuda': torch.cuda.set_per_process_memory_fraction(.75)
        self.loop=soft.SoftLoop('qwen0.5b',12,8,'open','Hello!',1,'float64',args.device,'native')
        if self.loop.sm.engine.revision!=plan['model_revision']:
            raise ValueError('model revision differs')
        norm=self.loop.model.model.norm
        x=torch.linspace(-1,1,norm.weight.numel(),device=self.loop.device,dtype=torch.float64)[None,None,:]
        sensitivity=float((norm(x*(1+1e-12))-norm(x)).abs().max())
        if not 0<sensitivity<1e-9 or not hasattr(self.loop.model.model.rotary_emb,'_inv_freq64'):
            raise ValueError('effective float64 gate failed')
        self.vector=soft.elpm.load_vectors('llm-phase-map/vectors-l12')['philosophy']
        self.origin=np.array(plan['start']); self.scales=np.array(plan['coordinate_scales'])
        self.anchor=DATA/plan['anchor_directory']
        for name, digest in plan['anchor_hashes'].items():
            if sha(self.anchor/name)!=digest: raise ValueError('anchor hash mismatch: '+name)
        points=np.load(self.anchor/'points.npy',allow_pickle=False)
        self.seed=points[6].copy()  # phase six passed the previous composed-map FD gate
        self.qs={}
        for shift in [0,6,12]:
            with np.load(self.anchor/f'phase-{(shift+6)%18}.npz',allow_pickle=False) as z:
                cols=np.concatenate([z['vectors'][:,:2].real,np.random.default_rng(91+shift).normal(size=(896,4))],axis=1)
                self.qs[shift]=np.linalg.qr(cols)[0]
        self.cache={}; self.warm=[]; self.attempts=0
        save(out/'runtime.json',{'norm_sensitivity':sensitivity,'rotary_float64':True,
             'device':torch.cuda.get_device_name() if args.device=='cuda' else 'cpu',
             'torch':torch.__version__,'cuda':torch.version.cuda,'threads':torch.get_num_threads(),
             'python':subprocess.check_output(['python','--version'],text=True).strip()})

    def cell(self, physical, beta):
        base=self.soft.SoftOrbit(self.loop,float(physical[0])*self.vector,10.0**float(physical[1]),chunk=4,jacobian_method='ad')
        return MemoryOrbit(base,beta)

    def apply(self, orb, points, q):
        w=lfb._rows(q,points[0])
        for point in points: w=orb.jvp_rows(point,w)
        return lfb._cols(w)

    def __call__(self,u,beta,cyclic=False):
        u=np.asarray(u,float); beta=float(beta); key=(*u,beta)
        if key in self.cache and (not cyclic or len(self.cache[key]['phases'])==3): return self.cache[key]
        if self.attempts>=self.plan['maximum_evaluations'] or time.monotonic()-self.started>self.args.seconds:
            raise TimeoutError('bounded local continuation limit')
        if abs(beta)>self.plan['maximum_abs_beta'] or np.max(abs(u))>self.plan['maximum_scaled_coordinate']:
            raise ValueError('continuation left declared parameter domain')
        number=self.attempts; self.attempts+=1
        dest=self.out/f'point-{number:04d}'; dest.mkdir()
        physical=self.origin+self.scales*u
        record={'index':number,'coordinate':u.tolist(),'beta':beta,'alpha':float(physical[0]),
                'log10_temperature':float(physical[1]),'qualified':False,'started_seconds':time.monotonic()-self.started}
        save(dest/'point.json',record)
        self.emit('point_start',**record)
        orb=self.cell(physical,beta)
        if self.warm:
            parent=min(self.warm,key=lambda r:np.linalg.norm(u-r[0])+abs(beta-r[1])/1e-6)
            seed=parent[2]; q=parent[3].copy()
        else: seed=self.seed; q=self.qs[0].copy()
        x=self.torch.as_tensor(seed.reshape((1,)+orb.shape),dtype=self.torch.float64,device=self.loop.device)
        history=[]
        for iteration in range(13):
            pts=orb.points(x,18); rel=float((pts[-1]-x).norm()/x.norm()); history.append(rel)
            if not np.isfinite(rel): raise ValueError('nonfinite orbit closure')
            if rel<1e-12: break
            x=pts[-1]
        record['closure_history']=history
        save(dest/'point.json',record)
        if history[-1]>=1e-12: raise ValueError('primitive-orbit fixed-point correction failed')
        returns=[float((p-x).norm()/x.norm()) for p in pts[1:]]
        pg=primitive_gate(returns,18)
        if not pg['closed'] or not pg['no_lower_period']: raise ValueError('primitive orbit gate failed')
        orbit=np.stack([lfb._flat(p) for p in pts[:-1]])
        np.save(dest/'points.npy',orbit)
        tokens=[int(orb.probabilities(p).argmax()) for p in pts[:-1]]
        if canonical(tokens)!=canonical([271,785,3405,271,785,13885]*3):
            raise ValueError('text family changed')
        record.update(primitive=pg,tokens=tokens,phases=[])
        for shift in ([0,6,12] if cyclic else [0]):
            rotated=pts[shift:-1]+pts[:shift]
            result=qualified_ritz(lambda v:self.apply(orb,rotated,v),q if shift==0 else self.qs[shift])
            vals=result['values']
            local={'phase':shift,'multipliers':[[float(v.real),float(v.imag)] for v in vals],
                   'relative_residuals':result['relative_residuals'].tolist(),
                   'tau':float(sum(vals[:2]).real),'delta':float(np.prod(vals[:2]).real),
                   'radius':float(max(abs(vals))),'history':result['history']}
            if local['radius']>=1: raise ValueError('periodic orbit lost attraction')
            np.savez_compressed(dest/f'phase-{shift}.npz',**{k:result[k] for k in ['values','vectors','direct','relative_residuals','q','w']})
            record['phases'].append(local)
            save(dest/'point.json',record)
            if shift==0: next_q=result['q']
        phase=record['phases'][0]
        if len(record['phases'])==3:
            pairs=[np.array([complex(*v) for v in p['multipliers'][:2]]) for p in record['phases']]
            discrepancies=[min(max(abs(pairs[0]-p)),max(abs(pairs[0]-p[::-1]))) for p in pairs]
            if max(discrepancies)>=1e-5*max(abs(pairs[0])): raise ValueError('cyclic pair disagreement')
            record['cyclic_pair_absolute_discrepancy']=float(max(discrepancies))
        record.update(qualified=True,tau=phase['tau'],delta=phase['delta'],radius=phase['radius'],seconds=time.monotonic()-self.started-record['started_seconds'])
        save(dest/'point.json',record)
        self.cache[key]=record; self.warm.append((u.copy(),beta,lfb._flat(x),next_q))
        self.emit('point_complete',index=number,beta=beta,tau=record['tau'],seconds=record['seconds'])
        return record

    def baseline(self):
        row=self(np.zeros(2),0,cyclic=True)
        dest=self.out/f"point-{row['index']:04d}"
        with np.load(self.anchor/'phase-6.npz',allow_pickle=False) as z: matrix=z['matrix']
        orbit=np.load(dest/'points.npy',allow_pickle=False)
        orb=self.cell(self.origin,0)
        points=[self.torch.as_tensor(p.reshape((1,)+orb.shape),dtype=self.torch.float64,device=self.loop.device) for p in orbit]
        rng=np.random.default_rng(4421)
        with np.load(dest/'phase-0.npz',allow_pickle=False) as z: vectors=z['vectors']
        directions=np.column_stack([rng.normal(size=896),vectors]); directions/=np.linalg.norm(directions,axis=0)
        direct=self.apply(orb,points,directions); dense=matrix@directions
        errors=np.linalg.norm(direct-dense,axis=0)/np.maximum(np.maximum(np.linalg.norm(direct,axis=0),np.linalg.norm(dense,axis=0)),1e-300)
        expected=json.loads((self.anchor/'point.json').read_text())
        tau_error=abs(row['tau']-expected['tau'])/expected['spectral_radius']
        spectral=[]
        for phase in row['phases']:
            with np.load(self.anchor/f"phase-{(phase['phase']+6)%18}.npz",allow_pickle=False) as z:
                reference=z['multipliers'][:2]
            current=np.array([complex(*v) for v in phase['multipliers'][:2]])
            pair_error=min(max(abs(current-reference)),max(abs(current-reference[::-1])))/max(abs(reference))
            determinant_error=abs(np.prod(current)-np.prod(reference))/abs(np.prod(reference))
            spectral.append({'phase':phase['phase'],'pair_relative_error':float(pair_error),
                             'determinant_relative_error':float(determinant_error)})
        result={'composed_jvp_relative_errors':errors.tolist(),'tau_error_relative_to_radius':tau_error,
                'cyclic_dense_comparisons':spectral,
                'passed':bool(max(errors)<1e-7 and tau_error<1e-5 and
                    all(p['pair_relative_error']<1e-5 and p['determinant_relative_error']<1e-5 for p in spectral)),
                'point':row}
        np.savez_compressed(self.out/'baseline-operator-check.npz',directions=directions,direct=direct,dense=dense,errors=errors)
        save(self.out/'baseline.json',result)
        if not result['passed']: raise ValueError('baseline dense/operator replay failed')
        return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',default='docs/baby-shrimp-zoom/saddle-memory-plan.json')
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--seconds',type=int,default=1200)
    parser.add_argument('--mode',choices=['pilot','continue'],default='pilot')
    parser.add_argument('--betas',type=float,nargs='+',default=[0,1e-7,-1e-7])
    args=parser.parse_args()
    if not 0<args.seconds<=14400: parser.error('runtime must be bounded within four hours')
    out=args.out; out.mkdir(parents=True,exist_ok=False)
    plan_path=ROOT/args.plan; plan=json.loads(plan_path.read_text())
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    def emit(stage,**kw):
        row={'time':datetime.now(timezone.utc).isoformat(),'stage':stage,**kw}
        with (out/'progress.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    sources=['tools/baby_shrimp_saddle_memory.py','tools/baby_shrimp_memory.py','tools/explore_llm_soft_loop.py',
             'tools/explore_llm_attractors.py','tools/explore_llm_phase_map.py','tools/llm_feigenbaum.py']
    save(out/'protocol.json',{'commit':commit,'plan':plan,'plan_sha256':sha(plan_path),
         'source_sha256':{p:sha(ROOT/p) for p in sources},'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
         'started_utc':datetime.now(timezone.utc).isoformat()})
    try:
        ev=Evaluator(args,plan,out,emit); baseline=ev.baseline()
        if args.mode=='pilot':
            rows=[ev(np.zeros(2),beta,cyclic=True) for beta in args.betas]
            result={'complete':True,'baseline':baseline,'rows':rows,'scope':'Timing and fixed-parameter derivative pilot; not saddle continuation.'}
        else:
            from trace_saddle_continuation import TraceSaddleContinuation
            engine=TraceSaddleContinuation(ev,stencils=(.1,.03),correction_tolerance=.003,max_iterations=6,
                max_evaluations=plan['maximum_evaluations'],trust_radius=2.,record=lambda row:emit('continuation',event=row))
            result=engine.continue_values(args.betas,np.zeros(2))
        save(out/'result.json',result)
    except BaseException as error:
        save(out/'failure.json',{'type':type(error).__name__,'error':str(error),'time':datetime.now(timezone.utc).isoformat()})
        raise


if __name__=='__main__': main()
