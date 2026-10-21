"""Measure explicit feedback-path sensitivity on the saved child orbit.

This is a baseline derivative, not a fresh curve or a native coupling claim.
The independently continued-orbit response is compared by the final auditor.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from baby_shrimp_saddle_memory import Evaluator, ROOT, save, sha
import numpy as np
from trace_pair_sensitivity import trace_pair_sensitivity, memory_tangent_derivative
import llm_feigenbaum as lfb


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',default='docs/baby-shrimp-zoom/saddle-memory-plan.json')
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--seconds',type=int,default=900)
    args=parser.parse_args()
    if not 0<args.seconds<=1800: parser.error('bounded baseline sensitivity required')
    args.out.mkdir(parents=True,exist_ok=False)
    plan=json.loads((ROOT/args.plan).read_text())
    def emit(stage,**kw):
        row={'time':datetime.now(timezone.utc).isoformat(),'stage':stage,**kw}
        with (args.out/'progress.jsonl').open('a') as f: f.write(json.dumps(row,allow_nan=False)+'\n')
        print(json.dumps(row,allow_nan=False),flush=True)
    source=['tools/baby_shrimp_memory_mechanism.py','tools/trace_pair_sensitivity.py',
            'tools/baby_shrimp_saddle_memory.py','tools/baby_shrimp_memory.py',
            'tools/explore_llm_soft_loop.py','tools/explore_llm_attractors.py','tools/explore_llm_phase_map.py']
    save(args.out/'protocol.json',{'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_sha256':{p:sha(ROOT/p) for p in source},'plan_sha256':sha(ROOT/args.plan),
        'scope':'Frozen-orbit explicit memory sensitivity; total response must re-solve the orbit.'})
    try:
        ev=Evaluator(args,plan,args.out,emit)
        orb=ev.cell(ev.origin,0)
        raw=np.load(ev.anchor/'points.npy',allow_pickle=False)
        rows=[]
        for shift in [6,0,12]:
            points=[ev.torch.as_tensor(p.reshape((1,)+orb.shape),device=ev.loop.device,dtype=ev.torch.float64)
                    for p in np.roll(raw,-shift,axis=0)]
            with np.load(ev.anchor/f'phase-{shift}.npz',allow_pickle=False) as z: matrix=z['matrix']
            callbacks=[lambda v,p=p:lfb._cols(orb.base.jvp_rows(p,lfb._rows(v,p))) for p in points]
            retained={}
            def derivative(v):
                w,z=memory_tangent_derivative(callbacks,v)
                retained.update(w=w,z=z,v=v)
                return z
            result=trace_pair_sensitivity(matrix,derivative)
            error=np.linalg.norm(retained['w']-matrix@retained['v'],axis=0)/np.maximum(np.linalg.norm(retained['w'],axis=0),1e-300)
            if max(error)>=1e-7: raise ValueError('explicit sensitivity composed baseline gate failed')
            numeric={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
            np.savez_compressed(args.out/f'phase-{shift}.npz',**numeric,baseline_jvp=retained['w'])
            row={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items()
                 if k not in ['right','left','applied_derivative']}
            row.update(phase=shift,composed_jvp_relative_errors=error.tolist())
            rows.append(row);save(args.out/'partial.json',rows);emit('phase_complete',phase=shift,trace_derivative=row['trace_derivative'])
        derivatives=np.array([r['trace_derivative'] for r in rows])
        spread=float(np.ptp(derivatives))
        if spread>1e-5*max(abs(derivatives)): raise ValueError('cyclic explicit sensitivities disagree')
        save(args.out/'result.json',{'complete':True,'rows':rows,'cyclic_derivative_spread':spread,
            'scope':'Derivative through the injected identity path at fixed original orbit. This alone does not measure total saddle-height response or identify native mode coupling.'})
    except BaseException as error:
        save(args.out/'failure.json',{'type':type(error).__name__,'error':str(error)})
        raise


if __name__=='__main__': main()
