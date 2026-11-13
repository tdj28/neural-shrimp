"""Post-test diagnostics of the frozen FGR3 table; never fit to test outcomes."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from fold_global_return import sha,write_json,backup_file
from audit_fold_return_followup import load_batch,verify_manifest
from audit_fold_return_inputs import independent_curve
from fold_return_surrogate import predict


def main(a):
    run,audit,out,backup=map(Path,(a.run,a.audit,a.out,a.backup))
    out.mkdir(parents=True,exist_ok=False);backup.mkdir(parents=True,exist_ok=False)
    binding=verify_manifest(run)
    summary=json.loads((audit/'audit.json').read_text())
    assert summary['bindings']['manifest_sha256']==binding['manifest_sha256']
    records=json.loads((audit/'events.json').read_text())
    rows=json.loads((audit/'rows.json').read_text())
    for f in json.loads((audit/'manifest.json').read_text()):
        assert sha(audit/f['path'])==f['sha256']
    x,_,_,n=load_batch(run/'trajectories')
    model=json.loads((run/'frozen-model.json').read_text())
    plan=json.loads((run/'plan.json').read_text())
    p=json.loads((Path(plan['parent_curve'])/'plan.json').read_text())
    with np.load(run/'inputs.npz') as z:d={k:z[k] for k in z.files}
    points=[];curves=[]
    for i,events in enumerate(records):
        q=np.array([e['v_over_U'] for e in events])
        original=np.array([x[e['exit'],i] for e in events])
        rebuilt=independent_curve(q,d,p['width'])
        relative=np.linalg.norm(original-rebuilt,axis=1)/np.linalg.norm(original,axis=1)
        curves.extend({'row':i,'exit':e['exit'],'q':float(value),'relative_state_error':float(error)}
                      for e,value,error in zip(events,q,relative))
        for j,e in enumerate(events):
            if j+1<len(events):
                actual='return';next_q=events[j+1]['v_over_U'];elapsed=events[j+1]['exit']-e['exit']
            elif rows[i]['outcome']['kind']=='reference_p3':
                actual='reference_p3';next_q=None;elapsed=None
            else:
                actual='censored';next_q=None;elapsed=None
            pred=predict(model,e['v_over_U'])
            points.append({'row':i,'exit':e['exit'],'q':e['v_over_U'],'actual':actual,
                           'next_q':next_q,'elapsed':elapsed,'prediction':pred,
                           'relative_reconstruction_error':float(relative[j])})
    diagnostics=[]
    for resolved in (False,True):
        subset=[r for r in points if r['actual']=='return' and r['prediction']['kind']=='return'
                and r['prediction']['branch_resolved']==resolved]
        errors=np.array([r['prediction']['next_q']-r['next_q'] for r in subset])
        diagnostics.append({'endpoints_same_return_time':resolved,'correctly_routed_returns':len(subset),
                             'coordinate_RMSE_U':float(np.sqrt(np.mean(errors**2))) if len(errors) else None,
                             'time_MAE_updates':float(np.mean([abs(r['prediction']['updates']-r['elapsed']) for r in subset])) if subset else None})
    error=np.array([r['relative_state_error'] for r in curves])
    result={'status':'post-test descriptive diagnostics; no refit or new model selection',
            'raw':binding,'audit_sha256':sha(audit/'audit.json'),'model_sha256':sha(run/'frozen-model.json'),
            'natural_exits':len(curves),'relative_reconstruction_error':{
                'median':float(np.median(error)),'maximum':float(max(error)),
                'percentiles_90_99':np.quantile(error,[.9,.99]).tolist()},
            'interval_diagnostics':diagnostics,
            'limits':'Tiny state reconstruction error does not bound amplified return error. This diagnostic cannot distinguish unresolved map structure from amplified curve error or missing state information.'}
    write_json(out/'points.json',points);write_json(out/'curve-errors.json',curves);write_json(out/'audit.json',result)
    manifest=[backup_file(p,out,backup) for p in sorted(out.iterdir()) if p.is_file()]
    write_json(out/'manifest.json',manifest);backup_file(out/'manifest.json',out,backup)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','audit','out','backup'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
