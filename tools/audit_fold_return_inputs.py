"""Extra independent input/target binding for the completed FGR3 study."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from fold_global_return import sha, write_json, backup_file


def independent_curve(q, d, width):
    v = np.asarray(q)*width
    z = (v-float(d['curve_mid']))/float(d['curve_half'])
    # Clenshaw evaluation, rather than the runner's Vandermonde matrix product.
    transverse = np.polynomial.chebyshev.chebval(z,d['curve_coef']).T
    return d['centre']+v[:,None]*d['right']+transverse


def training_inputs(run, require_complete=True):
    run=Path(run); plan=json.loads((run/'plan.json').read_text())
    parent=json.loads((Path(plan['parent_curve'])/'plan.json').read_text())
    with np.load(run/'inputs.npz') as z:d={k:z[k] for k in z.files}
    total=0; maximum=0.; records=[]
    for selection in sorted(run.glob('round-*-selection.json')):
        rid=int(selection.name.split('-')[1]); selected=json.loads(selection.read_text())['q']
        if rid==0:
            assert selected==np.linspace(*plan['training']['domain'],plan['training']['initial_points']).tolist()
        assert len(selected)==len(set(selected))
        for batch,lo in enumerate(range(0,len(selected),plan['batch_size'])):
            path=run/f'round-{rid:02d}-batch-{batch:03d}'
            if not (path/'result.json').exists():
                if require_complete:raise ValueError('Incomplete training batch')
                continue
            q=selected[lo:lo+plan['batch_size']]
            with np.load(path/'initial.npz') as z:
                initial=z['states']; active=z['active_rows']
            assert np.array_equal(active,np.arange(len(q)))
            expected=independent_curve(q,d,parent['width'])
            error=float(np.max(abs(expected-initial[:len(q)])))
            assert error<1e-14,(path,error)
            maximum=max(maximum,error)
            result=json.loads((path/'result.json').read_text())
            assert len(result['outcomes'])==len(q)
            records.extend({'q':float(value),'batch':path.name,'row':i,**outcome}
                           for i,(value,outcome) in enumerate(zip(q,result['outcomes'])))
            total+=len(q)
    if require_complete:
        model=json.loads((run/'model.json').read_text())
        assert sorted(records,key=lambda r:r['q'])==model['rows']
        assert total==plan['training']['maximum_points']
    return {'rows':total,'maximum_initial_component_error':maximum,
            'independent_Clenshaw_seed_check':True,'table_targets_match_recorded_batch_outcomes':require_complete}


def validation_inputs(run):
    run=Path(run);plan=json.loads((run/'plan.json').read_text())
    selection=json.loads((run/'selection.json').read_text())
    n=plan['validation']['rows'];rng=np.random.default_rng(plan['validation']['seed'])
    s=(np.arange(n)+rng.random(n))/n
    assert selection['s']==s.tolist()
    assert sha(selection['parent_inputs'])==selection['parent_sha256']
    with np.load(selection['parent_inputs']) as z:a,b=z['seeds'][[64,77]]
    expected=a+s[:,None]*(b-a)
    with np.load(run/'trajectories/initial.npz') as z:
        assert np.array_equal(expected,z['states'])
        assert np.array_equal(z['active_rows'],np.arange(n))
    assert sha(run/'frozen-model.json')==selection['model_sha256']
    return {'rows':n,'stratified_draw_reproduced':True,'fresh_initial_states_bitwise_reproduced':True,
            'frozen_table_binding_passed':True}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('training','validation','out','backup'):p.add_argument('--'+name,required=True)
    a=p.parse_args();out=Path(a.out);backup=Path(a.backup)
    out.mkdir(parents=True,exist_ok=False);backup.mkdir(parents=True,exist_ok=False)
    result={'training':training_inputs(a.training),'validation':validation_inputs(a.validation),
            'training_manifest_sha256':sha(Path(a.training)/'manifest.json'),
            'validation_manifest_sha256':sha(Path(a.validation)/'manifest.json'),
            'audit_source_sha256':sha(__file__)}
    write_json(out/'audit.json',result);backup_file(out/'audit.json',out,backup)
    print(json.dumps(result,indent=2))
