"""Post-acquisition independent geometry and primary-decision arithmetic."""
import argparse
import json
from pathlib import Path
import numpy as np


def audit(raw):
    result=json.loads((raw/'analysis.json').read_text())
    plan=json.loads((raw/'manifest.json').read_text())
    classes=[]
    certificates=0
    for i,point in enumerate(plan['points']):
        folder=raw/f'point-{i:02d}'
        geometry=json.loads((folder/'geometry.json').read_text())
        states=[]
        for j,start in enumerate(geometry['starts']):
            c=start['certificate']
            if c['verified'] and c['attracting']:
                m=np.array(c['monodromy'])
                v=np.linalg.eigvals(m)
                neutral=int(np.argmin(abs(v-1)))
                radius=float(max(abs(np.delete(v,neutral))))
                np.testing.assert_allclose(radius,c['transverse_spectral_radius'],rtol=1e-10,atol=1e-10)
                orbit=np.array(c['orbit_states'])
                closure=np.linalg.norm(orbit[-1]-orbit[0])/max(1,np.linalg.norm(orbit[0]))
                np.testing.assert_allclose(closure,c['relative_closure'],rtol=1e-5,atol=1e-12)
                assert radius<1-1e-5 and closure<=1e-8
                assert len(c['interior_maximum_times'])+1==c['peak_count']
                assert min(c['interior_maximum_return_distances'])>1e-7
                states.append('periodic')
                certificates+=1
            else:
                cells=[np.load(folder/f'geometry-{j}-{dt}.npz') for dt in [.01,.005]]
                states.append('irregular-supported' if all(int(z['period'])==0 and z['stats'][1]>.005 for z in cells) else 'unresolved')
        status=states[0] if states[0]==states[1] else 'mixed-or-unresolved'
        assert status==geometry['status']==result['points'][i]['status']
        classes.append(status)
    pairs=[]
    for i in range(25):
        if plan['points'][i]['alpha']==plan['points'][i+1]['alpha'] and set(classes[i:i+2])=={'periodic','irregular-supported'}:
            pairs.append([i,i+1] if classes[i]=='periodic' else [i+1,i])
    assert pairs==result['method_decisions'][1]['pairs']
    for k in range(2):
        differences=[]
        for periodic,irregular in pairs:
            p=np.load(raw/f'point-{periodic:02d}/readout-{k}-1-1-0.npz')['predictions'][:,0]
            q=np.load(raw/f'point-{irregular:02d}/readout-{k}-1-1-0.npz')['predictions'][:,0]
            y=np.tile([-1,-1,1,1],128)
            differences.append(((q-y)**2-(p-y)**2).reshape(128,4).mean(axis=1))
        d=np.mean(differences,axis=0)
        rng=np.random.default_rng(2026092305)
        draws=np.array([d[rng.integers(0,128,128)].mean() for _ in range(4000)])
        ci=np.quantile(draws,[.025,.975])
        saved=result['method_decisions'][k]
        np.testing.assert_allclose(d.mean(),saved['mean_periodic_minus_irregular'],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(ci,saved['conditional_context_interval'],rtol=1e-10,atol=1e-10)
    return dict(geometry_and_decision_arithmetic_passed=True,certificates=certificates,pairs=pairs,
                decision=result['decision'])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('raw',type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.raw),indent=2))
