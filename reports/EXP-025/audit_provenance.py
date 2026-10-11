"""Verify selective refinement provenance and independently recompute repeatability."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(raw):
    receipt=json.loads((raw/'receipt.json').read_text())
    parent=Path(receipt['parent'])
    assert hashlib.sha256((parent/'receipt.json').read_bytes()).hexdigest()==receipt['parent_receipt_sha256']
    methods=json.loads((raw/'method-provenance.json').read_text())
    summary=json.loads((raw/'analysis.json').read_text())
    preserved=0
    repeat_means=[]
    maximum_relative_difference=0.
    for i in range(26):
        folder=raw/f'point-{i:02d}'
        old=parent/f'point-{i:02d}'
        changed=[x['context'] for x in methods['refined_contexts'] if x['point']==i]
        unchanged=sorted(set(range(192))-set(changed))
        for f in folder.iterdir():
            if f.name.startswith(('geometry','repeat-')) or f.name in ['contexts.npz','terminal_states.npy','independent.npy']:
                assert f.read_bytes()==(old/f.name).read_bytes(),str(f)
                preserved+=1
        for k in range(2):
            for a in range(2):
                new=np.load(folder/f'traces-{k}-{a}.npy')
                original=np.load(old/f'traces-{k}-{a}.npy')
                np.testing.assert_array_equal(new[unchanged],original[unchanged])
        values=[]
        for k in range(2):
            traces=np.load(folder/f'repeat-{k}.npy')
            # Independent scalar formula for activation distance.
            d=np.sqrt(np.sum((np.tanh(traces[:,0,-1,:])-np.tanh(traces[:,1,-1,:]))**2,axis=1))
            d=np.r_[d[32:96],d[128:192]]
            mean=float(d.sum()/128)
            np.testing.assert_allclose(mean,summary['points'][i]['repeatability'][k]['final_mean'],rtol=1e-12,atol=1e-15)
            values.append(mean)
        repeat_means.append(values)
        maximum_relative_difference=max(maximum_relative_difference,abs(values[0]/values[1]-1))
    pairs=summary['method_decisions'][1]['pairs']
    ratios=[repeat_means[q][1]/repeat_means[p][1] for p,q in pairs]
    periodic=[repeat_means[i][1] for i,p in enumerate(summary['points']) if p['status']=='periodic']
    irregular=[repeat_means[i][1] for i,p in enumerate(summary['points']) if p['status']=='irregular-supported']
    return dict(selective_refinement_provenance_passed=True,unchanged_files=preserved,
        refined_contexts=len(methods['refined_contexts']),repeatability_arithmetic_passed=True,
        max_repeatability_mean_relative_step_difference=maximum_relative_difference,
        crossing_repeatability_ratios_irregular_over_periodic=ratios,
        every_periodic_setting_more_repeatable_than_every_irregular_setting=max(periodic)<min(irregular),
        periodic_mean_separation_range=[min(periodic),max(periodic)],
        irregular_mean_separation_range=[min(irregular),max(irregular)])


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('raw',type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.raw),indent=2))
