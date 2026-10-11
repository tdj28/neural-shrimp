"""Independent scalar-erf reconstruction of information bounds and aggregates."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np


def reference(means,sigma,labels):
    minus=[i for i in range(4) if labels[i]<0]
    plus=[i for i in range(4) if labels[i]>0]
    cdf=lambda x:.5*math.erfc(-x/math.sqrt(2))
    distances={(i,j):math.sqrt(sum((float(x)-float(y))**2 for x,y in zip(means[i],means[j]))) for i in minus for j in plus}
    upper=min(sum(cdf(distances[(minus[k],plus[(k+shift)%2])]/(2*sigma)) for k in range(2))/2 for shift in [0,1])
    nearest=1-sum(cdf(-x/(2*sigma)) for x in distances.values())/2
    positive=(means[plus[0]]+means[plus[1]])/2
    negative=(means[minus[0]]+means[minus[1]])/2
    weights=positive-negative;center=(positive+negative)/2
    norm=math.sqrt(sum(float(x)**2 for x in weights))
    linear=.5 if norm==0 else sum(cdf(labels[i]*sum(float(w)*float(x-c) for w,x,c in zip(weights,means[i],center))/(sigma*norm)) for i in range(4))/4
    return [max(.5,nearest,linear),upper,linear,nearest,min(distances.values())]


def audit(raw):
    receipt=json.loads((raw/'receipt.json').read_text())
    for entry in receipt['artifacts']:
        f=raw/entry['path'];assert f.stat().st_size==entry['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==entry['sha256']
    p=json.loads((raw/'manifest.json').read_text());parent=Path(p['parent'])
    summary=json.loads((raw/'analysis.json').read_text())
    tests=np.r_[32:96,128:192];labels=np.array([[-1,-1,1],[-1,1,-1],[1,-1,-1],[1,1,1]])
    maxerror=0.;checked=0
    for i in range(26):
        for ai in range(2):
            traces=[np.load(parent/f'point-{i:02d}/traces-{k}-{ai}.npy',mmap_mode='r') for k in range(2)]
            for wi,(lo,hi) in enumerate(p['windows']):
                for si,sigma in enumerate(p['sigmas']):
                    saved=np.load(raw/f'point-{i:02d}-{ai}-{wi}-{si}.npy')
                    # All primary contexts; fixed context subset for every secondary condition.
                    contexts=range(128) if ai==1 and wi==1 and si==1 else [0,32,64,127]
                    for k in range(2):
                        for c in contexts:
                            means=np.tanh(traces[k][tests[c],:,lo:hi+1]).reshape(4,-1)
                            for j in range(3):
                                alternative=np.array(reference(means,sigma,labels[:,j]))
                                error=float(np.max(abs(alternative-saved[k,c,j])))
                                maxerror=max(maxerror,error)
                                np.testing.assert_allclose(alternative,saved[k,c,j],atol=2e-12,rtol=2e-12)
                                checked+=1
                    row=summary['points'][i]['records'][ai*6+wi*3+si]
                    np.testing.assert_allclose(np.min(saved[:,:,:,0],axis=0).mean(axis=0),row['lower_mean'],atol=1e-14)
                    np.testing.assert_array_equal((np.min(saved[:,:,:,0],axis=0)>=.95).mean(axis=0),row['high_fraction'])
                    np.testing.assert_array_equal((np.max(saved[:,:,:,1],axis=0)<=.55).mean(axis=0),row['low_fraction'])
    return dict(independent_bounds_audit_passed=True,bound_vectors_checked=checked,max_discrepancy=maxerror,
                receipt_sha256=hashlib.sha256((raw/'receipt.json').read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('raw',type=Path)
    args=parser.parse_args();print(json.dumps(audit(args.raw),indent=2))
