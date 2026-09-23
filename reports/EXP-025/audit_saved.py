"""Independent feature/OLS arithmetic from EXP-025 saved states; no new IVPs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(directory):
    receipt = json.loads((directory/'receipt.json').read_text())
    for entry in receipt['artifacts']:
        f = directory/entry['path']
        assert f.stat().st_size == entry['bytes']
        assert hashlib.sha256(f.read_bytes()).hexdigest() == entry['sha256']
    plan = json.loads((directory/'manifest.json').read_text())
    summary = json.loads((directory/'analysis.json').read_text())
    tr = list(range(32))+list(range(96,128))
    te = list(range(32,96))+list(range(128,192))
    pattern = np.array([[-1,-1,1],[-1,1,-1],[1,-1,-1],[1,1,1]])
    ytrain, ytest = np.tile(pattern,(64,1)), np.tile(pattern,(128,1))
    maximum = 0.
    count = 0
    for i in range(26):
        folder = directory/f'point-{i:02d}'
        for k in range(2):
            for ai in range(2):
                states = np.load(folder/f'traces-{k}-{ai}.npy', mmap_mode='r')
                for wi, (lo,hi) in enumerate(plan['windows']):
                    raw = np.tanh(states[:,:,lo:hi+1]).reshape(192,4,-1)
                    for quadratic in range(2):
                        z = raw
                        if quadratic:
                            products = np.stack([raw[:,:,a]*raw[:,:,b] for a in range(raw.shape[-1])
                                                 for b in range(a,raw.shape[-1])],axis=-1)
                            z = np.concatenate([raw,products],axis=-1)
                        train,test = z[tr].reshape(256,-1),z[te].reshape(512,-1)
                        saved = np.load(folder/f'readout-{k}-{ai}-{wi}-{quadratic}.npz')
                        pred = test @ saved['weights']+saved['intercept']
                        loss = ((pred-ytest)**2).reshape(128,4,3).mean(axis=1)
                        np.testing.assert_allclose(loss,saved['context_loss'],rtol=1e-11,atol=1e-11)
                        row = summary['points'][i]['assays'][ai*4+wi*2+quadratic]
                        np.testing.assert_allclose(1-loss.mean(axis=0), row['r2'][k],rtol=1e-11,atol=1e-11)
                        # Independent augmented least squares for all affine fits
                        # and the primary-amplitude late quadratic fit at fine step.
                        if not quadratic or (k==1 and ai==1 and wi==1):
                            design = np.column_stack((np.ones(256),train))
                            penalty = np.eye(design.shape[1])*np.sqrt(256*plan['ridge'])
                            penalty[0,0] = 0.
                            aug = np.vstack((design,penalty))
                            targets = np.vstack((ytrain,np.zeros((len(penalty),3))))
                            coef = np.linalg.lstsq(aug,targets,rcond=None)[0]
                            alt = np.column_stack((np.ones(512),test))@coef
                            error = float(np.max(np.abs(alt-pred)))
                            maximum = max(maximum,error)
                            np.testing.assert_allclose(alt,pred,rtol=1e-7,atol=1e-7)
                        count += 1
    return dict(independent_arithmetic_passed=True, readouts_checked=count,
                max_prediction_difference=maximum, source=receipt['source_commit'],
                receipt_sha256=hashlib.sha256((directory/'receipt.json').read_bytes()).hexdigest())


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    print(json.dumps(audit(args.directory),indent=2))
