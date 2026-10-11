#!/usr/bin/env python3
"""Saved native half-return arithmetic only; no model or decomposition."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=Path('/data2/neural-shrimp-data/explore/clark-rnn')
PINS={'new-p10-boundary-slices-001':'18fa07a209e4697aeb3ec4159fff57867a1d1bfa3f4f1f5042dc6288cc308785','new-p10-slice-extension-001':'e79ad6abe815a35f7db015f6d013217aa123a494f088cf0746c04499629adc59'}
SELECT=[('new-p10-boundary-slices-001','boundary_result','g46-dir1-boundary'),('new-p10-slice-extension-001','boundary_result','extend-g42-dir-1-step159-fold'),('new-p10-slice-extension-001','boundary_result','extend-g44-dir-1-step107-fold'),('new-p10-slice-extension-001','native','extend-g42-dir1-step384'),('new-p10-slice-extension-001','native','extend-g44-dir1-step171'),('new-p10-slice-extension-001','native','extend-g44-dir1-step172'),('new-p10-slice-extension-001','native','extend-g46-dir-1-step4')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def half_metrics(cycle,js,m,q=None):
    h=np.eye(len(m));full=np.eye(len(m))
    for j in js[:5]:h=j@h
    for j in js:full=j@full
    scale=max(1,float(np.max(np.abs(cycle))));out={'half_return':h,'native_product':full,'antipodal_defect':float(np.max(np.abs(cycle+np.roll(cycle,5,axis=0)))/scale),'phase_jacobian_repetition_defect':float(np.max(np.abs(js[:5]-js[5:]))),'native_product_relative_error':float(np.linalg.norm(full-m,np.inf)/max(1,np.linalg.norm(m,np.inf))),'square_relative_defect':float(np.linalg.norm(m-h@h,np.inf)/max(1,np.linalg.norm(m,np.inf)))}
    if q is not None:out.update(half_mode_plus_residual=float(np.max(np.abs(h@q-q))/max(np.max(np.abs(q)),1e-100)),half_mode_minus_residual=float(np.max(np.abs(h@q+q))/max(np.max(np.abs(q)),1e-100)))
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();start=time.monotonic();a.out.mkdir(parents=True,exist_ok=False);rows=[]
    for name,pin in PINS.items():
        raw=BASE/name
        if sha(raw/'manifest.json')!=pin:raise ValueError('pin')
        inv=json.loads((raw/'manifest.json').read_text())['files']
        if set(inv)!={str(p.relative_to(raw)) for p in raw.rglob('*') if p.is_file() and p!=raw/'manifest.json'}:raise ValueError('inventory')
        for n,d in inv.items():
            p=raw/n
            if p.is_symlink() or sha(p)!=d['sha256'] or p.stat().st_size!=d['bytes']:raise ValueError('bytes')
    for index,(name,kind,tag) in enumerate(SELECT):
        files=[p for p in (BASE/name/'run').glob('*-'+kind+'.json') if json.loads(p.read_text())['tag']==tag]
        if not files:raise ValueError('fixed record missing')
        p=sorted(files)[-1];meta=json.loads(p.read_text());arr=p.with_name(meta['array_file'])
        if sha(arr)!=meta['array_sha256']:raise ValueError('array binding')
        with np.load(arr) as f:v={k:f[k].copy() for k in f.files}
        r=half_metrics(v['cycle'],v['phase_jacobians'],v['monodromy'],v.get('right_vector'))
        np.savez(a.out/f'half-{index}.npz',half_return=r.pop('half_return'),native_product=r.pop('native_product'))
        rows.append(dict(r,tag=tag,source_record=str(p),source_record_sha256=sha(p),source_array_sha256=sha(arr),selected_last_return=True,scope='Descriptive saved arithmetic; native records include the explicitly named unaccepted step172 proposal. No root/eigensystem recomputation.'))
    result={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':sha(Path(__file__)),'pins':PINS,'rows':rows,'elapsed_seconds':time.monotonic()-start,'scope':'M=H² is exact only under exact antipodal half-period symmetry. Floating defects diagnose consistency, not an exact symmetry theorem or bifurcation certificate. Fixed points of G=-F5 are symmetric primitive10 cycles; G is not the original F.'}
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');inv={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in a.out.iterdir() if p.is_file()};(a.out/'manifest.json').write_text(json.dumps({'files':inv},indent=2)+'\n')
    for p in a.out.iterdir():p.chmod(0o444)
    a.out.chmod(0o555);print(json.dumps(result))
if __name__=='__main__':main()
