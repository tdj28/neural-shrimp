"""Full composed float64 monodromy and finite numerical stability diagnostics."""
import argparse, hashlib, json, resource, sys, time
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def spectrum(matrix):
    vals,vecs=np.linalg.eig(matrix)
    if not np.isfinite(vals).all() or not np.isfinite(vecs).all():raise ValueError('nonfinite eigensystem')
    order=np.argsort(-np.abs(vals),kind='stable');vals=vals[order];vecs=vecs[:,order]
    residuals=np.max(np.abs(matrix@vecs-vecs*vals[None,:]),axis=0)
    return vals,vecs,residuals

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    c=json.loads(Path(a.card).read_text())
    if c['alpha']!=.06468 or c['log10T']!=-.148:raise ValueError('wrong exact cell')
    for e in c['inputs']:
        if sha(e['path'])!=e['sha256']:raise ValueError('input hash mismatch')
    states=np.load(c['states'],allow_pickle=False)
    if states.shape!=(513,2,896) or not np.isfinite(states).all():raise ValueError('invalid state history')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(Path(c['repository'])/'codex/crosscheck'))
    import torch
    from qwen_map import ManualQwenSoftMap
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    start=time.monotonic();m=ManualQwenSoftMap.from_pinned_files();cache=m.prefix_cache(c['alpha']);rows=[]
    with torch.no_grad():
        for j,p in enumerate((2,3)):
            folder=out/('p'+str(p));folder.mkdir();x=torch.from_numpy(states[-p,j].copy());eye=torch.eye(896,dtype=torch.float64);blocks=[];times=[]
            for k in range(0,896,8):
                t=time.monotonic();v=m.jvp(x[None,:].expand(8,-1).clone(),eye[k:k+8],c['alpha'],c['log10T'],p,cache)
                if v.shape!=(8,896) or not torch.isfinite(v).all():raise ValueError('invalid derivative block')
                b=v.numpy().copy();np.save(folder/f'columns-{k:03d}.npy',b);blocks.append(b);times.append(time.monotonic()-t)
                with (out/'events.jsonl').open('a') as f:f.write(json.dumps({'period':p,'columns_complete':k+8,'elapsed_seconds':time.monotonic()-start})+'\n')
            matrix=np.concatenate(blocks,axis=0).T.copy();np.save(folder/'monodromy.npy',matrix)
            t=time.monotonic();vals,vecs,residuals=spectrum(matrix);eigentime=time.monotonic()-t
            np.savez(folder/'eigensystem.npz',values=vals,vectors=vecs,residuals=residuals)
            critical=vecs[:,0];checks=[]
            # Split a complex vector into real directions, preserving both pieces.
            for label,vn in [('real',critical.real),('imag',critical.imag)]:
                norm=float(np.linalg.norm(vn))
                if norm==0:checks.append({'part':label,'zero_component':True});continue
                vn=vn/norm;v=torch.from_numpy(vn.copy());direct=m.jvp(x,v,c['alpha'],c['log10T'],p,cache).numpy();dense=matrix@vn
                fdrows=[];arrays={'direction':vn,'direct':direct,'dense':dense}
                for h in (1e-2,3e-3,1e-3,3e-4,1e-4,3e-5,1e-5):
                    plus=m.period_map(x+h*v,c['alpha'],c['log10T'],p,cache).numpy();minus=m.period_map(x-h*v,c['alpha'],c['log10T'],p,cache).numpy();arrays['plus_'+str(h)]=plus;arrays['minus_'+str(h)]=minus
                    fdrows.append({'h':h,'absolute_inf_error':float(np.max(np.abs((plus-minus)/(2*h)-direct)))})
                if not all(np.isfinite(v).all() for v in arrays.values()):raise ValueError('nonfinite critical check')
                np.savez(folder/('critical-'+label+'.npz'),**arrays)
                checks.append({'part':label,'zero_component':False,'dense_vs_direct_inf':float(np.max(np.abs(dense-direct))),'fd_ladder':fdrows})
            direct_complex=np.zeros(896,dtype=complex)
            for vn,factor in [(critical.real,1),(critical.imag,1j)]:
                if np.linalg.norm(vn)>0:direct_complex+=factor*m.jvp(x,torch.from_numpy(vn.copy()),c['alpha'],c['log10T'],p,cache).numpy()
            np.save(folder/'critical-direct-complex.npy',direct_complex)
            image=m.period_map(x,c['alpha'],c['log10T'],p,cache).numpy();cycle=states[-p:,j];sep=[float(np.max(np.abs(cycle[k]-cycle[(k+1)%p]))) for k in range(p)]
            row={'period':p,'state_index':513-p,'columns':896,'dense_seconds':sum(times),'eigen_seconds':eigentime,'spectral_radius':float(abs(vals[0])),'next_modulus':float(abs(vals[1])),'critical_multiplier':[float(vals[0].real),float(vals[0].imag)],'critical_direct_eigen_residual_inf':float(np.max(np.abs(direct_complex-vals[0]*critical))),'maximum_eigen_residual_inf':float(residuals.max()),'closure_inf':float(np.max(np.abs(image-x.numpy()))),'adjacent_separations':sep,'critical_checks':checks,'numerically_inside_unit_circle':bool(np.max(np.abs(vals))<1),'scope':'Finite float64 numerical diagnostic; no rigorous stability/root certificate.'}
            (folder/'summary.json').write_text(json.dumps(row,indent=2,allow_nan=False)+'\n');rows.append(row)
    for e in c['inputs']:
        if sha(e['path'])!=e['sha256']:raise ValueError('postrun input changed')
    (out/'report.json').write_text(json.dumps({'rows':rows,'elapsed_seconds':time.monotonic()-start,'rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss},indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
