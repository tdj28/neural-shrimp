"""Prepared Task34 exact-dyadic proposal producer; not an interval certificate."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from derivative_profile import validate_cells

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--card',required=True);ap.add_argument('--out',required=True);ap.add_argument('--cell',required=True);a=ap.parse_args()
    c=json.loads(Path(a.card).read_text())
    validate_cells(c['cells'])
    selected=[cell for cell in c['cells'] if cell['name']==a.cell]
    if len(selected)!=1:raise ValueError('unknown or repeated requested cell')
    cell=selected[0]
    prior=json.loads(Path(cell['settle_report']).read_text())
    if any(prior['cell'][k]!=cell[k] for k in ('name','alpha','log10T','period')):raise ValueError('prior exact cell differs')
    if not(prior['diagnostics']['finite_closure_pass'] and prior['diagnostics']['finite_primitive_screen_pass']):raise ValueError('cell not qualified')
    for e in c['inputs']:
        if sha(e['path'])!=e['sha256']:raise ValueError('input hash mismatch')
    states=np.load(cell['states'],allow_pickle=False)
    if states.shape!=(513,896) or not np.isfinite(states).all():raise ValueError('invalid history')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);repo=Path(c['repository'])
    sys.path[:0]=[str(repo/'codex/crosscheck'),str(repo/'codex/certify-llm')]
    import torch
    from qwen_map import ManualQwenSoftMap
    from center_seed import propose_product_and_inverse
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    m=ManualQwenSoftMap.from_pinned_files();cache=m.prefix_cache(cell['alpha']);rows=[]
    with torch.no_grad():
        for p in (cell['period'],):
            folder=out/cell['name'];folder.mkdir();centers=states[-p:].copy();matrices=[];phases=[]
            for k in range(p):
                x=torch.from_numpy(centers[k]);image=m(x,cell['alpha'],cell['log10T'],cache).numpy();residual=image-centers[(k+1)%p]
                rel=float(np.linalg.norm(residual)/np.linalg.norm(centers[(k+1)%p]))
                np.savez(folder/f'phase-{k}-residual.npz',center=centers[k],image=image,residual=residual)
                if not np.isfinite(rel) or rel>=1e-10:raise ValueError('phase closure failed')
                t=time.monotonic();A=m.jacobian(x,cell['alpha'],cell['log10T'],period=1,chunk_size=8,cache=cache).numpy().copy()
                np.save(folder/f'phase-{k}-matrix.npy',A)
                if A.shape!=(896,896) or not np.isfinite(A).all():raise ValueError('invalid phase matrix')
                matrices.append(A);phases.append({'phase':k,'relative_l2_closure':rel,'seconds':time.monotonic()-t})
            A=np.stack(matrices);product,B,left,right,diagnostics=propose_product_and_inverse(A)
            np.savez(folder/'preconditioner.npz',A=A,B=B,centers=centers,parameters=np.array([cell['alpha'],cell['log10T']]),product=product,left_inverse_defect=left,right_inverse_defect=right)
            rows.append({'cell':cell,'period':p,'phases':phases,'diagnostics':diagnostics,'product_order':'A[p-1] @ ... @ A[0]','certificate_passed':False})
    for e in c['inputs']:
        if sha(e['path'])!=e['sha256']:raise ValueError('postrun input changed')
    (out/'report.json').write_text(json.dumps({'rows':rows,'scope':'Finite exact-dyadic proposals only; no derivative enclosure or inverse proof.'},indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
