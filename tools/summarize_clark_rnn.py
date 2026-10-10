#!/usr/bin/env python3
"""Saved-only maps/component inventory; no trajectory or new decomposition."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def components(periods,positive):
    seen=set();out=[];ng,nd=periods.shape
    for i in range(ng):
        for k in range(nd):
            if (i,k) in seen or periods[i,k]<3:continue
            p=int(periods[i,k]);stack=[(i,k)];seen.add((i,k));cells=[];boundary=set()
            while stack:
                a,b=stack.pop();cells.append((a,b))
                for u,v in [(a-1,b),(a+1,b),(a,b-1),(a,b+1)]:
                    if 0<=u<ng and 0<=v<nd:
                        if periods[u,v]==p:
                            if (u,v) not in seen:seen.add((u,v));stack.append((u,v))
                        else:boundary.add((u,v))
            out.append({'period':p,'cells':sorted(cells),'size':len(cells),'touches_grid_edge':any(a in [0,ng-1] or b in [0,nd-1] for a,b in cells),'neighbor_positive_cells':sorted([x for x in boundary if positive[x]]),'boundary_cells':sorted(boundary)})
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--manifest-sha256',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();start=time.monotonic()
    if sha(a.raw/'manifest.json')!=a.manifest_sha256:raise ValueError('raw pin')
    inv=json.loads((a.raw/'manifest.json').read_text())['files']
    if set(inv)!={str(p.relative_to(a.raw)) for p in a.raw.rglob('*') if p.is_file() and p!=a.raw/'manifest.json'}:raise ValueError('inventory')
    for f,d in inv.items():
        p=a.raw/f
        if p.is_symlink() or sha(p)!=d['sha256'] or p.stat().st_size!=d['bytes']:raise ValueError('raw bytes')
    a.out.mkdir(parents=True,exist_ok=False)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True);reports=[]
    for col,n in enumerate([4,8,16]):
        for row,seed in enumerate([28170,28171]):
            folder=a.raw/'run'/f'n{n}-s{seed}'
            with np.load(folder/'batch.npz') as f:
                periods=f['period_proposals'].reshape(21,21,2);grid=f['grid'].reshape(21,21,2,2);grow=f['growth_blocks'].reshape(2,8,21,21,2).sum(axis=1)/2048/grid[:,:,0,1][None,:,:,None];positive=np.all(grow>.002,axis=(0,3));collapse=f['tangent_collapse_count'].reshape(21,21,2);positive &= np.all(collapse==0,axis=2)
            agree=periods[:,:,0]==periods[:,:,1];consensus=np.where(agree,periods[:,:,0],-1);comps=components(consensus,positive)
            qualifies={};duplicate=[]
            for f in sorted(folder.glob('cycle-*.json')):
                if '-event-' in f.name:continue
                q=json.loads(f.read_text());qualifies[q['index']]=q
            for i in sorted(qualifies):
                if i%2 or i+1 not in qualifies:continue
                q=qualifies[i];z=qualifies[i+1]
                if q['status']!='numerical_attracting_primitive' or z['status']!='numerical_attracting_primitive' or q['period']!=z['period']:continue
                c=np.load(folder/f'cycle-{i:04d}.npz')['cycle'];d=np.load(folder/f'cycle-{i+1:04d}.npz')['cycle'];scale=max(1,float(np.max(np.abs(c))),float(np.max(np.abs(d))))
                best=min((float(np.max(np.abs(c-sign*np.roll(d,k,axis=0)))/scale),sign,k) for sign in [1,-1] for k in range(len(c)))
                duplicate.append({'even_index':i,'odd_index':i+1,'phase_sign_match_error':best[0],'sign':best[1],'shift':best[2]})
            for comp in comps:
                comp['gain_range']=[float(grid[x[0],x[1],0,0]) for x in [min(comp['cells']),max(comp['cells'])]]
                ds=[float(grid[i,k,0,1]) for i,k in comp['cells']];comp['delta_range']=[min(ds),max(ds)]
                comp['qualified_seed_cases']=sum(qualifies.get(2*(i*21+k)+v,{}).get('status')=='numerical_attracting_primitive' for i,k in comp['cells'] for v in [0,1])
            reports.append({'n':n,'seed':seed,'two_start_period_disagreements':int(np.sum(~agree)),'components':comps,'two_start_phase_sign_matches':duplicate})
            ax=axes[row,col];masked=np.ma.masked_where(consensus<=0,np.log2(np.maximum(consensus,1)));im=ax.pcolormesh(grid[:,:,0,0],grid[:,:,0,1],masked,shading='nearest',vmin=0,vmax=6,cmap='viridis');ax.set_xscale('log');ax.set_facecolor('#dddddd');ax.set_title(f'N={n}, seed={seed}');ax.set_xlabel('gain g');ax.set_ylabel('leak / Euler step δ')
            gx=grid[:,:,0,0];dx=grid[:,:,0,1];ax.scatter(gx[positive],dx[positive],s=8,c='black',marker='.',label='both halves positive (finite time)');ax.scatter(gx[~agree],dx[~agree],s=12,c='red',marker='x',linewidths=.5,label='starts disagree')
    cb=fig.colorbar(im,ax=axes.ravel().tolist(),ticks=np.log2([1,2,4,8,16,32,64]));cb.ax.set_yticklabels(['1','2','4','8','16','32','64']);cb.set_label('full-state recurrence period proposal (two-start agreement)')
    fig.suptitle('Exploratory fixed random RNN panel; colored cells are not shrimp claims\nGray: no common period ≤64; black: finite-time positive growth; red ×: starts disagree',fontsize=13)
    fig.savefig(a.out/'panel.png',dpi=150);fig.savefig(a.out/'panel.pdf');plt.close(fig)
    result={'schema':'clark-rnn-saved-map-v1','raw':str(a.raw),'raw_manifest_sha256':a.manifest_sha256,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source_sha256':sha(Path(__file__)),'elapsed_seconds':time.monotonic()-start,'scope':'Descriptive post-outcome components. No new model/eigen/SVD/correction; no shrimp, chaos or continuous-time proof.','networks':reports}
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');files={str(p.relative_to(a.out)):{'sha256':sha(p),'bytes':p.stat().st_size} for p in a.out.iterdir() if p.is_file()};(a.out/'manifest.json').write_text(json.dumps({'files':files},indent=2)+'\n')
    for p in a.out.iterdir():p.chmod(0o444)
    a.out.chmod(0o555)
    print(json.dumps({'elapsed_seconds':result['elapsed_seconds'],'components':sum(len(x['components']) for x in reports),'out':str(a.out)}))
if __name__=='__main__':main()
