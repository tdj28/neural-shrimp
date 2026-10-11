#!/usr/bin/env python3
"""All-four saved N12 maps and component inventory; no target acquisition."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
RAW=Path('/data2/neural-shrimp-data/explore/clark-rnn/n12-matrices-panel-001')
PIN='44edb885b2ba884fdcb229eeabf4d3ae133e8cffc7a6fe289af5a530138f87d3'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();start=time.monotonic()
    if sha(RAW/'manifest.json')!=PIN:raise ValueError('pin')
    inv=json.loads((RAW/'manifest.json').read_text())['files']
    if set(inv)!={str(p.relative_to(RAW)) for p in RAW.rglob('*') if p.is_file() and p!=RAW/'manifest.json'}:raise ValueError('inventory')
    for name,d in inv.items():
        p=RAW/name
        if p.is_symlink() or sha(p)!=d['sha256'] or p.stat().st_size!=d['bytes']:raise ValueError('bytes')
    a.out.mkdir(parents=True,exist_ok=False)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True);reports=[];allc=[]
    for ax,seed in zip(axes.ravel(),range(28191,28195)):
        folder=RAW/'run'/f'n12-s{seed}'
        with np.load(folder/'batch.npz') as f:
            periods=f['period_proposals'].reshape(31,31,2);grid=f['grid'].reshape(31,31,2,2)
            halves=f['growth_blocks'].reshape(2,16,31,31,2).sum(axis=1)/4096/grid[:,:,0,1][None,:,:,None]
            pos=np.all(halves>.002,axis=(0,3))&np.all(f['tangent_collapse_count'].reshape(31,31,2)==0,axis=2)&np.all(periods==0,axis=2)
        consensus=np.where(periods[:,:,0]==periods[:,:,1],periods[:,:,0],-1);cs=json.loads((folder/'components.json').read_text())['components']
        qs={q['index']:q for f in folder.glob('cycle-*.json') if '-event-' not in f.name for q in [json.loads(f.read_text())]}
        dup=[]
        for i,q in sorted(qs.items()):
            if i%2 or i+1 not in qs or q['status']!='numerical_attracting_primitive' or qs[i+1]['status']!='numerical_attracting_primitive':continue
            with np.load(folder/f'cycle-{i:05d}.npz') as f:c=f['cycle']
            with np.load(folder/f'cycle-{i+1:05d}.npz') as f:d=f['cycle']
            if c.shape!=d.shape:raise ValueError('two-start period mismatch')
            scale=max(1,float(np.max(np.abs(c))),float(np.max(np.abs(d))))
            best=min((float(np.max(np.abs(c-sign*np.roll(d,k,axis=0)))/scale),sign,k) for sign in [1,-1] for k in range(len(c)))
            dup.append({'cell':i//2,'error':best[0],'sign':best[1],'shift':best[2]})
        for c in cs:
            ij=[divmod(k,31) for k in c['cells']];c['seed']=seed;c['gain_range']=[min(float(grid[i,j,0,0]) for i,j in ij),max(float(grid[i,j,0,0]) for i,j in ij)];c['delta_range']=[min(float(grid[i,j,0,1]) for i,j in ij),max(float(grid[i,j,0,1]) for i,j in ij)];c['grid_bounds']=[min(i for i,j in ij),max(i for i,j in ij),min(j for i,j in ij),max(j for i,j in ij)];c['qualified_seed_cases']=sum(qs.get(2*k+i,{}).get('status')=='numerical_attracting_primitive' for k in c['qualification_cells'] for i in [0,1]);allc.append(c)
        reports.append({'seed':seed,'components':cs,'two_start_phase_sign_distances':dup,'two_start_period_disagreements':int(np.sum(consensus==-1))})
        image=ax.pcolormesh(grid[:,:,0,0],grid[:,:,0,1],np.ma.masked_where(consensus<=0,np.log2(np.maximum(consensus,1))),shading='nearest',vmin=0,vmax=6,cmap='viridis')
        ax.set_xscale('log');ax.set_facecolor('#ddd');ax.scatter(grid[:,:,0,0][pos],grid[:,:,0,1][pos],s=6,color='black');ax.scatter(grid[:,:,0,0][consensus==-1],grid[:,:,0,1][consensus==-1],s=8,color='#d54b4b',marker='x',linewidths=.5)
        ax.set(title=f'N12, seed {seed}',xlabel='gain g',ylabel='leak / Euler step δ')
    colorbar=fig.colorbar(image,ax=axes.ravel().tolist(),ticks=np.log2([1,2,4,8,16,32,64]));colorbar.ax.set_yticklabels(['1','2','4','8','16','32','64']);colorbar.set_label('Two-start full-state period proposal')
    fig.suptitle('Four fixed new N12 random RNNs: all cells and components retained\nBlack: both-half, both-start finite positive growth; red ×: starts disagree',fontsize=12)
    fig.savefig(a.out/'panel.png',dpi=160);fig.savefig(a.out/'panel.pdf');plt.close(fig)
    ranked=sorted(allc,key=lambda c:(-(c['grid_edge_margin']>=2),-(c['halo_both_start_positive_growth']>=3),-c['count'],c['period'],c['seed'],c['cells'][0]))
    r={'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':sha(Path(__file__)),'raw':str(RAW),'manifest_sha256':PIN,'networks':reports,'ranked_component_keys':[[c['seed'],c['id']] for c in ranked],'leading_candidate':ranked[0],'elapsed_seconds':time.monotonic()-start,'scope':'Saved descriptive components and selected seed qualifications, not verified windows/shrimps/chaos. No model/derivative/eigen/SVD/solver/trajectory call.'}
    (a.out/'result.json').write_text(json.dumps(r,indent=2)+'\n');inv={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in a.out.iterdir() if p.is_file()};(a.out/'manifest.json').write_text(json.dumps({'files':inv},indent=2)+'\n')
    for p in a.out.iterdir():p.chmod(0o444)
    a.out.chmod(0o555);print(json.dumps({'candidate':r['leading_candidate'],'elapsed_seconds':r['elapsed_seconds']}))
if __name__=='__main__':main()
