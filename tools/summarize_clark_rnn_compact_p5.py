#!/usr/bin/env python3
"""Saved-only adapted refinement geometry inventory; no model or decomposition."""
import argparse,hashlib,importlib.util,json,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('old_summary',ROOT/'tools/summarize_clark_rnn.py');M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
sha=M.sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--raw',type=Path,required=True);ap.add_argument('--manifest-sha256',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();started=time.monotonic()
    if sha(a.raw/'manifest.json')!=a.manifest_sha256:raise ValueError('manifest pin')
    inv=json.loads((a.raw/'manifest.json').read_text())['files']
    if set(inv)!={str(p.relative_to(a.raw)) for p in a.raw.rglob('*') if p.is_file() and p!=a.raw/'manifest.json'}:raise ValueError('inventory')
    for name,d in inv.items():
        p=a.raw/name
        if Path(name).is_absolute() or '..' in Path(name).parts or p.is_symlink() or sha(p)!=d['sha256'] or p.stat().st_size!=d['bytes']:raise ValueError('raw identity')
    a.out.mkdir(parents=True,exist_ok=False)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5),constrained_layout=True);axes=np.atleast_2d(axes);reports=[]
    for row,kind in enumerate(['compact-p5']):
        folder=a.raw/'run'
        with np.load(folder/'inputs.npz') as f:gains=f['gains'];deltas=f['deltas']
        ng,nd=len(gains),len(deltas)
        with np.load(folder/'batch.npz') as f:
            p=f['period_proposals'].reshape(ng,nd,2);grid=f['grid'].reshape(ng,nd,2,2)
            halves=f['growth_blocks'].reshape(2,16,ng,nd,2).sum(axis=1)/4096/grid[:,:,0,1][None,:,:,None]
            collapse=f['tangent_collapse_count'].reshape(ng,nd,2)
        same=p[:,:,0]==p[:,:,1];consensus=np.where(same,p[:,:,0],-1);least=np.min(halves,axis=(0,3));positive=(least>.002)&np.all(collapse==0,axis=2)
        comps=M.components(consensus,positive);qualified={};duplicates=[];records=[]
        for path in sorted(folder.glob('cycle-*.json')):
            if '-event-' in path.name:continue
            q=json.loads(path.read_text());qualified[q['index']]=q
            record={k:q.get(k) for k in ['index','gain','delta','initial','period','status','radius','closure','closure_adjacent','eigen_residual','proper_divisor_separations']}
            with np.load(path.with_suffix('.npz')) as f:
                if 'eigenvalues' in f and 'eigenvalues_adjacent' in f:
                    e=f['eigenvalues'];e1=f['eigenvalues_adjacent'];record.update(multiplier_distance_plus_one=float(min(np.min(np.abs(e-1)),np.min(np.abs(e1-1)))),multiplier_distance_minus_one=float(min(np.min(np.abs(e+1)),np.min(np.abs(e1+1)))),eigenvalues=[[float(z.real),float(z.imag)] for z in e])
            records.append(record)
        for i,q in sorted(qualified.items()):
            if i%2 or i+1 not in qualified:continue
            r=qualified[i+1]
            if q['status']!='numerical_attracting_primitive' or r['status']!='numerical_attracting_primitive' or q['period']!=r['period']:continue
            with np.load(folder/f'cycle-{i:05d}.npz') as f:c=f['cycle']
            with np.load(folder/f'cycle-{i+1:05d}.npz') as f:d=f['cycle']
            scale=max(1,float(np.max(np.abs(c))),float(np.max(np.abs(d))))
            err,sign,shift=min((float(np.max(np.abs(c-sgn*np.roll(d,k,axis=0)))/scale),sgn,k) for sgn in [1,-1] for k in range(len(c)))
            duplicates.append({'even_index':i,'odd_index':i+1,'phase_sign_error':err,'sign':sign,'shift':shift})
        for c in comps:
            c['gain_range']=[min(float(gains[i]) for i,k in c['cells']),max(float(gains[i]) for i,k in c['cells'])];c['delta_range']=[min(float(deltas[k]) for i,k in c['cells']),max(float(deltas[k]) for i,k in c['cells'])]
            c['qualified_seed_cases']=sum(qualified.get(2*(i*nd+k)+v,{}).get('status')=='numerical_attracting_primitive' for i,k in c['cells'] for v in [0,1])
        reports.append({'kind':kind,'shape':[ng,nd],'components':comps,'records':records,'two_start_phase_sign_matches':duplicates,'positive_consensus_cells':int(np.sum(positive)),'start_period_disagreements':int(np.sum(~same))})
        gx,dx=grid[:,:,0,0],grid[:,:,0,1]
        ax=axes[row,0];im=ax.pcolormesh(gx,dx,np.ma.masked_where(consensus<=0,np.log2(np.maximum(consensus,1))),shading='nearest',vmin=0,vmax=6,cmap='viridis');ax.set_facecolor('#dddddd');ax.scatter(gx[~same],dx[~same],s=8,c='red',marker='x',linewidths=.4)
        inds=np.array([i for i,q in qualified.items() if i%2==0 and q['status']=='numerical_attracting_primitive']);cells=inds//2
        ax.scatter(gx.ravel()[cells],dx.ravel()[cells],s=19,facecolors='none',edgecolors='white',linewidths=.6)
        ax.set_title(f'{kind} patch: two-start full-state period; white = corrected')
        cb=fig.colorbar(im,ax=ax,ticks=np.log2([3,5,7,10,20,64]));cb.ax.set_yticklabels(['3','5','7','10','20','64'])
        ax=axes[row,1];im=ax.pcolormesh(gx,dx,least,shading='nearest',cmap='coolwarm',vmin=-.05,vmax=.05);ax.set_title(f'{kind}: minimum over two starts and two LE halves');fig.colorbar(im,ax=ax,extend='both',label='finite-time log growth / physical time')
        for ax in axes[row]:ax.set_xlabel('gain g');ax.set_ylabel('leak / Euler step δ')
    fig.suptitle('Adapted finite random RNN refinement: N4, fixed seed28173\n24000 burn +2000 tangent alignment +8192 measurement; numerical, not a shrimp certificate',fontsize=13)
    fig.savefig(a.out/'refinement.png',dpi=160);fig.savefig(a.out/'refinement.pdf');plt.close(fig)
    result={'schema':'clark-rnn-refinement-saved-geometry-v1','raw':str(a.raw),'manifest_sha256':a.manifest_sha256,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'source_sha256':{str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'tools/summarize_clark_rnn.py']},'elapsed_seconds':time.monotonic()-started,'scope':'Post-outcome descriptive saved reductions only; no model, derivative, decomposition, continuation, interval certificate or chaos/whole-geometry claim.','panels':reports}
    (a.out/'result.json').write_text(json.dumps(result,indent=2)+'\n');files={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in a.out.iterdir() if p.is_file()};(a.out/'manifest.json').write_text(json.dumps({'files':files},indent=2)+'\n')
    for p in a.out.iterdir():p.chmod(0o444)
    a.out.chmod(0o555)
    print(json.dumps({'out':str(a.out),'elapsed_seconds':result['elapsed_seconds'],'components':[len(p['components']) for p in reports]}))
if __name__=='__main__':main()
