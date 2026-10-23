"""Offline hash/array audit and scientific figures for HCM1; no model execution."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    a=argparse.ArgumentParser();a.add_argument('--raw',type=Path,required=True);a.add_argument('--worker',type=Path,required=True);a.add_argument('--out',type=Path,required=True);args=a.parse_args()
    r=args.raw;o=args.out;o.mkdir(parents=True,exist_ok=False)
    result=json.loads((args.worker/'result.json').read_text());source='cc6e66cceae417f67ffc26bd4b0b6571b88298d1'
    assert result['commit']==source and result['pod_id']=='54r7yo4rn29be3' and result['runner_returncode']==0
    assert sha(args.worker/'results.tgz')==result['local_sha256']==result['remote_sha256']
    manifest=json.loads((r/'manifest.json').read_text())
    for name,digest in manifest.items():assert sha(r/name)==digest,name
    plan=json.loads((r/'plan.json').read_text());runtime=json.loads((r/'runtime.json').read_text())
    assert runtime['source']==source
    plan_bytes=subprocess.check_output(['git','show',source+':docs/huginn-chaos-map-plan.json'])
    assert hashlib.sha256(plan_bytes).hexdigest()==sha(r/'plan.json')==runtime['plan_sha256']
    paths=['tools/huginn_chaos_map.py','tools/runpod_huginn_chaos_runner.sh','tools/runpod_soft_runner.sh','tools/explore_latent_loop.py','docs/huginn-chaos-map-plan.json']
    archive=subprocess.check_output(['git','archive','--format=tar',source,*paths])
    assert hashlib.sha256(archive).hexdigest()==result['upload']['code_sha256']
    events=[json.loads(x) for x in (r/'progress.jsonl').read_text().splitlines()]
    native=next(x for x in events if x['phase']=='native_gate');assert native['error']==native['repeat']==0
    throughput=next(x for x in events if x['phase']=='throughput_gate');assert throughput['projected_seconds_with_margin']<throughput['remaining_seconds']-600
    assert events[-1]['phase']=='complete' and events[-1]['cells']==1024
    z=dict(np.load(r/'map.npz'));expected={'cells':(1024,2),'lyapunov':(2,1024),'half_means':(2,2,1024),'recurrence_candidate':(2,1024),'norm_range':(2,2,1024),'lambda_axis':(32,),'steering_axis':(32,)}
    for key,shape in expected.items():assert z[key].shape==shape and np.isfinite(z[key]).all(),key
    chunks=[]
    for start in range(0,1024,16):
        q=dict(np.load(r/f'chunk-{start:04d}.npz'));chunks.append(q)
        assert q['final_state'].shape==(16,12,5280) and q['observations'].shape==(2,128,16,3)
        for key,value in q.items():assert np.isfinite(value).all(),(start,key)
        for key in ('cells','lyapunov','half_means','recurrence_candidate','norm_range'):
            want=z[key][start:start+16] if key=='cells' else z[key][...,start:start+16]
            assert np.array_equal(q[key],want),(start,key)
    expected_cells=np.array([(l,s) for l in z['lambda_axis'] for s in z['steering_axis']]);assert np.array_equal(z['cells'],expected_cells)
    l=z['lyapunov'];h=z['half_means'];p=z['recurrence_candidate'];both=(l>1e-3).all(axis=0);halves=(h>1e-3).all(axis=(0,1));agree=p[0]==p[1]
    native_idx=np.flatnonzero((z['cells'][:,0]==1)&(z['cells'][:,1]==0));assert len(native_idx)==1;i=int(native_idx[0])
    counts={str(int(k)):int(((p[0]==k)&agree).sum()) for k in np.unique(p)}
    audit=dict(source=source,pod_id=result['pod_id'],archive_sha256=result['local_sha256'],raw_manifest_sha256=sha(r/'manifest.json'),plan_sha256=runtime['plan_sha256'],raw_root=str(r),verified_files=len(manifest),verified_chunks=len(chunks),cells=1024,all_finite=True,native_gate=native,throughput_gate=throughput,positive_both_offsets=int(both.sum()),positive_both_halves_both_offsets=int(halves.sum()),positive_with_recurrence=int((both&(p>0).any(axis=0)).sum()),recurrence_agreement_counts=counts,recurrence_disagreement=int((~agree).sum()),offset_gap_max=float(abs(l[0]-l[1]).max()),offset_gap_median=float(np.median(abs(l[0]-l[1]))),half_gap_max=float(abs(h[:,0]-h[:,1]).max()),native=dict(index=i,exponents=l[:,i].tolist(),recurrence_candidates=p[:,i].tolist()),exponent_range=[float(l.min()),float(l.max())],norm_range=[float(z['norm_range'].min()),float(z['norm_range'].max())],limitations=['One short prompt, one steering direction, one initialization, float32 and finite windows','Positive growth and repeated returns are candidates; no primitive-orbit stability, float64 or JVP verification','Full tail states not retained, so recurrence labels cannot be independently recomputed from saved full-state tails; their implementation has synthetic tests','A positive finite-time exponent does not by itself distinguish sustained chaos from transients'])
    (o/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm,ListedColormap,BoundaryNorm
    plt.rcParams.update({'font.size':11,'axes.titleweight':'bold'})
    fig,axs=plt.subplots(1,2,figsize=(15,7),layout='constrained')
    fig.suptitle("Huginn’s internal loop: first exploratory map",fontsize=20)
    x=z['steering_axis'];y=z['lambda_axis'];s=np.min(l,axis=0).reshape(32,32)
    im=axs[0].pcolormesh(x,y,s,shading='nearest',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-.23,vcenter=0,vmax=.035),rasterized=True)
    axs[0].set_title('Do nearby internal states separate?')
    cb=fig.colorbar(im,ax=axs[0],shrink=.8);cb.set_label('Smaller measured exponent of the two offsets\nBlue: contraction · red: expansion')
    vals=[-1]+sorted(map(int,np.unique(p)));plot=np.where(agree,p[0],-1);idx=np.array([vals.index(int(k)) for k in plot]).reshape(32,32)
    colors=['#d9d9d9','#f6f1e8','#183c68','#3485b0','#61bead','#a8d283','#f1d16c','#eaa25a','#dc6b59','#ba588d','#8655a1','#584c91','#312a65'][:len(vals)]
    im2=axs[1].pcolormesh(x,y,idx,shading='nearest',cmap=ListedColormap(colors),norm=BoundaryNorm(np.arange(len(vals)+1)-.5,len(vals)),rasterized=True)
    cb2=fig.colorbar(im2,ax=axs[1],ticks=np.arange(len(vals)),shrink=.8)
    cb2.ax.set_yticklabels(['Offsets disagree' if k==-1 else 'No return ≤64' if k==0 else 'Fixed point candidate' if k==1 else f'{k}-step candidate' for k in vals])
    axs[1].set_title('Does the full internal state repeat?')
    for ax in axs:
        ax.plot(0,1,'*',ms=16,color='#ffcd00',mec='black',mew=1,zorder=5)
        ax.annotate('Original update, no steering',(0,1),xytext=(-.23,.968),arrowprops=dict(arrowstyle='-',color='black'),fontsize=10,bbox=dict(fc='white',alpha=.85,ec='none'))
        ax.set_xlabel('Internal steering strength (additive RMS)');ax.set_ylabel('Update size λ (1 = original step)')
        ax.set_xlim(-.26,.26);ax.set_ylim(.943,1.357)
    fig.supxlabel('32 × 32 settings · 512 settling steps + 1,024 measured steps at each of two offsets\nFloat32 screening: positive growth and repeated returns are candidates, not verified chaos or shrimps.',fontsize=11)
    fig.savefig(o/'huginn-internal-loop-map.png',dpi=180);fig.savefig(o/'huginn-internal-loop-map.pdf');plt.close(fig)
    (o/'artifact-hashes.json').write_text(json.dumps({p.name:sha(p) for p in o.iterdir() if p.is_file()},indent=2)+'\n')
    print(json.dumps(audit,indent=2))

if __name__=='__main__':main()
