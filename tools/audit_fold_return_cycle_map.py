"""Rehash and independently audit the complete paired departure-to-departure map."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from fold_global_return import sha,backup_file,write_json
from fold_return_cycle_map import audit_departure


def history(root,initial,n,expected_dimension=896):
    arrays=[initial[None]]
    for p in sorted(root.glob('updates-*.npz')):
        lo,hi=map(int,p.stem.split('-')[1:]);assert lo==sum(len(a) for a in arrays)
        with np.load(p) as z:
            x=z['states'];assert x.shape==(hi-lo+1,n,expected_dimension) and x.dtype==np.float64 and np.isfinite(x).all()
            arrays.append(x)
    return np.concatenate(arrays)


def audit(args):
    run,out=Path(args.run),Path(args.out);out.mkdir(parents=True,exist_ok=False)
    meta=json.loads((run/'metadata.json').read_text());backup=Path(meta['backup'])
    manifest=json.loads((run/'manifest.json').read_text());complete=json.loads((run/'complete.json').read_text())
    assert run.stat().st_dev!=backup.stat().st_dev
    assert sha(run/'manifest.json')==sha(backup/'manifest.json')==complete['manifest_sha256']==complete['backup_manifest_sha256']
    for f in manifest:
        for base in (run,backup):assert (base/f['path']).stat().st_size==f['bytes'] and sha(base/f['path'])==f['sha256']
    for name,digest in meta['source_hashes'].items():
        code=subprocess.check_output(['git','show',meta['source_commit']+':tools/'+name],cwd=ROOT)
        assert hashlib.sha256(code).hexdigest()==digest
    plan=json.loads((run/'plan.json').read_text());result=json.loads((run/'result.json').read_text())
    assert sha(run/'plan.json')==meta['plan_sha256'] and sha(run/'inputs.npz')==plan['inputs_sha256']
    parent=Path(plan['parent']);assert sha(parent/'manifest.json')==plan['parent_manifest_sha256']
    pr=json.loads((parent/'result.json').read_text());pp=json.loads((parent/'plan.json').read_text());pm=json.loads((parent/'metadata.json').read_text())
    for k in ('prefix_ids','direction_sha256','model_revision'):assert meta[k]==pm[k]
    assert plan['source']==pp['source'] and plan['epsilon']==pp['epsilon']==1e-5
    gate=json.loads((run/'replay-gate.json').read_text());assert max(gate.values())<1e-11
    with np.load(run/'inputs.npz') as z:d={k:z[k] for k in z.files}
    with np.load(parent/'inputs.npz') as z:pd={k:z[k] for k in z.files}
    ph=history(parent,pd['seeds'],64)
    expected=np.array([ph[length,i] for i,length in enumerate(pr['valid_through'])])
    assert np.array_equal(expected,d['seeds'])
    x=history(run,d['seeds'],64);rows=[]
    for i,(r,o,length) in enumerate(zip(plan['rows'],result['outcomes'],result['valid_through'])):
        assert np.all(x[length+1:,i]==x[length,i])
        if not d['active'][i]:assert length==0 and o['kind']=='excluded_P3';continue
        delta=x[:length+1,i]-d['centre'];v=delta@d['left'];t=np.linalg.norm(delta-v[:,None]*d['right'],axis=1)
        independent=audit_departure(v,t,plan['width'])
        assert independent is not None and independent.keys()==o.keys(),(i,independent,o)
        for key in independent:
            if key in ('v','transverse'):assert abs(independent[key]-o[key])<1e-12
            else:assert independent[key]==o[key],(i,key)
        assert o['kind']=='next_natural_exit',('nonstandard passage',i,o)
        q0=float((pd['seeds'][i]-d['centre'])@d['left']/plan['width'])
        rows.append({**r,**o,'initial_departure_v_over_U':q0,'next_departure_v_over_U':o['v']/plan['width'],
                     'departure_to_departure_updates':pr['valid_through'][i]+o['update']})
    pairs=[]
    for a,b in zip(rows[::2],rows[1::2]):
        assert a['pair']==b['pair'] and a['id']%2==0 and b['id']==a['id']+1
        pairs.append({'pair':a['pair'],'parent_row':a['parent_row'],'split':a['split'],
                      'v0':a['initial_departure_v_over_U'],'observed_next_v':a['next_departure_v_over_U'],
                      'predicted_next_v':b['next_departure_v_over_U'],
                      'next_v_error':b['next_departure_v_over_U']-a['next_departure_v_over_U'],
                      'cycle_updates':a['departure_to_departure_updates'],
                      'cycle_time_error':b['departure_to_departure_updates']-a['departure_to_departure_updates']})
    test=[p for p in pairs if p['split']=='test'];assert len(test)==28
    summary={'status':'audited one complete return cycle', 'execution_source':meta['source_commit'],
             'analysis_source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'analysis_sha256':sha(__file__),'manifest_sha256':sha(run/'manifest.json'),'plan_sha256':sha(run/'plan.json'),
             'raw_path':str(run),'backup_path':str(backup),'all_manifest_hashes_verified_on_separate_disks':True,
             'files_per_disk':len(manifest),'bytes_per_disk':sum(f['bytes'] for f in manifest),'replay_gate':gate,
             'heldout_returning_pairs':len(test),'control_pairs':len(pairs)-len(test),
             'excluded_P3_pairs':int((~d['active']).sum()//2),
             'max_complete_cycle_time_error':max(abs(p['cycle_time_error']) for p in test),
             'max_next_departure_v_over_U_error':max(abs(p['next_v_error']) for p in test),
             'complete_cycle_updates_range':[min(p['cycle_updates'] for p in test),max(p['cycle_updates'] for p in test)],
             'wall_seconds':result['wall_seconds'],
             'limits':['Same parameter setting and natural exit curve as FGR2C.','One complete cycle only; accuracy under indefinite iteration is untested.','Full network still evaluates the global excursion and local passage.']}
    write_json(out/'rows.json',rows);write_json(out/'pairs.json',pairs);write_json(out/'audit.json',summary)
    render(out,pairs,summary)
    write_json(out/'files.json',{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()})
    for p in out.iterdir():backup_file(p,out,backup/'audit')
    print(json.dumps(summary,indent=2))


def render(out,pairs,summary,background=None):
    import math
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    test=[p for p in pairs if p['split']=='test']
    fig,axes=plt.subplots(1,2,figsize=(13.5,6.5),layout='constrained')
    fig.suptitle('The next complete return cycle can be predicted from one exit coordinate',fontsize=18,fontweight='bold')
    if background:
        groups={}
        for e in background:groups.setdefault(e['row'],[]).append(e)
        points=[]
        for group in groups.values():
            for a,b in zip(group,group[1:]):
                if b['visit']==a['visit']+1:points.append((a['exit_v_over_U'],b['exit_v_over_U']))
        if points:
            points=np.array(points)
            axes[0].scatter(points[:,0],points[:,1],s=5,c='#bbbbbb',alpha=.6,label='Other measured natural return cycles',zorder=0)
    q0=[p['v0'] for p in test]
    axes[0].scatter(q0,[p['observed_next_v'] for p in test],s=36,c='#086a92',label='Original full state')
    axes[0].scatter(q0,[p['predicted_next_v'] for p in test],s=80,facecolors='none',edgecolors='#bd643a',label='Reconstructed from v alone')
    axes[0].set(xlabel='Position at this departure, v / U',ylabel='Position at the next departure, v / U',title='A  Excursion → slow passage → next departure')
    axes[0].legend(fontsize=9)
    axes[1].scatter(q0,[max(abs(p['next_v_error']),1e-16) for p in test],s=36,c='#086a92')
    axes[1].set(xlabel='Position at this departure, v / U',ylabel='Absolute prediction error at next departure (U)',yscale='log',title='B  Residual errors, shown on a logarithmic scale')
    for ax in axes:ax.grid(alpha=.18)
    e=summary['max_next_departure_v_over_U_error'];scale=10**(1-math.floor(math.log10(e))) if e else 1;upper=math.ceil(e*scale)/scale
    fig.supxlabel(f"{len(test)} held-out returning pairs · all complete-cycle timing errors ≤ {summary['max_complete_cycle_time_error']} model updates\n"
                  f"All next-departure errors ≤ {upper:.2g} U. The three P3 pairs are excluded here; their destination was checked separately.\n"
                  'The full network evaluates the reconstructed states. One parameter setting; no guarantee for indefinite iteration.',fontsize=10)
    fig.savefig(out/'complete-return-map.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    audit(p.parse_args())
