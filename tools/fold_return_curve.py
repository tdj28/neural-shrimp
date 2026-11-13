"""FGR2C: test whether the scalar natural-exit coordinate retains return information.

Fit only development exits from the declared immutable prefix, then reconstruct
one natural exit per held-out initial trajectory. Run paired original and
reconstructed states through the unchanged network using the FGR1 runner.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from fold_global_return import backup_file,sha,write_json
from audit_fold_return_line import visits_from_mask,capture_trace
from audit_fold_global_return import independent_return,independent_recurrence


def fit_curve(states, centre, left, right, degree=7):
    v=(states-centre)@left
    mid=(v.min()+v.max())/2;half=(v.max()-v.min())/2
    if half<=0:raise ValueError('Nondegenerate coordinate interval required')
    w=states-centre-v[:,None]*right
    design=np.polynomial.chebyshev.chebvander((v-mid)/half,degree)
    coef=np.linalg.lstsq(design,w,rcond=None)[0]
    coef-=(coef@left)[:,None]*right
    return {'mid':mid,'half':half,'coef':coef}


def reconstruct(v, curve, centre, right):
    design=np.polynomial.chebyshev.chebvander((np.asarray(v)-curve['mid'])/curve['half'],len(curve['coef'])-1)
    return centre+np.asarray(v)[:,None]*right+design@curve['coef']


def development_crossvalidation(states, groups, centre, left, right):
    records={}
    for degree in (2,3,5,7):
        predicted=np.zeros_like(states)
        for group in np.unique(groups):
            train=groups!=group;test=~train
            curve=fit_curve(states[train],centre,left,right,degree)
            predicted[test]=reconstruct((states[test]-centre)@left,curve,centre,right)
        error=np.linalg.norm(predicted-states,axis=1)/np.linalg.norm(states,axis=1)
        records[str(degree)]={'median_relative_state_error':float(np.median(error)),
                              'max_relative_state_error':float(error.max())}
    return records


def prepare(args):
    run,out=Path(args.run),Path(args.out);out.mkdir(parents=True,exist_ok=False)
    plan=json.loads((run/'plan.json').read_text());meta=json.loads((run/'metadata.json').read_text())
    backup=Path(meta['backup'])
    with np.load(run/'inputs.npz') as z:d={k:z[k] for k in z.files}
    source_files=[{'path':'inputs.npz','sha256':sha(run/'inputs.npz')},
                  {'path':'plan.json','sha256':sha(run/'plan.json')},
                  {'path':'metadata.json','sha256':sha(run/'metadata.json')}]
    arrays=[d['seeds'][None]]
    for lo in range(1,640,128):
        p=run/f'updates-{lo:05d}-{lo+127:05d}.npz'
        assert sha(p)==sha(backup/p.name)
        source_files.append({'path':p.name,'sha256':sha(p)})
        with np.load(p) as z:arrays.append(z['states'])
    x=np.concatenate(arrays);train=[];training_rows=[];selected=[]
    for i,row in enumerate(plan['rows']):
        delta=x[:,i]-d['centre'];v=delta@d['left'];t=np.linalg.norm(delta-v[:,None]*d['right'],axis=1)
        visits=visits_from_mask((abs(v)<plan['width'])&(t<plan['transverse_limit']))
        exits=[]
        for j,a in enumerate(visits):
            if a['right_censored'] or a['last_inside']-a['entry']+2<20:continue
            nxt=visits[j+1]['entry'] if j+1<len(visits) else len(x)
            indices=np.arange(a['last_inside']+2,min(nxt,a['last_inside']+34),2)
            indices=indices[(v[indices]>=3*plan['width'])&(t[indices]<plan['transverse_limit'])]
            if len(indices):exits.append(int(indices[0]))
        if row['split']=='development':
            for j in exits:train.append(x[j,i]);training_rows.append(i)
        elif row['split']=='test' or i==0:
            if not exits:raise ValueError(f'No natural exit for requested heldout row {i}')
            j=exits[0];selected.append({'parent_row':i,'parent_update':j,'split':row['split'],'v':float(v[j])})
    train=np.array(train);curve=fit_curve(train,d['centre'],d['left'],d['right'])
    assert len(selected)==32 and all(r['parent_row'] not in training_rows for r in selected)
    observed=np.array([x[r['parent_update'],r['parent_row']] for r in selected])
    v=np.array([r['v'] for r in selected]);predicted=reconstruct(v,curve,d['centre'],d['right'])
    assert np.max(abs((predicted-d['centre'])@d['left']-v))<1e-13
    seeds=np.stack([observed,predicted],axis=1).reshape(64,896)
    rows=[]
    for j,r in enumerate(selected):
        for kind in ('original','curve_reconstruction'):
            rows.append({'id':len(rows),'pair':j,'kind':kind,**r})
    np.savez(out/'inputs.npz',seeds=seeds,centre=d['centre'],left=d['left'],right=d['right'],
             replay_states=d['replay_states'],replay_F=d['replay_F'],replay_G=d['replay_G'],
             reference_orbit=d['reference_orbit'],curve_coef=curve['coef'],curve_mid=curve['mid'],curve_half=curve['half'],
             train_states=train,training_rows=np.array(training_rows))
    new={k:plan[k] for k in ('source','epsilon','width','transverse_limit')}
    new.update(name='FGR2C',status='exploratory adaptive reduction test',rows=rows,
               threads=8,chunk_updates=128,max_updates=1024,inputs_sha256=sha(out/'inputs.npz'),
               parent=str(run),parent_source=meta['source_commit'],parent_files=source_files,
               selection='First natural exit after >=20 channel updates per odd heldout row and row0 control; no heldout outcome used in fitting.',
               fit='Degree7 Chebyshev transverse curve from all143 development natural exits in first640 updates; coefficients projected into kernel of left critical vector.',
               rationale='Degrees2,3,5,7 examined with development-row crossvalidation only; degree7 max relative reconstruction error2.251e-13 in that prefix.',
               stop='First persistent channel return, outside recurrence candidate, or1024 updates. Does not test indefinite itinerary prediction.',
               comparisons='Route, return update and return coordinate for paired exact and reconstructed states; verify exact states against parent histories; verify P3 phase matches independently.')
    write_json(out/'plan.json',new);write_json(out/'saved-events.json',{'parent_prefix':source_files})
    print(json.dumps({'pairs':len(selected),'development_exits':len(train),
                      'heldout_initial_state_relative_errors':(np.linalg.norm(predicted-observed,axis=1)/np.linalg.norm(observed,axis=1)).tolist()},indent=2))


def audit(args):
    run,out=Path(args.run),Path(args.out);out.mkdir(parents=True,exist_ok=False)
    meta=json.loads((run/'metadata.json').read_text());backup=Path(meta['backup'])
    plan=json.loads((run/'plan.json').read_text());result=json.loads((run/'result.json').read_text())
    manifest=json.loads((run/'manifest.json').read_text());complete=json.loads((run/'complete.json').read_text())
    assert run.stat().st_dev!=backup.stat().st_dev
    assert sha(run/'manifest.json')==sha(backup/'manifest.json')==complete['manifest_sha256']
    for r in manifest:
        for base in (run,backup):assert sha(base/r['path'])==r['sha256'] and (base/r['path']).stat().st_size==r['bytes']
    assert sha(run/'inputs.npz')==plan['inputs_sha256'] and sha(run/'plan.json')==meta['plan_sha256']
    root=Path(__file__).resolve().parents[1]
    sources={'fold_global_return.py':meta['source_sha256'],**meta['runtime_source_sha256']}
    for name,digest in sources.items():
        assert hashlib.sha256(subprocess.check_output(['git','show',meta['source_commit']+':tools/'+name],cwd=root)).hexdigest()==digest
    gate=json.loads((run/'replay-gate.json').read_text())
    assert gate['passed'] and max(gate['one_update_max_abs'],gate['two_updates_max_abs'])<1e-11
    parent=Path(plan['parent'])
    for r in plan['parent_files']:assert sha(parent/r['path'])==r['sha256']
    pm=json.loads((parent/'metadata.json').read_text())
    for k in ('direction_sha256','prefix_text','prefix_ids','model_revision'):assert meta[k]==pm[k]
    with np.load(run/'inputs.npz') as z:d={k:z[k] for k in z.files}
    curve=fit_curve(d['train_states'],d['centre'],d['left'],d['right'])
    np.testing.assert_allclose(curve['coef'],d['curve_coef'],rtol=1e-6,atol=1e-15)
    x=[d['seeds'][None]]
    for p in sorted(run.glob('updates-*.npz')):
        lo,hi=map(int,p.stem.split('-')[1:]);assert lo==sum(len(a) for a in x)
        with np.load(p) as z:
            assert z['states'].shape==(hi-lo+1,64,896) and np.isfinite(z['states']).all()
            assert z['states'].dtype==np.float64
            assert z['top_ids'].shape==z['top_probabilities'].shape==(hi-lo+1,64,8)
            assert np.isfinite(z['top_probabilities']).all() and ((z['top_probabilities']>=0)&(z['top_probabilities']<=1)).all()
            assert (z['top_probabilities'].sum(axis=-1)<=1+1e-12).all()
            assert ((z['top_ids']>=0)&(z['top_ids']<151936)).all()
            assert np.array_equal(z['valid_through'],np.minimum(result['valid_through'],hi))
            x.append(z['states'])
    x=np.concatenate(x);rows=[]
    with np.load(parent/'inputs.npz') as z:original=[z['seeds'][None]]
    needed=max(r['parent_update']+n for r,n in zip(plan['rows'],result['valid_through']))
    parent_replay_files=[]
    for path in sorted(parent.glob('updates-*.npz')):
        assert sha(path)==sha(Path(pm['backup'])/path.name)
        parent_replay_files.append({'path':path.name,'sha256':sha(path)})
        with np.load(path) as z:
            original.append(z['states']);parent_valid=z['valid_through'].copy()
        if sum(len(a) for a in original)>needed:break
    original=np.concatenate(original);assert len(original)>needed
    replay_errors=[]
    for i,(row,o,length) in enumerate(zip(plan['rows'],result['outcomes'],result['valid_through'])):
        assert row['id']==i and row['parent_row'] not in d['training_rows']
        hist=x[:length+1,i];ret=independent_return(hist,d['centre'],d['left'],d['right'],plan['width'])
        if length+1<len(x):assert np.all(x[length+1:,i]==hist[-1])
        if row['kind']=='original':
            count=min(length,int(parent_valid[row['parent_row']])-row['parent_update'])+1
            assert count>1
            truth=original[row['parent_update']:row['parent_update']+count,row['parent_row']]
            error=float(np.max(np.linalg.norm(hist[:count]-truth,axis=1)/np.linalg.norm(truth,axis=1)))
            replay_errors.append(error)
            assert error<1e-7,('heldout saved-state replay',i,error)
        if o['kind']=='return':
            assert ret and length==ret['confirmed_update'] and ret['entry_update']==o['entry_update']
            assert abs(ret['entry_v']-o['entry_v'])<1e-13
        elif o['kind']=='outside_recurrence_candidate':
            assert ret is None and independent_recurrence(hist)['lag']==o['lag']
            cap=capture_trace(hist,d['reference_orbit']);assert cap['last128_ordered'] and cap['last128_max_relative_distance']<1e-9
        else:assert o['kind']=='unresolved_at_horizon' and length==1024
        rows.append({**row,**o,'valid_updates':length})
    pairs=[]
    for i in range(32):
        a,b=rows[2*i:2*i+2]
        both=a['kind']==b['kind']=='return'
        pairs.append({'pair':i,'parent_row':a['parent_row'],'split':a['split'],
                      'same_route':a['kind']==b['kind'],'original_route':a['kind'],'reconstructed_route':b['kind'],
                      'return_time_difference':b['entry_update']-a['entry_update'] if both else None,
                      'return_v_over_U_error':(b['entry_v']-a['entry_v'])/plan['width'] if both else None,
                      'initial_relative_state_error':float(np.linalg.norm(d['seeds'][2*i]-d['seeds'][2*i+1])/np.linalg.norm(d['seeds'][2*i]))})
    test=[r for r in pairs if r['split']=='test'];returned=[r for r in test if r['return_time_difference'] is not None]
    summary={'status':'audited paired scalar-curve reconstruction test',
             'execution_source':meta['source_commit'],'manifest_sha256':sha(run/'manifest.json'),'plan_sha256':sha(run/'plan.json'),
             'raw_path':str(run),'backup_path':str(backup),'hash_verified_on_separate_disks':True,
             'test_pairs':len(test),'all_test_routes_agree':all(r['same_route'] for r in test),
             'returned_test_pairs':len(returned),'original_P3_test_pairs':sum(r['original_route']=='outside_recurrence_candidate' for r in test),
             'max_return_time_difference':max((abs(r['return_time_difference']) for r in returned),default=None),
             'max_return_v_over_U_error':max((abs(r['return_v_over_U_error']) for r in returned),default=None),
             'max_test_initial_relative_state_error':max(r['initial_relative_state_error'] for r in test),
             'max_original_parent_trajectory_replay_relative_error':max(replay_errors),
             'parent_replay_files':parent_replay_files,'replay_gate':gate,
             'development_crossvalidation':development_crossvalidation(d['train_states'],d['training_rows'],d['centre'],d['left'],d['right']),
             'wall_seconds':result['total_wall_seconds'],
             'limitation':'Predicts one excursion from natural exits at one parameter setting; the full network still evaluates the global map. Not a universal closed formula or indefinite itinerary guarantee.'}
    write_json(out/'rows.json',rows);write_json(out/'pairs.json',pairs);write_json(out/'audit.json',summary)
    render(out,d,rows,pairs,summary,plan['width'])
    write_json(out/'files.json',{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()})
    for p in out.iterdir():backup_file(p,out,backup/'audit')
    print(json.dumps(summary,indent=2))


def render(out,data,rows,pairs,summary,width):
    import math
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(13.5,6.5),layout='constrained')
    fig.suptitle('One coordinate predicts the next excursion from natural exits',fontsize=19,fontweight='bold')
    v=(data['train_states']-data['centre'])@data['left']
    w=data['train_states']-data['centre']-v[:,None]*data['right']
    mean=w.mean(axis=0);_,_,vh=np.linalg.svd(w-mean,full_matrices=False)
    q=np.linspace(v.min(),v.max(),300)
    c={'mid':float(data['curve_mid']),'half':float(data['curve_half']),'coef':data['curve_coef']}
    fit=reconstruct(q,c,data['centre'],data['right'])-data['centre']-q[:,None]*data['right']
    axes[0].plot(q/width,(fit-mean)@vh[0],color='#bd643a',lw=2,label='Curve fitted on development exits')
    axes[0].scatter(v/width,(w-mean)@vh[0],s=9,c='#aaaaaa',label='Development exits')
    originals=data['seeds'][::2];tv=(originals-data['centre'])@data['left']
    tw=originals-data['centre']-tv[:,None]*data['right']
    axes[0].scatter(tv[1:]/width,(tw[1:]-mean)@vh[0],s=34,facecolors='none',edgecolors='#086a92',label='Held-out natural exits',zorder=3)
    axes[0].set(xlabel='Exit position along the slow direction, v / U',ylabel='Leading transverse coordinate',title='A  Natural exits follow a narrow curved path')
    axes[0].legend(fontsize=9)
    for j,p in enumerate(pairs):
        if p['split']!='test':continue
        a,b=rows[2*j:2*j+2]
        if a['kind']=='return':
            axes[1].scatter(tv[j]/width,a['entry_v']/width,s=36,c='#086a92',label='Original full state' if j==1 else None)
        if b['kind']=='return':
            axes[1].scatter(tv[j]/width,b['entry_v']/width,s=80,facecolors='none',edgecolors='#bd643a',label='Reconstructed from v alone' if j==1 else None)
        if a['kind']=='outside_recurrence_candidate':
            axes[1].scatter(tv[j]/width,-1.13,marker='x',s=60,c='#883f6e')
    axes[1].set(xlabel='Exit position v / U (the only reconstruction input)',ylabel='Next entry position, v / U',title='B  Paired network runs: observed and predicted',ylim=(-1.22,.2))
    axes[1].axhline(-1.05,color='.7',lw=.7);axes[1].text(3.03,-1.14,'× Approaches the 3-cycle',fontsize=9,color='#883f6e')
    axes[1].legend(fontsize=9,loc='upper right')
    for ax in axes:ax.grid(alpha=.18)
    e=summary['max_return_v_over_U_error']
    scale=10**(1-math.floor(math.log10(e))) if e else 1
    upper=math.ceil(e*scale)/scale
    text=(f"{summary['test_pairs']} held-out starting trajectories · {summary['returned_test_pairs']} return pairs · "
          f"{summary['original_P3_test_pairs']} pairs approach the verified 3-cycle\n"
          f"Largest return-time discrepancy: {summary['max_return_time_difference']} updates; "
          f"all entry discrepancies ≤ {upper:.2g} U.\n"
          'The full network evaluates each excursion. This validates a scalar initial-state reduction at one setting,\n'
          'not a universal closed formula or indefinitely accurate trajectory prediction.')
    fig.supxlabel(text,fontsize=10);fig.savefig(out/'scalar-return-validation.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','audit'])
    p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    args=p.parse_args();{'prepare':prepare,'audit':audit}[args.command](args)
