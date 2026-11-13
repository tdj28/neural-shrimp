"""Independent full-history FGR2 audit and grouped predictive comparisons."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'tools'))
from fold_global_return import backup_file, sha, write_json
from audit_fold_global_return import independent_recurrence, quadratic


def visits_from_mask(mask):
    """Direct parity-run extraction, independent of the runtime visit helper."""
    mask = np.asarray(mask, dtype=bool)
    visits = []
    for phase in (0, 1):
        indices = np.flatnonzero(mask[phase::2])*2+phase
        if not len(indices):
            continue
        groups = np.split(indices, np.flatnonzero(np.diff(indices) != 2)+1)
        for g in groups:
            if len(g) >= 3:
                visits.append({'entry':int(g[0]), 'last_inside':int(g[-1]),
                               'phase':phase, 'readings':len(g),
                               'right_censored':bool(g[-1] >= len(mask)-2)})
    visits.sort(key=lambda r:r['entry'])
    assert all(b['entry'] > a['last_inside'] for a,b in zip(visits,visits[1:]))
    return visits


def capture_trace(x, orbit, tolerance=1e-9):
    distances = np.sqrt(np.sum((x[:,None]-orbit[None])**2,axis=2))
    distances /= np.maximum(np.linalg.norm(orbit,axis=1),1e-12)[None]
    phase = np.argmin(distances,axis=1)
    error = distances[np.arange(len(x)),phase]
    good = error <= tolerance
    # Reject a phase-order violation on either side, not just unordered proximity.
    good[1:] &= (phase[1:]-phase[:-1])%len(orbit) == 1
    starts = np.flatnonzero(np.convolve(good.astype(int),np.ones(128,dtype=int),'valid')==128)
    return {'first_128_match_start':int(starts[0]) if len(starts) else None,
            'last128_max_relative_distance':float(error[-128:].max()),
            'last128_ordered':bool(np.all((phase[-127:]-phase[-128:-1])%len(orbit)==1))}


def grouped_metrics(observed, predicted, groups):
    error = np.asarray(predicted)-np.asarray(observed)
    return {'RMSE':np.sqrt(np.mean(error**2,axis=0)).tolist(),
            'MAE':np.mean(np.abs(error),axis=0).tolist(),
            'equal_row_RMSE':np.sqrt(np.mean([np.mean(error[np.asarray(groups)==g]**2,axis=0)
                                            for g in np.unique(groups)],axis=0)).tolist()}


def interpolate(train_x, train_y, test_x):
    """Linear interpolation with disclosed endpoint clipping; duplicate x averaged."""
    x, inverse = np.unique(train_x, return_inverse=True)
    y = np.zeros((len(x),train_y.shape[1]))
    count = np.bincount(inverse)
    np.add.at(y,inverse,train_y)
    y /= count[:,None]
    return np.column_stack([np.interp(test_x,x,y[:,j]) for j in range(y.shape[1])])


def line_prediction(rows):
    """First-return routing along the controlled line; finite-horizon labels only."""
    s=np.array([r['s'] for r in rows])
    labels=np.array([r['first_return'] is not None for r in rows],dtype=int)
    train=np.array([r['split']=='development' for r in rows])
    test=np.array([r['split']=='test' for r in rows])
    nearest=np.abs(s[test,None]-s[None,train]).argmin(axis=1)
    predicted=labels[train][nearest]
    transitions=[]
    for a,b in zip(rows,rows[1:]):
        if (a['first_return'] is None)!=(b['first_return'] is None):
            transitions.append({'left_row':a['id'],'right_row':b['id'],'s_interval':[a['s'],b['s']]})
    return {'label':'one or more channel visits within each retained trajectory',
            'adjacent_route_changes':transitions,'nearest_neighbor_route_accuracy':float(np.mean(predicted==labels[test])),
            'train_label_counts':dict(Counter(labels[train].tolist())),
            'test_label_counts':dict(Counter(labels[test].tolist())),
            'confusion_actual_rows_predicted_columns':[[int(np.sum((labels[test]==a)&(predicted==p))) for p in (0,1)] for a in (0,1)]}


def natural_prediction(records, states, centre, left, right, width):
    """Prespecified simple models. No heldout initial-state row enters a fit."""
    if not records:
        return {'status':'no natural exits'}
    v=(states-centre)@left
    transverse=states-centre-v[:,None]*right
    train=np.array([r['split']=='development' for r in records])
    test=np.array([r['split']=='test' for r in records])
    if train.sum()<10 or test.sum()<2:
        return {'status':'insufficient grouped development/test exits'}
    mean=transverse[train].mean(axis=0)
    _,sv,vh=np.linalg.svd(transverse[train]-mean,full_matrices=False)
    pc=(transverse-mean)@vh[:3].T
    feature=np.column_stack([v/width,pc])
    mu,sd=feature[train].mean(axis=0),feature[train].std(axis=0)
    sd[sd<1e-14]=1.
    z=(feature-mu)/sd
    geometry=interpolate(v[train],states[train],v[test])
    geometry_error=np.linalg.norm(geometry-states[test],axis=1)
    centered_scale=np.sqrt(np.mean(np.sum((states[test]-states[train].mean(axis=0))**2,axis=1)))
    answer={'status':'exploratory, complete starting trajectories held out',
            'train_exits':int(train.sum()),'test_exits':int(test.sum()),
            'train_initial_rows':sorted(set(r['row'] for r in records if r['split']=='development')),
            'test_initial_rows':sorted(set(r['row'] for r in records if r['split']=='test')),
            'transverse_PC_variance_fraction':(sv[:5]**2/np.sum(sv**2)).tolist(),
            'state_curve_interpolation_RMSE':float(np.sqrt(np.mean(geometry_error**2))),
            'state_curve_error_over_test_spread':float(np.sqrt(np.mean(geometry_error**2))/centered_scale),
            'state_curve_max_relative_error':float(np.max(geometry_error/np.linalg.norm(states[test],axis=1))),
            'v_over_U_range':[float(v.min()/width),float(v.max()/width)],
            'test_exits_beyond_train_v_range':int(((v[test]<v[train].min())|(v[test]>v[train].max())).sum())}
    returned=np.array([r['next_entry'] is not None for r in records])
    tr,te=train&returned,test&returned
    target=np.array([[r['next_entry_v_over_U'],r['excursion_updates']] if r['next_entry'] is not None else [0.,0.] for r in records])
    answer['return_train_exits']=int(tr.sum());answer['return_test_exits']=int(te.sum())
    if tr.sum()>=10 and te.sum()>=2:
        predictions={'constant':np.tile(target[tr].mean(axis=0),(te.sum(),1)),
                     'v_linear_interpolation':interpolate(v[tr],target[tr],v[te])}
        for name,features in [('v_quadratic',z[:,:1]),('v_plus_three_transverse_PCs_quadratic',z)]:
            design=quadratic(features);penalty=.01*np.eye(design.shape[1]);penalty[0,0]=0
            coef=np.linalg.solve(design[tr].T@design[tr]+penalty,design[tr].T@target[tr])
            predictions[name]=design[te]@coef
        for k in (3,8):
            d=np.linalg.norm(z[te,None]-z[None,tr],axis=2)
            ix=np.argsort(d,axis=1)[:,:min(k,tr.sum())]
            weights=1/np.maximum(np.take_along_axis(d,ix,axis=1),1e-9)
            predictions[f'four_coordinate_neighbor_{k}']=(target[tr][ix]*weights[:,:,None]).sum(axis=1)/weights.sum(axis=1)[:,None]
        groups=np.array([r['row'] for r in records])[te]
        answer['target_order']=['next_entry_v_over_U','individual_updates_to_entry']
        answer['models']={name:grouped_metrics(target[te],pred,groups) for name,pred in predictions.items()}
        answer['observations']=target[te].tolist()
        answer['predictions']={k:p.tolist() for k,p in predictions.items()}
        answer['test_event_ids']=np.flatnonzero(te).tolist()
    labels=np.array([1 if r['next_entry'] is not None else 0 if r['final_outcome']=='reference_p3' else -1 for r in records])
    ctr,cte=train&(labels>=0),test&(labels>=0)
    answer['route_counts_train']=dict(Counter(labels[ctr].tolist()))
    answer['route_counts_test']=dict(Counter(labels[cte].tolist()))
    if ctr.any() and cte.any():
        d=np.linalg.norm(z[cte,None]-z[None,ctr],axis=2)
        predicted=labels[ctr][d.argmin(axis=1)]
        answer['nearest_neighbor_route_accuracy']=float(np.mean(predicted==labels[cte]))
        answer['route_confusion_rows_actual_cols_predicted']=[[int(np.sum((labels[cte]==a)&(predicted==p))) for p in (0,1)] for a in (0,1)]
        answer['route_label_order']=['P3 before another channel visit','channel return']
    return answer


def audit(args):
    run,out=Path(args.run),Path(args.out);out.mkdir(parents=True,exist_ok=False)
    meta=json.loads((run/'metadata.json').read_text());backup=Path(meta['backup'])
    manifest=json.loads((run/'manifest.json').read_text());complete=json.loads((run/'complete.json').read_text())
    assert run.stat().st_dev != backup.stat().st_dev
    assert sha(run/'manifest.json')==complete['manifest_sha256']==sha(backup/'manifest.json')==complete['backup_manifest_sha256']
    for r in manifest:
        for base in (run,backup):
            assert (base/r['path']).stat().st_size==r['bytes'] and sha(base/r['path'])==r['sha256']
    for name,digest in meta['runtime_sources'].items():
        code=subprocess.check_output(['git','show',meta['source_commit']+':tools/'+name],cwd=ROOT)
        assert hashlib.sha256(code).hexdigest()==digest
    plan=json.loads((run/'plan.json').read_text());result=json.loads((run/'result.json').read_text())
    assert sha(run/'plan.json')==meta['plan_sha256'] and sha(run/'inputs.npz')==plan['inputs_sha256']
    prior,cycle=Path(plan['prior_path']),Path(plan['cycle_path'])
    assert sha(prior/'manifest.json')==plan['prior_manifest_sha256']
    assert sha(prior/'inputs.npz')==plan['prior_inputs_sha256']
    assert sha(cycle/'verification.json')==plan['cycle_verification_sha256']
    assert sha(cycle/'orbit-and-monodromy.npz')==plan['cycle_arrays_sha256']
    prior_meta=json.loads((prior/'metadata.json').read_text())
    prior_plan=json.loads((prior/'plan.json').read_text())
    assert plan['source']==prior_plan['source']
    for key in ('direction_sha256','prefix_text','prefix_ids','model_revision'):
        assert meta[key]==prior_meta[key],key
    cv=json.loads((cycle/'verification.json').read_text())
    assert cv['model_revision']==meta['model_revision'] and cv['spectral_radius']<1
    gate=json.loads((run/'replay-gate.json').read_text())
    assert gate['passed'] and max(gate['one_update_max_abs'],gate['two_update_max_abs'])<1e-11
    assert plan['epsilon']==1e-5 and plan['max_updates']==4096 and plan['threads']==8
    with np.load(run/'inputs.npz') as z:inp={k:z[k] for k in z.files}
    with np.load(prior/'inputs.npz') as z:
        expected=z['seeds'][64]+inp['s'][:,None]*(z['seeds'][77]-z['seeds'][64])
        expected[0],expected[-1]=z['seeds'][64],z['seeds'][77]
        assert np.array_equal(inp['seeds'],expected)
    assert inp['seeds'].shape==(64,896) and np.array_equal(inp['s'],np.linspace(0,1,64))
    arrays=[inp['seeds'][None]];next_start=1
    for p in sorted(run.glob('updates-*.npz')):
        start,end=map(int,p.stem.split('-')[1:]);assert start==next_start;next_start=end+1
        with np.load(p) as z:
            x,ids,prob=z['states'],z['top_ids'],z['top_probabilities']
            assert x.shape==(end-start+1,64,896) and x.dtype==np.float64 and np.isfinite(x).all()
            assert ids.shape==prob.shape==(end-start+1,64,8) and np.isfinite(prob).all()
            assert ((prob>=0)&(prob<=1)).all() and (prob.sum(axis=-1)<=1+1e-12).all()
            assert ((ids>=0)&(ids<151936)).all()
            assert np.array_equal(z['valid_through'],np.minimum(result['valid_through'],end))
            arrays.append(x)
    histories=np.concatenate(arrays);del arrays
    rows,events,exit_states=[],[],[];channel=np.zeros((64,len(histories)),dtype=bool)
    for i,(row,o,length) in enumerate(zip(plan['rows'],result['outcomes'],result['valid_through'])):
        assert row['id']==i and 256<=length<=4096
        x=histories[:length+1,i]
        if length+1<len(histories):assert np.all(histories[length+1:,i]==x[-1])
        cap=capture_trace(x,inp['reference_orbit'])
        if o['kind']=='reference_p3':
            assert cap['first_128_match_start'] is not None and cap['last128_ordered'] and cap['last128_max_relative_distance']<=1e-9
        elif o['kind']=='other_recurrence_candidate':
            rec=independent_recurrence(x);assert rec and rec['lag']==o['lag']
        else:
            assert o['kind'] in ('unresolved_at_horizon','wall_time_limit') and cap['first_128_match_start'] is None
            assert independent_recurrence(x) is None
            if o['kind']=='unresolved_at_horizon':assert length==4096
        delta=x-inp['centre'];v=delta@inp['left'];t=np.linalg.norm(delta-v[:,None]*inp['right'],axis=1)
        mask=(np.abs(v)<plan['width'])&(t<plan['transverse_limit'])
        channel[i,:length+1]=mask;visits=visits_from_mask(mask)
        rows.append({**row,**o,**cap,'valid_updates':int(length),'visits':visits,'visit_count':len(visits),
                     'first_return':visits[0]['entry'] if visits else None,
                     'initial_v_over_U':float(v[0]/plan['width'])})
        for j,a in enumerate(visits):
            nxt=visits[j+1]['entry'] if j+1<len(visits) else None
            if a['right_censored'] or a['last_inside']-a['entry']+2<20:continue
            stop=nxt if nxt is not None else len(x)
            indices=np.arange(a['last_inside']+2,min(stop,a['last_inside']+34),2)
            candidates=indices[(v[indices]>=3*plan['width'])&(t[indices]<plan['transverse_limit'])]
            if not len(candidates):continue
            exit_step=int(candidates[0]);exit_states.append(x[exit_step])
            events.append({'event':len(events),'row':i,'split':row['split'],'visit':j,
                           'channel_entry':a['entry'],'channel_last_inside':a['last_inside'],
                           'channel_updates':a['last_inside']-a['entry']+2,'exit':exit_step,
                           'exit_v_over_U':float(v[exit_step]/plan['width']),
                           'exit_transverse':float(t[exit_step]),'next_entry':nxt,
                           'next_entry_v_over_U':None if nxt is None else float(v[nxt]/plan['width']),
                           'excursion_updates':None if nxt is None else nxt-exit_step,
                           'final_outcome':o['kind']})
    exit_states=np.array(exit_states)
    prediction=natural_prediction(events,exit_states,inp['centre'],inp['left'],inp['right'],plan['width'])
    summary={'status':'audited exploratory repeated returns','execution_source':meta['source_commit'],
             'analysis_source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
             'analysis_sha256':sha(__file__),'manifest_sha256':sha(run/'manifest.json'),
             'plan_sha256':sha(run/'plan.json'),'raw_path':str(run),'backup_path':str(backup),
             'hashes_verified_on_separate_disks':True,'manifest_files_per_disk':len(manifest),
             'bytes_per_disk':sum(r['bytes'] for r in manifest),'replay_gate':gate,
             'counts':dict(Counter(o['kind'] for o in result['outcomes'])),
             'returned_at_least_once':sum(bool(r['visit_count']) for r in rows),
             'returned_then_P3':sum(bool(r['visit_count']) and r['kind']=='reference_p3' for r in rows),
             'total_channel_visits':sum(r['visit_count'] for r in rows),
             'natural_exits':len(events),'wall_seconds':result['wall_seconds'],
             'batch_iterations':len(histories)-1,'retained_valid_updates':sum(result['valid_through']),
             'prediction':prediction,'initial_line_prediction':line_prediction(rows),'limits':plan['limits']}
    write_json(out/'rows.json',rows);write_json(out/'events.json',events);write_json(out/'audit.json',summary)
    np.savez(out/'natural-exits.npz',states=exit_states,channel=channel)
    render(out,rows,events,summary,channel)
    write_json(out/'files.json',{p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in out.iterdir() if p.is_file()})
    for p in out.iterdir():
        if p.is_file():backup_file(p,out,backup/'audit')
    print(json.dumps({k:v for k,v in summary.items() if k!='prediction'},indent=2))


def render(out,rows,events,summary,channel):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(14,7),layout='constrained',gridspec_kw={'width_ratios':[2,1]})
    fig.suptitle('Does a return to the slow channel last?',fontsize=21,fontweight='bold')
    display=np.zeros(channel.shape)
    # Mark both adjacent steps for legibility; the audit retains the actual parity.
    display[channel|np.c_[channel[:,1:],np.zeros(len(rows),dtype=bool)]]=1
    for i,r in enumerate(rows):
        if r['kind']=='reference_p3':display[i,r['first_128_match_start']:r['valid_updates']+1]=2
        display[i,r['valid_updates']+1:]=np.nan
    cmap=ListedColormap(['#eef1f4','#db9e3b','#557c92']);cmap.set_bad('white')
    axes[0].imshow(display,origin='lower',aspect='auto',interpolation='nearest',
                   extent=[0,channel.shape[1]-1,-.5/63,1+.5/63],cmap=cmap,vmin=0,vmax=2)
    axes[0].set(xlabel='Individual model updates',ylabel='Starting-state interpolation s',title='A  Complete histories, including later returns')
    axes[1].scatter([r['s'] for r in rows],[r['visit_count'] for r in rows],c=['#557c92' if r['kind']=='reference_p3' else '#ad563c' for r in rows],s=34)
    axes[1].set(xlabel='Starting-state interpolation s',ylabel='Persistent channel visits',title='B  Returns before capture or the horizon')
    axes[1].grid(alpha=.18)
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([],[],color='#557c92',marker='o',ls='',label='Matched the 3-cycle'),
                            Line2D([],[],color='#ad563c',marker='o',ls='',label='Unresolved at 4,096 updates')],fontsize=9)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color='#db9e3b',label='Slow channel'),Patch(color='#557c92',label='Matched attracting 3-cycle'),Patch(color='#eef1f4',label='Elsewhere')],loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3,fontsize=9)
    fig.supxlabel('All 64 initial states have the same slow coordinate; only transverse components change.\n'
                  'Qwen2.5-0.5B · α = αc + 0.00001 · log₁₀T = −0.115 · patched float64 CPU\n'
                  'Visits: three same-phase readings with |v| < U and transverse remainder < 0.05. P3: 128 ordered matches at relative 10⁻⁹.\n'
                  'Blank after a blue segment: simulation stopped after the cycle check; no later states are plotted.\n'
                  'Continued recurrence within the horizon does not establish permanent trapping or chaos.',fontsize=10)
    fig.savefig(out/'repeated-returns.png',dpi=180);plt.close(fig)
    returns=[r for r in events if r['next_entry'] is not None]
    fig,axes=plt.subplots(1,2,figsize=(14,6.5),layout='constrained')
    fig.suptitle('Testing a simpler return rule on naturally reached exits',fontsize=19,fontweight='bold')
    for split,color in [('development','#aaaaaa'),('test','#0b6995'),('control','#d79b36')]:
        rr=[r for r in returns if r['split']==split]
        axes[0].scatter([r['exit_v_over_U'] for r in rr],[r['next_entry_v_over_U'] for r in rr],s=12,c=color,alpha=.65,label=split)
    axes[0].set(xlabel='Naturally reached exit position, v / U',ylabel='Next channel entry, v / U',title='A  Exits after at least 20 channel updates')
    axes[0].legend(fontsize=9)
    pred=summary['prediction']
    if 'models' in pred:
        names=list(pred['models']);values=[pred['models'][n]['equal_row_RMSE'][0] for n in names]
        axes[1].barh(np.arange(len(names)),values,color=['#aaa']+['#0b6995']*(len(names)-1))
        axes[1].set_yticks(np.arange(len(names)),[n.replace('_',' ') for n in names],fontsize=9)
        axes[1].invert_yaxis();axes[1].set(xlabel='Held-out entry prediction error (RMSE in v / U)',title='B  Entire starting trajectories held out')
    for ax in axes:ax.grid(alpha=.18)
    fig.supxlabel('Deliberately perturbed initial exits are excluded. Fits use development trajectories only.\n'
                  'Prediction errors concern returns that occur; capture classification is audited separately.\n'
                  'Exploratory result at one parameter setting; a small error here is not a universal return law.',fontsize=10)
    fig.savefig(out/'natural-return-prediction.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    audit(p.parse_args())
