#!/usr/bin/env python3
"""Read-only independent EXP007 outcome audit; never imports experiment code."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np

SOURCE = 'ac098c3e3833daaf27c7dc2869d45e8b7085f948'
ROOT = Path(__file__).resolve().parents[1]
ARMS = ['atlas','local','tangent','axes','uniform','search','hybrid']
RADII = [.00025,.001,.004,.016]
PHASES = [337,1012,1687,2362]
DTS = [.01,.005]
SUBSETS = {'local':[16,17], 'tangent':[18,19], 'axes':[0,4,8,12],
           'uniform':list(range(16)), 'search':list(range(21))}
STATS = {'metric_blocks':0, 'rms_values':0, 'maximum_rms_discrepancy':0.,
         'raw_traces':0, 'maximum_unit_norm_error':0., 'cadence_flags':0}

def read(path):
    return json.loads(path.read_text())

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def close(actual,expected):
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-14)

def rms(error):
    return float(np.sqrt(np.average(np.square(error)))), np.sqrt(np.average(np.square(error),axis=0))

def independent_reference(ref,phase):
    """Use NumPy's scalar interpolation with a separately appended endpoint."""
    x=np.remainder(phase,1.)*len(ref)
    knots=np.arange(len(ref)+1,dtype=float)
    extended=np.vstack([ref,ref[0]])
    return np.column_stack([np.interp(x,knots,extended[:,j]) for j in range(3)])

def check_rms(saved,error):
    aggregate,components=rms(error)
    close([saved['aggregate'],*saved['per_component']],[aggregate,*components])
    STATS['rms_values']+=4
    STATS['maximum_rms_discrepancy']=max(STATS['maximum_rms_discrepancy'],
        float(np.max(np.abs(np.array([saved['aggregate'],*saved['per_component']])-
                            np.array([aggregate,*components])))))
    return aggregate

def metric_check(saved,ref,times,values,T):
    assert saved['period']==T
    offset=saved['phase_offset_cycles']
    assert -.5<=offset<.5
    phase=times/T
    synchronous=values-independent_reference(ref,phase)
    aligned=values-independent_reference(ref,phase+offset)
    check_rms(saved['synchronous_rms'],synchronous)
    result=check_rms(saved['aligned_rms'],aligned)
    n=max(1,int(np.ceil((times[-1]-times[0])/(T*saved['window_cycles']))))
    window_ids=np.minimum(np.floor((times-times[0])/(T*saved['window_cycles'])).astype(int),n-1)
    assert len(saved['windows'])==len(np.unique(window_ids))
    for w,index in zip(saved['windows'],np.unique(window_ids)):
        mask=window_ids==index
        assert w['index']==index and w['sample_count']==int(mask.sum())
        assert w['first_sample_time']==float(times[mask][0])
        assert w['last_sample_time']==float(times[mask][-1])
        check_rms(w['synchronous_rms'],synchronous[mask])
        check_rms(w['aligned_rms'],aligned[mask])
    STATS['metric_blocks']+=1
    STATS['cadence_flags']+=int(not saved['sampling']['candidate_cadence_resolves_reference_spectrum'])
    return result

def directions(anchor,local,radius):
    angles=np.arange(16)*(2*np.pi/16)
    result=np.column_stack([np.cos(angles),np.sin(angles)])
    result[::4]=[[1,0],[0,1],[-1,0],[0,-1]]
    result=np.vstack([result,local,-local,anchor['tangent'],
                      -np.asarray(anchor['tangent']),anchor['atlas_margin'][str(radius)]['direction']])
    assert result.shape==(21,2)
    norms=np.linalg.norm(result,axis=1)
    np.testing.assert_allclose(norms,1,rtol=0,atol=1e-12)
    STATS['maximum_unit_norm_error']=max(STATS['maximum_unit_norm_error'],float(np.max(np.abs(norms-1))))
    return result

def check_raw(raw,initial,anchor,direction,radius,dt,duration):
    expected_time=np.arange(round(duration/.1)+1)*.1
    expected_time[-1]=duration
    np.testing.assert_array_equal(raw['times'],expected_time)
    np.testing.assert_array_equal(raw['initial'],initial)
    np.testing.assert_array_equal(raw['states'][0],initial)
    np.testing.assert_array_equal(raw['delta'],radius*direction)
    np.testing.assert_array_equal([raw['alpha'],raw['gamma']],
         np.asarray([anchor['alpha'],anchor['gamma']])+radius*direction)
    assert float(raw['dt'])==dt
    assert raw['states'].shape==(len(expected_time),3) and np.isfinite(raw['states']).all()
    STATS['raw_traces']+=1

def pick(previews,radius):
    by={r['candidate']:r for r in previews}
    assert len(by)==len(previews)==21 and set(by)==set(range(21))
    chosen={'atlas':{'status':'complete','candidate':20,'preview_calls':0}}
    for arm,subset in SUBSETS.items():
        rows=[by[i] for i in subset]
        valid=all(r['status']=='complete' and isinstance(r.get('score'),(float,int))
                  and not isinstance(r.get('score'),bool) and np.isfinite(r['score'])
                  and 0<=r['score']<=2 for r in rows)
        chosen[arm]={'status':'complete' if valid else 'selector_failed',
                     'candidate':min(rows,key=lambda r:(r['score'],r['candidate']))['candidate'] if valid else None,
                     'preview_calls':len(subset)}
    source='local' if radius<=.004 else 'atlas'
    chosen['hybrid']={**chosen[source],'source_policy':source}
    return chosen

def make_summary(lookup,eligible):
    rng=np.random.default_rng(20260912)
    draws=rng.integers(len(eligible),size=(10000,len(eligible))) if eligible else None
    results={}
    for dt in DTS:
        blocks={}
        for radius in RADII:
            blocks[str(radius)]={}
            for arm in ARMS:
                by_anchor=[]
                for aid in eligible:
                    rows=[lookup[(aid,phase,radius,arm,dt)] for phase in PHASES]
                    by_anchor.append(float(np.mean([r['_tail'] for r in rows]))
                                     if all(r['status']=='complete' for r in rows) else None)
                blocks[str(radius)][arm]=by_anchor
        blocks['pooled']={arm:[float(np.average([blocks[str(r)][arm][i] for r in RADII]))
                   if all(blocks[str(r)][arm][i] is not None for r in RADII) else None
                   for i in range(len(eligible))] for arm in ARMS}
        results[str(dt)]={}
        for key,errors in blocks.items():
            means={a:float(np.average(v)) if v and all(x is not None for x in v) else None for a,v in errors.items()}
            contrasts={}
            for comparator in ['local','tangent','axes','uniform','search','atlas']:
                if means['hybrid'] is None or means[comparator] is None:
                    contrasts[comparator]={'status':'undefined_incomplete'}
                else:
                    contrast=np.asarray(errors[comparator])-np.asarray(errors['hybrid'])
                    bootstrap_means=np.average(contrast[draws],axis=1)
                    contrasts[comparator]={'status':'complete','benefit':float(np.average(contrast)),
                          'ci95':np.quantile(bootstrap_means,[.025,.975]).tolist(),
                          'positive_anchors':int(np.count_nonzero(contrast>0)),
                          'per_anchor':contrast.tolist()}
            results[str(dt)][key]={'means':means,'per_anchor':errors,'comparator_minus_hybrid':contrasts}
    return results

def compare_nested(got,want):
    if isinstance(got,dict):
        assert set(got)==set(want)
        for k in got:compare_nested(got[k],want[k])
    elif isinstance(got,list):
        assert len(got)==len(want)
        for a,b in zip(got,want):compare_nested(a,b)
    elif isinstance(got,(float,int)) and not isinstance(got,bool):close(got,want)
    else:assert got==want,(got,want)

def distribution(values):
    values=np.asarray(values)
    return {'n':len(values),'mean':float(np.mean(values)),'median':float(np.median(values)),
            'p95':float(np.quantile(values,.95)),'maximum':float(np.max(values))} if len(values) else {'n':0}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=ROOT,help='Repository containing the frozen source and old inputs')
    parser.add_argument('--attempt',type=Path,help='Defaults to REPO/evidence/EXP-007/attempt-001')
    parser.add_argument('--out',type=Path,help='Optional JSON output; the attempt is always read-only')
    args=parser.parse_args();directory=args.attempt or args.repo/'evidence'/'EXP-007'/'attempt-001';started=time.perf_counter()
    if args.out is not None:
        if args.out.resolve().is_relative_to(directory.resolve()):
            parser.error('--out must be outside the immutable attempt directory')
    receipt=read(directory/'receipt.json');p=read(directory/'manifest.json')
    assert receipt['status']=='complete' and receipt['source_commit']==SOURCE
    assert not (directory/'timeout.json').exists() and not (directory/'failure.json').exists()
    assert sha(directory/'manifest.json')==receipt['manifest_sha256']
    frozen=subprocess.check_output(['git','show',SOURCE+':experiments/hopfield/EXP-007-plan.json'],cwd=args.repo)
    assert frozen==(directory/'manifest.json').read_bytes()
    assert p['arms']==ARMS and p['radii']==RADII and p['phases']==PHASES and p['dt']==DTS
    assert p['primary_dt']==p['decision_dt']==.005 and p['bootstrap_seed']==20260912
    assert p['probe_duration']==300 and p['preview_duration']==75 and p['bootstrap_samples']==10000
    artifacts={a['path']:a for a in receipt['artifacts']};assert len(artifacts)==len(receipt['artifacts'])
    for name,a in artifacts.items():
        raw=directory/name
        assert raw.stat().st_size==a['bytes'] and sha(raw)==a['sha256'],name
    for a in p['source_artifacts']:assert sha(args.repo/a['path'])==a['sha256'],a['path']
    print(json.dumps({'stage':'receipt_hashes','artifacts':len(artifacts),'sources':len(p['source_artifacts'])}),flush=True)
    anchors={a['id']:a for a in p['anchors']};assert len(anchors)==12
    baselines=read(directory/'baselines.json');assert set(baselines)==set(anchors)
    eligible=[a for a in anchors if baselines[a]['eligible']]
    assert eligible==receipt['eligible_anchors']
    contexts={};decisions={};selection_counts={a:Counter() for a in ARMS}
    preview_status=Counter();preview_walls=[];sensitivity_walls=[];sensitivity_nfev=[]
    for aid in eligible:
        cert=baselines[aid]['certificate'];T=cert['period']
        assert cert['verified'] and cert['attracting']
        assert [s['period'] for s in baselines[aid]['steps']]==[9,9]
        orbit=np.asarray(cert['orbit_states'])[:-1];ref=np.asarray(cert['orbit_activations'])[:-1]
        assert orbit.shape==ref.shape==(2700,3);close(ref,np.tanh(orbit))
        for phase in PHASES:
            initial=orbit[phase];epoch=phase*T/2700
            f=directory/'sensitivity'/f'{aid}-p{phase}'
            meta=read(f.with_suffix('.json'));sensitivity_walls.append(meta['wall_seconds']);sensitivity_nfev.append(meta['nfev'])
            with np.load(f.with_suffix('.npz'),allow_pickle=False) as raw:
                np.testing.assert_array_equal(raw['states'][0],initial)
                mask=raw['times']>75;assert mask.sum()==2250
                S=raw['activation_sensitivity'][mask].reshape(-1,2);v=raw['activation_velocity'][mask].ravel()
                residual=S-np.outer(v,np.dot(v,S)/np.dot(v,v)) if np.dot(v,v)>0 else S.copy()
                G=residual.T@residual/len(v);close(G,raw['tail_gram'])
                eigenvalues,eigenvectors=np.linalg.eigh(G);local=eigenvectors[:,0]
                first=np.flatnonzero(abs(local)>1e-14)[0]
                if local[first]<0:local=-local
                close(raw['tail_direction'],local);local=raw['tail_direction'].copy()
            contexts[(aid,phase)]=(initial,ref,T,epoch,local)
            for radius in RADII:
                cid=f'{aid}-p{phase}-r{radius:g}';d=read(directory/'decisions'/f'{cid}.json')
                assert (d['anchor'],d['phase'],d['radius'])==(aid,phase,radius)
                ds=directions(anchors[aid],local,radius)
                for preview in d['previews']:
                    idx=preview['candidate'];np.testing.assert_array_equal(preview['direction'],ds[idx])
                    preview_status[preview['status']]+=1;preview_walls.append(preview['wall_seconds'])
                    if preview['status']=='complete':
                        with np.load(directory/preview['raw'],allow_pickle=False) as raw:
                            check_raw(raw,initial,anchors[aid],ds[idx],radius,.005,75.)
                            score=metric_check(preview['metrics'],ref,raw['times']+epoch,np.tanh(raw['states']),T)
                        close(preview['score'],score)
                        assert preview['score']==preview['metrics']['aligned_rms']['aggregate']
                choice=pick(d['previews'],radius);assert d['policies']==choice
                for arm,c in choice.items():selection_counts[arm][str(c['candidate'])]+=1
                decisions[(aid,phase,radius)]=(d,ds)
    preview_expected={'planned':4032,'eligible':len(eligible)*336,
           'baseline_not_evaluable':(12-len(eligible))*336,'status_counts':dict(preview_status),
           'sum_concurrent_call_wall_seconds':float(sum(preview_walls))}
    compare_nested(read(directory/'preview-census.json'),preview_expected)
    assert len(list((directory/'decisions').glob('*.json')))==len(eligible)*16
    print(json.dumps({'stage':'previews','status':dict(preview_status),'eligible':eligible}),flush=True)
    records=[json.loads(line) for line in (directory/'records.jsonl').read_text().splitlines()]
    lookup={}
    for row in records:
        key=tuple(row[k] for k in ['anchor','phase','radius','arm','dt']);assert key not in lookup;lookup[key]=row
        aid,phase,radius,arm,dt=key
        assert row['id']==f'{aid}-p{phase}-r{radius:g}-{arm}-dt{dt:g}'
        if aid not in eligible:assert row['status']=='baseline_not_evaluable';continue
        initial,ref,T,epoch,local=contexts[(aid,phase)]
        if radius==0:direction=np.zeros(2)
        else:
            decision,ds=decisions[(aid,phase,radius)];choice=decision['policies'][arm]
            if choice['status']!='complete':assert row['status']=='selector_failed';continue
            direction=ds[choice['candidate']]
        assert row['status'] in ['complete','failed']
        if row['status']!='complete':continue
        assert row['phase_epoch']==epoch and row['raw']==f"trials/{row['id']}.npz"
        np.testing.assert_array_equal(row['direction'],direction)
        np.testing.assert_array_equal(row['delta'],radius*direction)
        np.testing.assert_array_equal([row['alpha'],row['gamma']],np.asarray([anchors[aid]['alpha'],anchors[aid]['gamma']])+radius*direction)
        with np.load(directory/row['raw'],allow_pickle=False) as raw:
            check_raw(raw,initial,anchors[aid],direction,radius,dt,300.)
            tail=raw['times']>75;assert tail.sum()==2250 and raw['times'][tail][0]==75.10000000000001
            row['_tail']=metric_check(row['tail_metrics'],ref,raw['times'][tail]+epoch,np.tanh(raw['states'][tail]),T)
            row['_full']=metric_check(row['full_metrics'],ref,raw['times']+epoch,np.tanh(raw['states']),T)
    expected={(a,ph,r,arm,dt) for a in anchors for ph in PHASES for dt in DTS
               for r,arm in [(0.,'zero')]+[(r,arm) for r in RADII for arm in ARMS]}
    assert set(lookup)==expected and len(records)==2784
    summary=make_summary(lookup,eligible);saved_summary=read(directory/'summary.json')
    assert saved_summary['planned']==2784 and saved_summary['eligible_anchors']==eligible
    status=dict(Counter(r['status'] for r in records));assert saved_summary['status_counts']==status
    zero={};cross=[];cross_by_arm={a:[] for a in ARMS+['zero']}
    tangent_axes={'conditions':0,'identical_actions':0,'identical_preview_states':0,
                  'target_pairs':0,'identical_target_states':0,'identical_tail_metric_objects':0,'axis_direction_counts':{}}
    axis_counts=Counter()
    for (aid,phase,radius),(d,ds) in decisions.items():
        ti=d['policies']['tangent']['candidate'];ai=d['policies']['axes']['candidate']
        tangent_axes['conditions']+=1
        tangent_axes['identical_actions']+=int(np.array_equal(ds[ti],ds[ai]))
        axis_counts[str(ds[ai].tolist())]+=1
        preview_by={r['candidate']:r for r in d['previews']}
        with np.load(directory/preview_by[ti]['raw'],allow_pickle=False) as x,np.load(directory/preview_by[ai]['raw'],allow_pickle=False) as y:
            tangent_axes['identical_preview_states']+=int(np.array_equal(x['states'],y['states']))
        for dt in DTS:
            xrow=lookup[(aid,phase,radius,'tangent',dt)];yrow=lookup[(aid,phase,radius,'axes',dt)]
            tangent_axes['target_pairs']+=1
            with np.load(directory/xrow['raw'],allow_pickle=False) as x,np.load(directory/yrow['raw'],allow_pickle=False) as y:
                tangent_axes['identical_target_states']+=int(np.array_equal(x['states'],y['states']))
            tangent_axes['identical_tail_metric_objects']+=int(xrow['tail_metrics']==yrow['tail_metrics'])
    tangent_axes['axis_direction_counts']=dict(axis_counts)
    for dt in DTS:
        compare_nested(saved_summary['steps'][str(dt)]['panels'],summary[str(dt)])
        z=[lookup[(a,ph,0.,'zero',dt)] for a in eligible for ph in PHASES]
        zval=[r['_tail'] for r in z if r['status']=='complete']
        zero[str(dt)]={'complete':len(zval),'planned':len(z),'mean':float(np.mean(zval)) if zval and len(zval)==len(z) else None,
                       'maximum':float(max(zval)) if zval and len(zval)==len(z) else None}
        compare_nested(saved_summary['steps'][str(dt)]['zero'],zero[str(dt)])
    for key,row in lookup.items():
        if row['dt']!=.005:continue
        other=lookup[(*key[:-1],.01)]
        if row['status']==other['status']=='complete':
            value=abs(row['_tail']-other['_tail']);cross.append(value);cross_by_arm[row['arm']].append(value)
    actual_files={str(f.relative_to(directory)) for f in directory.rglob('*') if f.is_file() and 'build' not in f.relative_to(directory).parts}
    extra=sorted(actual_files-set(artifacts)-{'receipt.json'})
    assert not extra,extra
    costs={
        'collection_receipt_wall_seconds_before_verification':receipt['wall_seconds'],
        'actual_preview_calls':len(preview_walls),'sum_concurrent_preview_call_wall_seconds':float(sum(preview_walls)),
        'sensitivity_solves':len(sensitivity_walls),'sensitivity_solver_wall_seconds':float(sum(sensitivity_walls)),
        'sensitivity_solver_nfev':int(sum(sensitivity_nfev)),
        'actual_target_records':len(records),'completed_target_calls':status.get('complete',0),
        'sum_concurrent_target_call_wall_seconds':float(sum(r.get('wall_seconds',0) for r in records)),
        'counterfactual_preview_calls_per_policy_all_eligible_conditions':
           {a:(len(eligible)*16*{'atlas':0,'local':2,'tangent':2,'axes':4,'uniform':16,'search':21}.get(a,0)
               if a!='hybrid' else len(eligible)*4*3*2) for a in ARMS},
        'actual_shared_pool_caveat':'Actual forecasts are shared; budgets are separate declared information-access costs, not separately benchmarked policy runtime.',
        'paid_compute':receipt['paid_compute'],'paid_review':receipt['paid_review']}
    output={'source_commit':SOURCE,'manifest_sha256':receipt['manifest_sha256'],
       'verified_artifacts':len(artifacts),'verified_artifact_bytes':sum(a['bytes'] for a in artifacts.values()),
       'verified_source_artifacts':len(p['source_artifacts']),
       'target_status':status,'preview_census':preview_expected,'eligible_anchors':eligible,
       'ineligible_baselines':{a:b for a,b in baselines.items() if not b['eligible']},
       'metric_verification':STATS,'independent_summary':summary,'zero':zero,
       'cross_step_absolute_tail_rms_difference':distribution(cross),
       'cross_step_by_arm':{a:distribution(v) for a,v in cross_by_arm.items()},
       'selection_index_counts':{a:dict(v) for a,v in selection_counts.items()},'cost':costs,
       'tangent_axes_identity':tangent_axes,
       'frozen_first_eligible_example':None if not eligible else {
           'anchor':eligible[0],'phase':337,'radius':.016,'dt':.005,
           'tail_rms':{a:lookup[(eligible[0],337,.016,a,.005)].get('_tail') for a in ARMS}},
       'full_window_secondary_means':{str(dt):{a:float(np.mean([r['_full'] for r in records if r['dt']==dt and r['arm']==a]))
                                             if all(r['status']=='complete' for r in records if r['dt']==dt and r['arm']==a) else None
                                             for a in ARMS} for dt in DTS},
       'audit_wall_seconds':time.perf_counter()-started}
    if args.out is not None:args.out.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'stage':'complete','output':str(args.out) if args.out is not None else None,'target_status':status,
       'primary_pooled':summary['0.005']['pooled'],'cross_step':output['cross_step_absolute_tail_rms_difference'],
       'zero':zero,'cost':costs,'verification':STATS}),flush=True)

if __name__=='__main__':main()
