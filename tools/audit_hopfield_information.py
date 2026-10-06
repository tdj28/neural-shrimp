#!/usr/bin/env python3
"""Independent read-only EXP008 raw/feature/decision audit; no experiment imports.

This reconstructs saved phases, not new ODE trajectories or phase optima.
Model hashes and saved predictions are checked; a separate audit owns ridge/CV replay.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.ndimage import label
from scipy.spatial import cKDTree
ROOT=Path(__file__).resolve().parents[1]
SOURCE='3c5604497ec57b19d62fbb595da7b190102f4b33'
MANIFEST='fb0e546947e0f36b77ff4978f60769db3541fc64330022bd664da22c20c321cd'
PHASES=[675,1350]
RADII=[.00025,.001,.004,.016]
DTS=[.01,.005]
LEARNERS=['baseline','geometry','shuffled_geometry']
POLICIES=LEARNERS+['min_preview']
LAMBDAS=np.array([1e-6,1e-4,1e-2,1.,100.])
STATS={'metric_blocks':0,'rms_values':0,'maximum_rms_discrepancy':0.,'raw_traces':0,'cadence_flags':0}

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


def fid(a,ph,r,c):
    return f'{a}-p{ph}-r{r:g}-c{c:02d}'


def tid(a,ph,r,c,dt):
    stem=f'{a}-p{ph}-r0-zero' if c==-1 else fid(a,ph,r,c)
    return f'{stem}-dt{dt:g}'


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def valid(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and np.isfinite(value) and 0<=value<=2


def compare(got,want,*,rtol=1e-12,atol=1e-14):
    if isinstance(want,dict):
        assert isinstance(got,dict) and set(got)==set(want)
        for k in want:compare(got[k],want[k],rtol=rtol,atol=atol)
    elif isinstance(want,(list,tuple)):
        assert len(got)==len(want)
        for g,w in zip(got,want):compare(g,w,rtol=rtol,atol=atol)
    elif isinstance(want,(float,np.floating)):
        np.testing.assert_allclose(got,want,rtol=rtol,atol=atol)
    else:assert got==want,(got,want)


def raw_check(raw,anchor,initial,direction,radius,dt,duration):
    times=np.arange(round(duration/.1)+1)*.1;times[-1]=duration
    np.testing.assert_array_equal(raw['times'],times)
    np.testing.assert_array_equal(raw['initial'],initial)
    np.testing.assert_array_equal(raw['states'][0],initial)
    np.testing.assert_array_equal(raw['delta'],radius*direction)
    np.testing.assert_array_equal([raw['alpha'],raw['gamma']],np.array([anchor['alpha'],anchor['gamma']])+radius*direction)
    assert raw['states'].shape==(len(times),3) and np.isfinite(raw['states']).all()
    assert float(raw['dt'])==dt
    STATS['raw_traces']+=1


def make_directions(local):
    angles=np.arange(16)*(2*np.pi/16)
    d=np.column_stack([np.cos(angles),np.sin(angles)])
    d[::4]=[[1,0],[0,1],[-1,0],[0,-1]]
    d=np.vstack([d,local,-local]);np.testing.assert_allclose(np.linalg.norm(d,axis=1),1,rtol=0,atol=1e-12)
    return d


def read_atlas(repo,core):
    arrays=[]
    for p in sorted((repo/'evidence/EXP-004/attempt-001/focus').glob('row-*.npz')):
        with np.load(p,allow_pickle=False) as raw:arrays.append({k:raw[k].copy() for k in ['period','alpha','gamma']})
    q=np.array([r['period'] for r in arrays]);alpha=arrays[0]['alpha'];gamma=np.array([r['gamma'] for r in arrays])
    ids,_=label(q==9,np.ones((3,3),int));inside=ids==ids[tuple(core)]
    exterior_ij=np.argwhere(~inside)
    outside=np.column_stack([np.broadcast_to(alpha,q.shape)[~inside],np.broadcast_to(gamma[:,None],q.shape)[~inside]])
    interior=np.column_stack([np.broadcast_to(alpha,q.shape)[inside],np.broadcast_to(gamma[:,None],q.shape)[inside]])
    return dict(alpha=alpha,gamma=gamma,inside=inside,outside=outside,outside_ij=exterior_ij,
                outside_tree=cKDTree(outside),inside_tree=cKDTree(interior))


def geo_check(atlas,anchor,direction,radius,saved):
    point=np.array([anchor['alpha'],anchor['gamma']]);destination=point+radius*direction
    delta=atlas['outside']-point;squared=np.einsum('ij,ij->i',delta,delta)
    minimum=float(squared.min());tolerance=max(1e-28,32*np.finfo(float).eps*minimum)
    k=int(np.flatnonzero(squared<=minimum+tolerance)[0]);margin=float(np.linalg.norm(delta[k]));normal=delta[k]/margin
    a,g=atlas['alpha'],atlas['gamma']
    edge=float(min(destination[0]-a[0],a[-1]-destination[0],destination[1]-g[0],g[-1]-destination[1]))
    covered=edge>=0
    col=int(np.argmin(abs(a-destination[0])));row=int(np.argmin(abs(g-destination[1])))
    member=bool(atlas['inside'][row,col]) if covered else None
    pixel=[a[col],g[row]]
    signed=float((atlas['outside_tree'] if member else atlas['inside_tree']).query(pixel)[0])*(1 if member else -1) if covered else None
    censored=not covered or abs(signed)>=edge
    features=[margin,0. if signed is None else signed,float(bool(member)),
              float(np.clip(np.dot(direction,normal),-1,1)),float(covered),float(censored)]
    close(saved['geometry'],features)
    diagnostics=saved['geometry_diagnostics']
    compare(diagnostics['destination'],destination.tolist())
    assert diagnostics['nearest_pixel_row']==row and diagnostics['nearest_pixel_column']==col
    assert diagnostics['destination_inside_crop']==covered and diagnostics['destination_in_core_component']==member
    compare(diagnostics['destination_signed_pixel_margin'],signed)
    close(diagnostics['destination_crop_edge_lower_bound'],edge)
    assert diagnostics['margin_crop_censored']==censored
    np.testing.assert_array_equal(diagnostics['anchor_nearest_non_core_pixel'],atlas['outside_ij'][k])
    close(diagnostics['anchor_outward'],normal)


def base_check(row,anchor,context,names):
    # Construct columns by name independently, rather than calling the producer.
    a,g=anchor['alpha'],anchor['gamma'];r=row['radius'];direction=context['directions'][row['candidate']]
    dx,dy=r*direction;da,dg=a+dx,g+dy;T=context['T'];rho=context['rho'];phase=row['phase']
    gram=context['gram'];delta=np.array([dx,dy]);preview=row['metrics']
    local=float(np.sqrt(max(0.,delta@gram['projected']@delta)))
    unprojected=float(np.sqrt(max(0.,delta@gram['unprojected']@delta)))
    score=preview['aligned_rms']['aggregate'];shift=preview['phase_offset_cycles']
    first=preview['windows'][0]['aligned_rms']['aggregate'];last=preview['windows'][-1]['aligned_rms']['aggregate']
    clearance=min(da-1.056,1.08-da,dg-10.14,10.55-dg)
    values=dict(alpha=a,gamma=g,destination_alpha=da,destination_gamma=dg,alpha_squared=a*a,alpha_gamma=a*g,gamma_squared=g*g,
       phase_sin=np.sin(2*np.pi*phase/2700),phase_cos=np.cos(2*np.pi*phase/2700),period=T,log_rho=np.log(rho),log_rho_per_time=np.log(rho)/T,
       radius=r,log_radius=np.log(r),radius_squared=r*r,direction_alpha=direction[0],direction_gamma=direction[1],
       delta_alpha=dx,delta_gamma=dy,delta_alpha_squared=dx*dx,delta_cross=dx*dy,delta_gamma_squared=dy*dy,
       alpha_delta_alpha=a*dx,alpha_delta_gamma=a*dy,gamma_delta_alpha=g*dx,gamma_delta_gamma=g*dy,
       rectangle_contains_destination=float(clearance>=0),signed_crop_clearance=clearance,
       local_projected_rmse=local,local_unprojected_rmse=unprojected,local_phase_cycles=float(gram['coeff']@delta/T),
       preview_rmse=score,preview_synchronous_rmse=preview['synchronous_rms']['aggregate'],
       preview_phase_sin=np.sin(2*np.pi*shift),preview_phase_cos=np.cos(2*np.pi*shift),
       preview_first_window_rmse=first,preview_last_window_rmse=last,preview_window_change=last-first,
       preview_rmse_squared=score*score,local_projected_mse=local*local,preview_times_local=score*local,preview_times_radius=score*r)
    for j in range(3):
        values[f'initial_u{j+1}']=context['reference'][phase,j]
        values[f'orbit_std_u{j+1}']=context['reference'].std(axis=0)[j]
        values[f'preview_u{j+1}_rmse']=preview['aligned_rms']['per_component'][j]
    assert len(values)==len(names)==51 and set(values)==set(names)
    close(row['base'],[values[k] for k in names])


def matrices(rows,anchors):
    ids=[a['id'] for a in anchors]
    by={(r['anchor'],r['phase'],r['radius'],r['candidate']):r for r in rows}
    donor={aid:ids[(i+5)%12] for i,aid in enumerate(ids)}
    base=np.asarray([r['base'] for r in rows]);geometry=np.asarray([r['geometry'] for r in rows])
    shuffled=np.asarray([by[(donor[r['anchor']],r['phase'],r['radius'],r['candidate'])]['geometry'] for r in rows])
    return base,geometry,shuffled


def model_check(packet,features,targets,anchors,decisions,names,geo_names):
    contents={k:v for k,v in packet.items() if k!='packet_sha256'}
    digest=hashlib.sha256(json.dumps(contents,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    assert packet['packet_sha256']==digest
    if packet['status']!='complete':return {'status':packet['status'],'limitation':'No complete model for saved prediction audit'}
    dev=features['development'];test=features['test'];dev_m=matrices(dev,anchors['development']);test_m=matrices(test,anchors['test'])
    lookup={r['id']:r for r in targets['development']};y=np.array([lookup[r['id']+'-dt0.005']['rmse'] for r in dev])
    groups=np.array([r['block'] for r in dev]);unique=sorted(set(groups));assert len(unique)==12
    assert packet['development_row_ids']==[r['id'] for r in dev] and packet['development_rows']==1728
    assert packet['development_groups']==unique and packet['development_group_counts']==dict(Counter(groups))
    h=hashlib.sha256()
    for array in (*dev_m,y):
        h.update(str(array.shape).encode());h.update(np.ascontiguousarray(array,dtype='<f8').tobytes())
    h.update(json.dumps([[r['id'] for r in dev],groups.tolist()],separators=(',',':')).encode())
    assert h.hexdigest()==packet['development_data_sha256']
    test_lookup={r['id']:r for r in targets['test']}
    yt=np.array([test_lookup[row['id']+'-dt0.005']['rmse'] for row in test])
    result={}
    for policy,tmat in zip(LEARNERS,(test_m[0],np.column_stack(test_m[:2]),np.column_stack((test_m[0],test_m[2])))):
        saved=packet['models'][policy]
        assert saved['feature_names']==(names if policy=='baseline' else names+geo_names)
        assert saved['cost']=={'cv_folds':12,'ridge_solves':65}
        prediction=saved['intercept']+((tmat-np.array(saved['feature_mean']))/np.array(saved['feature_scale']))@np.array(saved['coefficients'])
        assert np.isfinite(prediction).all()
        result[policy]={'chosen_lambda':saved['chosen_lambda'],'test_prediction_mse':float(np.mean((prediction-yt)**2)),
             'predictions_below_zero':int(sum(prediction<0)),'predictions_above_two':int(sum(prediction>2)),
             'minimum_prediction':float(min(prediction)),'maximum_prediction':float(max(prediction))}
        for i,d in enumerate(decisions):
            chunk=prediction[i*18:(i+1)*18]
            close(d['predictions'][policy],chunk)
            winner=int(np.argmin(chunk))
            assert d['policies'][policy]['candidate']==winner
            close(d['policies'][policy]['selection_score'],chunk[winner])
            assert d['model_packet_sha256']==digest
    return result


def summary_check(p,targets,decisions,saved):
    test=targets['test'];by={(r['anchor'],r['phase'],r['radius'],r['candidate'],r['dt']):r for r in test}
    origins=[a for a in p['anchors'] if a['role']=='test'];dby={(d['anchor'],d['phase'],d['radius']):d for d in decisions}
    draws=np.random.default_rng(20260913).integers(12,size=(10000,12));steps={};oracle_by_step={}
    for dt in DTS:
        values={r:{k:[] for k in POLICIES} for r in RADII};regrets={r:{k:[] for k in POLICIES} for r in RADII};oracle_by_step[str(dt)]={}
        for radius in RADII:
            oracles=[]
            for anchor in origins:
                per={k:[] for k in POLICIES};per_regret={k:[] for k in POLICIES};phase_oracle=[]
                for phase in PHASES:
                    rows=[by[(anchor['id'],phase,radius,c,dt)] for c in range(18)]
                    complete=all(r['status']=='complete' and valid(r.get('rmse')) for r in rows)
                    oracle=min(r['_rms'] for r in rows) if complete else None;phase_oracle.append(oracle)
                    d=dby[(anchor['id'],phase,radius)]
                    for k in POLICIES:
                        choice=d['policies'][k];ok=complete and choice['status']=='complete'
                        error=rows[choice['candidate']]['_rms'] if ok else None
                        per[k].append(error);per_regret[k].append(error-oracle if ok else None)
                oracles.append(float(np.mean(phase_oracle)) if all(v is not None for v in phase_oracle) else None)
                for k in POLICIES:
                    values[radius][k].append(float(np.mean(per[k])) if all(v is not None for v in per[k]) else None)
                    regrets[radius][k].append(float(np.mean(per_regret[k])) if all(v is not None for v in per_regret[k]) else None)
            oracle_by_step[str(dt)][str(radius)]=oracles
        for container in [values,regrets]:
            container['pooled']={k:[float(np.mean([container[r][k][i] for r in RADII]))
                if all(container[r][k][i] is not None for r in RADII) else None for i in range(12)] for k in POLICIES}
        panels={}
        for radius,v in values.items():
            means={k:{'selected_rmse':float(np.mean(v[k])) if all(x is not None for x in v[k]) else None,
                       'regret':float(np.mean(regrets[radius][k])) if all(x is not None for x in regrets[radius][k]) else None} for k in POLICIES}
            contrasts={}
            for name,left,right in [('baseline_minus_geometry','baseline','geometry'),('baseline_minus_shuffled','baseline','shuffled_geometry'),
                                    ('shuffled_minus_geometry','shuffled_geometry','geometry'),('min_preview_minus_geometry','min_preview','geometry')]:
                if any(x is None for x in v[left]+v[right]):contrasts[name]={'status':'undefined_incomplete'};continue
                dif=np.array(v[left])-np.array(v[right]);ci=np.quantile(dif[draws].mean(axis=1),[.025,.975])
                contrasts[name]={'status':'complete','mean_benefit':float(np.mean(dif)),'ci95':ci.tolist(),'positive_blocks':int(sum(dif>0)),'per_block':dif.tolist()}
            panel={'means':means,'per_block':{k:{'selected_rmse':v[k],'regret':regrets[radius][k]} for k in POLICIES},'contrasts':contrasts}
            prediction_blocks={k:{m:[] for m in ['mse','minimum','maximum','below_zero_fraction','above_two_fraction','out_of_range_fraction']} for k in LEARNERS}
            doses=RADII if radius=='pooled' else [radius]
            for anchor in origins:
                for policy in LEARNERS:
                    condition_metrics=[]
                    for r in doses:
                        for phase in PHASES:
                            d=dby[(anchor['id'],phase,r)]
                            pred=d.get('predictions',{}).get(policy)
                            available=pred is not None and len(pred)==18 and np.isfinite(pred).all()
                            rows=[by[(anchor['id'],phase,r,c,dt)] for c in range(18)]
                            if available:
                                pred=np.asarray(pred)
                                mse=float(np.mean((pred-np.array([row['_rms'] for row in rows]))**2)) if all(row['status']=='complete' for row in rows) else None
                                condition_metrics.append(dict(mse=mse,minimum=float(min(pred)),maximum=float(max(pred)),
                                    below_zero_fraction=float(np.mean(pred<0)),above_two_fraction=float(np.mean(pred>2)),
                                    out_of_range_fraction=float(np.mean((pred<0)|(pred>2)))))
                            else:condition_metrics.append({k:None for k in prediction_blocks[policy]})
                    for metric in prediction_blocks[policy]:
                        vals=[x[metric] for x in condition_metrics]
                        aggregate=None if any(x is None for x in vals) else (min(vals) if metric=='minimum' else max(vals) if metric=='maximum' else float(np.mean(vals)))
                        prediction_blocks[policy][metric].append(aggregate)
            prediction={}
            for policy,metrics in prediction_blocks.items():
                ag={k:None if any(x is None for x in v) else (min(v) if k=='minimum' else max(v) if k=='maximum' else float(np.mean(v))) for k,v in metrics.items()}
                prediction[policy]={'status':'complete' if ag['mse'] is not None else 'undefined_incomplete',
                                    'range_status':'complete' if ag['minimum'] is not None else 'undefined_incomplete',**ag}
            panel.update(prediction=prediction,prediction_per_block=prediction_blocks)
            compare(saved['steps'][str(dt)]['panels'][str(radius)],panel)
            panels[str(radius)]=panel
        steps[str(dt)]=panels
    all_records=targets['development']+targets['test']
    gate=len(all_records)==7008 and all(r['status']=='complete' and valid(r.get('rmse')) for r in all_records)
    assert saved['primary_data_gate']['passed']==gate
    primary=steps['0.005']['pooled']['contrasts']['baseline_minus_geometry'] if gate else {'status':'undefined_incomplete','reason':'A required target or numerical sham is incomplete'}
    compare(saved['primary'],primary)
    return steps,oracle_by_step


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--repo',type=Path,default=ROOT)
    parser.add_argument('--attempt',type=Path);parser.add_argument('--out',type=Path)
    args=parser.parse_args();directory=args.attempt or args.repo/'evidence/EXP-008/attempt-001';start=time.perf_counter()
    if args.out and args.out.resolve().is_relative_to(directory.resolve()):parser.error('Audit output must be outside immutable attempt')
    # Completion is required before inspecting any outcome-bearing artifact.
    receipt=read(directory/'receipt.json');assert receipt['status']=='complete'
    assert receipt['source_commit']==SOURCE and receipt['manifest_sha256']==MANIFEST
    assert not (directory/'failure.json').exists() and not (directory/'timeout.json').exists()
    p=read(directory/'manifest.json');assert sha(directory/'manifest.json')==MANIFEST
    frozen=subprocess.check_output(['git','show',SOURCE+':experiments/hopfield/EXP-008-plan.json'],cwd=args.repo)
    assert frozen==(directory/'manifest.json').read_bytes()
    artifacts={r['path']:r for r in receipt['artifacts']};assert len(artifacts)==len(receipt['artifacts'])
    for name,entry in artifacts.items():assert sha(directory/name)==entry['sha256'] and (directory/name).stat().st_size==entry['bytes'],name
    for item in p['source_artifacts']+p['source_files']:assert sha(args.repo/item['path'])==item['sha256'],item['path']
    print(json.dumps({'stage':'hashes','artifacts':len(artifacts)}),flush=True)
    assert p['phases']==PHASES and p['radii']==RADII and p['dt']==DTS and p['planned_targets']==7008 and p['planned_previews']==3456
    names=p['base_feature_names'];geo_names=p['geometry_feature_names'];assert len(names)==len(set(names))==51 and len(geo_names)==6
    anchors={role:[a for a in p['anchors'] if a['role']==role] for role in ['development','test']}
    assert all(len(v)==12 for v in anchors.values());atlas=read_atlas(args.repo,p['selection']['core_index'])
    aa={a['id']:a for a in p['anchors']};features={};targets={};contexts={};eligible={};sens_cost=[];sens_nfev=[]
    for role in ['development','test']:
        baseline=read(directory/f'{role}-baselines.json');assert set(baseline)=={a['id'] for a in anchors[role]}
        eligible[role]=[a for a,b in baseline.items() if b['eligible']]
        for aid,b in baseline.items():
            if not b['eligible']:continue
            cert=b['certificate'];assert cert['verified'] and cert['attracting'];assert [s['period'] for s in b['steps']]==[9,9]
            orbit=np.asarray(cert['orbit_states'])[:-1];ref=np.asarray(cert['orbit_activations'])[:-1]
            assert orbit.shape==ref.shape==(2700,3);close(ref,np.tanh(orbit))
            for ph in PHASES:
                file=directory/'sensitivity'/f'{aid}-p{ph}.npz'
                if not file.exists():continue
                meta=read(file.with_suffix('.json'));sens_cost.append(meta['wall_seconds']);sens_nfev.append(meta['nfev'])
                with np.load(file,allow_pickle=False) as raw:
                    np.testing.assert_array_equal(raw['states'][0],orbit[ph]);np.testing.assert_array_equal(raw['state_sensitivity'][0],np.zeros((3,2)))
                    tail=raw['times']>75;assert tail.sum()==2250
                    S=raw['activation_sensitivity'][tail].reshape(-1,2);v=raw['activation_velocity'][tail].ravel()
                    coeff=(v@S)/(v@v);residual=S-v[:,None]*coeff
                    G=residual.T@residual/len(v);U=S.T@S/len(v)
                    close(raw['tail_gram'],G);e,ev=np.linalg.eigh(G);local=ev[:,0]
                    if local[np.flatnonzero(abs(local)>1e-14)[0]]<0:local=-local
                    close(raw['tail_direction'],local)
                    context={'directions':make_directions(raw['tail_direction']),'initial':orbit[ph],'reference':ref,
                         'T':cert['period'],'epoch':ph*cert['period']/2700,'rho':cert['transverse_spectral_radius'],
                         'gram':{'projected':G,'unprojected':U,'coeff':coeff}}
                    contexts[(aid,ph)]=context
        rows=read(directory/f'{role}-features.json');features[role]=rows
        assert rows==read_lines(directory/f'{role}-previews.jsonl')
        expected=[fid(a['id'],ph,r,c) for a in anchors[role] for ph in PHASES for r in RADII for c in range(18)]
        assert [r['id'] for r in rows]==expected and len(rows)==1728
        for row in rows:
            a=aa[row['anchor']];assert row['role']==role and row['block']==a['block_id']
            if row['status']!='complete':continue
            context=contexts[(row['anchor'],row['phase'])];direction=context['directions'][row['candidate']]
            np.testing.assert_array_equal(row['direction'],direction);np.testing.assert_array_equal(row['delta'],row['radius']*direction)
            geo_check(atlas,a,direction,row['radius'],row)
            with np.load(directory/row['raw'],allow_pickle=False) as raw:
                raw_check(raw,a,context['initial'],direction,row['radius'],.005,75)
                value=metric_check(row['metrics'],context['reference'],raw['times']+context['epoch'],np.tanh(raw['states']),context['T'])
            close(row['preview_score'],value);base_check(row,a,context,names)
        records=read_lines(directory/f'{role}-targets.jsonl');targets[role]=records
        expected={tid(a['id'],ph,r,c,dt) for a in anchors[role] for ph in PHASES
                   for r in [0.,*RADII] for c in ([-1] if r==0 else range(18)) for dt in DTS}
        assert len(records)==len(expected)==3504 and {r['id'] for r in records}==expected
        for row in records:
            a=aa[row['anchor']];assert row['role']==role and row['block']==a['block_id']
            assert row['source_commit']==SOURCE and row['manifest_sha256']==MANIFEST
            assert row['id']==tid(row['anchor'],row['phase'],row['radius'],row['candidate'],row['dt'])
            if row['status']!='complete':continue
            context=contexts[(row['anchor'],row['phase'])];direction=np.zeros(2) if row['candidate']==-1 else context['directions'][row['candidate']]
            np.testing.assert_array_equal(row['direction'],direction);np.testing.assert_array_equal(row['delta'],row['radius']*direction)
            assert row['phase_epoch']==context['epoch']
            with np.load(directory/row['raw'],allow_pickle=False) as raw:
                raw_check(raw,a,context['initial'],direction,row['radius'],row['dt'],300)
                tail=raw['times']>75;assert tail.sum()==2250
                row['_rms']=metric_check(row['tail_metrics'],context['reference'],raw['times'][tail]+context['epoch'],np.tanh(raw['states'][tail]),context['T'])
            close(row['rmse'],row['_rms'])
        print(json.dumps({'stage':role,'previews':dict(Counter(r['status'] for r in rows)),
                          'targets':dict(Counter(r['status'] for r in records))}),flush=True)
    seal=read(directory/'model-seal.json');dseal=read(directory/'test-decision-seal.json');begin=read(directory/'test-target-started.json')
    assert seal['model_sha256']==sha(directory/'model.json') and seal['source_commit']==SOURCE
    assert seal['development_targets_sha256']==sha(directory/'development-targets.jsonl') and seal['development_features_sha256']==sha(directory/'development-features.json')
    assert dseal['model_seal_sha256']==sha(directory/'model-seal.json') and dseal['features_sha256']==sha(directory/'test-features.json')
    assert dseal['decisions_sha256']==sha(directory/'test-decisions.json')
    assert begin['model_seal_sha256']==sha(directory/'model-seal.json') and begin['decision_seal_sha256']==sha(directory/'test-decision-seal.json')
    assert receipt['started_at']<=seal['sealed_at']<=dseal['sealed_at']<=begin['started_at']<=receipt['ended_at']
    assert all(r['test_decision_seal_sha256']==begin['decision_seal_sha256'] for r in targets['test'])
    decisions=read(directory/'test-decisions.json');assert len(decisions)==96
    for i,d in enumerate(decisions):
        part=features['test'][i*18:(i+1)*18];first=part[0]
        assert (d['anchor'],d['block'],d['phase'],d['radius'])==(first['anchor'],first['block'],first['phase'],first['radius'])
        assert d['candidate_ids']==list(range(18)) and d['required_preview_calls']=={k:18 for k in POLICIES}
        assert d['preview_scores']==[r['preview_score'] for r in part]
        if all(r['status']=='complete' for r in part):assert d['policies']['min_preview']['candidate']==int(np.argmin(d['preview_scores']))
    packet=read(directory/'model.json');fit=model_check(packet,features,targets,anchors,decisions,names,geo_names)
    saved=read(directory/'summary.json');steps,oracles=summary_check(p,targets,decisions,saved)
    zero={};cross=[];cross_test=[]
    for role,records in targets.items():
        by={r['id']:r for r in records}
        for dt in DTS:
            z=[r for r in records if r['candidate']==-1 and r['dt']==dt];vals=[r['_rms'] for r in z if r['status']=='complete']
            zero[f'{role}-{dt}']={'planned':24,'complete':len(vals),'mean':float(np.mean(vals)) if len(vals)==24 else None,'maximum':max(vals) if len(vals)==24 else None}
            compare(saved['zero'][f'{role}-{dt}'],zero[f'{role}-{dt}'])
        for r in records:
            if r['dt']!=.005:continue
            other=by[tid(r['anchor'],r['phase'],r['radius'],r['candidate'],.01)]
            if r['status']==other['status']=='complete':
                difference=abs(r['_rms']-other['_rms']);cross.append(difference)
                if role=='test':cross_test.append(difference)
    distribution=lambda x: {'n':len(x),'mean':float(np.mean(x)),'median':float(np.median(x)),'p95':float(np.quantile(x,.95)),'maximum':float(max(x))} if x else {'n':0}
    test_lookup={(r['anchor'],r['phase'],r['radius'],r['candidate'],r['dt']):r for r in targets['test']}
    selected_cross={}
    for policy in POLICIES:
        differences=[]
        for d in decisions:
            c=d['policies'][policy]['candidate']
            if c is None:continue
            key=(d['anchor'],d['phase'],d['radius'],c)
            fine,coarse=test_lookup[(*key,.005)],test_lookup[(*key,.01)]
            if fine['status']==coarse['status']=='complete':differences.append(abs(fine['_rms']-coarse['_rms']))
        selected_cross[policy]=distribution(differences)
    example=next(d for d in decisions if d['anchor']==anchors['test'][0]['id'] and d['phase']==675 and d['radius']==.016)
    example_actual=[test_lookup[(example['anchor'],675,.016,c,.005)].get('_rms') for c in range(18)]
    costs={'collection_before_verification_seconds':receipt['wall_seconds'],'model_fit_and_write_seconds':seal['fit_seconds'],
           'sensitivity_solves':len(sens_cost),'sensitivity_solver_seconds':sum(sens_cost),'sensitivity_nfev':sum(sens_nfev),
           'summed_overlapping_preview_call_seconds':sum(r['wall_seconds'] for rows in features.values() for r in rows),
           'summed_overlapping_target_call_seconds':sum(r['wall_seconds'] for rows in targets.values() for r in rows),
           'paid_compute':receipt['paid_compute'],'paid_review':receipt['paid_review']}
    output={'source_commit':SOURCE,'manifest_sha256':MANIFEST,'receipt_sha256':sha(directory/'receipt.json'),
       'verified_artifacts':len(artifacts),'verified_bytes':sum(r['bytes'] for r in artifacts.values()),
       'verified_old_artifacts':len(p['source_artifacts']),'verified_source_files':len(p['source_files']),
       'eligibility':eligible,'preview_status':{role:dict(Counter(r['status'] for r in rows)) for role,rows in features.items()},
       'target_status':{role:dict(Counter(r['status'] for r in rows)) for role,rows in targets.items()},
       'metric_verification':STATS,'feature_rows_verified':sum(r['status']=='complete' for rows in features.values() for r in rows),
       'saved_model_checks':fit,'steps':steps,'oracle_per_block':oracles,'zero':zero,
       'cross_step_all':distribution(cross),'cross_step_test':distribution(cross_test),'costs':costs,
       'cross_step_selected_policies':selected_cross,
       'decision_identity':{k:sum(d['policies'][k]['candidate']==d['policies']['geometry']['candidate'] for d in decisions) for k in POLICIES},
       'first_frozen_example':{**example,'actual_fine_tail_rmse':example_actual},
       'audit_wall_seconds':time.perf_counter()-start,
       'limits':['Saved-phase reconstruction, not new trajectories or a newly optimized phase','This audit checks saved model inputs and predictions; independent ridge/CV replay is a separate audit','Blocks are a conditional descriptive sample from the already seen atlas']}
    if args.out:args.out.write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'stage':'complete','output':str(args.out),'primary':steps['0.005']['pooled'],
                     'model_check':fit,'metric_check':STATS,'cross_step':output['cross_step_all']}),flush=True)


if __name__=='__main__':main()
