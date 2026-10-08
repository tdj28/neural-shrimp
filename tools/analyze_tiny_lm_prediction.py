"""Analyze only frozen six-trajectory calibration data; never launch policies/assessment."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import subprocess

import scipy

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from neural_shrimp.tiny_lm_prediction import (FAMILIES,PENALTIES,burst_labels,
    family_columns,fit_logistic,predict,prediction_metrics)


def split_mask(record, split):
    lower,upper={'fit':(32,383),'tune':(400,583),'test':(600,783)}[split]
    t=np.arange(len(record['y']))
    return (t>=lower)&(t<=upper)&np.isfinite(record['y'])


def select_model(records, family):
    columns=family_columns(records[0]['groups'],family)
    pieces=[(r['X'][split_mask(r,'fit')][:,columns],r['y'][split_mask(r,'fit')]) for r in records]
    X=np.concatenate([p[0] for p in pieces]); y=np.concatenate([p[1] for p in pieces])
    candidates=[]
    for penalty in PENALTIES:
        model=fit_logistic(X,y,penalty)
        errors=[]
        for r in records:
            mask=split_mask(r,'tune')
            if mask.any():
                errors.append(float(np.mean((predict(model,r['X'][mask][:,columns])-r['y'][mask])**2)))
        if not errors:
            raise ValueError('no uncensored tuning rows; calibration uninformative')
        candidates.append({'penalty':penalty,'mean_trajectory_tune_brier':float(np.mean(errors)),'model':model})
    best=min(candidates,key=lambda r:(r['mean_trajectory_tune_brier'],-r['penalty']))
    model=best['model']
    threshold=float(np.quantile(predict(model,X),.95))
    return {'columns':columns.tolist(),'model':model,'alarm_threshold':threshold,
            'selection':[{k:v for k,v in c.items() if k!='model'} for c in candidates]}


def distinct_episodes(labels):
    indices=np.flatnonzero(labels==1)
    return int(0 if not len(indices) else 1+np.sum(np.diff(indices)>16))


def score_record(record, selected):
    mask=split_mask(record,'test'); indices=np.flatnonzero(mask)
    p=predict(selected['model'],record['X'][mask][:,selected['columns']])
    y=record['y'][mask]; threshold=selected['alarm_threshold']
    metrics=prediction_metrics(y,p,threshold)
    metrics['distinct_label_episodes']=distinct_episodes(record['y'][mask])
    delays=[]
    for t in indices[(p>=threshold)&(y==1)]:
        history=record['ce'][t-31:t+1];median=np.median(history)
        level=median+max(.1,3*1.4826*np.median(abs(history-median)))
        hits=np.flatnonzero(record['ce'][t+1:min(t+17,len(record['ce']))]>level)+1
        guard=record['guard_step']
        if guard is not None and t<guard<=t+16:
            hits=np.r_[hits,guard-t]
        if len(hits):delays.append(int(min(hits)))
    metrics['lead_time_at_true_positive_alarms_steps']=delays
    metrics['prediction']=p.tolist();metrics['steps']=indices.tolist()
    return metrics


def analyze(records):
    if sorted((r['seed'],r['factor']) for r in records)!=[(s,f) for s in (28151,28152) for f in (.8,1.6,2.5)]:
        raise ValueError('expected exactly six calibration cases; profile/assessment rejected')
    for r in records:
        if r['role']!='calibration':raise ValueError('not calibration data')
        expected=burst_labels(r['ce'],r['guard_step'])
        np.testing.assert_equal(r['y'],expected)
    output={'models':{},'per_trajectory':{},'leave_one_seed_out':{},'gate':{},
            'scope':'Development calibration only; time-held and seed-held prediction; no assessment/policies launched. Average precision is the PR summary; no independent-step uncertainty claim.'}
    for family in FAMILIES:
        selected=select_model(records,family);output['models'][family]=selected
        output['per_trajectory'][family]=[{'seed':r['seed'],'factor':r['factor'],**score_record(r,selected)} for r in records]
        output['leave_one_seed_out'][family]={}
        for held in (28151,28152):
            model=select_model([r for r in records if r['seed']!=held],family)
            output['leave_one_seed_out'][family][str(held)]={'selected_penalty':model['model']['penalty'],
                'scores':[{'factor':r['factor'],**score_record(r,model)} for r in records if r['seed']==held]}
    for seed in (28151,28152):
        a=[r for r in output['per_trajectory']['B2'] if r['seed']==seed]
        b=[r for r in output['per_trajectory']['B3'] if r['seed']==seed]
        def mean_defined(rows,key):
            values=[r[key] for r in rows if r[key] is not None]
            return float(np.mean(values)) if len(values)==3 else None
        brier2,brier3=mean_defined(a,'brier'),mean_defined(b,'brier')
        seed_records=[r for r in records if r['seed']==seed]
        target=np.concatenate([r['y'][split_mask(r,'test')] for r in seed_records])
        ap2=prediction_metrics(target,np.concatenate([r['prediction'] for r in a]))['average_precision']
        ap3=prediction_metrics(target,np.concatenate([r['prediction'] for r in b]))['average_precision']
        episodes=sum(r['distinct_label_episodes'] for r in a)
        passed=(None not in (brier2,brier3,ap2,ap3) and brier3<=.95*brier2 and ap3>=ap2 and episodes>=5)
        output['gate'][str(seed)]={'brier_B2':brier2,'brier_B3':brier3,'AP_B2':ap2,'AP_B3':ap3,
                                  'episodes':episodes,'pass':bool(passed)}
    output['calibration_gate_pass']=all(r['pass'] for r in output['gate'].values())
    output['next_action']='Requires independent review; this analyzer launches no further work.'
    return output


def verify_inventory(folder, manifest):
    names = [f['path'] for f in manifest['files']]
    if len(names) != len(set(names)):
        raise ValueError('duplicate manifest path')
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p.name != 'manifest.json'}
    if actual != set(names):
        raise ValueError('manifest must cover the exact collection file inventory')
    for item in manifest['files']:
        name = Path(item['path']); path = folder/name
        if name.is_absolute() or '..' in name.parts or not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError('unsafe manifest path')
        if path.is_symlink() or path.stat().st_size != item['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('manifest mismatch: '+str(name))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('collection',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();manifest_path=args.collection/'manifest.json';manifest=json.loads(manifest_path.read_text())
    verify_inventory(args.collection,manifest)
    if args.output.resolve().is_relative_to(args.collection.resolve()):
        raise ValueError('analysis output must be outside the immutable collection')
    records=[]
    for path in sorted(args.collection.glob('seed-*/result.json')):
        r=json.loads(path.read_text())
        with np.load(path.with_name('features.npz')) as data:
            records.append({**r,'groups':r['feature_groups'],'X':data['X'],'y':data['burst'],'ce':data['ce']})
    source_paths=['src/neural_shrimp/tiny_lm_prediction.py','tools/analyze_tiny_lm_prediction.py']
    source_hashes={}
    for name in source_paths:
        actual=(ROOT/name).read_bytes()
        if actual != subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT):
            raise ValueError('analysis source must match committed HEAD: '+name)
        source_hashes[name]=hashlib.sha256(actual).hexdigest()
    analysis_identity={'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                       'source_sha256':source_hashes,'numpy_version':np.__version__,'scipy_version':scipy.__version__}
    result=analyze(records)
    result['analysis_identity']=analysis_identity
    result['input_manifest_sha256']=hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with args.output.open('x') as handle:json.dump(result,handle,indent=2,allow_nan=False);handle.write('\n')


if __name__=='__main__':main()
