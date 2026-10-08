"""Bounded independent saved-evidence audit; never trains or evaluates a corpus."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

SOURCE='b9e5b8c3973e8bd776402b0cb3d8d61609bb71f1'
REVIEW='39ba17219075c92cd33038b032501bb60da6fd25'
PRODUCER='9f98bb6df2fdf6fb08773715fd04839f91492fa3'
PAIRS=[(s,f) for s in (28151,28152) for f in (.8,1.6,2.5)]
FAMILIES=('B0','B1','B2','B3')


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def labels(loss,guard):
    y=np.full(len(loss),np.nan)
    for t in range(31,len(loss)):
        if guard is not None and t<guard<=t+16:y[t]=1.;continue
        if t+16>=len(loss):continue
        past=loss[t-31:t+1];center=np.median(past)
        level=center+max(.1,3*1.4826*np.median(np.abs(past-center)))
        y[t]=float(any(value>level for value in loss[t+1:t+17]))
    return y


def mask(y,split):
    lo,hi={'fit':(32,383),'tune':(400,583),'test':(600,783)}[split]
    t=np.arange(len(y));return (t>=lo)&(t<=hi)&np.isfinite(y)


def metrics(y,p,threshold):
    if not len(y):return dict(rows=0,brier=None,average_precision=None,false_alarms_per_100=None)
    assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    ap=None
    if 0<sum(y)<len(y):
        npos=float(sum(y));tp=0.;n=0;ap=0.
        for score in sorted(set(p.tolist()),reverse=True):
            same=p==score;delta=float(y[same].sum());tp+=delta;n+=int(same.sum())
            ap+=(delta/npos)*(tp/n)
    return dict(rows=len(y),positive_rows=int(sum(y)),brier=float(sum((p-y)**2)/len(y)),
                average_precision=ap,false_alarms_per_100=float(100*sum((p>=threshold)&(y==0))/len(y)))


def near(actual,expected):
    if expected is None:assert actual is None,(actual,expected)
    else:np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=5e-15)


def logistic(model,X):
    mean=np.asarray(model['mean']);scale=np.asarray(model['scale']);weights=np.asarray(model['weights'])
    X=np.where(np.isfinite(X),X,mean)
    z=(X-mean)/scale@weights[1:]+weights[0]
    p=np.empty(len(z));positive=z>=0
    p[positive]=1/(1+np.exp(-z[positive]));ez=np.exp(z[~positive]);p[~positive]=ez/(1+ez)
    return p


def audit(collection,analysis_path,collector_launcher,analysis_launcher):
    started=time.monotonic();manifest_path=collection/'manifest.json';manifest=json.loads(manifest_path.read_text())
    entries=manifest['files'];names=[e['path'] for e in entries]
    assert len(names)==len(set(names))
    actual={str(p.relative_to(collection)) for p in collection.rglob('*') if p.is_file() and p.name!='manifest.json'}
    assert actual==set(names),'collection file inventory mismatch'
    for e in entries:
        p=collection/e['path'];assert not p.is_symlink() and p.resolve().is_relative_to(collection.resolve())
        assert p.stat().st_size==e['bytes'] and sha(p)==e['sha256'],e['path']
    assert manifest['metadata']['source_commit']==SOURCE
    for name,digest in manifest['metadata']['source_sha256'].items():
        assert hashlib.sha256(subprocess.check_output(['git','show',SOURCE+':'+name])).hexdigest()==digest
    launch=json.loads(collector_launcher.read_text());al=json.loads(analysis_launcher.read_text())
    assert launch['returncode']==0 and al['returncode']==0
    for log in (launch,al):
        assert log['source_commit']==SOURCE and log['review_commit']==REVIEW
        assert set(log['threads'].values())=={'1'}
    receipt=json.loads(subprocess.check_output(['git','show',PRODUCER+':docs/explore-tiny-lm-calibration-receipt.json']))
    assert receipt['source_commit']==SOURCE and receipt['independent_source_review_commit']==REVIEW
    assert receipt['manifest_sha256']==sha(manifest_path) and receipt['analysis']['sha256']==sha(analysis_path)
    assert receipt['manifest_files']==len(entries) and receipt['manifest_file_bytes']==sum(e['bytes'] for e in entries)
    for name,record in [('calibration',launch),('analysis',al)]:
        bound=receipt['launchers'][name];assert bound['record']==record
        assert sha(Path(bound['log_path']))==bound['log_sha256']==record['log_sha256']
        assert sha(Path(bound['launcher_path']))==bound['launcher_sha256']
    # Field spellings can vary; require both exact identity strings in the durable launcher.
    assert SOURCE in json.dumps(launch) and REVIEW in json.dumps(launch)
    assert not (collection/'failure.json').exists()
    records={};case_summaries=[]
    for case in sorted(collection.glob('seed-*')):
        if not case.is_dir():continue
        r=json.loads((case/'result.json').read_text());key=(r['seed'],r['factor'])
        assert key not in records and key in PAIRS and r['role']=='calibration'
        assert r['requested_updates']==800 and r['outcome'] in ('completed','numerical_guard_stop')
        state_count=0;all_states=[]
        for p in sorted(case.glob('states-*.npy')):
            lo,hi=map(int,p.stem.split('-')[1:]);state=np.load(p,mmap_mode='r')
            assert lo==state_count and len(state)==hi-lo+1 and state.shape[1]==15776
            assert np.isfinite(state).all() or r['outcome']=='numerical_guard_stop'
            state_count+=len(state);all_states.append(state)
        assert state_count==r['completed_updates']+1==r['state_rows']
        with np.load(case/'features.npz') as z:X,ce,y=z['X'],z['ce'],z['burst']
        assert X.shape==(r['completed_updates'],81) and len(ce)==len(y)==len(X)
        assert np.isfinite(ce).all();np.testing.assert_equal(y,labels(ce,r['guard_step']))
        trace=[json.loads(line) for line in (case/'trace.jsonl').read_text().splitlines()]
        assert [row['step'] for row in trace]==list(range(len(ce)))
        np.testing.assert_equal([row['ce'] for row in trace],ce)
        if r['outcome']=='completed':assert len(ce)==800 and r['guard_step'] is None
        else:
            guard=json.loads((case/'guard.json').read_text());assert r['guard_step']==len(ce)==guard['step']
        names={name:i for i,name in enumerate(r['feature_names'])};planes=[]
        for lag in range(8):
            prefix='gram' if lag==0 else f'gram_history_lag{lag}'
            ab,ac,bc=[X[:,names[f'{prefix}_{j}']] for j in (3,4,5)]
            den1=np.maximum(0.,1-ab*ab);den2=np.maximum(0.,1-bc*bc)
            valid=np.isfinite(den1+den2+ac)&(den1>=1e-4)&(den2>=1e-4)
            p=np.full(len(X),np.nan);p[valid]=np.clip((ab[valid]*bc[valid]-ac[valid])/np.sqrt(den1[valid]*den2[valid]),-1,1)
            planes.append(p)
        planes=np.asarray(planes);mean=np.mean(planes,axis=0);std=np.std(planes,axis=0)
        # Startup rows need fewer than four directions and are never fitted.
        for name,expected in [('plane_coherence',planes[0]),('plane_mean8',mean),('plane_std8',std)]:
            np.testing.assert_allclose(X[32:,names[name]],expected[32:],rtol=2e-10,atol=2e-12,equal_nan=True)
        np.testing.assert_equal(X[32:,names['plane_unavailable']],(~np.isfinite(planes[0]))[32:])
        np.testing.assert_equal(X[32:,names['plane_history_unavailable']],(~np.isfinite(mean))[32:])
        r.update(X=X,ce=ce,y=y);records[key]=r
        case_summaries.append(dict(seed=key[0],factor=key[1],outcome=r['outcome'],completed_updates=len(ce),
            state_rows=state_count,guard_step=r['guard_step'],labels_available=int(np.isfinite(y).sum()),
            positive_labels=int(np.nansum(y)),split_rows={s:int(mask(y,s).sum()) for s in ('fit','tune','test')}))
    assert sorted(records)==PAIRS
    analysis=json.loads(analysis_path.read_text());assert analysis['input_manifest_sha256']==sha(manifest_path)
    identity=analysis['analysis_identity'];assert identity['commit']==SOURCE
    for name,digest in identity['source_sha256'].items():
        assert hashlib.sha256(subprocess.check_output(['git','show',SOURCE+':'+name])).hexdigest()==digest
    summary_scores={};pooled={}
    for family in FAMILIES:
        selected=analysis['models'][family];allowed={'common',*FAMILIES[:FAMILIES.index(family)+1]}
        groups=records[PAIRS[0]]['feature_groups'];cols=[i for i,g in enumerate(groups) if g in allowed]
        assert selected['columns']==cols
        train=np.concatenate([r['X'][mask(r['y'],'fit')][:,cols] for r in records.values()])
        model=selected['model'];assert model['training_rows']==len(train)
        count=np.isfinite(train).sum(0);mean=np.divide(np.nansum(train,0),count,out=np.zeros(len(cols)),where=count>0)
        scale=np.maximum(np.where(np.isfinite(train),train,mean).std(0),1e-8)
        near(model['mean'],mean);near(model['scale'],scale)
        near(selected['alarm_threshold'],np.quantile(logistic(model,train),.95))
        candidates=selected['selection'];assert sorted(c['penalty'] for c in candidates)==[.01,.1,1.,10.]
        winner=min(candidates,key=lambda x:(x['mean_trajectory_tune_brier'],-x['penalty']))
        assert winner['penalty']==model['penalty']
        tune_errors=[]
        for r in records.values():
            use=mask(r['y'],'tune')
            if use.any():tune_errors.append(np.mean((logistic(model,r['X'][use][:,cols])-r['y'][use])**2))
        near(winner['mean_trajectory_tune_brier'],np.mean(tune_errors))
        rows=analysis['per_trajectory'][family];assert sorted((r['seed'],r['factor']) for r in rows)==PAIRS
        checked=[]
        for score in rows:
            r=records[(score['seed'],score['factor'])];use=mask(r['y'],'test');steps=np.flatnonzero(use)
            assert score['steps']==steps.tolist();p=np.asarray(score['prediction']);y=r['y'][use]
            near(p,logistic(model,r['X'][use][:,cols]))
            metric=metrics(y,p,selected['alarm_threshold'])
            for name,value in metric.items():near(score[name],value)
            positives=steps[y==1];episodes=0 if not len(positives) else 1+int(sum(np.diff(positives)>16))
            assert score['distinct_label_episodes']==episodes
            checked.append(dict(seed=r['seed'],factor=r['factor'],**metric,episodes=episodes))
        summary_scores[family]=checked
        for seed in (28151,28152):
            scores=[s for s in rows if s['seed']==seed]
            truth=np.concatenate([records[(seed,s['factor'])]['y'][mask(records[(seed,s['factor'])]['y'],'test')] for s in scores])
            prob=np.concatenate([s['prediction'] for s in scores])
            pooled[family,seed]=metrics(truth,prob,selected['alarm_threshold'])
        for held,entry in analysis['leave_one_seed_out'][family].items():
            assert entry['selected_penalty'] in [.01,.1,1.,10.]
            for score in entry['scores']:
                r=records[(int(held),score['factor'])];y=r['y'][mask(r['y'],'test')];p=np.asarray(score['prediction'])
                for name in ('rows','positive_rows','brier','average_precision'):
                    if name in score:near(score[name],metrics(y,p,.95).get(name))
    gates={}
    for seed in (28151,28152):
        a=[r for r in summary_scores['B2'] if r['seed']==seed];b=[r for r in summary_scores['B3'] if r['seed']==seed]
        b2=None if any(r['brier'] is None for r in a) else float(np.mean([r['brier'] for r in a]))
        b3=None if any(r['brier'] is None for r in b) else float(np.mean([r['brier'] for r in b]))
        ap2=pooled['B2',seed]['average_precision'];ap3=pooled['B3',seed]['average_precision'];episodes=sum(r['episodes'] for r in a)
        passed=bool(None not in (b2,b3,ap2,ap3) and b3<=.95*b2 and ap3>=ap2 and episodes>=5)
        expected=dict(brier_B2=b2,brier_B3=b3,AP_B2=ap2,AP_B3=ap3,episodes=episodes,**{'pass':passed})
        for k,v in expected.items():near(analysis['gate'][str(seed)][k],v)
        gates[str(seed)]=expected
    assert analysis['calibration_gate_pass']==all(g['pass'] for g in gates.values())
    assert receipt['gate']==analysis['gate'] and receipt['calibration_gate_pass']==analysis['calibration_gate_pass']
    for case in receipt['cases']:
        raw=records[(case['seed'],case['factor'])]
        for key in ('eta','outcome','guard_step','completed_updates','state_rows','elapsed_seconds','feature_seconds','initial_ce','last_preupdate_ce'):
            assert case[key]==raw[key]
        for split in ('fit','tune','test'):
            selected=raw['y'][mask(raw['y'],split)]
            assert case['labels'][split]=={'finite':len(selected),'positive':int(sum(selected))}
    return dict(passed=True,source_commit=SOURCE,source_review_commit=REVIEW,collection=str(collection),
        manifest_sha256=sha(manifest_path),analysis_sha256=sha(analysis_path),collector_launcher_sha256=sha(collector_launcher),
        analysis_launcher_sha256=sha(analysis_launcher),manifest_files=len(entries),manifest_bytes=sum(e['bytes'] for e in entries),
        cases=case_summaries,gate=gates,calibration_gate_pass=analysis['calibration_gate_pass'],elapsed_seconds=time.monotonic()-started,
        scope='Saved-evidence audit only: no training, gradient recomputation, optimization refit, assessment or policy. Recomputed selected-model predictions, train-only statistics, selected tuning score, all saved Brier/AP and aggregate gate; alternate penalty fits and LOSO coefficients are not independently refitted.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('collection','analysis','collector-launcher','analysis-launcher','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();result=audit(a.collection,a.analysis,a.collector_launcher,a.analysis_launcher)
    with a.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
