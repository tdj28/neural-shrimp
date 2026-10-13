"""Compare completed independent rows to predeclared manuscript/reference values.

Comparison never feeds back into fitted models or resampling.
"""
import json
from pathlib import Path
import numpy as np


def lookup(data,path):
    for key in path.split('.'):
        data=data[key]
    return data


def check(identifier,actual,expected,tolerance,reference,context=None):
    a,e=np.asarray(actual,dtype=float),np.asarray(expected,dtype=float)
    if a.shape!=e.shape:raise ValueError('comparison shape mismatch '+identifier)
    delta=a-e
    return {'id':identifier,'actual':actual,'expected':expected,'delta':delta.tolist(),
            'tolerance':tolerance,'within_rounding':bool(np.all(np.isfinite(a)) and np.all(np.abs(delta)<=tolerance+1e-14)),
            'reference':reference,'context':context}


def descriptive_claims(result,registry):
    rows=[]
    for c in registry['claims']:
        ident=c['id']; bits=ident.split('.')
        if bits[0]=='capture': value=lookup(result['capture'][bits[1]],'.'.join(bits[2:]))
        elif bits[:2]==['history','contrast']:value=result['history_contrasts'][bits[2]]['difference']
        elif bits[:2]==['history','ci']:value=result['history_contrasts'][bits[2]]['ci95']
        elif bits[0]=='history':value=lookup(result['history_pooled'][bits[1]],'.'.join(bits[2:]))
        elif bits[0]=='controller':value=lookup(result['controller'][bits[1]],'.'.join(bits[2:]))
        else:continue
        rows.append(check(ident,value,c['expected'],c['rounding_tolerance'],c['reference'],c['note']))
    return rows


def controller_reference(result,published):
    mapping={'hostile_share':'hostile_5_39','hostile_ci95':'ci_5_39','late_share':'hostile_35_39',
             'late_ci95':'ci_35_39','mean_counter':'mean_counter','gratitude_share':'gratitude_share_5_39',
             'efficiency':'efficiency_vs_constant','efficiency_ci95':'efficiency_ci'}
    rows=[]
    for name,r in result['controller'].items():
        old=published['arms'][name.removeprefix('p1-')]
        for key,target in mapping.items():
            if key in r:
                rows.append(check(name+'.'+key,r[key],old[target],1e-10,'p1-controller/controller.json'))
    return rows


def detector_rows(directory):
    rows={}
    for p in sorted(Path(directory).glob('event*-h*.json')):
        d=json.loads(p.read_text())
        for row in d['rows']:
            who=row.get('predictor',row.get('test'))
            rows[(d['run_length'],d['horizon'],who)]=row
    return rows


def detection_claims(rows,registry,experiment):
    comparisons=[]
    for c in registry['claims']:
        bits=c['id'].split('.')
        if bits[0]!=experiment:continue
        key=(int(bits[1][1:]),int(bits[2][1:]),bits[3])
        row=rows[key]
        value=lookup(row,'.'.join(bits[4:])) if bits[4]!='flagged' else row['flagged_at_train_10pct_fpr'][bits[5]]
        comparisons.append(check(c['id'],value,c['expected'],c['rounding_tolerance'],c['reference'],c['status']))
    return comparisons


def detection_reference(rows,published,experiment):
    comparisons=[]
    for old in published['results']:
        run=2 if 'two consecutive' in old['event'] else 1
        who=old.get('predictor',old.get('test'))
        if who!='pooled':who=who.rsplit('/',1)[-1]
        key=(run,old['h'],who)
        row=rows[key]
        for field in ('rows','events','auc_baseline','auc_with_predictor','increment','ci95'):
            comparisons.append(check(f'{experiment}.c{run}.h{old["h"]}.{who}.{field}',row[field],old[field],
                0 if field in ('rows','events') else 1e-10,'published comparison JSON',
                'L7 two-consecutive reference may predate amendment2; difference must be preserved' if experiment=='L7' and run==2 else None))
        for field,value in old.get('flagged_at_train_10pct_fpr',{}).items():
            comparisons.append(check(f'{experiment}.c{run}.h{old["h"]}.{who}.flagged.{field}',
                row['flagged_at_train_10pct_fpr'][field],value,1e-10,'published comparison JSON'))
    return comparisons


def l7_oof_operating_point(path):
    """Explicit descriptive threshold on all negative OOF scores, not an L8 threshold."""
    with np.load(path,allow_pickle=False) as data:
        y=data['y']; result={}
        for key in ('baseline','D3_state_probe'):
            score=data[key]; threshold=float(np.quantile(score[y==0],.9))
            result[key]={'threshold':threshold,'flagged_positive':float(np.mean(score[y==1]>threshold)),
                         'realized_fpr':float(np.mean(score[y==0]>threshold))}
    return {'definition':'pooled negative out-of-fold90thpercentile; strict>, descriptive not pre-frozen threshold',
            'models':result}
