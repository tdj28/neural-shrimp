"""Task11 classifier robustness, authored without importing producer analyses.

Task6's separately authored reducers/model definitions are explicitly reused.
No second-classifier outcome for discarded regenerate originals is constructed.
"""
import numpy as np
from definitions import ALL,CAPTURE,HISTORY,L6,L10,TRAIN,TEST,POOLS,CONTRASTS
from task6_reducers import capture_row,difference_interval,mean_interval,controller_rows,increment
from task6_detection import make_table,frozen_fit,frozen_score,operating_point


def agreement(first,second):
    a,b=np.asarray(first)>.5,np.asarray(second)>.5
    if a.shape!=b.shape or a.ndim!=2: raise ValueError('agreement requires paired dialogue/turn shapes')
    counts={f'first{x}_second{y}':int(((a==x)&(b==y)).sum()) for x in (0,1) for y in (0,1)}
    return {'counts':counts,'turns':a.size,'dialogues':len(a),'agreement':float((a==b).mean()),
            'first_only':float((a&~b).mean()),'second_only':float((~a&b).mean()),
            'first_share':float(a.mean()),'second_share':float(b.mean()),
            'share_difference_second_minus_first':float(b.mean()-a.mean())}


def l10_shown(data,draws=4000):
    """Shown replies only; fresh declared arm-order RNG, not the would-be bootstrap."""
    rng=np.random.default_rng(0);out={};means={}
    for name in L10:
        arm=data[name];a=arm['score']>.5;v=a[:,5:40].mean(1);late=a[:,35:40].mean(1)
        means[name]=v
        out[name]={'shown_hostile_share':float(v.mean()),'shown_ci95':mean_interval(v,rng,draws),
                   'shown_late_share':float(late.mean()),'shown_late_ci95':mean_interval(late,rng,draws),
                   'mean_counter':float(np.abs(arm['counter'][:,5:40]).mean()),
                   'regenerated_5_39':float(arm['regenerated'][:,5:40].mean()),
                   'regenerated_5_14':float(arm['regenerated'][:,5:15].mean()),
                   'regenerated_30_39':float(arm['regenerated'][:,30:40].mean()),
                   'would_be_second_classifier':'unavailable: discarded original replies were not retained'}
    curve=L10[:3];doses=[0.,.02,.04]
    for name in L10[3:]:
        dose=out[name]['mean_counter'];v=means[name]
        point=float(v.mean()-np.interp(dose,doses,[means[c].mean() for c in curve]))
        boot=np.empty(draws)
        for k in range(draws):
            own=v[rng.integers(0,len(v),len(v))].mean()
            ref=[means[c][rng.integers(0,len(means[c]),len(means[c]))].mean() for c in curve]
            boot[k]=own-np.interp(dose,doses,ref)
        out[name].update(shown_efficiency=point,shown_efficiency_ci95=np.percentile(boot,[2.5,97.5]).tolist(),
                         dose_outside_constant_range=bool(dose<0 or dose>.04))
    return out


def one_descriptive(data,instrument,draws):
    rows={a:{**data[a],'score':data[a][instrument]} for a in ALL}
    pools={n:np.concatenate([rows[a]['score'] for a in arms]) for n,arms in POOLS.items()}
    contrasts={a+'-minus-'+b:difference_interval((pools[a][:,35:40]>.5).mean(1),
                 (pools[b][:,35:40]>.5).mean(1),draws) for a,b in CONTRASTS}
    return {'capture':{a:capture_row(rows[a]['score']) for a in CAPTURE},
            'history_seed_rows':{a:capture_row(rows[a]['score']) for a in HISTORY},
            'history_pooled':{n:capture_row(s) for n,s in pools.items()},'history_contrasts':contrasts,
            'l6':controller_rows({a:rows[a] for a in L6},draws),'l10_shown':l10_shown(rows,draws)}


def direction(v):
    return 'positive' if v>0 else 'negative' if v<0 else 'zero'


def evidence(v,interval):
    return {'direction':direction(v),'ci_excludes_zero':bool(interval[0]>0 or interval[1]<0),
            'positive_lower_bound':bool(interval[0]>0)}


def conclusion_rows(outputs):
    """Frozen descriptive criteria; mixed/negative values never removed."""
    rows=[]
    for kind in ('share','stay'):
        armnames=['p1-h040','p1-h050','p1-h060','p1-h070']
        row={'claim':'L2 strictly increasing '+kind,'classifiers':{}}
        for c,o in outputs.items():
            vals=[o['capture'][a][kind] if kind=='share' else o['capture'][a]['stay']['probability'] for a in armnames]
            row['classifiers'][c]={'values':vals,'holds':all(v is not None for v in vals) and all(x<y for x,y in zip(vals,vals[1:]))}
        row['criterion_changed']=row['classifiers']['first']['holds']!=row['classifiers']['second']['holds'];rows.append(row)
    for a,b in CONTRASTS:
        key=a+'-minus-'+b;row={'claim':'history '+key,'classifiers':{}}
        for c,o in outputs.items():
            r=o['history_contrasts'][key];row['classifiers'][c]=evidence(r['difference'],r['ci95'])
        row['criterion_changed']=row['classifiers']['first']!=row['classifiers']['second'];rows.append(row)
    for family,names,point,ci in [('l6',L6[1:5],'efficiency','efficiency_ci95'),
                                  ('l10_shown',L10[3:],'shown_efficiency','shown_efficiency_ci95')]:
        for name in names:
            row={'claim':family+' '+name+' minus same-run constant dose response','classifiers':{}}
            for c,o in outputs.items():
                r=o[family][name];row['classifiers'][c]=evidence(r[point],r[ci])
            row['criterion_changed']=row['classifiers']['first']!=row['classifiers']['second'];rows.append(row)
    return rows


def descriptive(data,draws=4000):
    if set(data)!=set(ALL): raise ValueError('exact all44 required')
    outputs={c:one_descriptive(data,c,draws) for c in ('first','second')}
    return {'classifiers':outputs,'agreement':{a:agreement(data[a]['first'],data[a]['second']) for a in ALL},
            'conclusion_checks':conclusion_rows(outputs),'actual_sample_args':{a:data[a]['args'] for a in ALL},
            'conventions':{'threshold':'strictly >0.5, both stored float32 outputs',
             'bootstrap':'4000 dialogue draws; seed0; fixed arm/order and independent contrast sampling',
             'l10_scope':'shown replies only; no would-be or basin-prevention robustness inference',
             'status':'exploratory classifier robustness; original headlines/design unchanged'}}


def detection(data,instrument,draws=2000):
    train=[{**data[a],'score':data[a][instrument]} for a in TRAIN]
    test=[{**data[a],'score':data[a][instrument]} for a in TEST]
    tab=make_table(train,3,1,'baseline_state');other=make_table(test,3,1,'baseline_state')
    diagnostics=[];rows=[]
    reason=('training set lacks one class' if len(np.unique(tab['y']))!=2 else
            'zero observed at-risk test rows' if not len(other['y']) else None)
    if reason is not None:
        return {'instrument':instrument,'admissible':False,'reason':reason,
                'rows':[],'n_train':len(tab['y']),'positive_train_windows':int(tab['y'].sum()),
                'n_test':len(other['y']),'diagnostics':[],'warnings':[],
                'all_logistic_converged':False,'positive_pooled_lower_bound':None}, {
                    'y':other['y'],'group':other['group'],'arm':other['arm'],'turn':other['turn']}
    fitted=frozen_fit(tab,diagnostics);scores,thresholds=frozen_score(fitted,other)
    for name,mask in [('pooled',np.ones(len(other['y']),bool)),
                      (TEST[0],other['arm']==0),(TEST[1],other['arm']==1)]:
        y=other['y'][mask];g=other['group'][mask];s=[v[mask] for v in scores]
        if len(np.unique(y))!=2:
            row={'test':name,'admissible':False,'reason':'test row lacks one class',
                 'rows':len(y),'events':int(y.sum()),'dialogues':len(np.unique(g)),
                 'auc_baseline':None,'auc_with_predictor':None,'increment':None,'ci95':None,
                 'bootstrap_draws':draws,'bootstrap_valid':0,'bootstrap_seed':0,'thresholds':thresholds}
        else:
            row=increment(y,g,s[0],s[1],draws)
            row.update(test=name,admissible=True,thresholds=thresholds,
                       flagged_at_train_10pct_fpr=operating_point(y,s,thresholds))
        rows.append(row)
    warnings=[w for d in diagnostics for w in d.get('warnings',[])]
    converged=all(not d.get('warnings') and max(d.get('iterations',[0]))<5000 for d in diagnostics)
    admissible=bool(rows[0]['admissible'] and converged)
    result={'instrument':instrument,'event':'first hostile','horizon':3,'rows':rows,'admissible':admissible,
            'n_train':len(tab['y']),'positive_train_windows':int(tab['y'].sum()),
            'n_test':len(other['y']),'diagnostics':diagnostics,'warnings':warnings,
            'all_logistic_converged':converged,
            'positive_pooled_lower_bound':bool(rows[0]['ci95'][0]>0) if admissible else None,
            'scope':'exploratory: classifier replaces labels AND score-derived baseline features; L7-only fits'}
    arrays={'y':other['y'],'group':other['group'],'arm':other['arm'],'turn':other['turn'],
            'baseline':scores[0],'with_state':scores[1]}
    return result,arrays
