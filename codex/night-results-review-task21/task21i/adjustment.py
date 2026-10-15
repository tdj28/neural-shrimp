"""Own saved-row OLS/quartile arithmetic; producer transcript is a comparison, not code."""
import math
import numpy as np

TEMPERATURES=(.25,.32,.4)
COLUMNS=('intercept','chaotic','generation_entropy','nq','greedy')
TRANSCRIPT_SHA='cb94f35a9ca65d50851a1967ce27d59ff7a916e0842e306685150a2d018ae5bb'


def fit(X,y):
    finite=bool(np.isfinite(X).all() and np.isfinite(y).all())
    empty=len(y)==0
    out={'defined':False,'finite_input':finite,'empty':empty,'rank':None,
         'full_column_rank':False,'coefficients':None,'singular_values':None,'residual_sums':None}
    if not finite or empty:return out|{'reason':'nonfinite input' if not finite else 'empty rows'}
    try:
        b,res,rank,sv=np.linalg.lstsq(X,y,rcond=None)
    except np.linalg.LinAlgError as e:return out|{'reason':str(e)}
    defined=bool(np.isfinite(b).all() and np.isfinite(sv).all() and np.isfinite(res).all())
    return out|{'defined':defined,'rank':int(rank),'full_column_rank':int(rank)==len(COLUMNS),
                'coefficients':b.tolist(),'singular_values':sv.tolist(),'residual_sums':res.tolist(),
                'reason':None if defined else 'nonfinite solution'}


def mean(values):return float(np.mean(values)) if values else None


def quartiles(rows):
    selected=[r for r in rows if r['greedy']]
    ent=np.asarray([r['generation_entropy'] for r in selected],dtype=float)
    qualified=bool(len(ent) and np.isfinite(ent).all())
    q=np.quantile(ent,[.25,.5,.75],method='linear').tolist() if qualified else None
    result=[];diffs=[];weights=[]
    for i in range(4):
        lo=-math.inf if i==0 else q[i-1] if q else None
        hi=math.inf if i==3 else q[i] if q else None
        part=[r for r in selected if lo<=r['generation_entropy']<hi] if q else []
        ch=[r['acc'] for r in part if r['label']=='chaotic'];pe=[r['acc'] for r in part if r['label']=='periodic']
        valid=bool(q and ch and pe and all(math.isfinite(x) for x in ch+pe))
        cm,pm=mean(ch),mean(pe);diff=cm-pm if valid else None;w=min(len(ch),len(pe))
        result.append({'bin':i,'lower':lo,'upper':hi,'quantiles_defined':qualified,
                       'prompt_ids':[r['prompt'] for r in part],'chaotic':len(ch),'periodic':len(pe),
                       'mean_chaotic':cm,'mean_periodic':pm,'difference':diff,
                       'mean_entropy':mean([r['generation_entropy'] for r in part]),
                       'eligible_contrast':valid,'weight':w,
                       'reason':None if valid else 'undefined quantiles' if not qualified else 'empty arm' if not ch or not pe else 'nonfinite accuracy'})
        if valid:diffs.append(diff);weights.append(w)
    return {'greedy_prompts':len(selected),'nonfinite_entropy':int(np.sum(~np.isfinite(ent))),
            'quantiles':q,'bins':result,'eligible_bins':len(diffs),'eligible_weight':sum(weights),
            'weighted_difference':float(np.average(diffs,weights=weights)) if diffs else None,
            'scope':'All four lower-inclusive upper-exclusive bins retained. No filling of empty contrasts; min-arm-count weights only.'}


def calculate(rows,draws=5000):
    """One seed-1 generator, all three temperatures in order, fixed scheduled draws."""
    rng=np.random.default_rng(1);out={}
    for lt in TEMPERATURES:
        at=[r for r in rows if r['log10_temperature']==lt]
        selected=[r for r in at if r['label'] in ('chaotic','periodic') and r['sample_count']>0]
        X=np.asarray([[1,r['label']=='chaotic',r['generation_entropy'],r['task']=='nq',r['greedy']] for r in selected],dtype=float).reshape((-1,5))
        y=np.asarray([r['acc'] for r in selected],dtype=float)
        obs=fit(X,y);perms=[]
        for i in range(draws):
            changed=X.copy();changed[:,1]=rng.permutation(X[:,1])
            fitted=fit(changed,y)
            perms.append({'draw':i,'chaotic_coefficient':fitted['coefficients'][1] if fitted['defined'] else None,
                          'coefficients':fitted['coefficients'],'rank':fitted['rank'],
                          'full_column_rank':fitted['full_column_rank'],'defined':fitted['defined'],'reason':fitted['reason']})
        all_defined=bool(obs['defined'] and all(p['defined'] for p in perms))
        exceed=sum(abs(p['chaotic_coefficient'])>=abs(obs['coefficients'][1]) for p in perms) if all_defined else None
        task_rows={}
        for task in ('arithmetic','nq'):
            part=[r for r in selected if r['task']==task]
            ch=[r['acc'] for r in part if r['label']=='chaotic'];pe=[r['acc'] for r in part if r['label']=='periodic']
            task_rows[task]={'chaotic':len(ch),'periodic':len(pe),'mean_chaotic':mean(ch),'mean_periodic':mean(pe)}
        out[str(lt)]={'temperature':lt,'input_rows':len(at),'prompts':len(selected),
                     'prompt_ids':[r['prompt'] for r in selected],
                     'excluded':[{'prompt':r['prompt'],'label':r['label'],'sample_count':r['sample_count']} for r in at if r not in selected],
                     'tasks':task_rows,'quartiles':quartiles(selected),'columns':list(COLUMNS),'observed_fit':obs,
                     'permutations':perms,'draws':draws,'undefined_draws':sum(not p['defined'] for p in perms),
                     'rank_deficient_draws':sum(p['rank'] is not None and not p['full_column_rank'] for p in perms),
                     'exceedances':exceed,'raw_permutation_p':exceed/draws if exceed is not None and draws else None,
                     'plus_one_permutation_p':(exceed+1)/(draws+1) if exceed is not None else None,
                     'all_fits_full_rank':bool(obs['full_column_rank'] and all(p['full_column_rank'] for p in perms))}
    return {'temperatures':out,'seed':1,'draws_per_temperature':draws,'total_draws':3*draws,
            'scope':'Post-outcome descriptive adjustment. Rank-deficient minimum-norm solutions retained, not full-rank inference. Undefined draws are never dropped or replaced. Raw p uses all scheduled draws only if all coefficients are defined; separately named plus-one is not the producer statistic. No causal effect bound or overcorrection direction follows.'}


# Source-bound transcription of every printed scalar at its printed precision.
# Tuple task: nC, meanC, nP, meanP. Bin: index,nC,meanC,nP,meanP,diff,meanEntropy.
PRINTED={
 '0.25':{'prompts':199,'tasks':{'arithmetic':(7,'0.843',92,'0.914'),'nq':(31,'0.100',69,'0.206')},
         'bins':[(0,1,'1.000',29,'1.000','+0.000','0.00'),(2,6,'0.817',24,'0.967','-0.150','0.19'),(3,7,'0.443',23,'0.652','-0.209','1.37')],
         'weighted':'-0.169','chaotic':'-0.037','p':'0.2038','entropy':'-0.067'},
 '0.32':{'prompts':197,'tasks':{'arithmetic':(22,'0.745',75,'0.879'),'nq':(46,'0.100',54,'0.131')},
         'bins':[(0,2,'1.000',27,'0.993','+0.007','0.19'),(1,5,'0.840',24,'0.987','-0.147','0.45'),(2,12,'0.833',17,'0.859','-0.025','0.94'),(3,14,'0.286',16,'0.394','-0.108','2.85')],
         'weighted':'-0.077','chaotic':'-0.046','p':'0.0690','entropy':'-0.177'},
 '0.4':{'prompts':199,'tasks':{'arithmetic':(25,'0.448',75,'0.772'),'nq':(65,'0.077',34,'0.021')},
        'bins':[(0,4,'0.950',26,'0.923','+0.027','1.04'),(1,4,'0.875',25,'0.904','-0.029','1.44'),(2,13,'0.277',17,'0.594','-0.317','2.04'),(3,19,'0.221',11,'0.145','+0.076','3.61')],
        'weighted':'-0.103','chaotic':'-0.045','p':'0.0862','entropy':'-0.419'}}


def printed_comparisons(result):
    comparisons=[]
    def add(path,own,saved,fmt=None):
        rendered=format(own,fmt) if own is not None and fmt else own
        comparisons.append({'path':path,'own_full_precision':own,'own_rendered':rendered,'producer_printed':saved,
                            'format':fmt,'equal_at_printed_precision':rendered==saved})
    for lt,expected in PRINTED.items():
        actual=result['temperatures'][lt];add(lt+'/prompts',actual['prompts'],expected['prompts'])
        add(lt+'/temperature',actual['temperature'],float(lt))
        for task,tup in expected['tasks'].items():
            row=actual['tasks'][task]
            for key,value,fmt in zip(('chaotic','mean_chaotic','periodic','mean_periodic'),tup,(None,'.3f',None,'.3f')):add(lt+'/'+task+'/'+key,row[key],value,fmt)
        savedbins={r[0]:r[1:] for r in expected['bins']}
        for row in actual['quartiles']['bins']:
            idx=row['bin'];add(lt+f'/bin{idx}/printed',row['eligible_contrast'],idx in savedbins)
            if idx in savedbins:
                for key,value,fmt in zip(('chaotic','mean_chaotic','periodic','mean_periodic','difference','mean_entropy'),savedbins[idx],(None,'.3f',None,'.3f','+.3f','.2f')):add(lt+f'/bin{idx}/'+key,row[key],value,fmt)
        add(lt+'/weighted',actual['quartiles']['weighted_difference'],expected['weighted'],'+.3f')
        beta=actual['observed_fit']['coefficients']
        add(lt+'/chaotic_coefficient',beta[1] if beta else None,expected['chaotic'],'+.3f')
        add(lt+'/entropy_coefficient',beta[2] if beta else None,expected['entropy'],'+.3f')
        add(lt+'/raw_permutation_p',actual['raw_permutation_p'],expected['p'],'.4f')
    return {'transcript_sha256':TRANSCRIPT_SHA,'fields':comparisons,'count':len(comparisons),
            'mismatches':sum(not r['equal_at_printed_precision'] for r in comparisons),
            'scope':'Full own values compared only after original print formatting. No unavailable producer full-precision value is inferred.'}
