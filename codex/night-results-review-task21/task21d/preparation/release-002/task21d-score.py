"""Independent finite-grid fold-window rescorer. Never imports producer code.

This module defines arithmetic only. No file is read and no target is run at import.
The caller must authenticate a reviewed input/card and retain all attempted results.
"""
import math
import numpy as np

COORDINATES = ('log10_temperature', 'alpha')
GRID_COUNTS = (20001, 2001)
SPREAD_LIMIT = 0.10


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('finite scalar required')
    return float(value)


def integer(value, minimum=0):
    v=finite(value)
    if v!=int(v) or v<minimum: raise ValueError('integer out of range')
    return int(v)


def collect(cells, fold, q, coordinate):
    """Literal source ordering, with input-index/run/exclusion/duplicate ledgers."""
    if coordinate not in COORDINATES: raise ValueError('unknown coordinate')
    q=integer(q,1); fold=finite(fold)
    pts=[]
    for i,c in enumerate(cells):
        x=finite(c[coordinate]); cl=c['class']
        if not isinstance(cl,str): raise ValueError('class must be a string')
        p=integer(c['period']) if cl=='periodic' else 0
        pts.append({'input_index':i,'coordinate':x,'effective_period':p,'saved_class':cl,'saved_period':c.get('period')})
    pts.sort(key=lambda c:(c['coordinate'],c['effective_period']))
    rs=[]
    for c in pts:
        if not rs or c['effective_period']!=rs[-1]['period']:
            rs.append({'index':len(rs),'period':c['effective_period'],'lo':c['coordinate'],'hi':c['coordinate'],'cell_indices':[]})
        rs[-1]['hi']=c['coordinate']; rs[-1]['cell_indices'].append(c['input_index'])
    for r in rs: r['cells']=len(r['cell_indices'])
    duplicates=[]
    for p in pts:
        if duplicates and p['coordinate']==duplicates[-1]['coordinate']:
            duplicates[-1]['input_indices'].append(p['input_index'])
        else: duplicates.append({'coordinate':p['coordinate'],'input_indices':[p['input_index']]})
    duplicates=[d for d in duplicates if len(d['input_indices'])>1]
    base=[r for r in rs if r['period']>0 and q%r['period']==0]
    if not base: raise ValueError('no base-period or divisor run on line')
    chosen=min(base,key=lambda r:min(abs(r['lo']-fold),abs(r['hi']-fold)))
    side=1. if (chosen['lo']+chosen['hi'])/2<fold else -1.
    choices={}; ledger=[]
    for r in rs:
        i,p,lo,hi,n=r['index'],r['period'],r['lo'],r['hi'],r['cells']
        neighbors=[j for j in (i-1,i+1) if 0<=j<len(rs)]
        doubling=p%2==0 and any(rs[j]['period']==p//2 for j in neighbors)
        reasons=[]
        if p<=q: reasons.append('period_not_above_base')
        if n<2: reasons.append('single_cell_unresolved')
        if doubling: reasons.append('adjacent_half_period_run')
        if side*((lo+hi)/2-fold)<=0: reasons.append('not_beyond_fold_by_run_centre')
        row={'run_index':i,'period':p,'neighbor_runs':neighbors,'exclusion_reasons':reasons,'eligible':not reasons}
        ledger.append(row)
        if reasons: continue
        # The historical strict comparison selects hi on an equal-distance tie.
        edge=lo if abs(lo-fold)<abs(hi-fold) else hi
        w={'period':p,'laps':p/q,'edge':edge,'d':abs(edge-fold),'d_centre':abs((lo+hi)/2-fold),'cells':n}
        row['candidate']=w
        if p not in choices or w['d']<choices[p][0]['d']:
            if p in choices: ledger[choices[p][1]]['period_choice']='replaced_by_nearer_run'
            choices[p]=(w,i); row['period_choice']='selected_nearest_per_period'
        else: row['period_choice']='farther_or_equal_repeated_period'
    windows=sorted([v[0] for v in choices.values()],key=lambda w:w['d'])
    return windows, side, {'sorted_cells':pts,'runs':rs,'duplicate_coordinates':duplicates,
                          'base_run_candidates':[r['index'] for r in base], 'selected_base_run':chosen['index'],
                          'run_dispositions':ledger}


def fan_windows(windows,q,mode):
    """Keep the historical stop rule separate from the amended skip rule."""
    if mode not in ('original_stop','amended_skip'): raise ValueError('unknown fan mode')
    if not windows: return []
    chain=[windows[0]]
    if mode=='original_stop':
        for w in windows[1:]:
            if w['period']!=chain[-1]['period']-q: break
            chain.append(w)
    else:
        by_period={w['period']:w for w in windows}
        while True:
            candidate=by_period.get(chain[-1]['period']-q)
            if candidate is None or candidate['d']<=chain[-1]['d']: break
            chain.append(candidate)
    return chain


def grid_fit(windows,key='d',upper=10.):
    """Enumerate exactly the two frozen grids, preserving their order and first tie.

    Chunked NumPy arithmetic independently minimizes the declared spread. The
    grids, objective and mean are unchanged; every producer delta is reportable.
    """
    if not windows: raise ValueError('empty fit')
    x=np.asarray([finite(w['laps']) for w in windows],dtype=np.float64)
    d=np.asarray([finite(w[key]) for w in windows],dtype=np.float64)
    if np.any(x<=1) or np.any(d<0): raise ValueError('invalid laps or distances')
    lower=-float(x.min())+1e-3
    grid=np.concatenate((np.linspace(lower,upper,GRID_COUNTS[0]),np.linspace(lower,lower+1.,GRID_COUNTS[1])))
    selected=None; nonfinite=[]; ties=0
    for first in range(0,len(grid),512):
        g=grid[first:first+512]
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            vals=d[None,:]*(x[None,:]+g[:,None])**2
            means=vals.mean(axis=1); spreads=(vals.max(axis=1)-vals.min(axis=1))/means
        for j,(delta,c,spread) in enumerate(zip(g,means,spreads)):
            idx=first+j; c=float(c); spread=float(spread)
            if not math.isfinite(c) or not math.isfinite(spread): nonfinite.append(idx)
            # Like the original scalar loop, a first NaN stays selected; it is
            # explicitly marked undefined, not silently skipped or qualified.
            if selected is None or spread<selected['spread']:
                selected={'C':c,'delta':float(delta),'spread':spread,'grid_index':idx}; ties=1
            elif spread==selected['spread']: ties+=1
    selected['audit']={'lower':lower,'upper':upper,'counts':list(GRID_COUNTS),
        'candidate_count':len(grid),'first_minimum_tie_count':ties,'nonfinite_candidate_indices':nonfinite,
        'defined':math.isfinite(selected['C']) and math.isfinite(selected['spread'])}
    return selected


def score_variant(windows,fold,side,q,a,b,mode):
    chain=fan_windows(windows,q,mode); ab=abs(a*b)
    result={'fold':fold,'side':side,'base_period':q,'windows':windows,'fan':[w['period'] for w in chain],
            'ab':ab,'a':a,'b':b,'fits':[]}
    audit=[]
    for n in range(4,len(chain)+1):
        f=grid_fit(chain[:n]); bounded=grid_fit(chain[:n],upper=1.); centre=grid_fit(chain[:n],key='d_centre')
        fa={}
        for label,obj in [('free',f),('bounded',bounded),('centre',centre)]:
            fa[label]={'grid_index':obj.pop('grid_index'),**obj.pop('audit')}
            obj['passage_fraction']=float(np.sqrt(obj['C']*ab)/np.pi)
        f.update(windows=n,offset_physical=bool(f['delta']<=1),bounded=bounded,centre=centre)
        result['fits'].append(f); audit.append({'windows':n,'fit_grids':fa})
    fits=result['fits']
    result['law_holds']=bool(fits and fits[0]['spread']<=SPREAD_LIMIT)
    result['law_holds_bounded']=bool(fits and fits[0]['bounded']['spread']<=SPREAD_LIMIT)
    passed=0
    for f in fits:
        if not f['spread']<=SPREAD_LIMIT: break
        passed=f['windows']
    result['windows_meeting_law']=passed
    return result, audit


def rescore(cells,nf,q,coordinate='log10_temperature'):
    fold=finite(nf['source'][coordinate]); a=finite(nf['a']); b=finite(nf['b'])
    if not math.isfinite(a*b): raise ValueError('normal-form product is nonfinite')
    ws,side,ledger=collect(cells,fold,q,coordinate)
    variants={}; grids={}
    for mode in ('original_stop','amended_skip'):
        variants[mode],grids[mode]=score_variant(ws,fold,side,q,a,b,mode)
    return {'variants':variants,'selection_ledger':ledger,'fit_grid_audit':grids,
            'coordinate':coordinate,
            'normal_form_interpretation':{'signed_a_times_b':a*b,'side_times_a_times_b':side*a*b,
                'bottleneck_sign_condition':side*a*b>0,'extra_scoring_gate':False},
            'scope':'Unblinded finite saved-line rescore; native period/fold/derivative qualification inherited.',
            'scientific_claims_accepted':False}


def compare_leaves(original,independent,path=''):
    """All scalar/string/list-shape leaves; retain every finite nonzero delta."""
    rows=[]
    if isinstance(original,dict) and isinstance(independent,dict):
        for k in sorted(original.keys()|independent.keys()):
            p=path+'/'+str(k).replace('~','~0').replace('/','~1')
            if k not in original or k not in independent:
                rows.append({'path':p,'disposition':'missing_key','original_present':k in original,'independent_present':k in independent})
            else: rows.extend(compare_leaves(original[k],independent[k],p))
    elif isinstance(original,list) and isinstance(independent,list):
        rows.append({'path':path+'/length','original':len(original),'independent':len(independent),'equal':len(original)==len(independent)})
        for i in range(max(len(original),len(independent))):
            if i>=len(original) or i>=len(independent): rows.append({'path':path+'/'+str(i),'disposition':'missing_index'})
            else: rows.extend(compare_leaves(original[i],independent[i],path+'/'+str(i)))
    else:
        row={'path':path,'original':original,'independent':independent,'equal':type(original)==type(independent) and original==independent}
        if not isinstance(original,bool) and not isinstance(independent,bool) and isinstance(original,(int,float)) and isinstance(independent,(int,float)):
            if math.isfinite(original) and math.isfinite(independent):
                row['independent_minus_original']=independent-original; row['absolute_difference']=abs(independent-original)
            else: row['disposition']='nonfinite'
        rows.append(row)
    return rows
