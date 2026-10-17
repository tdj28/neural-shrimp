"""Independent E5 saved-data arithmetic; no producer imports or model execution."""
import math
import numpy as np

SEED = 20261607
MASKED = (151645, 151643)


def clean(x):
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not math.isfinite(x):
        return {'nonfinite': 'nan' if math.isnan(x) else ('+inf' if x > 0 else '-inf')}
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    return x


def periods(tokens, window, maximum):
    a = np.asarray(tokens)
    if a.ndim < 1 or a.shape[-1] < window or not 1 <= maximum < window:
        raise ValueError('invalid periodic-window contract')
    if a.dtype.kind not in 'iu': raise ValueError('token IDs must be integers')
    w = a[..., -window:]
    q = np.zeros(a.shape[:-1], dtype=np.int16)
    for lag in range(1, maximum + 1):
        fits = (w[..., lag:] == w[..., :-lag]).all(axis=-1)
        q[(q == 0) & fits] = lag
    return q


def tails(tokens, q):
    a, q = np.asarray(tokens), np.asarray(q)
    if q.shape != a.shape[:-1]: raise ValueError('tail-period shape')
    out = np.zeros(q.shape, dtype=np.int32)
    for ix in np.ndindex(q.shape):
        lag = int(q[ix])
        if lag <= 0: continue
        different = np.flatnonzero(a[ix][lag:] != a[ix][:-lag])
        out[ix] = a.shape[-1] if len(different) == 0 else a.shape[-1] - int(different[-1]) - 1
    return out


def rep_onset(grid, fractions):
    g, f = np.asarray(grid, float), np.asarray(fractions, float)
    if len(g) < 2 or f.shape != g.shape or not np.isfinite(f).all(): raise ValueError('onset grid/fractions')
    hit = np.flatnonzero(f < .5)
    if not len(hit):
        return dict(value=None, censored='above', analysis_value=float(g[-1] + (g[1] - g[0])), index=None, monotone=None)
    j = int(hit[0]); monotone = bool(np.all(f[j:] < .5))
    if j == 0:
        return dict(value=None, censored='below', analysis_value=float(g[0] - (g[1] - g[0])), index=0, monotone=monotone)
    value = float(g[j-1] + (f[j-1] - .5) / (f[j-1] - f[j]) * (g[j] - g[j-1]))
    return dict(value=value, censored=None, analysis_value=value, index=j, monotone=monotone)


def label_onset(grid, labels, target):
    g = np.asarray(grid, float)
    hits = [i for i, x in enumerate(labels) if x == target]
    if not hits: return dict(value=None, censored='above', analysis_value=float(g[-1] + (g[1]-g[0])), index=None)
    j = hits[0]
    return dict(value=float(g[j]), censored=None, analysis_value=float(g[j]), index=j)


def ranks(a):
    """Average ranks, last axis; pairwise comparisons are independent of SciPy."""
    a = np.asarray(a, float)
    less = (a[..., None, :] < a[..., :, None]).sum(axis=-1)
    equal = (a[..., None, :] == a[..., :, None]).sum(axis=-1)
    r = less + .5 * (equal + 1)
    return np.where(np.isfinite(a).all(axis=-1, keepdims=True), r, np.nan)


def pearson(a, b):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    ac, bc = a-a.mean(axis=-1, keepdims=True), b-b.mean(axis=-1, keepdims=True)
    den = np.sqrt((ac*ac).sum(axis=-1)*(bc*bc).sum(axis=-1))
    out = np.full(den.shape, np.nan)
    np.divide((ac*bc).sum(axis=-1), den, out=out, where=den > 0)
    return out


def spearman(a, b): return pearson(ranks(a), ranks(b))


def partial(a, b, c):
    ra, rb, rc = ranks(a), ranks(b), ranks(c)
    ab, ac, bc = pearson(ra,rb), pearson(ra,rc), pearson(rb,rc)
    den = np.sqrt(np.maximum((1-ac*ac)*(1-bc*bc), 0))
    out = np.full(den.shape, np.nan)
    np.divide(ab-ac*bc, den, out=out, where=den > 0)
    return out


def draw_indices(n, count, clusters=None):
    rng = np.random.default_rng(SEED)
    if clusters is None: return [rng.integers(0, n, size=n) for _ in range(count)]
    clusters = np.asarray(clusters)
    keys = sorted(set(clusters.tolist()))
    members = [np.flatnonzero(clusters == k) for k in keys]
    return [np.concatenate([members[i] for i in rng.integers(0, len(keys), size=len(keys))]) for _ in range(count)]


def boot_values(stat, indices):
    # Equal cluster sizes are guaranteed by E5; unequal manufactured clusters remain supported.
    out = []
    for lo in range(0, len(indices), 256):
        part = indices[lo:lo+256]
        if len({len(i) for i in part}) == 1: out.extend(np.asarray(stat(np.stack(part))).tolist())
        else: out.extend(float(stat(i)) for i in part)
    return np.asarray(out, float)


def interval(v, clusters=None):
    v = np.asarray(v, float); ok = np.isfinite(v)
    ci = np.quantile(v[ok], [.025, .975], method='linear').tolist() if ok.any() else [None,None]
    return dict(ci=ci, draws=len(v), nonfinite_draws=int((~ok).sum()), level=.95, clusters=clusters)


def matched(frac, labels):
    f, l = np.asarray(frac,float), np.asarray(labels)
    c, p, irregular = l=='chaotic', l=='periodic', l=='irregular'
    nc, np_ = c.sum(axis=-2), p.sum(axis=-2)
    mc = np.divide(np.where(c,f,0).sum(axis=-2), nc, out=np.full(nc.shape,np.nan), where=nc>0)
    mp = np.divide(np.where(p,f,0).sum(axis=-2), np_, out=np.full(np_.shape,np.nan), where=np_>0)
    eligible = (nc>0)&(np_>0)
    w = np.divide(nc*np_, nc+np_, out=np.zeros(nc.shape,float), where=(nc+np_)>0)
    w = np.where(eligible,w,0)
    numerator = np.where(eligible,w*(mc-mp),0).sum(axis=-1); denominator=w.sum(axis=-1)
    value = np.divide(numerator,denominator,out=np.full(denominator.shape,np.nan),where=denominator>0)
    return value, (nc,np_,irregular.sum(axis=-2),mc,mp,w)


def statistics(rows, grid, n_boot=10000):
    """Original paired endpoint definitions; all Monte Carlo values are retained separately."""
    n=len(rows); x=np.array([r['T_chaos']['analysis_value'] for r in rows]); y=np.array([r['T_rep']['analysis_value'] for r in rows]); z=np.array([r['log10_T_H'] for r in rows])
    f=np.array([r['repeat_fraction'] for r in rows]); l=np.array([r['labels'] for r in rows])
    iid=draw_indices(n,n_boot); op=[r['opener_index'] for r in rows]; sy=[r['system_index'] for r in rows]
    all_values={}
    def boot(name, stat, indices=iid, nc=None):
        v=boot_values(stat,indices); all_values[name]=v
        return interval(v,nc)
    primary=dict(spearman=float(spearman(x,y)), bootstrap=boot('primary',lambda i:spearman(x[i],y[i])),
        cluster_bootstrap_openers=boot('primary_openers',lambda i:spearman(x[i],y[i]),draw_indices(n,n_boot,op),len(set(op))),
        cluster_bootstrap_systems=boot('primary_systems',lambda i:spearman(x[i],y[i]),draw_indices(n,n_boot,sy),len(set(sy))))
    rng=np.random.default_rng(SEED+1)
    pv=np.array([float(spearman(x,rng.permutation(y))) for _ in range(n_boot)])
    all_values['permutation']=pv
    hits=int((np.abs(pv)>=abs(primary['spearman'])-1e-12).sum())
    primary['permutation_p']=(hits+1)/(n_boot+1)
    pdiag=dict(exceedances=hits, raw_fraction=hits/n_boot, plus_one=primary['permutation_p'], nonfinite_draws=int((~np.isfinite(pv)).sum()), observed_defined=bool(np.isfinite(primary['spearman'])), qualified_p=primary['permutation_p'] if np.isfinite(primary['spearman']) and np.isfinite(pv).all() else None)
    keep=np.array([r['T_chaos']['censored'] is None and r['T_rep']['censored'] is None for r in rows])
    primary['sensitivity']={'uncensored_only':{'n':int(keep.sum()),'spearman':float(spearman(x[keep],y[keep])) if keep.sum()>=3 else None},'censored_at_grid_edge':{'spearman':float(spearman(np.minimum(x,grid[-1]),np.clip(y,grid[0],grid[-1])))},'T_nonperiodic_instead_of_T_chaos':{'spearman':float(spearman([r['T_nonperiodic']['analysis_value'] for r in rows],y))},'grid_step':float(grid[1]-grid[0])}
    gap=y[keep]-x[keep]
    descriptive=dict(uncensored_pairs=int(keep.sum()),median_log10_T_rep_minus_log10_T_chaos=float(np.median(gap)) if len(gap) else None,quartiles_gap=np.quantile(gap,[.25,.75]).tolist() if len(gap) else None,prompts_T_rep_below_T_chaos=int((gap<0).sum()),pearson_uncensored=float(pearson(x[keep],y[keep])) if keep.sum()>=3 else None)
    secondary=dict(h_target_nats=2.,partial_spearman=float(partial(x,y,z)),bootstrap=boot('partial',lambda i:partial(x[i],y[i],z[i])),spearman_T_H_T_rep=float(spearman(z,y)),bootstrap_T_H_T_rep=boot('H_rep',lambda i:spearman(z[i],y[i])),spearman_T_H_T_chaos=float(spearman(z,x)),bootstrap_T_H_T_chaos=boot('H_chaos',lambda i:spearman(z[i],x[i])),T_H_range_log10=[float(z.min()),float(z.max())])
    value,(nc,np_,ni,mc,mp,w)=matched(f,l); per_t=[]
    for j in range(len(grid)):
        a=dict(t_index=j,n_chaotic=int(nc[j]),n_periodic=int(np_[j]),n_irregular=int(ni[j]),mean_chaotic=float(mc[j]) if nc[j] else None,mean_periodic=float(mp[j]) if np_[j] else None,log10_T=float(grid[j]))
        if nc[j] and np_[j]:a.update(difference=float(mc[j]-mp[j]),weight=float(w[j]))
        per_t.append(a)
    sm=dict(difference=float(value),per_temperature=per_t,bootstrap=boot('matched',lambda i:matched(f[i],l[i])[0]),temperatures_with_both=int(((nc>0)&(np_>0)).sum()))
    return dict(primary=primary,secondary_partial=secondary,secondary_matched=sm,descriptive=descriptive), dict(draws=all_values,permutation_diagnostics=pdiag,bootstrap_indices={'iid':iid,'openers':draw_indices(n,n_boot,op),'systems':draw_indices(n,n_boot,sy)})


def entropy(z, temperature):
    z=np.asarray(z,float); a=z/temperature; a=a-a.max(); w=np.exp(a); total=w.sum()
    return float(np.log(total)-(w*a).sum()/total)


def entropy_root(z,target=2.):
    """Independent 64-step bisection in log T on source bracket [-12,12]."""
    z=np.asarray(z,float)
    if z.ndim!=1 or not np.isfinite(z).all() or not 0<target<math.log(len(z)): raise ValueError('entropy input')
    lo,hi=-12.,12.; hlo,hhi=entropy(z,math.exp(lo)),entropy(z,math.exp(hi))
    if not hlo <= target <= hhi:
        return dict(defined=False,reason='target outside fixed numerical bracket',bracket_entropy=[hlo,hhi],T=None,residual=None)
    for _ in range(64):
        mid=(lo+hi)/2
        if entropy(z,math.exp(mid))<target:lo=mid
        else:hi=mid
    t=math.exp((lo+hi)/2); residual=entropy(z,t)-target
    return dict(defined=abs(residual)<=1e-8,T=t,residual=residual,logT_bracket=[lo,hi],bracket_entropy=[hlo,hhi],method='64 bisections in natural-log T; finite bracket, not interval arithmetic')


def histogram(q):
    return {str(int(k)):int(v) for k,v in zip(*np.unique(q[q>0],return_counts=True))}


def sample_summary(tokens, window=96, qmax=48):
    q=periods(tokens,window,qmax); qd=periods(tokens,window,max(qmax,min(48,window-1))); tail=tails(tokens,qd)
    result=dict(repeat_fraction=(q>0).mean(axis=1).tolist(),repeat_periods=histogram(q),diag_periods_up_to_48=histogram(qd),median_tail_length_by_T=np.median(tail,axis=1).tolist(),fraction_period_up_to_48_by_T=(qd>0).mean(axis=1).tolist(),fraction_tail_at_least_64_by_T=(tail>=64).mean(axis=1).tolist(),fraction_window64_by_T=(periods(tokens,64,min(qmax,63))>0).mean(axis=1).tolist())
    return result,dict(periods=q,diagnostic_periods=qd,tail_lengths=tail)


def loop_evidence(z, cfg):
    """Saved scalar/array consistency; never reconstructs an unsaved native history."""
    n=len(z['label']); rows=[]; virtual=[]; aggregate=[]
    if not np.isfinite(z['x_end']).all():aggregate.append('non-finite final states')
    known={'periodic','chaotic','irregular'}
    if not set(z['label'].tolist())<=known:aggregate.append('unknown labels '+str(sorted(set(z['label'].tolist())-known)))
    checks_by_message={
        'periodic cell with a period outside 1..qmax':[], 'non-periodic cell with a period':[],
        'periodic cell whose closure residual is not below tol':[],
        'non-periodic cell without finite exponents at both offsets':[],
        'chaotic cell whose exponents are not all above lambda_min':[],
        'irregular cell whose exponents are all above lambda_min':[],
        'retired cell not labeled periodic':[], 'cells retired in a run without --retire':[]}
    for j in range(n):
        lab=str(z['label'][j]); per=lab=='periodic'; retired=bool(z['retired'][j]); supp=bool(z['supplementary'][j]); p=int(z['period'][j]); lam=z['lyapunov'][:,j]; pos=bool((lam>cfg['lambda_min']).all())
        bad=[per and not 1<=p<=cfg['qmax'],not per and p!=0,per and not retired and not supp and not z['residual'][j]<cfg['tol'],not per and not np.isfinite(lam).all(),lab=='chaotic' and not pos,lab=='irregular' and pos,retired and not per,retired and not cfg['retire']]
        for key,fail in zip(checks_by_message,bad):
            if fail:checks_by_message[key].append(j)
        q=int(z['retire_q_tol'][j] if retired else z['q_supplementary'][j] if supp else z['q_end'][j])
        cls='periodic' if q>0 else ('hidden chaos' if z['lead_period'][j]>=1 else 'visible chaos') if pos else 'irregular, exponent not positive at both offsets'
        expected='periodic' if cls=='periodic' else 'chaotic' if pos else 'irregular'
        lead_check=None
        if z['twins'][j] and not supp and not retired:
            seq=z['lead'][:,j]; lead_check=0
            for lag in range(1,65):
                if len(seq)>2*lag and np.all(seq[lag:]==seq[:-lag]):lead_check=lag;break
        row={k:z[k][j] for k in ('label','raw_class','period','q_transient','q_end','q_supplementary','twins','supplementary','retired','retire_step','retire_q','retire_q_tol','lead_period','entropy_mean','entropy_sd','residual','cell_prompt','cell_tindex')}
        row.update(cell=j,lyapunov=lam,collapses=z['collapses'][:,j],source_gate_failures=[k for k,v in checks_by_message.items() if j in v],final_state_finite=bool(np.isfinite(z['x_end'][j]).all()),source_class_recomputed=cls,source_class_match=cls==str(z['raw_class'][j]),source_label_match=expected==lab,source_period_match=p==(q if cls=='periodic' else 0),lead_period_recomputed=lead_check,lead_period_match=None if lead_check is None else lead_check==int(z['lead_period'][j]),native_period_history='not saved; scalar association only',residual_gate_applies=bool(per and not retired and not supp),supplementary_lead_history_available=not supp)
        rows.append(row)
        if z['retire_step'][j]>=0 and not retired and (not per or p!=int(z['retire_q_tol'][j])):
            virtual.append(dict(cell=j,label=lab,period=p,retire_step=int(z['retire_step'][j]),retire_q_tol=int(z['retire_q_tol'][j])))
    aggregate.extend(k for k,v in checks_by_message.items() if v)
    return dict(rows=rows,gate_failures=aggregate,virtual_retirement_mismatches=virtual,
        source_gate_pass=not aggregate,classification_mismatches=[r['cell'] for r in rows if not (r['source_class_match'] and r['source_label_match'] and r['source_period_match'])],lead_mismatches=[r['cell'] for r in rows if r['lead_period_match'] is False],retirement_recorded=int((z['retire_step']>=0).sum()),collapses_total=int(z['collapses'].sum()))


def sample_evidence(z, shape, vocab):
    t=z['tokens']; e=z['mean_entropy']; failures=[]
    if t.shape!=tuple(shape):failures.append('tokens of shape '+str(t.shape)+', expected '+str(tuple(shape)))
    if t.min()<0 or t.max()>=vocab:failures.append('token ids outside the vocabulary')
    if np.isin(t,MASKED).any():failures.append('masked end-of-turn tokens present')
    if not np.isfinite(e).all():failures.append('non-finite sampling entropies')
    return dict(gate_failures=failures,source_gate_pass=not failures,token_min=int(t.min()),token_max=int(t.max()),masked_token_count=int(np.isin(t,MASKED).sum()),nonfinite_entropy_count=int((~np.isfinite(e)).sum()),shape=list(t.shape),prefix=z['prefix'],T_H_float32=z['T_H_float32'],T_H_float32_scope='saved scalar; float32 first-step logits not saved',mean_entropy_scope='saved path means; sampled-step logits not saved')


def prompt_summary(pid, names, grid, loop, sample, entropy_record, window, qmax):
    data, witness=sample_summary(sample['tokens'],window,qmax)
    labels=loop['labels']; tc=label_onset(grid,labels,'chaotic'); tn=label_onset(grid,['non-periodic' if a!='periodic' else a for a in labels],'non-periodic')
    th=entropy_record['T_H']; thm=entropy_record['T_H_masked']
    row=dict(prompt=pid,system_index=pid//8,opener_index=pid%8,system=names['system_prompts'][pid//8],opener=names['openers'][pid%8],labels=labels,T_chaos=tc,T_nonperiodic=tn,T_rep=rep_onset(grid,data['repeat_fraction']),log10_T_H=float(np.log10(th)),T_H=th,log10_T_H_masked=float(np.log10(thm)),log10_T_H_float32_sampler=float(np.log10(sample['T_H_float32'])),first_step_entropy_by_T=entropy_record['entropy_by_T'],first_irregular_index=next((i for i,s in enumerate(labels) if s=='irregular'),None),virtual_retirement_mismatches=loop['virtual_retirement_mismatches'],mean_sampling_entropy_by_T=sample['mean_entropy'].mean(axis=1).tolist(),lyapunov_by_T=loop['lyapunov'],period_by_T=loop['period'],gate_failures={'loop':loop['gate_failures'],'samples':sample['saved_gate_failures']},**data)
    return row,witness


def aggregate(rows,grid,n_boot=10000):
    ent=np.array([r['first_step_entropy_by_T'] for r in rows]); res=dict(n_prompts=len(rows),prompts=rows,first_step_entropy_range=dict(min_over_grid=float(ent.min()),max_over_grid=float(ent.max()),h_target=2.,prompts_reaching_target_within_grid=int(((ent[:,0]<=2)&(ent[:,-1]>=2)).sum())),virtual_retirement_mismatches=sum(len(r['virtual_retirement_mismatches']) for r in rows),censoring={
        'T_chaos_never_chaotic':[r['prompt'] for r in rows if r['T_chaos']['censored']],
        'T_rep_below_grid':[r['prompt'] for r in rows if r['T_rep']['censored']=='below'],
        'T_rep_above_grid':[r['prompt'] for r in rows if r['T_rep']['censored']=='above'],
        'T_rep_nonmonotone':[r['prompt'] for r in rows if r['T_rep']['monotone'] is False]})
    s,w=statistics(rows,grid,n_boot);res.update(s);return res,w


def comparison(own, saved, path=''):
    """Every leaf retained, no numerical agreement tolerance."""
    out=[]
    if isinstance(own,dict) and isinstance(saved,dict):
        for k in sorted(set(own)|set(saved)):
            p=path+'/'+str(k)
            if k not in own:out.append(dict(path=p,status='missing_own',saved=clean(saved[k])))
            elif k not in saved:out.append(dict(path=p,status='own_only',own=clean(own[k])))
            else:out.extend(comparison(own[k],saved[k],p))
    elif isinstance(own,(list,tuple,np.ndarray)) and isinstance(saved,(list,tuple,np.ndarray)):
        for i in range(max(len(own),len(saved))):
            p=path+'/'+str(i)
            if i>=len(own):out.append(dict(path=p,status='missing_own',saved=clean(saved[i])))
            elif i>=len(saved):out.append(dict(path=p,status='own_only',own=clean(own[i])))
            else:out.extend(comparison(own[i],saved[i],p))
    else:
        a,b=clean(own),clean(saved); rec=dict(path=path,own=a,saved=b,status='exact' if a==b else 'different')
        if isinstance(own,(int,float,np.number)) and not isinstance(own,(bool,np.bool_)) and isinstance(saved,(int,float)) and not isinstance(saved,bool) and math.isfinite(float(own)) and math.isfinite(float(saved)):
            rec.update(signed_difference=float(own)-float(saved),absolute_difference=abs(float(own)-float(saved)))
        out.append(rec)
    return out


def development_choice(samples, loop, grid, gates_ok, virtual_count, loop_seconds, sample_seconds, middle_entropies):
    variants={}; fractions={}
    for name,w,q in [('card',96,32),('period_48',96,48),('window_64',64,32)]:
        fr=[(periods(s['tokens'],w,q)>0).mean(axis=1) for s in samples]
        on=[rep_onset(grid,f) for f in fr];fractions[name]=fr
        variants[name]=dict(window=w,qmax=q,censored=sum(a['censored'] is not None for a in on),censored_below=sum(a['censored']=='below' for a in on),censored_above=sum(a['censored']=='above' for a in on),T_rep_log10=[a['value'] for a in on])
    degenerate=variants['card']['censored']>=4
    long=any(np.any(a>b) for a,b in zip(fractions['period_48'],fractions['card']))
    order=['period_48','window_64'] if long else ['window_64','period_48']
    chosen=next((v for v in order if variants[v]['censored']<=3),'card') if degenerate else 'card'
    th=np.array(loop['T_H']);lth=np.log10(th);outside=int(((lth<grid[0])|(lth>grid[-1])).sum())
    h=round(float(np.median(middle_entropies))*4)/4 if outside>=4 else 2.
    retire=virtual_count==0 and gates_ok; n=len(loop['label']); tw=int(loop['twins'].sum())
    original_steps=2000*n+2000*tw
    projected=int(np.where(loop['retire_step']>=0,np.minimum(loop['retire_step'],2000),2000).sum()+2000*tw)
    rate=original_steps/loop_seconds;loop_h=(projected if retire else original_steps)/rate*3/3600
    sample_h=float(np.mean(sample_seconds))*24/3600;total=loop_h+sample_h;samples_n=16 if total<=7 else 12 if total<=9.5 else 8
    return dict(gates_all_pass=gates_ok,retire=retire,repetition_rule=dict(variants=variants,card_rule_degenerate=degenerate,chosen=chosen,rep_window=variants[chosen]['window'],rep_qmax=variants[chosen]['qmax']),entropy_level=dict(log10_T_H_at_2_nats=lth.tolist(),outside_grid=outside,h_target=h,changed=h!=2.),heldout_samples=dict(dev_loop_row_steps=original_steps,dev_loop_seconds=loop_seconds,row_steps_per_second=rate,projected_loop_hours=loop_h,projected_sampling_hours_at_16=sample_h,projected_total_hours=total,samples_per_cell=samples_n),rep_window=variants[chosen]['window'],rep_qmax=variants[chosen]['qmax'],h_target=h,samples=samples_n)


def definitions(res):
    res['primary']['definition']='Spearman correlation of log10 T_chaos and log10 T_rep over prompts; censored values one grid step beyond the grid edge'
    res['secondary_partial']['definition']='partial Spearman correlation of log10 T_chaos and log10 T_rep given log10 T_H (Pearson partial correlation of the ranks)'
    res['secondary_matched']['definition']='per temperature, mean repetition fraction over chaotic cells minus the mean over periodic cells; combined with weights n_c n_p / (n_c + n_p)'
    return res
