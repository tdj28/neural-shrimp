"""Independent saved-data checks for NEED_TO_AUDIT items 6--13. No producer imports/models.

The compared subset is explicit: all horizon/PIF64/pilot scalar summaries, selective
control rates/contrasts, primary quantization endpoints and row gates, all steering
history contrasts/rates, and held-out hidden-readout labels, controls and state gates.
Uncompared diagnostics and historical generation are not certified by this audit.
"""
import argparse, collections, hashlib, itertools, json, math, pathlib, re, time, traceback, warnings
import numpy as np
from scipy.stats import t as student_t

P = pathlib.Path
PIN = {}
CHECKS = []
NOTES = []
ROOT = None
DATA = P('/data2/neural-shrimp-data')

def pin(p):
    p = P(p).resolve()
    if not (str(p).startswith(str(DATA)+'/') or str(p).startswith(str(ROOT)+'/')):
        raise ValueError('outside declared read roots: '+str(p))
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    row={'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
    if str(p) in PIN and PIN[str(p)]!=row: raise ValueError('input changed '+str(p))
    PIN[str(p)]=row
    return p

def js(p): return json.loads(pin(p).read_text())
def doc(n): return js(ROOT/'docs/pi'/n)
def recs(p):
    with pin(p).open() as f: return [json.loads(x) for x in f if x.strip()]
def nz(p): return np.load(pin(p),allow_pickle=False)
def resolve(s):
    if s.startswith('../neural-shrimp-data/'):return DATA/s[len('../neural-shrimp-data/'):]
    return P(s)
def unique(rows,key):
    d={}
    for r in rows:
        k=key(r)
        if k in d:raise ValueError('duplicate key '+str(k))
        d[k]=r
    return d
def norm(v):
    if isinstance(v,np.ndarray):return norm(v.tolist())
    if isinstance(v,np.generic):return norm(v.item())
    if isinstance(v,float) and not math.isfinite(v):return str(v)
    if isinstance(v,dict):return {str(k):norm(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [norm(x) for x in v]
    return v
def compare(label,own,published):
    """Compare each independently derived leaf; retain every finite nonzero delta."""
    if isinstance(own,dict):
        for k,v in own.items():
            if k not in published:CHECKS.append({'path':label+'/'+str(k),'missing_published':norm(v)})
            else:compare(label+'/'+str(k),v,published[k])
    elif isinstance(own,(list,tuple)):
        if len(own)!=len(published):CHECKS.append({'path':label,'lengths':[len(own),len(published)]})
        else:
            for i,v in enumerate(own):compare(label+'/'+str(i),v,published[i])
    else:
        a,b=norm(own),norm(published); r={'path':label,'own':a,'published':b,'exact':a==b}
        if isinstance(a,(int,float)) and not isinstance(a,bool) and isinstance(b,(int,float)):
            r.update(signed_delta=float(a-b),absolute_delta=float(abs(a-b)))
        CHECKS.append(r)
def rate(flags,z=1.959964):
    flags=list(flags);n=len(flags);k=int(sum(flags))
    if not n:return {'k':0,'n':0,'rate':float('nan'),'ci95':[float('nan')]*2}
    q=k/n;den=1+z*z/n;mid=(q+z*z/(2*n))/den
    half=z*math.sqrt(q*(1-q)/n+z*z/(4*n*n))/den
    return {'k':k,'n':n,'rate':q,'ci95':[mid-half,mid+half]}
def boot(x,seed):
    if not len(x):return [None,None]
    a=np.asarray(x,float);rng=np.random.default_rng(seed)
    return np.quantile(a[rng.integers(len(a),size=(10000,len(a)))].mean(1),[.025,.975]).tolist()
def unit_mean(pairs):
    d=collections.defaultdict(list)
    for u,v in pairs:d[u].append(float(v))
    return {k:float(np.mean(v)) for k,v in sorted(d.items())}
def manifest_check(d,kind):
    rr=d.get('raw',{});base=resolve(rr['root']) if rr else resolve(d['storage_root'])
    entries=rr['files'] if rr else d['raw_files']
    for e in entries:
        p=pin(base/e['path']);a=PIN[str(p)]
        if a['sha256']!=e['sha256'] or a['bytes']!=e['bytes']:
            raise ValueError('published raw hash mismatch '+str(p))
    NOTES.append({'component':kind,'original_manifest_entries_authenticated':len(entries)})

def false_chaos():
    pub=js(ROOT/'paper/llm/figures/fig-false-chaos.json')
    d=js(DATA/'explore'/pub['run']/'false-chaos.json');eps=np.array(d['eps']);C=d['conditions']
    clean=np.array(C['clean']['exponent']);mean=clean.mean(1)
    out={'clean_exponent_per_cell':mean.tolist(),'clean_spread_over_offsets':np.ptp(clean,axis=1).tolist(),'prediction':{}}
    all_rows=[]
    for key,v in C.items():
        y=np.asarray(v['exponent'],float)
        if key.startswith('noise-'):
            pr=np.maximum(mean[:,None],np.log(np.sqrt(2)*v['sigma']/eps)[None,:]);err=y-pr
            out['prediction'][key]={'sigma':v['sigma'],'max_abs_error':float(np.max(np.abs(err))),
                'max_abs_error_per_cell':np.max(np.abs(err),axis=1).tolist()}
        for i,j in itertools.product(range(y.shape[0]),range(y.shape[1])):
            all_rows.append({'condition':key,'cell':i,'eps':float(eps[j]),'exponent':float(y[i,j]),
                'prediction':float(pr[i,j]) if key.startswith('noise-') else None,
                'signed_error':float(err[i,j]) if key.startswith('noise-') else None})
    f=C['float32-final-norm'];frac=np.array(f['merges'])/np.maximum(np.array(f['merges'])+np.array(f['counted']),1)
    out['float32_final_norm']={'exponent':f['exponent'],'merged_fraction':frac.tolist()}
    compare('false-chaos',out,pub)
    return {'all_rows':all_rows,'clean_extrema':[[float(min(a)),float(max(a))] for a in clean],
        'float32_eps_ge_1e-6_extrema':[[float(np.min(a[eps>=1e-6])),float(np.max(a[eps>=1e-6]))] for a in np.array(f['exponent'])],
        'float32_small_offset_merge_extrema':[int(np.min(np.array(f['merges'])[:,eps<=1e-11])),int(np.max(np.array(f['merges'])[:,eps<=1e-11]))],
        'free_twins':{k:v['free_log10_distance'] for k,v in C.items()}}

def paired_rows(run,fill=False):
    a=js(run/'run.json')['args'];rows=recs(run/'records.jsonl');by=collections.defaultdict(dict)
    for r in rows:
        if not fill and r['index']<0:continue
        if r['layout'] in by[r['index']]:
            if not fill:raise ValueError('duplicate pair '+str(run))
            old=by[r['index']][r['layout']]
            NOTES.append({'component':'pilot_source_last_write','run':str(run),'index':r['index'],'layout':r['layout'],
                          'old_answer':old['answer'],'new_answer':r['answer'],'same_record':old==r})
        by[r['index']][r['layout']]=r
    incomplete=[i for i,v in by.items() if len(v)!=2]
    if incomplete:
        if not fill:raise ValueError('incomplete pairs '+str(incomplete))
        NOTES.append({'component':'pilot_source_unpaired','run':str(run),'indices':incomplete})
        by={i:v for i,v in by.items() if len(v)==2}
    return a,{i:tuple(v[k] for k in sorted(v)) for i,v in by.items()}

def horizon():
    d=doc('h-horizon-summary.json');outputs=[]
    for ix,pub in enumerate(d['configs']):
        run=resolve(pub['run']);cfg,pairs=paired_rows(run);N=cfg['soft_steps']
        steps=sorted(map(int,next(iter(pairs.values()))[0]['readout_texts']))+[N]
        out={'problems':len(pairs),'readout_steps':steps,'text_agree':{},'answer_agree':{},'accuracy':{},'settled':{}}
        horizons={};firsts=[]
        for i,(a,b) in pairs.items():
            horizons[i]=next((s for s in steps if (a['text'] if s==N else a['readout_texts'][str(s)])!=(b['text'] if s==N else b['readout_texts'][str(s)])),None)
            firsts.append(next((j for j,(x,y) in enumerate(zip(a['top1'],b['top1'])) if x!=y),None))
        for s in steps:
            texts=[(a['text'],b['text']) if s==N else (a['readout_texts'][str(s)],b['readout_texts'][str(s)]) for a,b in pairs.values()]
            answers=[(a['answer'],b['answer']) if s==N else (a['readouts'][str(s)],b['readouts'][str(s)]) for a,b in pairs.values()]
            out['text_agree'][str(s)]=rate(x==y for x,y in texts)
            out['answer_agree'][str(s)]=rate(x==y for x,y in answers)
            out['accuracy'][str(s)]=[rate(v[k] is not None and v[k]==p[k]['gold'] for v,p in zip(answers,pairs.values())) for k in (0,1)]
            out['settled'][str(s)]=rate(v[0]==p[0]['answer'] for v,p in zip(answers,pairs.values()))
        hs=[s for s in horizons.values() if s is not None];fd=[f for f in firsts if f is not None]
        out.update(horizon_counts={str(s):list(horizons.values()).count(s) for s in steps},horizon_beyond=list(horizons.values()).count(None),
            horizon_median_among_split=float(np.median(hs)) if hs else None,trajectory_divergence=rate(f is not None for f in firsts),first_divergence_median=float(np.median(fd)) if fd else None)
        j=pub.get('join');join=None
        if j:
            fs=unique(recs(resolve(j['f64_config'])/'records.jsonl'),lambda r:r['index']);common=sorted(set(fs)&set(pairs))
            levels=sorted({t['eps0'] for r in fs.values() for t in r['twins']})
            sep={i:any(t['first_above_dmax'] is not None for t in fs[i]['twins'] if t['eps0']==levels[0]) for i in common}
            flips={i:pairs[i][0]['text']!=pairs[i][1]['text'] for i in common}
            join={'common_problems':len(common),'f64_separated_smallest_offset':rate(sep.values()),
                  'bf16_text_flip_given_f64_separated':rate(flips[i] for i in common if sep[i]),
                  'bf16_text_flip_given_f64_not_separated':rate(flips[i] for i in common if not sep[i]),
                  'f64_text_flip_by_offset':{f'{e:g}':rate(any(t['text_flip'] for t in fs[i]['twins'] if t['eps0']==e) for i in common) for e in levels}}
            join['bf16_horizon_vs_f64_separation']={}
            for yes,key in [(True,'median_horizon_separated'),(False,'median_horizon_not_separated')]:
                hh=[horizons[i] for i in common if sep[i]==yes and horizons[i] is not None]
                join['bf16_horizon_vs_f64_separation'][key]=float(np.median(hh)) if hh else None
            out['join']=join
        compare('horizon/'+str(ix),out,pub)
        diagnostic=None
        if j:
            diagnostic={f'{e:g}':rate(any(t['flip'] for t in fs[i]['twins'] if t['eps0']==e) for i in common) for e in levels}
        outputs.append({'T':cfg['temperature'],'alpha':cfg['alpha'],**out,'independent_f64_numeric_answer_flips':diagnostic})
    return outputs

def pif64():
    outputs={}
    for name in ['pif64-full.json','pif64-topn15.json']:
        pub=doc(name);run=resolve(pub['f64_run']);rows=recs(run/'records.jsonl');cfg=js(run/'run.json')['args']
        by=unique(rows,lambda r:r['index']);e=cfg['eps0'];S=cfg['soft_steps'];cl={}
        for i,r in by.items():cl[i]='separated' if any(x is not None for x in r['first_above_dmax']) else 'grew' if any(x>e for x in r['final_distance']) else 'contracted'
        net=[math.log(f/e)/(S-1) if f>0 else -math.inf for r in rows for f in r['final_distance']]
        ss=[s for r in rows for s in r['first_above_dmax'] if s is not None]
        out={'problems':len(rows),'accuracy':rate(r['correct'] for r in rows),'twin_flip_any':rate(r['twin_flips']>0 for r in rows),
            'twin_flip_pairs':rate(a!=r['answer'] for r in rows for a in r['twin_answers']),
            'class_counts':{c:list(cl.values()).count(c) for c in ['separated','grew','contracted']},
            'merged_twins':sum(f==0 for r in rows for f in r['final_distance']),'twins':len(net),
            'net_rate_quantiles':{str(q):float(np.quantile(net,q)) for q in [.1,.25,.5,.75,.9]},
            'separation_step_median':float(np.median(ss)) if ss else None,'separation_step_quartiles':np.quantile(ss,[.25,.75]).tolist() if ss else None,
            'flip_by_class':{},'entropy_mean_by_class':{},'gap_min_by_class':{}}
        for c in out['class_counts']:
            rs=[r for r in rows if cl[r['index']]==c]
            out['flip_by_class'][c]=rate(r['twin_flips']>0 for r in rs)
            for key,field in [('entropy_mean_by_class','entropy_mean'),('gap_min_by_class','gap_min')]:out[key][c]=float(np.median([r[field] for r in rs])) if rs else None
        compare(name,out,pub);outputs[name]=out
        out['strict_all_twins_below_initial']=sum(all(f<e for f in r['final_distance']) for r in rows)
        out['twin_flip_count_inconsistencies']=[r['index'] for r in rows if r['twin_flips']!=sum(x!=r['answer'] for x in r['twin_answers'])]
        if 'join' in pub:
            bcfg,pairs=paired_rows(resolve(pub['join']['bf16_run']));ids=sorted(set(pairs)&set(by));j={
                'common_problems':len(ids),'bf16_flip':rate(pairs[i][0]['answer']!=pairs[i][1]['answer'] for i in ids),
                'bf16_divergence':rate(any(pairs[i][0][k]!=pairs[i][1][k] for k in ['tokens','top1']) for i in ids),
                'accuracy_f64_on_common':rate(by[i]['correct'] for i in ids),
                'accuracy_bf16_layouts_on_common':[rate(pairs[i][k]['correct'] for i in ids) for k in [0,1]]}
            compare(name+'/join',j,pub['join']);out['join_checked']=j
    return outputs

def pilots():
    out={}
    for name in ['e1-pilot-summary.json','e4-pilot-summary.json']:
        out[name]={}
        for s,pub in doc(name).items():
            cfg,pairs=paired_rows(resolve(s),fill=True);pp=list(pairs.values());fl=[a['answer']!=b['answer'] for a,b in pp]
            first=[next((i for i,(x,y) in enumerate(zip(a['top1'],b['top1'])) if x!=y),None) for a,b in pp]
            rr=rate(fl);v={'pairs':len(pp),'answer_flips':rr['k'],'flip_rate':rr['rate'],'flip_ci95':rr['ci95'],
                'trajectory_divergence':sum(a['tokens']!=b['tokens'] or a['top1']!=b['top1'] for a,b in pp),
                'median_first_divergence_step':float(np.median([f for f in first if f is not None])) if any(f is not None for f in first) else None,
                'accuracy':{str(la):float(np.mean([p[k]['correct'] for p in pp])) for k,la in enumerate(sorted(cfg['layouts'][:2]))}}
            if cfg['mode']=='soft':
                for flag,key in [(True,'mean_gain_flipped'),(False,'mean_gain_kept')]:
                    vals=[np.mean(a['gain']) for (a,b),f in zip(pp,fl) if f==flag];v[key]=float(np.mean(vals)) if vals else None
            compare(name+'/'+s,v,pub);out[name][s]={'negative_index_pairs':sum(i<0 for i in pairs),**v}
    return out

def selective():
    d=doc('c12-1-summary.json');manifest_check(d,'c12-1');out={};problems={p['uid']:p for p in doc('c12-1-problems.json')['problems']}
    for name,pub in d['runs'].items():
        rows=recs(resolve(pub['dir'])/'records.jsonl');base=unique([r for r in rows if r['kind']=='baseline'],lambda r:(r['uid'],r['T'],r['S']))
        arms=[r for r in rows if r['kind']=='arm'];unique([r for r in arms if r['arm']!='zero'],lambda r:(r['arm'],r.get('batch'),r['uid'],r['T'],r['S'],r['inj'],r['rho']))
        issues=[];counts=collections.Counter();score_deltas=[]
        for r in [r for r in rows if r['kind'] in ['baseline','arm']]:
            text=r['text'];i=text.lower().rfind('answer is');nums=re.findall(r'-?\d[\d,]*\.?\d*',text[i+9:] if i>=0 else text)
            ans=None
            if nums:
                val=float((nums[0] if i>=0 else nums[-1]).replace(',','').rstrip('.'));ans=str(int(val)) if val==int(val) else str(val)
            p=problems[r['uid']];flags={'answer':ans,'correct':ans is not None and ans==p['gold'],'target_hit':ans is not None and ans==p['target'],'invalid':ans is None}
            bad={k:{'own':v,'saved':r[k]} for k,v in flags.items() if v!=r[k]}
            if bad:score_deltas.append({'uid':r['uid'],'S':r['S'],'T':r['T'],'arm':r.get('arm'),'differences':bad})
        for r in arms:
            if (r['uid'],r['T'],r['S']) not in base:issues.append('missing baseline')
            if r['arm'].startswith('pga') and r['realized_rel']>r['rho']*(1+1e-4):issues.append('pga radius exceeded')
            counts[r['arm']]+=1
        groups=collections.defaultdict(list)
        for r in arms:groups[(r['S'],r['inj'],r['rho'],r['arm'])].append(r)
        def stats(rs):
            um=unit_mean((r['uid'],r['target_hit']) for r in rs);n=len(rs);k=sum(r['target_hit'] for r in rs)
            return {'pairs':n,'problems':len(um),'target_hits':k,'rate':k/n if n else None,'problem_weighted':float(np.mean(list(um.values()))) if um else None,
                'boot95_problems':boot(list(um.values()),12004),'answer_changed':sum(r['changed'] for r in rs),'invalid':sum(r['invalid'] for r in rs),'correct_kept':sum(r['correct'] for r in rs)}
        rebuilt={}
        for key,p in pub['rates'].items():
            m=re.fullmatch(r'S(\d+)_(early|late)_rho([\d.]+)_(.+)',key)
            if m is None:raise ValueError('unexpected selective rate key: '+key)
            S,inj,rho,arm=m.groups();group=(int(S),inj,float(rho),arm)
            if group not in groups:raise ValueError('missing selective source group: '+key)
            rs=groups[group];eligible=[r for r in rs if base[(r['uid'],r['T'],r['S'])]['correct']]
            q={'eligible':stats(eligible),'unconditional':stats(rs)}
            # Compact receipt intentionally omits some original-analysis fields.
            q={k:{kk:vv for kk,vv in v.items() if kk in p[k]} for k,v in q.items()}
            compare('c12-1/'+name+'/rates/'+key,q,p);rebuilt[key]=q
        contrasts={}
        for key,p in pub['contrasts'].items():
            if '_early-minus-late' in key:
                m=re.match(r'S(\d+)_rho([\d.]+)_(.+)_early-minus-late',key);S,rho,arm=m.groups();a=groups[(int(S),'early',float(rho),arm)];b=groups[(int(S),'late',float(rho),arm)]
            else:
                m=re.match(r'S(\d+)_(early|late)_rho([\d.]+)_(.+)-minus-(.+)',key);S,inj,rho,aa,bb=m.groups();a=groups[(int(S),inj,float(rho),aa)];b=groups[(int(S),inj,float(rho),bb)]
            am=unique(a,lambda r:(r['uid'],r['T']));bm=unique(b,lambda r:(r['uid'],r['T']));ks=sorted(set(am)&set(bm));ks=[k for k in ks if base[(k[0],k[1],am[k]['S'])]['correct']]
            vals=unit_mean((k[0],float(am[k]['target_hit'])-float(bm[k]['target_hit'])) for k in ks)
            q={'estimate':float(np.mean(list(vals.values()))) if vals else None,'boot95_problems':boot(list(vals.values()),12004),'pairs':len(ks),'problems':len(vals),
                'a_only':sum(am[k]['target_hit'] and not bm[k]['target_hit'] for k in ks),'b_only':sum(bm[k]['target_hit'] and not am[k]['target_hit'] for k in ks)}
            compare('c12-1/'+name+'/contrasts/'+key,q,p);contrasts[key]=q
        out[name]={'rows':len(rows),'gate_issues':issues,'scoring_deltas':score_deltas,'rates':rebuilt,'contrasts':contrasts,'zero_changed':sum(r['changed'] for r in arms if r['arm']=='zero')}
    return out

def quantization():
    d=doc('c12-2-summary.json');manifest_check(d,'c12-2');cfg=d['settings'];base=resolve(d['storage_root'])/d['heldout_dir'];vars=['ref','q8','q4','noise'];cats={v:collections.defaultdict(list) for v in vars};gates=[];kl=collections.defaultdict(list)
    for f in sorted((base/'batches').glob('b*-ref.npz')):
        tag=f.name[:-7];batch={}
        for v in vars:
            with nz(f.with_name(tag+v+'.npz')) as z:
                a={k:z[k] for k in ['prompt','t_index','start','cls','sub','period','lam','q_end','q_twin','q_detect','confirmed','residual','x_end','batch_prompts']}
            n=len(a['cls']);per=a['cls']==0;twin=(a['q_end']==0)&~per;ch=a['cls']==1;un=a['cls']==2
            verified=a['confirmed'].astype(bool)&(a['q_detect']>0)&(a['q_end']==a['q_detect']);expected_cls=np.full(n,2);expected_cls[verified]=0
            expected_cls[(~verified)&(a['q_end']==0)&np.all(a['lam']>cfg['lambda_min'],axis=0)&(a['q_twin']==0)]=1
            checks={'row_count':n==len(a['batch_prompts'])*len(cfg['log_t'])*cfg['starts'],'finite_end':bool(np.isfinite(a['x_end']).all()),
                'classification_reconstructed':bool(np.array_equal(expected_cls,a['cls'])),
                'no_pending':bool(np.all(a['sub']!=5)),'periodic_residual':bool(np.all(a['residual'][per]<cfg['tol'])),
                'confirmed':bool(np.all(a['confirmed'][per])),'period_range':bool(np.all((a['period'][per]>=1)&(a['period'][per]<=8)&(a['period'][per]==a['q_detect'][per]))),
                'twins_present':bool(not np.isnan(a['lam'][:,twin]).any()),'no_extra_twins':bool(np.isnan(a['lam'][:,~twin]).all()),
                'chaos_gate':bool(np.all(a['lam'][:,ch]>cfg['lambda_min']) and np.all(a['q_twin'][ch]==0)),
                'unresolved_labels':bool(np.isin(a['sub'][un],[1,2,3,4]).all())}
            gates.append({'batch':tag,'variant':v,'n':n,'checks':checks});batch[v]=a
            for k in ['prompt','t_index','start','cls','period','sub']:cats[v][k].append(a[k])
        for k in ['prompt','t_index','start']:
            if not all(np.array_equal(batch['ref'][k],batch[v][k]) for v in vars):raise ValueError('unaligned '+tag)
        with nz(f.with_name(tag+'kl.npz')) as z:
            for k in z.files:kl[k].append(z[k])
    cats={v:{k:np.concatenate(a) for k,a in d_.items()} for v,d_ in cats.items()};kl={k:np.concatenate(a) for k,a in kl.items()};ref=cats['ref'];pids=sorted(set(ref['prompt'].tolist()))
    assert pids==d['heldout_prompts_run'];unique([tuple(map(int,r)) for r in zip(ref['prompt'],ref['t_index'],ref['start'])],lambda x:x)
    shares={v:np.array([np.mean(cats[v]['cls'][ref['prompt']==p]==0) for p in pids]) for v in vars};ends={};changes={};ones={}
    for v in vars[1:]:
        diff=shares[v]-shares['ref'];sd=float(np.std(diff,ddof=1));mean=float(np.mean(diff));half=float(student_t.ppf(.975,len(diff)-1))*sd/math.sqrt(len(diff));q={'n':len(diff),'mean':mean,'sd':sd,'ci':[mean-half,mean+half],'bootstrap_ci':boot(diff,20261018),'share_ref_mean':float(shares['ref'].mean()),'share_variant_mean':float(shares[v].mean()),'prompts_with_any_shift':int(np.sum(diff!=0))}
        compare('c12-2/endpoints/'+v,q,d['primary_endpoint'] if v=='q4' else d['secondary_endpoints'][v+'_minus_ref']);ends[v]=q
        change=(cats[v]['cls']!=ref['cls'])|((cats[v]['cls']==0)&(ref['cls']==0)&(cats[v]['period']!=ref['period']))
        q={'class_or_period_changed':{'count':int(sum(change)),'of':len(change),'pooled_share':float(np.mean(change))},'class_changed':int(sum(cats[v]['cls']!=ref['cls'])),'period_changed_both_periodic':int(sum((cats[v]['cls']==0)&(ref['cls']==0)&(cats[v]['period']!=ref['period'])))}
        compare('c12-2/changes/'+v,q,d['changes'][v]);changes[v]=q
        mask=kl['attractor'];a={k[len(v)+2:]:val[mask] for k,val in kl.items() if k.startswith(v+'__')}
        q={'states':int(sum(mask)),'kl_T_mean':float(np.mean(a['kl_T'])),'kl_T_median':float(np.median(a['kl_T'])),'kl_T_p90':float(np.quantile(a['kl_T'],.9)),'kl_T_p99':float(np.quantile(a['kl_T'],.99)),'kl_T_max':float(max(a['kl_T'])),'kl_1_mean':float(np.mean(a['kl_1'])),'kl_1_median':float(np.median(a['kl_1'])),'tv_T_mean':float(np.mean(a['tv_T'])),'top1_agreement':float(np.mean(a['top1']))}
        compare('c12-2/one_step/'+v,q,d['one_step'][v]['attractor']);ones[v]=q
    return {'gates':gates,'prompts':pids,'endpoints':ends,'changes':changes,'one_step':ones,'scope':'Primary held-out 2-start endpoint; all declared raw hashes authenticated. Extension/development secondary arithmetic not recomputed.'}

def steering():
    d=doc('c12-4-summary.json');manifest_check(d,'c12-4');out={}
    for name,pub in d['analyses'].items():
        run=resolve(pub['run']);rows=[r for r in recs(run/'records.jsonl') if r['kind']=='row'];info=js(run/'run.json')['history'][-1]
        if 'histories' not in info:
            out[name]={'rows':len(rows),'model':info.get('model_path'),'started':info.get('started'),'finished':info.get('finished'),'scope':'source/row/manifest association; historical calibration arithmetic not recomputed'};continue
        unique(rows,lambda r:(r['uid'],r['start'],r['history']));by=collections.defaultdict(list)
        issues=[];parse_mismatches=[]
        for r in rows:
            by[r['history']].append(r)
            if len(r['alpha'])!=info['n_steps'] or len(r['top1'])!=info['n_steps']:issues.append('length')
            if r['history']!='unsteered' and (any(a!=round(info['choices']['alpha_final'],8) for a in r['alpha'][info['P']:]) or r['alpha_pre']!=info['choices']['alpha_final']):issues.append('dwell or prefix differs')
            for s,ro in r['readouts'].items():
                m=re.match(r'(allow|deny)(?![A-Za-z])',re.sub(r'^[\s\"\x27*:`]*','',ro['text']),re.I);de=m[1].upper() if m else 'UNPARSEABLE'
                if de!=ro['decision']:parse_mismatches.append([r['uid'],r['start'],r['history'],s,de,ro['decision']])
        if len(rows)!=len(info['uids'])*len(info['starts'])*len(info['histories']):issues.append('cohort size')
        def units(rs,S,metric,cls):
            def val(r):
                ro=r['readouts'][str(S)]
                return ro['zA']-ro['zD'] if metric=='margin' else float(ro['decision']==({'allow':'ALLOW','deny':'DENY','unparseable':'UNPARSEABLE'}.get(metric,r['gold'])))
            return unit_mean((r['uid'],val(r)) for r in rs if cls is None or r['cls']==cls)
        con={};rates={}
        metrics={'false_allow_prohibited':('allow','prohibited'),'false_deny_permitted':('deny','permitted'),'unparseable':('unparseable',None),'margin_prohibited':('margin','prohibited'),'margin_permitted':('margin','permitted'),'accuracy':('correct',None)}
        for h,rs in by.items():
            rates[h]={}
            for S in info['readouts']:
                rates[h][str(S)]={}
                for key,(m,c) in metrics.items():
                    u=units(rs,S,m,c);vals=list(u.values());q={'instances':len(u),'mean':float(np.mean(vals)),'boot95_instances':boot(vals,12403)}
                    compare('c12-4/'+name+'/rates/'+h+'/'+str(S)+'/'+key,q,pub['rates'][h][str(S)][key]);rates[h][str(S)][key]=q
                if h!='direct':
                    ck=f'{h}-minus-direct_S{S}';con[ck]={}
                    for key,(m,c) in metrics.items():
                        if key=='accuracy':continue
                        a=units(rs,S,m,c);b=units(by['direct'],S,m,c);ks=sorted(set(a)&set(b));vals=[a[k]-b[k] for k in ks];q={'estimate':float(np.mean(vals)),'boot95_instances':boot(vals,12403),'instances':len(vals),'instances_positive':sum(v>0 for v in vals),'instances_negative':sum(v<0 for v in vals)}
                        compare('c12-4/'+name+'/contrasts/'+ck+'/'+key,q,pub['contrasts'][ck][key]);con[ck][key]=q
        out[name]={'rows':len(rows),'gate_issues':issues,'decision_parser_deltas':parse_mismatches,'rates':rates,'contrasts':con,'choices_sha256':info.get('choices_sha256'),'model':info['model_path'],'started':info['started'],'finished':info.get('finished')}
    return out

def hidden_label(text,typ):
    text=re.split(r'<\|im_end\|>|<\|endoftext\|>',text)[0];i=text.find('\n',len(text)-len(text.lstrip()));text=text if i<0 else text[:i]
    word=r'[^\W\d_]+(?:[\x27’-][^\W\d_]+)*';strip=' \t\n\r.,;:!?\"\x27*`()[]{}<>_~#’“”-'
    words=list(re.finditer(word,text));lab='other'
    if typ=='yesno':
        lab=next((m[0].lower() for m in words if m[0].lower() in ['yes','no']),'other');fmt=bool(words) and words[0][0].lower() in ['yes','no'] and not text[:words[0].start()].strip(strip)
    elif typ=='word':lab=words[0][0].lower() if words else 'other';fmt=bool(re.fullmatch(word,text.strip(strip)))
    else:
        m=re.search(r'\d+',text);lab=str(int(m[0])) if m and 1<=int(m[0])<=10 else 'other';m=re.match(r'\d+',text.lstrip(strip));fmt=bool(m and 1<=int(m[0])<=10)
    return text,lab,bool(fmt)

def hidden():
    d=doc('c12-5-summary.json');manifest_check(d,'c12-5');run=DATA/'explore/pi/c12-5/heldout-1';tr=js(run/'trajectories.json');cells={c['uid']:c for c in tr['cells']};info=js(run/'run.json')['history'][-1]
    rows=[r for r in recs(run/'readouts.jsonl') if r['kind']=='readout'];by=unique(rows,lambda r:(r['query'],r['cell'],r['layout'],r['k']))
    choices=doc('c12-5-choices.json');pub=d['analyses']['explore/pi/c12-5/heldout-1/analysis.json'];qres={};gates=[];parse_deltas=[]
    with nz(run/'trajectories.npz') as z:
        for i,uid0 in enumerate(z['uids']):
            uid=str(uid0);c=cells[uid];col=c['collect'];start=int(z['record_start']);leads=z['leads'][i];hist=[leads[s-70-start:s-start] for s in col]
            history_ok=all(len(h)==70 and np.array_equal(h,hist[0]) for h in hist);qual=history_ok and (float(z['exponent'][i])>1e-3 and int(z['period_end'][i])==0 if c['kind']=='chaotic' else int(z['period_end'][i])==7)
            if qual!=c['qualifies']:raise ValueError('qualification differs '+uid)
            states=z['states'][i];den=float(np.linalg.norm(states,axis=1).mean());dist=[float(np.linalg.norm(states[x]-states[y])/den) for x,y in itertools.combinations(range(16),2)]
            ds={'min':min(dist),'median':float(np.median(dist)),'max':max(dist)};compare('c12-5/state_distances/'+uid,ds,c['pair_distance'])
            gates.append({'cell':uid,'history_identical':history_ok,'qualifies_own':qual,'qualifies_saved':c['qualifies'],'finite_states':bool(np.isfinite(states).all()),'pair_distance':ds})
            for q in info['queries']:
                for k in range(16):
                    for layout in ['main','recompute']:
                        if (q['id'],uid,layout,k) not in by:raise ValueError('missing readout')
                    if by[(q['id'],uid,'main',k)]['pos0_argmax']!=int(leads[col[k]-start]):raise ValueError('next lead mismatch')
                if (q['id'],uid,'reference',-1) not in by:raise ValueError('missing reference')
    for q in info['queries']:
        qid=q['id'];fixed=str(choices['queries'][q['type']]['fixed_token']);stats=[];negative=[]
        for uid,c in cells.items():
            mr=[by[(qid,uid,'main',k)] for k in range(16)];rc=[by[(qid,uid,'recompute',k)] for k in range(16)]
            labels=[];probs=[]
            for r in mr:
                answer,lab,fmt=hidden_label(r['raw'],r['qtype']);labels.append(lab);probs.append(r['p_fixed'][fixed])
                if (answer,lab,fmt)!=(r['answer'],r['label'],r['in_format']):parse_deltas.append([qid,uid,r['k'],answer,lab,fmt,r['answer'],r['label'],r['in_format']])
            pairs=list(itertools.combinations(range(16),2));dif=sum(labels[a]!=labels[b] for a,b in pairs);tdif=sum(mr[a]['ids']!=mr[b]['ids'] for a,b in pairs);pdif=sum(format(probs[a],'.6g')!=format(probs[b],'.6g') for a,b in pairs)
            negative.append({'cell':uid,'ids_differs':sum(a['ids']!=b['ids'] for a,b in zip(mr,rc)),'p_max_abs_diff':max(abs(a['p_fixed'][fixed]-b['p_fixed'][fixed]) for a,b in zip(mr,rc))})
            if c['kind']=='chaotic' and c['split']=='heldout' and c['qualifies']:
                stats.append({'cell':uid,'differing_pairs':dif,'text_differing_pairs':tdif,'probability_sigdiff_pairs':pdif,'pairs':len(pairs)})
        n=len(stats);vals=[r['differing_pairs']/r['pairs'] for r in stats];anydiff=sum(x>0 for x in vals)
        primary={'mean':float(np.mean(vals)),'cells_with_any_disagreement':anydiff,'boot95_cells':boot(vals,12502)}
        if anydiff==0:primary['cells_with_any_upper95_one_sided_clopper_pearson']=-math.expm1(math.log(.05)/n)
        compare('c12-5/primary/'+qid,primary,d['results']['primary']['per_query'][qid])
        group={'cells':n,'positive_share':{'mean':float(np.mean([s['probability_sigdiff_pairs']/120 for s in stats]))},'text_disagreement':{'cells_with_any':sum(s['text_differing_pairs']>0 for s in stats)}}
        compare('c12-5/groups/'+qid,group,pub['groups'][qid]['chaotic-heldout'])
        qres[qid]={'primary':primary,'group':group,'cells':stats,'negative_controls':negative}
    return {'gates':gates,'queries':qres,'parser_changes_from_original_records':parse_deltas,'readouts':len(rows),'scope':'held-out original states/readouts and source-defined label extraction; no model rerun or development query reselection'}

def controls():
    assert rate([True,False])['rate']==.5
    assert boot([0,0,0],1)==[0,0]
    assert hidden_label(' Yes\nQuestion: no','yesno')==(' Yes','yes',True)
    assert hidden_label('13','rating')[1]=='other'
    assert unit_mean([(2,1),(1,0),(2,0)])=={1:0.,2:.5}
    try:unique([1,1],lambda x:x)
    except ValueError:pass
    else:raise AssertionError('duplicate gate')
    return {'groups':6,'passed':6}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source-packet',type=P,required=True);ap.add_argument('--out',type=P,required=True);ap.add_argument('--controls-only',action='store_true');ap.add_argument('--components',nargs='+',default=['false_chaos','horizon','pif64','pilots','selective','quantization','steering','hidden']);a=ap.parse_args();ROOT=a.source_packet.resolve();a.out.mkdir(parents=True,exist_ok=False);tic=time.monotonic()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');result={'controls':controls()}
        if not a.controls_only:
            for key,fn in [('false_chaos',false_chaos),('horizon',horizon),('pif64',pif64),('pilots',pilots),('selective',selective),('quantization',quantization),('steering',steering),('hidden',hidden)]:
                if key not in a.components:continue
                try:
                    value=fn();(a.out/(key+'.json')).write_text(json.dumps(norm(value),indent=2)+'\n');result[key]={'completed':True}
                except Exception as err:
                    (a.out/(key+'-failure.txt')).write_text(traceback.format_exc());result[key]={'completed':False,'error':repr(err)}
            for path in list(PIN):pin(path)
        result.update(elapsed_seconds=time.monotonic()-tic,input_files=len(PIN),input_bytes=sum(p['bytes'] for p in PIN.values()),compared_leaves=len(CHECKS),nonexact_leaves=sum(not c.get('exact',False) for c in CHECKS),warnings=[str(w.message) for w in caught],notes=NOTES)
        (a.out/'checks.json').write_text(json.dumps(norm(CHECKS),indent=2)+'\n');(a.out/'inputs.json').write_text(json.dumps(list(PIN.values()),indent=2)+'\n');(a.out/'report.json').write_text(json.dumps(norm(result),indent=2)+'\n');print(json.dumps(norm(result)))
    if any(isinstance(v,dict) and v.get('completed') is False for v in result.values()):raise SystemExit(1)
