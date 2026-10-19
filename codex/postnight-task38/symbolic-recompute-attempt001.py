"""Task38 retained-output replay, independent of producer numerical imports."""
import os, sys, json, hashlib, resource, signal, time, itertools, collections, traceback
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS, (8*1024**3, 8*1024**3))
resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
signal.alarm(120)
import numpy as np
from scipy.ndimage import distance_transform_edt, distance_transform_cdt
from scipy.spatial import cKDTree
BASE=Path('/data2/neural-shrimp-data/explore/llm-soft-loop/pod64')
OUT=Path(sys.argv[2]); OUT.mkdir(exist_ok=False)
MAN={}; started=time.monotonic()
def memory():
    values=dict(s.split(':',1) for s in Path('/proc/meminfo').read_text().splitlines())
    available=int(values['MemAvailable'].split()[0])*1024
    if available<16*1024**3: raise RuntimeError('available RAM below 16GiB; stopping own saved audit')
    return available
def pin(path):
    path=Path(path)
    if str(path) not in MAN:
        h=hashlib.sha256()
        with path.open('rb') as f:
            while chunk:=f.read(4*1024**2): h.update(chunk)
        MAN[str(path)]={'sha256':h.hexdigest(),'bytes':path.stat().st_size}
    return path
def js(path): return json.loads(pin(path).read_text())
def nz(path): return np.load(pin(path), allow_pickle=False)
def rot(ids):
    ids=tuple(map(int,ids)); return min(ids[i:]+ids[:i] for i in range(len(ids))) if ids else ()
def order(a,b):
    parity=False
    for x,y in zip(a,b):
        if x!=y: return (1 if 'LCR'.index(x)>'LCR'.index(y) else -1)*(-1 if parity else 1)
        if x=='C': return 0
        if x=='R': parity=not parity
    return 0
def maxrot(seq):
    rotations=[seq[i:]+seq[:i] for i in range(len(seq))]
    best=0
    for i in range(1,len(seq)):
        if order(rotations[i]*20,rotations[best]*20)>0: best=i
    return best,rotations[best]
def admissible(kmax=9):
    out=[]
    for k in range(1,kmax+1):
        for bits in itertools.product('LR',repeat=k-1):
            w=''.join(bits); seq=w+'C'
            if all(order((seq[j:]+seq[:j])*20,seq*20)<0 for j in range(1,k)): out.append(w)
    return set(out)
MSS=admissible()
def code(cos,q):
    k=q//6; n=len(cos); phases=[]
    for j in range(6):
        c=cos[[n-1-j-6*i for i in range(k)][::-1]]
        if not np.isfinite(c).all(): return None,'missing',{}
        seq=''.join('L' if v>0 else 'R' for v in c); shift,word=maxrot(seq)
        remaining=np.delete(abs(c),(shift-1)%k)
        phases.append((word, bool(not len(remaining) or min(remaining)>=.5)))
    usable=[w for w,ok in phases if ok]
    if not usable: return None,'too few usable phases',{}
    votes=collections.Counter(w[:-1] for w in usable).most_common()
    w,v=votes[0]
    if v<3 or (len(votes)>1 and votes[1][1]==v): return None,'phases disagree',{}
    xv=collections.Counter(s[-1] for s in usable if s[:-1]==w).most_common()
    word=w+'|'+(xv[0][0] if xv[0][1]>=2*v/3 else 'C')
    info={'usable':len(usable),'votes':v,'all_usable_agree_W':len(votes)==1}
    return word,('ok' if w in MSS else 'not an MSS word'),info
def words():
    lo,hi=map(int,sys.argv[3:5]); info=js(BASE/'sd/words/words.json'); saved=nz(BASE/'sd/words/words.npz')
    labels=list(saved['labels']); complete=[]; totals=collections.Counter(); rejects=collections.Counter()
    for band in range(lo,hi):
        available=memory(); rel=info['args']['list'][band]; path=BASE/'bands'/Path(rel).name
        meta=js(path/'scan.json'); z=nz(path/'scan.npz'); hr=nz(BASE/f'bands/hr10-c{band:02d}/scan.npz')
        fam={f['family']:rot(f.get('token_ids',[])) for f in meta['families']}; body=tuple(info['body_text_ids'])
        per=z['period']; cosine=z['word_cos']; family=z['family']; status=np.zeros(per.shape,int); found={}; reasons=collections.Counter(); weakdisagree=0
        for r,c in zip(*np.nonzero(per)):
            q=int(per[r,c]); ids=fam.get(int(family[r,c]),()); k=q//6
            if q%6 or k>8 or ids!=rot(body*k): status[r,c]=1;continue
            word,reason,details=code(cosine[r,c],q)
            if reason!='ok':
                status[r,c]=2;reasons[reason]+=1
                if reason=='not an MSS word': rejects[word]+=1
            else:
                status[r,c]=3;found[(r,c)]=word
                weakdisagree+=not details['all_usable_agree_W']
        r0=40*band; check=saved['status'][r0:r0+40]
        label_mismatch=sum(str(labels[int(saved['word_index'][r0+r,c])])!=word for (r,c),word in found.items())
        m=(hr['period']>0)&(hr['period']<=32)
        row={'band':band,'raw':str(path),'shape':list(per.shape),'cosine_dtype':str(cosine.dtype),
             'status_mismatches':int((check!=status).sum()),'label_mismatches':label_mismatch,
             'coded':len(found),'C':sum(w.endswith('|C') for w in found.values()),'reasons':dict(reasons),
             'accepted_despite_usable_W_disagreement':weakdisagree,'min_available_bytes':available,
             'periodic_up_to32':int(m.sum()),'periodic_period_differences':int((per[m]!=hr['period'][m]).sum()),
             'periodic_exponent_bit_differences':int((z['lyapunov'][m]!=hr['lyapunov'][m]).sum()),
             'chaos_label_differences':int(((z['lyapunov']>.001)!=(hr['lyapunov']>.001)).sum()),
             'grid_equal':bool(np.array_equal(z['alpha'],hr['alpha']) and np.array_equal(z['log10_temperature'],hr['log10_temperature']))}
        if 'rows_from_checkpoint' in meta:
            cp=nz(BASE/'bands'/Path(meta['args']['partial']).name/'scan-partial.npz')
            rr=nz(BASE/'bands'/Path(meta['args']['rerun']).name/'scan.npz'); nr=int(meta['rows_from_checkpoint'])
            merge={}
            for key in ('period','lyapunov','entropy','text_period','word_cos','word_log_growth'):
                shape=z[key].shape; left=cp[key][:nr*shape[1]].reshape(nr,*shape[1:]); merged=np.concatenate([left,rr[key]])
                merge[key]=bool(np.array_equal(merged,z[key],equal_nan=True))
            rerunmask=m[nr:]; row.update(rows_checkpoint=nr,rows_rerun=40-nr,merge_arrays_equal=merge,
                rerun_periodic_up_to32=int(rerunmask.sum()),rerun_period_differences=int((per[nr:][rerunmask]!=hr['period'][nr:][rerunmask]).sum()),
                checkpoint_period_differences=int((per[:nr][m[:nr]]!=hr['period'][:nr][m[:nr]]).sum()))
            cp.close();rr.close()
        complete.append(row);totals.update({'coded':row['coded'],'C':row['C'],**reasons});z.close();hr.close()
        print(json.dumps(row),flush=True)
    st=saved['status']; bad=st==2; inside=st>=2
    edges={metric:dict(count=int((distance_transform_cdt(inside,metric=metric)[bad]<=2).sum()),denominator=int(bad.sum())) for metric in ('chessboard','taxicab')}
    edges['euclidean']={'count':int((distance_transform_edt(inside)[bad]<=2).sum()),'denominator':int(bad.sum())}
    return {'bands':complete,'totals':dict(totals),'MSS_rejected_labels':dict(rejects),'edges_from_saved_status':edges,
            'saved_global_counts':{str(k):int((st==k).sum()) for k in range(4)}}
def threshold(x,sign,weak):
    use=~weak; xx=x[use]; ll=sign[use]; count=len(xx); nL=int(ll.sum()); minority=min(nL,count-nL)
    if count<10 or minority==0:return 0,minority
    sortedL=ll[np.argsort(xx)]; prefix=np.r_[0,np.cumsum(sortedL)]
    accurate=prefix+(count-nL)-(np.arange(count+1)-prefix)
    return int(count-max(accurate.max(),(count-accurate).max())),minority
def bend(xy):
    _,neigh=cKDTree(xy).query(xy,k=min(40,len(xy))); centered=xy[neigh]-xy[neigh].mean(axis=1,keepdims=True)
    xx=(centered[...,0]**2).mean(1); yy=(centered[...,1]**2).mean(1); xyv=(centered[...,0]*centered[...,1]).mean(1)
    return float(np.percentile(abs(np.degrees(.5*np.arctan2(2*xyv,xx-yy))),95))
def turns(x,y):
    edges=np.linspace(min(x),max(x),17); bins=np.clip(np.digitize(x,edges)-1,0,15)
    meds=[np.median(y[bins==i]) for i in range(16) if (bins==i).sum()>=4]
    directions=[]; acc=0
    for delta in np.diff(meds):
        acc+=delta
        if abs(acc)>.03*np.ptp(y):directions.append(np.sign(acc));acc=0
    return int(sum(directions[i]!=directions[i-1] for i in range(1,len(directions))))
def returnmaps():
    record=js(BASE/'sd/returnmaps/returnmaps.json'); body=tuple(record['summary']['body_text_ids']); rows=[]; mismatches=[]
    original={(c['band'],c['row'],c['col']):c for c in record['cells']}
    for band,rel in enumerate(record['args']['list']):
        memory();path=BASE/'bands'/Path(rel).name;meta=js(path/'scan.json');z=nz(path/'scan.npz')
        text={f['text_family']:rot(f['token_ids']) for f in meta['texts']}
        projection=z['return_projection'];cosine=z['word_cos']; domain=np.zeros(z['period'].shape,bool)
        for r in range(domain.shape[0]):
            for c in range(domain.shape[1]):domain[r,c]=int(z['text_period'][r,c])==6 and text.get(int(z['text_family'][r,c]))==body
        for r,c in zip(*np.nonzero(domain & (z['lyapunov']>.001))):
            phases=[];orig=original[(band,int(r),int(c))]
            for phase in range(6):
                s=projection[r,c,phase,:,0].astype(float); cs=cosine[r,c,phase+6::6][:len(s)-1]
                weak=abs(cs)<.5; errors,minority=threshold(s[:len(cs)],cs>0,weak)
                bd=bend(projection[r,c,phase,:len(cs),:].astype(float)); use=weak.mean()<=.1 and minority>=5 and bd<45
                nt=turns(s[:-1],s[1:]); row=dict(phase=phase,errors=errors,minority=minority,bend_deg=bd,
                    weak_fraction=float(weak.mean()),usable=bool(use),turning_points_robust=nt)
                for key,value in row.items():
                    old=orig['phases'][phase][key]
                    if isinstance(value,float):same=np.isclose(value,old,rtol=1e-10,atol=1e-12)
                    else:same=value==old
                    if not same:mismatches.append([band,int(r),int(c),phase,key,value,old])
                phases.append(row)
            used=[p for p in phases if p['usable']]; pooled=sum(p['errors'] for p in used)/sum(p['minority'] for p in used)
            best=min(used or phases,key=lambda p:p['bend_deg'])
            rows.append(dict(alpha=orig['alpha'],log10_temperature=orig['log10_temperature'],band=band,row=int(r),col=int(c),
                phases=phases,pooled_error_over_minority=pooled,max_phase_error_over_minority=max(p['errors']/p['minority'] for p in used),
                saved_old_flag=orig['unimodal_all_phases'],pass_pooled_5percent=bool(pooled<=.05),
                straightest_phase=best['phase'],straightest_turns=best['turning_points_robust']))
        z.close()
    passed=[r for r in rows if r['pass_pooled_5percent']]; fail=[r for r in rows if not r['pass_pooled_5percent']]
    return dict(rows=rows,mismatches=mismatches,chaotic_six_band=len(rows),pooled_pass=len(passed),zero_pooled=sum(r['pooled_error_over_minority']==0 for r in passed),
                max_pooled_pass=max(r['pooled_error_over_minority'] for r in passed),old_flag_pass=sum(r['saved_old_flag'] for r in rows),
                exceptions=len(fail),exceptions_more_than_one_turn=sum(r['straightest_turns']>1 for r in fail))
def verify():
    wordmeta=js(BASE/'sd/words/words.json'); words=nz(BASE/'sd/words/words.npz'); labels=words['labels']; rows=[]
    summary=js(BASE/'sd/verify-summary/verify-summary.json')
    for name in ('verify-a','verify-b'):
        rec=js(BASE/f'sd/{name}/verify.json')
        for c in rec['cells']:
            r=np.argmin(abs(words['log10_temperature']-c['log10_temperature']));col=np.argmin(abs(words['alpha']-c['alpha']))
            label=str(labels[int(words['word_index'][r,col])]);ph=c.get('phases',[])
            clean=[p for p in ph if p['separable'] in (True,None) and p['min_abs_cos']>=.5]
            kinds={p['word'] for p in clean}; confirmed=bool(clean and len(kinds)==1 and next(iter(kinds))[:-1]+'|'+next(iter(kinds))[-1]==label)
            pair=c['status']=='leading multiplier complex or not simple'
            rows.append(dict(alpha=c['alpha'],log10_temperature=c['log10_temperature'],period=c['period'],scan_label=label,
                converged=c['converged'],last_residual=c['newton_residuals'][-1],leading_multiplier=c['leading_multiplier'],
                clean_phases=len(clean),clean_words=list(kinds),word_confirmed=confirmed,complex_or_nonsimple=pair,
                original_status=c['status'],all_phases_agree=c.get('phases_agree'),parity=c.get('parity_matches_multiplier'),
                complete_original=c))
    counts=dict(total=len(rows),confirmed_words=sum(r['word_confirmed'] for r in rows),complex_or_nonsimple=sum(r['complex_or_nonsimple'] for r in rows),
        no_clean_real=sum(not r['clean_phases'] and not r['complex_or_nonsimple'] for r in rows),
        all_phase_disagree=sum(r['all_phases_agree'] is False for r in rows),max_residual=max(r['last_residual'] for r in rows))
    return dict(rows=rows,counts=counts,summary_counts={k:v for k,v in summary.items() if k!='rows'})
try:
    memory();pin(__file__)
    result={'words':words,'returnmaps':returnmaps,'verify':verify}[sys.argv[1]]()
    result.update(elapsed_seconds=time.monotonic()-started,source_sha256=MAN[str(Path(__file__))]['sha256'],
        python=sys.version,numpy=np.__version__,limits={'AS_GiB':8,'cpu_seconds':120,'alarm_seconds':120,'threads':1,'nice':os.getpriority(os.PRIO_PROCESS,0)})
    (OUT/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
except BaseException as exc:
    (OUT/'failure.json').write_text(json.dumps({'type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc()},indent=2)+'\n');raise
finally:
    (OUT/'input-manifest.json').write_text(json.dumps(MAN,indent=2)+'\n')
