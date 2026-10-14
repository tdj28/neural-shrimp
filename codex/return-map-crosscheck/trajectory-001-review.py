"""Prospective Task19 saved-trajectory checks; no model, fitting or Git import.

Run only after Root reviews the actual hash-bound card and Inference seals the
original attempt/closure. Negative scientific outcomes are retained, not failed.
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import time
import traceback

RAW = Path('/data2/neural-shrimp-data/codex/llm-return-map-crosscheck')
ATTEMPT = RAW/'trajectory-001'
AUTHORITY = RAW/'input-authority-001'
AUTHORITY_SHA = 'd3014365b8445f8508271959137b922f694f0519665e9dfcb8f459e49517d2e8'
# The literal hashes below are the accepted source, not imports of its reducers.
SCIENCE = {
 'codex/return-map-crosscheck/reduce.py':'d87ec765cde3c0cc1145e6225b9b98b9a9c85468df85368c938c571aebda098b',
 'codex/return-map-crosscheck/trajectory.py':'8da808f5413d4f7caca49b6c0cba4c8182ff4322c91e123513e9ccb3e9737524',
 'codex/return-map-crosscheck/runtime.py':'41f0f19f023f2bbd5f6842929bb07cad70b86ea8d0399eec717a22ae478d468a',
 'codex/hostile-fold-crosscheck/assets.py':'06477ee36a78f15ee8f1e8ae196dc53ae230ff19368fbda638eb6a6e0ef758e0',
 'codex/crosscheck/qwen_map.py':'493a3d1d57fec227f5ba2e3f8f9cfcfdd02d22495d1d44e6449ede901310ae6e'}
LIMITS = dict(cpu_threads=1,nice=10,memory_gib=4,timeout_seconds=120,
              maximum_output_mib=16,minimum_start_available_gib=20,minimum_runtime_available_gib=16)
ATOL, RTOL = 5e-15, 5e-13  # Saved binary64 arithmetic check, not a new science gate.
REPORT = dict(status='started',checks=[],arithmetic=[],scope='saved arithmetic and associations only',
              model_called=False,fit_recomputed=False,global_nearest_search_repeated=False)
START = None


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def read(p):
    def nonfinite(x):raise ValueError('nonfinite JSON literal '+x)
    return json.loads(Path(p).read_text(),parse_constant=nonfinite)


def require(ok,name):
    REPORT['checks'].append(dict(name=name,passed=bool(ok)))
    if not ok:raise ValueError(name)


def tick():
    require(time.monotonic()-START<120,'saved review deadline')
    available=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
    require(available>=16*1024**3,'runtime headroom16GiB')


def close(actual,expected,name):
    a,b=np.asarray(actual),np.asarray(expected)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all(),name+' finite shape')
    d=a-b;limit=ATOL+RTOL*np.abs(b)
    REPORT['arithmetic'].append(dict(name=name,count=int(a.size),nonzero=int(np.count_nonzero(d)),
        max_absolute=float(np.max(np.abs(d))) if a.size else 0.,atol=ATOL,rtol=RTOL,
        min_signed=float(np.min(d)) if a.size else 0.,max_signed=float(np.max(d)) if a.size else 0.))
    require(bool(np.all(np.abs(d)<=limit)),name+' saved arithmetic')


def member(root,name):
    p=root/name
    require(not Path(name).is_absolute() and '..'not in Path(name).parts and p.resolve()==p and not p.is_symlink(),'literal inventory member '+name)
    return p


def inventory(root,rows,label,extras=()):
    names=[r['path']for r in rows]
    require(len(names)==len(set(names)),label+' unique inventory')
    for r in rows:
        p=member(root,r['path']);require(p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],label+'/'+r['path'])
    actual={str(p.relative_to(root))for p in root.rglob('*')if p.is_file()}
    require(actual==set(names)|set(extras),label+' exact original inventory')


def literal(source,name):
    for node in ast.parse(source.read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name)and t.id==name for t in node.targets):return ast.literal_eval(node.value)
    raise ValueError('missing captured literal '+name)


def provenance(card):
    require(card['schema']=='codex-task19-trajectory-saved-review-v1' and card['admission']=='reviewed' and card['limits']==LIMITS,'exact reviewed saved audit card')
    require(card['script_sha256']==sha(__file__),'exact own review source')
    require(sha(ATTEMPT/'receipt.json')==card['receipt_sha256'],'original completed receipt pin')
    receipt=read(ATTEMPT/'receipt.json');inventory(ATTEMPT,receipt['artifacts'],'original attempt',('receipt.json',))
    require(receipt['status']=='return_map_completed' and receipt['exit_code']==0 and receipt['timed_out']is False
            and receipt['worker_termination_verified']is True and receipt['uncertain_ownership']==[], 'original computational closure')
    require(receipt['source_commit']==card['source_commit'] and receipt['plan_sha256']==card['plan_sha256'],'actual source/plan identity')
    manifest=read(ATTEMPT/'manifest.json');require(manifest['source_commit']==card['source_commit'],'captured original source identity')
    require(manifest['plan_sha256']==card['plan_sha256'] and manifest['command']==['/data2/neural-shrimp/.venv/bin/python','-I','-B','-u',str(ATTEMPT/'source/codex/return-map-crosscheck/runtime.py'),'--plan',str(ATTEMPT/'source/codex/return-map-crosscheck/trajectory-001-plan.json'),'--output',str(ATTEMPT/'run')],'exact original command association')
    rows=manifest['source'];require(len(rows)==10 and len({r['path']for r in rows})==10,'ten-file source capsule')
    for r in rows:
        p=member(ATTEMPT/'source',r['path']);require(p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],'captured source '+r['path'])
    for name,h in SCIENCE.items():require(sha(ATTEMPT/'source'/name)==h,'accepted scientific bytes '+name)
    cp=ATTEMPT/'source/codex/return-map-crosscheck/trajectory-001-plan.json';plan=read(cp)
    require(sha(cp)==card['plan_sha256'] and plan['admission']=='reviewed' and plan['source_sha256']==SCIENCE,'actual reviewed trajectory plan')
    require(plan['mode']=='trajectory' and plan['input_manifest_sha256']==AUTHORITY_SHA and plan['experiment']==dict(alpha=.0655,log10_temperature=.1899492494639028,transient=1000,record=2000,offsets=[1e-6,1e-9],seed=0,primary_phase=6,maximum_period=64,threshold=1e-8),'literal actual experiment')
    require((ATTEMPT/'run/original-plan.json').read_bytes()==cp.read_bytes(),'original worker plan bytes')
    binding=read(ATTEMPT/'run/input-binding.json');require(binding['plan']==plan and binding['source_commit']==card['source_commit'],'worker source/input binding')
    require(sha(AUTHORITY/'manifest.json')==AUTHORITY_SHA,'immutable producer snapshot')
    inputs=read(AUTHORITY/'manifest.json');inventory(AUTHORITY,inputs['files'],'original input',('manifest.json',))
    require(binding['input_manifest']==inputs and len(inputs['files'])==10,'all ten original input bindings')
    # Coordinator supplies the actual accepted authority and literal assertions,
    # after reaping. No assumed closure schema, fresh Git or process polling here.
    inherited=card['coordinator_authority'];p=Path(inherited['path'])
    require(sha(p)==inherited['sha256'] and bool(inherited['assertions']),'reviewed coordinator authority pin/assertions')
    authority=read(p)
    for assertion in inherited['assertions']:
        value=authority
        for key in assertion['keys']:value=value[key]
        require(value==assertion['equals'],'accepted closure/source authority '+repr(assertion['keys']))
    REPORT['authority']=dict(receipt_sha256=card['receipt_sha256'],source_commit=card['source_commit'],
        plan_sha256=card['plan_sha256'],coordinator=inherited,
        original_git_and_outer_process_closure='inherited from this Root-accepted coordinator authority; not re-executed')
    return receipt,plan


def array(name,shape,dtype='float64'):
    a=np.load(ATTEMPT/'run'/name,mmap_mode='r',allow_pickle=False)
    require(a.shape==shape and str(a.dtype)==dtype and np.isfinite(a).all(),'complete array '+name)
    return a


def period(tokens):
    return next((q for q in range(1,65)if len(tokens)>2*q and np.array_equal(tokens[q:],tokens[:-q])),0)


def twins_and_readouts(result):
    initial=array('initial-state.npy',(896,));directions=array('initial-twins-directions.npy',(2,896))
    twins=array('initial-twins-state.npy',(3,896));eps=np.array([1e-6,1e-9]);data={}
    for phase,n,batch in [('transient',1000,1),('record',2000,3)]:
        for key in ('input','state'):data[phase,key]=array(phase+'-'+key+'.npy',(n,batch,896))
        for key in ('lead','pmax','entropy'):
            a=array(phase+'-'+key+'.npy',(n,batch),'int64'if key=='lead'else'float64');data[phase,key]=a
            require(bool(np.all((a>=0)&(a<151936)))if key=='lead'else bool(np.all((a>0)&(a<=1)))if key=='pmax'else bool(np.all(a>=0)),'readout range '+phase+key)
    require(np.array_equal(data['transient','input'][0,0],initial),'fixed retained initial input')
    require(np.array_equal(data['transient','input'][1:],data['transient','state'][:-1]),'entire transient state chain')
    seed=np.random.default_rng(0).normal(size=(2,896));seed/=np.linalg.norm(seed,axis=1)[:,None]
    require(np.array_equal(seed,directions),'fixed seed0 unit directions')
    close(twins,np.concatenate([data['transient','state'][-1],data['transient','state'][-1,0]+directions*eps[:,None]]),'initial twin offsets')
    require(np.array_equal(data['record','input'][0],twins),'first native twin input')
    ren=array('record-renormalized_state.npy',(2000,3,896))
    require(np.array_equal(data['record','input'][1:],ren[:-1]),'entire base/twin renormalized input chain')
    growth={k:array('record-'+k+'.npy',(2000,2))for k in ('raw_distance','used_distance','log_growth','clamped')}
    dif=data['record','state'][:,1:]-data['record','state'][:,:1]
    raw=np.linalg.norm(dif,axis=2);used=np.maximum(raw,eps*1e-30);logs=np.log(used/eps)
    for k,v in [('raw_distance',raw),('used_distance',used),('log_growth',logs),('clamped',(raw<eps*1e-30).astype(float))]:close(growth[k],v,'all2000 '+k)
    expected=np.concatenate([data['record','state'][:,:1],data['record','state'][:,:1]+dif*(eps/used)[:,:,None]],axis=1)
    close(ren,expected,'every full-state renormalization')
    require(np.array_equal(ren[:,0],data['record','state'][:,0]),'base state never renormalized')
    require(result['floors_used']==growth['clamped'].sum(0).astype(int).tolist(),'both complete floor counts')
    close(result['lyapunov'],logs.mean(0),'both full mean exponents')
    require([(r['start'],r['stop'])for r in result['disjoint_windows']]==[(i,i+500)for i in range(0,2000,500)],'all four frozen disjoint windows')
    for r in result['disjoint_windows']:close(r['means'],logs[r['start']:r['stop']].mean(0),'window '+str(r['start']))
    require([r['stop']for r in result['running_means']]==list(range(100,2001,100)),'all twenty running windows')
    for r in result['running_means']:close(r['means'],logs[:r['stop']].mean(0),'running '+str(r['stop']))
    events=[read_line for line in (ATTEMPT/'run/events.jsonl').read_text().splitlines()if (read_line:=json.loads(line))]
    expected=[(e,'transient',i)for i in range(1000)for e in ('input','return')]+[('initial_twins','record',-1)]+[(e,'record',i)for i in range(2000)for e in ('input','return','growth')]
    require([(r['event'],r['phase'],r['index'])for r in events]==expected,'complete ordered8001 native events')
    close([r['seconds']for r in events if r['event']=='return'],result['seconds_native_calls'],'all3000 recorded call times')
    REPORT['twins']=dict(lyapunov=result['lyapunov'],floors=result['floors_used'],disjoint=result['disjoint_windows'],running=result['running_means'],
        every_growth_row=[dict(index=i,raw_distance=growth['raw_distance'][i].tolist(),used_distance=growth['used_distance'][i].tolist(),
        log_growth=growth['log_growth'][i].tolist(),clamped=growth['clamped'][i].tolist(),arithmetic_delta=(growth['log_growth'][i]-logs[i]).tolist())for i in range(2000)])
    return data


def state_and_cycle(data,result):
    s=data['record','state'][:,0];lead=data['record','lead'][:,0];q=period(lead);diag=result['state_diagnostics'];scale=max(1.,float(np.linalg.norm(s[-1])))
    distances=[float(np.linalg.norm(s[-lag-1]-s[-1])/scale)for lag in range(1,65)]
    require([r['lag']for r in diag['all_lags']]==list(range(1,65)),'all64 lag dispositions')
    close([r['relative_distance']for r in diag['all_lags']],distances,'all64 state lag values');close(diag['scale'],scale,'state lag scale')
    require(diag['period']==next((i+1 for i,d in enumerate(distances)if d<1e-8),0) and diag['tolerance']==1e-8,'literal finite state detector')
    close(diag['diameter_relative'],2*np.linalg.norm(s[-400:]-s[-400:].mean(0),axis=1).max()/scale,'last400 radius diameter')
    require(result['lead_period']==q and result['lead_cycle']==(lead[:q].tolist()if q else None),'actual complete own token cycle')
    require(result['token_counts']==[dict(token=int(t),count=int((lead==t).sum()))for t in np.unique(lead)],'every token count')
    require(result['primary_seven_phase_clock_matches']==(q==7),'q7 primary clock disposition')
    positive=all(v>1e-3 for v in result['lyapunov'])
    require(result['full_offsets_both_positive']==positive and result['numerical_repeating_text_chaos']==bool(q and not diag['period'] and positive),'finite scientific classification including negatives')
    with np.load(AUTHORITY/'trace-best/trace.npz',allow_pickle=False)as z:
        require(set(z.files)=={'lead','pmax','entropy','states'},'original trace schema')
        old=z['lead'].copy()  # No producer states/probabilities decoded.
    require(old.shape==(2000,) and np.issubdtype(old.dtype,np.integer) and np.all((old>=0)&(old<151936)),'original complete leading-token record')
    oldq=period(old);equal=q==oldq and q>0 and np.array_equal(lead[:q],old[:oldq])
    rotations=[i for i in range(q)if q==oldq and np.array_equal(lead[:q],np.roll(old[:q],-i))]if q else []
    producer=read(AUTHORITY/'trace-best/hidden-chaos.json')['cells'][0]
    REPORT['cycle']=dict(own_period=q,producer_period=oldq,own_ids=lead[:q].tolist()if q else None,
        producer_ids=old[:oldq].tolist()if oldq else None,exact_same_phase_cycle_equal=bool(equal),cyclic_shifts=rotations,
        complete2000_ids_equal=bool(np.array_equal(lead,old)),producer_original_labels=producer.get('lead_text'),
        producer_label_token_ids=producer.get('lead_tokens'),label_provenance='quoted immutable producer JSON; no tokenizer/model decoded',
        finite_state_diagnostics=diag,full_offsets_both_positive=positive,numerical_repeating_text_chaos=result['numerical_repeating_text_chaos'])
    p=data['record','pmax'][:,0];ent=data['record','entropy'][:,0]
    for name,value in [('mean',p.mean()),('sd',p.std()),('entropy_mean',ent.mean()),('entropy_sd',ent.std())]:close(result['probability_summary'][name],value,'readout '+name)
    return s,p,q


def bands(s,q,result):
    if not q:require('bands'not in result,'undefined-period bands absent');return
    b=result['bands'];pairs=b['pairs'];require([r['phases']for r in pairs]==[[i,j]for i in range(q)for j in range(i+1,q)],'all actual phase band pairs')
    scale=float(np.linalg.norm(s,axis=1).mean());centers=np.stack([s[k::q].mean(0)for k in range(q)])
    radii=np.array([np.linalg.norm(s[k::q]-centers[k],axis=1).max()for k in range(q)])
    close(b['mean_state_norm'],scale,'mean state band scale');close(b['radii_relative'],radii/scale,'every relative band radius')
    for r in pairs:
        a,c=r['phases'];i,j=r['sample_indices'];require(type(i)is int and type(j)is int and 0<=i<2000 and 0<=j<2000 and i%q==a and j%q==c,'literal saved nearest pair '+str([a,c]))
        close(r['sample_distance'],np.linalg.norm(s[i]-s[j]),'saved nearest witness '+str([a,c]))
        center=float(np.linalg.norm(centers[a]-centers[c]));close(r['centroid_distance'],center,'centroid '+str([a,c]));close(r['bounding_ball_gap'],center-radii[a]-radii[c],'ball gap '+str([a,c]))
    for key,field in [('minimum_sample_distance_relative','sample_distance'),('minimum_centroid_distance_relative','centroid_distance'),('minimum_bounding_ball_gap_relative','bounding_ball_gap')]:
        if pairs:close(b[key],min(r[field]for r in pairs)/scale,key)
        else:require(b[key]is None,key+' undefined for q1')
    require(b['invariant_set_separation_proved']is False,'no invariant band claim')
    REPORT['bands']=dict(**b,independent_check='all saved pair witnesses, indices, centroids, radii and aggregates; exhaustive-minimum completeness inherited from reviewed own runtime')


def fit_fields(f,x,n,basis,degree):
    require(f['basis']==basis and type(f['degree'])is int and f['degree']==degree,'literal polynomial basis/degree')
    lo,hi=float(x.min()),float(x.max());shift=lo if basis=='power'else(hi+lo)/2;scale=hi-lo if basis=='power'else(hi-lo)/2
    require(f['shift']==shift and f['scale']==scale,'literal saved x-derived polynomial domain')
    require(f['points']==n and len(f['coefficients'])==degree+1 and scale>0,'retained polynomial shape/domain')
    require(type(f['rank'])is int and 0<=f['rank']<=degree+1 and type(f['rank_full'])is bool and f['rank_full']==(f['rank']==degree+1),'polynomial numeric-rank disposition')
    critical=f['critical_points'];require(f['stationary_count']==len(critical),'all stationary dispositions')
    roots=f['derivative_roots_scaled'];physical=sorted(a*f['scale']+f['shift']for a,b in roots if abs(b)<1e-9 and x.min()<a*f['scale']+f['shift']<x.max())
    close([c['x']for c in critical],physical,'saved stationary root coordinate conversion')
    count=sum(abs(b*(f['scale']if f['basis']=='chebyshev'else 1.))<1e-9 and x.min()<a*f['scale']+f['shift']<x.max()for a,b in roots)
    require(f['producer_stationary_count']==count,'literal original root-tolerance disposition')
    for c in critical:
        kind='quadratic_maximum'if c['left_derivative']>0>c['right_derivative']and c['curvature']<0 else 'quadratic_minimum'if c['left_derivative']<0<c['right_derivative']and c['curvature']>0 else 'unresolved_or_degenerate'
        require(c['kind']==kind,'fitted curvature/sign classification')
    require(f['fitted_quadratic_maximum_count']==sum(c['kind']=='quadratic_maximum'for c in critical),'fitted maximum count')
    require((f['exponent_per_repetition']is None)==(f['zero_derivatives']>0),'undefined derivative retained')


def fit_dispositions(rows,p,q,label):
    require([r['phase']for r in rows]==list(range(q)),label+' all phase rows');summaries=[]
    for r in rows:
        pk=p[r['phase']::q];x,y=pk[:-1],pk[1:];n=len(x)
        require(r['observations']==len(pk) and r['pairs']==n,label+' actual phase counts')
        expected='insufficient_repetitions'if len(p[::q])<30 or n<10 else 'constant_coordinate'if np.ptp(x)<=0 or np.ptp(y)<=0 else 'fitted'
        require(r['disposition']==expected,label+' phase disposition '+str(r['phase']))
        if expected!='fitted':summaries.append(r);continue
        f=r['original'];fit_fields(f,x,n,'power',min(8,max(2,(len(p[::q])-1)//6)));require(f['degree']==min(8,max(2,(len(p[::q])-1)//6))and f['points']==n,'literal original degree/points')
        row=dict(phase=r['phase'],observations=len(pk),pairs=n,disposition=expected,original=f)
        for mode,kind in [('amended_random','random'),('purged_contiguous','purged_contiguous')]:
            c=r[mode];fit_fields(c['fit'],x,n,'chebyshev',c['selected_degree']);require(c['kind']==kind,'literal CV kind');partition=c['partition']
            order=np.random.default_rng(0).permutation(n)if kind=='random'else np.arange(n)
            require(len(partition)==5,'all five folds')
            for part,test in zip(partition,np.array_split(order,5)):
                mask=np.ones(n,dtype=bool);mask[test]=False
                if kind!='random':mask[max(0,int(test[0])-1):min(n,int(test[-1])+2)]=False
                require(part==dict(train=np.flatnonzero(mask).tolist(),test=test.tolist()),'exact frozen train/test pair partition')
            candidates=c['candidates'];require([a['degree']for a in candidates]==[d for d in range(4,25,2)if d<=max(4,(n-1)//6)],'all fixed candidate degrees')
            for a in candidates:
                require([f['fold']for f in a['folds']]==list(range(5)),'every candidate fold')
                for fold,part in zip(a['folds'],partition):
                    rank=fold['rank'];mse=fold['mse'];train=np.asarray(part['train']);test=np.asarray(part['test'])
                    require(type(rank)is int and 0<=rank<=a['degree']+1 and type(fold['rank_full'])is bool and fold['rank_full']==(rank==a['degree']+1),'candidate fold numeric-rank disposition')
                    require(type(mse)in(float,int) and math.isfinite(mse) and mse>=0,'finite nonnegative saved fold MSE')
                    lo,hi=float(x[train].min()),float(x[train].max());outside=int(((x[test]<lo)|(x[test]>hi)).sum())
                    require(fold['train_min']==lo and fold['train_max']==hi and type(fold['test_outside_train_range'])is int and fold['test_outside_train_range']==outside,'literal fold train-range/outside metadata')
                require(a['all_fold_ranks_full']==all(f['rank_full']for f in a['folds']),'candidate rank gate')
                close(a['cv_residual_relative'],math.sqrt(sum(f['mse']for f in a['folds'])/5)/float(np.ptp(y)),'equal-fold MSE summary')
            selected=min(candidates,key=lambda a:a['cv_residual_relative'])
            require(c['selected_degree']==selected['degree'] and c['cv_residual_relative']==selected['cv_residual_relative'],'saved minimum-CV selection')
            require(c['producer_value_fit_qualified']==(selected['cv_residual_relative']<.01),'literal producer value gate')
            require(c['qualified_value_fit']==bool(selected['all_fold_ranks_full']and c['fit']['rank_full']and selected['cv_residual_relative']<.01),'own disclosed rank+value gate')
            row[mode]={k:v for k,v in c.items()if k!='partition'}
            row[mode]['partition_sha256']=hashlib.sha256(json.dumps(partition,sort_keys=True).encode()).hexdigest()
        summaries.append(row)
    REPORT[label]=summaries


def aggregate_dispositions(rows,q,result):
    fitted=[r for r in rows if r['disposition']=='fitted'];summary=result['all_phase_summary'];full=float(np.mean(result['lyapunov']))
    for mode in ('original','amended_random','purged_contiguous'):
        kept=([min(fitted,key=lambda r:r['original']['residual_relative'])]if fitted else [])if mode=='original'else [r for r in fitted if r[mode]['producer_value_fit_qualified'if mode=='amended_random'else'qualified_value_fit']]
        fits=[r['original']if mode=='original'else r[mode]['fit']for r in kept];v=summary[mode]
        require(v['phases']==[r['phase']for r in kept],'all aggregate selected phases '+mode)
        close(v['full_twin_mean'],full,'aggregate full comparator '+mode)
        require(v['original_value_fit_qualified']==(fits[0]['residual_relative']<.01 if mode=='original'and fits else None),'original fit gate '+mode)
        exponents=[f['producer_floor_exponent']/q for f in fits];unfloored=[None if f['exponent_per_repetition']is None else f['exponent_per_repetition']/q for f in fits]
        if exponents:close(v['spread'],np.ptp(exponents),'aggregate spread '+mode)
        else:require(v['spread']is None,'undefined spread '+mode)
        close(v['exponents_per_step'],exponents,'aggregate historical exponents '+mode);require(v['unfloored_exponents_per_step']==unfloored,'aggregate undefined exponents '+mode)
        estimate=(float(np.median(unfloored))if unfloored and all(x is not None for x in unfloored)else None)if mode=='purged_contiguous'else(float(np.median(exponents))if exponents else None)
        if estimate is None:require(v['estimate']is None and v['relative_discrepancy']is None,'undefined aggregate '+mode)
        else:
            close(v['estimate'],estimate,'aggregate median '+mode)
            if full:close(v['relative_discrepancy'],(estimate-full)/full,'aggregate comparison '+mode)
            else:require(v['relative_discrepancy']is None,'zero comparator disposition')
        require(v['rank_warning_phases']==[r['phase']for r in kept if (not r['original']['rank_full']if mode=='original'else not r[mode]['qualified_value_fit'])],'every aggregate rank warning '+mode)
        require(v['has_one_stationary_point']==any(f['producer_stationary_count']==1 for f in fits) and v['has_one_quadratic_maximum']==any(f['stationary_count']==1 and f['fitted_quadratic_maximum_count']==1 for f in fits),'aggregate fitted shape dispositions '+mode)


def leaves(left,right,path=''):
    if isinstance(left,dict)and isinstance(right,dict):return [r for k in sorted(set(left)|set(right))for r in leaves(left.get(k),right.get(k),path+'/'+k)]
    if isinstance(left,list)and isinstance(right,list):return [r for i in range(max(len(left),len(right)))for r in leaves(left[i]if i<len(left)else None,right[i]if i<len(right)else None,path+'/'+str(i))]
    numeric=isinstance(left,(float,int))and not isinstance(left,bool)and isinstance(right,(float,int))and not isinstance(right,bool)
    return [dict(path=path,own=left,producer=right,difference=left-right if numeric else None,equal=left==right)]


def producer_deltas(result):
    old=read(AUTHORITY/'trace-best/hidden-chaos.json')['cells'][0];b=read(AUTHORITY/'trace-best/bands-summary.json')
    comp=result['producer_comparison'];require(comp['original']==old and result['producer_bands_original']==b,'literal entire original summaries')
    close(comp['exponent_differences'],np.asarray(result['lyapunov'])-old['lyapunov'],'both producer exponent deltas')
    require(comp['lead_period_equal']==(result['lead_period']==old['lead_period'])and comp['state_period_equal']==(result['state_diagnostics']['period']==old['period'])and comp['late_state_equality_required']is False and comp['disagreement_is_retained']is True,'original comparison scope')
    rows=result.get('fixed_phase6_producer_comparison',[])
    if result['fixed_phase6']['disposition']=='fitted':
        f=result['fixed_phase6']['original'];own=dict(phase=6,degree=f['degree'],residual_rel=f['residual_relative'],exponent_per_repetition=f['exponent_per_repetition'],critical_points=[c['x']for c in f['critical_points']],curvature_at_critical_points=[c['curvature']for c in f['critical_points']])
        require(rows==leaves(own,{k:b['return_map'].get(k)for k in own}),'complete original phase6 leaf associations')
    else:require(not rows,'unfitted primary comparison absent')
    for r in rows:
        numeric=isinstance(r['own'],(int,float))and not isinstance(r['own'],bool)and isinstance(r['producer'],(int,float))and not isinstance(r['producer'],bool)
        require(r['equal']==(r['own']==r['producer']),'comparison leaf equality '+r['path'])
        if numeric:close(r['difference'],r['own']-r['producer'],'comparison delta '+r['path'])
        else:require(r['difference']is None,'nonnumeric delta absent')
    if result['lead_period']:
        v=result['bands']['minimum_centroid_distance_relative'];d=result['centroid_separation_difference']
        if v is None:require(d is None,'undefined centroid delta')
        else:close(d,v-b['min_centroid_separation_rel'],'original centroid comparison')
    REPORT['producer_comparisons']=dict(exponents=comp,phase6_all_leaves=rows,nonzero_numeric=[r for r in rows if r['difference']not in (None,0)],
        original_bands=b,centroid_delta=result.get('centroid_separation_difference'))


def review(card):
    receipt,plan=provenance(card);tick();result=read(ATTEMPT/'run/result.json');status=read(ATTEMPT/'run/status.json')
    require(status['mode']=='trajectory'and status['status']=='completed'and status['computational_completed']is True and status['source_commit']==card['source_commit']and status['certificate_passed']is False and status['experimental_claims']is False,'original worker completion contract')
    require(status['backend']['numpy']=='2.5.3'and status['backend']['torch']=='2.14.1+cu130'and status['backend']['cpu_threads']==2,'original numerical backend')
    for k,v in dict(transient_steps=1000,recorded_steps=2000,native_step_calls=3000,row_forwards=7000,seed=0,offsets=[1e-6,1e-9]).items():require(result[k]==v,'full fixed computation '+k)
    model=read(ATTEMPT/'run/model.json');source=ATTEMPT/'source/codex/crosscheck/qwen_map.py'
    require(model['prefix_ids']==list(literal(source,'PREFIX_IDS'))and model['model_asset_sha256']=={k:literal(source,'PINNED_SHA256')[k]for k in ('model.safetensors','config.json')},'declared original model/prefix pins')
    for k,v in dict(alpha=.0655,log10_temperature=.1899492494639028,layers=24,vocab=151936,dimension=896,steering_layer=12,full_vocabulary_readout=True,full_probability_vectors_retained=False).items():require(model[k]==v,'fixed map metadata '+k)
    close(model['temperature'],10.**.1899492494639028,'recorded physical temperature')
    require(model['vector_sha256']==literal(ATTEMPT/'source/codex/hostile-fold-crosscheck/assets.py','VECTOR_SHA256'),'captured hostile vector metadata pin')
    require(result['post_asset_hashes_equal']is True,'original runtime post-asset result')
    data=twins_and_readouts(result);tick();s,p,q=state_and_cycle(data,result);bands(s,q,result)
    fixed=read(ATTEMPT/'run/fixed-seven-phase-fits.json');fit_dispositions(fixed,p,7,'fixed_seven_fits')
    require(result['fixed_phase6']==fixed[6]and result['primary_phase6_clock_qualified']==(q==7 and fixed[6]['disposition']=='fitted'),'actual primary phase6 disposition')
    if q:
        actual=read(ATTEMPT/'run/all-phase-fits.json');fit_dispositions(actual,p,q,'actual_period_fits')
        if q==7:require(actual==fixed,'same actual seven-phase fits')
        aggregate_dispositions(actual,q,result);REPORT['all_phase_summary_retained']=result['all_phase_summary']
    producer_deltas(result);tick()
    REPORT.update(status='saved_review_completed',passed=True,original_worker_status=status,
        data_arithmetic_scope='All input/output bytes hashed. Array nonzero differences retained as counts and signed extrema; every twin growth row retained separately.',
        limitations=['No model evaluation, eigensolve, fit, bootstrap or exhaustive pair search repeated.',
        'Readout lead/probability/entropy correctness and embedding start inherit captured model implementation; full logits/probability vectors and model weights are not reloaded.',
        'Polynomial coefficients/roots/derivative values are retained and gate-checked, not refitted/re-solved.',
        'Original Git byte provenance and outer process closure inherit exact coordinator authority; all captured bytes/files are independently hashed.',
        'Finite-horizon agreement is exploratory; no asymptotic chaos, invariant bands or invertible conjugacy proved.'])


def main():
    global START,np
    parser=argparse.ArgumentParser();parser.add_argument('--card',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    card=read(a.card);require(card['admission']=='reviewed','prepared review card cannot run')
    require(str(a.output)==card['output'] and a.output==RAW/'trajectory-001-review-001','literal exclusive saved review output');a.output.mkdir(parents=False,exist_ok=False);START=time.monotonic()
    try:
        require(os.getpriority(os.PRIO_PROCESS,0)==10,'exact external nice10')
        hard=resource.getrlimit(resource.RLIMIT_AS)[1];require(hard==resource.RLIM_INFINITY or hard>=4*1024**3,'inherited AS bound available');resource.setrlimit(resource.RLIMIT_AS,(4*1024**3,4*1024**3))
        for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
        import numpy as np
        require(np.__version__=='2.5.3','reviewed NumPy2.5.3')
        available=int(next(x.split()[1]for x in Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024
        require(available>=20*1024**3,'startup headroom20GiB');review(card)
    except BaseException as e:REPORT.update(status='saved_review_failed',passed=False,error=repr(e),traceback=traceback.format_exc())
    REPORT.update(seconds=time.monotonic()-START,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,card_sha256=sha(a.card),script_sha256=sha(__file__))
    raw=(json.dumps(REPORT,indent=2,allow_nan=False)+'\n').encode()
    if len(raw)>16*1024**2:
        REPORT['passed']=False;raw=json.dumps(dict(status='saved_review_failed',passed=False,error='report exceeded16MiB',original_card_sha256=sha(a.card))).encode()
    (a.output/'report.json').write_bytes(raw)
    (a.output/'manifest.json').write_text(json.dumps(dict(files=[dict(path='report.json',bytes=len(raw),sha256=sha(a.output/'report.json'))]),indent=2)+'\n')
    return 0 if REPORT.get('passed')and len(raw)<=16*1024**2 else 1

if __name__=='__main__':raise SystemExit(main())
