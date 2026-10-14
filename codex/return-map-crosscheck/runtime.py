"""Task19 finite own-map trajectory or saved65-cell reduction, never a proof.

An external owned supervisor supplies hard timeout and sampled RSS/headroom.
No producer Python module is imported. Captured source/plan pins are mandatory.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
OWNED=Path('/data2/neural-shrimp-data/codex/llm-return-map-crosscheck')
AUTHORITY=OWNED/'input-authority-001'
AUTHORITY_SHA='d3014365b8445f8508271959137b922f694f0519665e9dfcb8f459e49517d2e8'
SOURCE_COMMIT='93dfb7617df34e2878da4ee0c6b9ded0b0e54362'
ALPHA=.0655; LOG_T=.1899492494639028
SOURCES=('codex/return-map-crosscheck/reduce.py','codex/return-map-crosscheck/trajectory.py',
         'codex/return-map-crosscheck/runtime.py','codex/hostile-fold-crosscheck/assets.py','codex/crosscheck/qwen_map.py')
CAPS={mode:dict(cpu_threads=cpu,nice=10,memory_gib=mem,timeout_seconds=seconds,
    minimum_start_available_gib=start,minimum_runtime_available_gib=16,maximum_output_mib=512)
    for mode,cpu,mem,seconds,start in [('trajectory',2,12,4200,36),('saved',1,4,300,20)]}
EXPECTED_FILES={'trace-best/trace.npz','trace-best/hidden-chaos.json','trace-best/bands-summary.json',
    'h3-cells/sequences.npz','h3-cells/hidden-chaos.json','h3-cells/return-maps.json',
    'docs/h3-repeating-text-cells.json','assignment-and-addendum.md','source-export-manifest.json','readiness.json'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def available():
    return int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024


def compare_values(own,producer,path=''):
    """Retain every leaf/delta, including absent, unequal and nonnumeric fields."""
    if isinstance(own,dict) and isinstance(producer,dict):
        return [r for k in sorted(set(own)|set(producer))
                for r in compare_values(own.get(k),producer.get(k),path+'/'+k)]
    if isinstance(own,list) and isinstance(producer,list):
        return [r for i in range(max(len(own),len(producer))) for r in compare_values(
            own[i] if i<len(own) else None,producer[i] if i<len(producer) else None,path+'/'+str(i))]
    numbers=isinstance(own,(float,int)) and not isinstance(own,bool) and isinstance(producer,(float,int)) and not isinstance(producer,bool)
    return [dict(path=path,own=own,producer=producer,difference=own-producer if numbers else None,equal=own==producer)]


def validate(plan):
    if set(plan)!={'schema','admission','mode','envelope','source_sha256','input_manifest_sha256','experiment','scope'}:
        raise ValueError('literal complete plan keys required')
    if plan['schema']!='codex-return-map-v1' or plan['admission']!='reviewed' or plan['mode'] not in CAPS:
        raise ValueError('reviewed fixed mode required')
    if plan['envelope']!=CAPS[plan['mode']] or any(type(v)is not int for v in plan['envelope'].values()):
        raise ValueError('fixed resource envelope required')
    if plan['experiment']!=dict(alpha=ALPHA,log10_temperature=LOG_T,transient=1000,record=2000,
        offsets=[1e-6,1e-9],seed=0,primary_phase=6,maximum_period=64,threshold=1e-8):
        raise ValueError('no retuned parameters/clock/offsets allowed')
    if plan['input_manifest_sha256']!=AUTHORITY_SHA or set(plan['source_sha256'])!=set(SOURCES):
        raise ValueError('exact input/source inventory required')
    for s in plan['source_sha256'].values():
        if not isinstance(s,str) or re.fullmatch('[0-9a-f]{64}',s)is None:raise ValueError('full source hash required')
    return plan


def authenticate():
    if sha(AUTHORITY/'manifest.json')!=AUTHORITY_SHA:raise ValueError('original authority changed')
    m=json.loads((AUTHORITY/'manifest.json').read_text())
    if m['schema']!='codex-return-map-input-snapshot-v1' or m['source_commit']!=SOURCE_COMMIT:
        raise ValueError('original source authority mismatch')
    if len(m['files'])!=10 or {r['path'] for r in m['files']}!=EXPECTED_FILES:
        raise ValueError('all original/snapshot members required exactly once')
    for r in m['files']:
        p=AUTHORITY/r['path']
        if p.is_symlink() or p.resolve()!=p or p.stat().st_size!=r['bytes'] or sha(p)!=r['sha256']:
            raise ValueError('input file identity mismatch: '+r['path'])
    return m


class Recording:
    """Preallocated full native state input/return retention and phase journal."""
    def __init__(self,out,check,transient=1000,record=2000,dimension=896):
        import numpy as np
        self.out=out;self.check=check;self.arrays={};self.journal=(out/'events.jsonl').open('x')
        for phase,n,batch in [('transient',transient,1),('record',record,3)]:
            for name,shape,dtype in [('input',(n,batch,dimension),np.float64),('state',(n,batch,dimension),np.float64),
                ('lead',(n,batch),np.int64),('pmax',(n,batch),np.float64),('entropy',(n,batch),np.float64)]:
                a=np.lib.format.open_memmap(out/f'{phase}-{name}.npy',mode='w+',dtype=dtype,shape=shape)
                a[:]=np.nan if dtype==np.float64 else -1;self.arrays[phase,name]=a
        for name,shape in [('log_growth',(record,2)),('raw_distance',(record,2)),('used_distance',(record,2)),
                            ('clamped',(record,2)),('renormalized_state',(record,3,dimension))]:
            a=np.lib.format.open_memmap(out/f'record-{name}.npy',mode='w+',dtype=np.float64,shape=shape)
            a[:]=np.nan;self.arrays['record',name]=a
        self.flush()

    def flush(self):
        for a in self.arrays.values():a.flush()
        self.journal.flush()

    def __call__(self,event,phase,index,value):
        import numpy as np
        if event=='initial_twins':
            for k in ('state','directions'):np.save(self.out/f'initial-twins-{k}.npy',value[k],allow_pickle=False)
        else:
            mapping={'state':'input'} if event=='input' else ({k:k for k in ('state','lead','pmax','entropy')} if event=='return' else {k:k for k in value})
            for src,dst in mapping.items():
                self.arrays[phase,dst][index]=value[src]
                self.arrays[phase,dst].flush()
        self.journal.write(json.dumps(dict(event=event,phase=phase,index=index,
            seconds=value.get('seconds')),allow_nan=False)+'\n');self.journal.flush()
        self.check()


def run_trajectory(out,check):
    import numpy as np
    import torch
    import safetensors
    R=module(HERE/'reduce.py','task19_reduce');T=module(HERE/'trajectory.py','task19_trajectory')
    A=module(ROOT/'codex/hostile-fold-crosscheck/assets.py','task19_assets')
    from codex.crosscheck.qwen_map import SNAPSHOT,PINNED_SHA256
    if sha(A.VECTOR)!=A.VECTOR_SHA256:raise ValueError('unchanged hostile vector required')
    with np.load(A.VECTOR,allow_pickle=False) as z:vector=z[A.VECTOR_KEY].copy()
    if vector.dtype!=np.float64 or vector.shape!=(896,) or not np.isfinite(vector).all():raise ValueError('full hostility vector required')
    began=time.monotonic();model=A.load_model(vector)
    torch.set_grad_enabled(False)
    if model.d!=896 or model.layers!=24 or model.vocab!=151936 or model.steering_layer!=12 or model.embedding.dtype!=torch.float64:
        raise ValueError('full original independent map architecture required')
    alpha=torch.tensor(ALPHA,dtype=torch.float64);temperature=10.**torch.tensor(LOG_T,dtype=torch.float64)
    cache=model.prefix_cache(alpha);initial=model.initial_state().numpy().copy()
    write(out/'model.json',dict(load_and_prefix_seconds=time.monotonic()-began,alpha=ALPHA,
        log10_temperature=LOG_T,temperature=float(temperature),prefix_ids=model.prefix_ids.tolist(),
        layers=model.layers,vocab=model.vocab,dimension=model.d,steering_layer=12,
        vector_sha256=A.VECTOR_SHA256,full_vocabulary_readout=True,full_probability_vectors_retained=False,
        model_asset_sha256={k:PINNED_SHA256[k] for k in ('model.safetensors','config.json')},
        retained_readout='leading token/probability/entropy for every native row; full state input/return'))
    np.save(out/'initial-state.npy',initial,allow_pickle=False);np.save(out/'vector.npy',vector,allow_pickle=False)
    record=Recording(out,check);timings=[]
    def step(x):
        check();begin=time.monotonic()
        logits=model.logits(torch.from_numpy(x),alpha,cache=cache)
        p=torch.softmax(logits/temperature,dim=-1,dtype=torch.float64)
        state=p@model.embedding;top=p.max(dim=-1)
        entropy=-(p*torch.log(torch.clamp(p,min=1e-300))).sum(-1)
        elapsed=time.monotonic()-begin;timings.append(elapsed)
        return dict(state=state.numpy().copy(),lead=top.indices.numpy().copy(),pmax=top.values.numpy().copy(),
                    entropy=entropy.numpy().copy(),seconds=elapsed)
    result=T.advance(step,initial,record)
    record.flush();record.journal.close();check()
    s=record.arrays['record','state'][:,0];lead=record.arrays['record','lead'][:,0];pmax=record.arrays['record','pmax'][:,0]
    q=R.symbolic_period(lead);diag=T.state_diagnostics(s)
    result.update(lead_period=q,lead_cycle=lead[:q].tolist() if q else None,state_diagnostics=diag,
        primary_seven_phase_clock_matches=(q==7),
        token_counts=[dict(token=int(t),count=int((lead==t).sum())) for t in np.unique(lead)],
        full_offsets_both_positive=all(v>1e-3 for v in result['lyapunov']),
        numerical_repeating_text_chaos=bool(q and not diag['period'] and all(v>1e-3 for v in result['lyapunov'])),
        seconds_native_calls=timings,readout_clock='p(input_t); state=F(input_t) on same recorded row')
    # Fixed disclosed phase6 remains an explicit primary, even if the token q changes.
    primary=R.phase_fits(pmax,7)
    write(out/'fixed-seven-phase-fits.json',primary)
    result['fixed_phase6']=primary[6]
    result['primary_phase6_clock_qualified']=(q==7 and primary[6]['disposition']=='fitted')
    if q:
        all_phases=primary if q==7 else R.phase_fits(pmax,q)
        write(out/'all-phase-fits.json',all_phases)
        result['all_phase_summary']=R.aggregate(all_phases,q,float(np.mean(result['lyapunov'])))
        result['bands']=R.sample_bands(s,q)
    result['probability_summary']=dict(mean=float(pmax.mean()),sd=float(pmax.std()),
        entropy_mean=float(record.arrays['record','entropy'][:,0].mean()),
        entropy_sd=float(record.arrays['record','entropy'][:,0].std()))
    # Producer results are read only after the own trajectory and fits have finished.
    producer=json.loads((AUTHORITY/'trace-best/hidden-chaos.json').read_text())
    original=producer['cells'][0]
    if len(producer['cells'])!=1 or (original['alpha'],original['log10_temperature'])!=(ALPHA,LOG_T):
        raise ValueError('literal comparison cell identity required')
    result['producer_comparison']=dict(original=original,
        exponent_differences=(np.asarray(result['lyapunov'])-np.asarray(original['lyapunov'])).tolist(),
        lead_period_equal=q==original['lead_period'],state_period_equal=diag['period']==original['period'],
        late_state_equality_required=False,disagreement_is_retained=True)
    old_bands=json.loads((AUTHORITY/'trace-best/bands-summary.json').read_text())
    if (old_bands['alpha'],old_bands['log10_temperature'])!=(ALPHA,LOG_T):raise ValueError('original band cell differs')
    if primary[6]['disposition']=='fitted':
        f=primary[6]['original'];oldfit=old_bands['return_map']
        ours=dict(phase=6,degree=f['degree'],residual_rel=f['residual_relative'],
            exponent_per_repetition=f['exponent_per_repetition'],
            critical_points=[c['x'] for c in f['critical_points']],
            curvature_at_critical_points=[c['curvature'] for c in f['critical_points']])
        result['fixed_phase6_producer_comparison']=compare_values(ours,{k:oldfit.get(k) for k in ours})
    result['producer_bands_original']=old_bands
    if q:
        distance=result['bands']['minimum_centroid_distance_relative']
        result['centroid_separation_difference']=distance-old_bands['min_centroid_separation_rel'] if distance is not None else None
    for name in ('model.safetensors','config.json'):
        if sha(SNAPSHOT/name)!=PINNED_SHA256[name]:raise ValueError('admitted model asset changed: '+name)
    if sha(A.VECTOR)!=A.VECTOR_SHA256:raise ValueError('admitted hostile vector changed')
    result['post_asset_hashes_equal']=True
    return result


def classify(info,q):
    positive=all(v>1e-3 for v in info['lyapunov'])
    return 'periodic' if info['period']>0 else ('hidden chaos' if positive and q else 'visible chaos' if positive else 'irregular')


def admit_sequences(cells,info,lead,pmax):
    import numpy as np
    if len(cells)!=65 or len(info)!=65 or lead.shape!=(2000,65) or pmax.shape!=(2000,65):raise ValueError('all65 original ordered cells and2000 samples required')
    if not np.issubdtype(lead.dtype,np.integer) or np.any((lead<0)|(lead>=151936)) or pmax.dtype!=np.float64 or not np.isfinite(pmax).all() or np.any((pmax<=0)|(pmax>1)):
        raise ValueError('finite original probabilities/integer tokens required')
    for c,i in zip(cells,info):
        if c!=[i['log10_temperature'],i['alpha']] or i['alpha']!=ALPHA or len(i['lyapunov'])!=2 or i['eps']!=[1e-6,1e-9] or not np.isfinite(i['lyapunov']).all() or type(i['period'])is not int or not 0<=i['period']<=64 or type(i['lead_period'])is not int or not 0<=i['lead_period']<=64:
            raise ValueError('literal original cell/order/twin definition mismatch')


def run_saved(out,check):
    import numpy as np
    R=module(HERE/'reduce.py','task19_reduce')
    document=json.loads((AUTHORITY/'docs/h3-repeating-text-cells.json').read_text())
    original=json.loads((AUTHORITY/'h3-cells/hidden-chaos.json').read_text())
    comparison=json.loads((AUTHORITY/'h3-cells/return-maps.json').read_text())
    with np.load(AUTHORITY/'h3-cells/sequences.npz',allow_pickle=False) as z:
        if set(z.files)!={'lead','pmax'}:raise ValueError('literal saved sequence fields required')
        lead,pmax=z['lead'].copy(),z['pmax'].copy()
    admit_sequences(document['cells'],original['cells'],lead,pmax)
    args=original['args']
    for k,v in dict(transient=1000,record=2000,eps=[1e-6,1e-9],model='qwen0.5b',concept='hostility-vs-kindness',layer=12).items():
        if args[k]!=v:raise ValueError('frozen original sequence recipe mismatch: '+k)
    rows=[]
    for index,info in enumerate(original['cells']):
        check();q=R.symbolic_period(lead[:,index]);kind=classify(info,q)
        if q!=info['lead_period']:raise ValueError('saved symbolic period does not reproduce')
        row=dict(index=index,cell=document['cells'][index],class_=kind,text_period=q,
            producer_lyapunov=info['lyapunov'],producer_state_period=info['period'],
            no_independent_fullstate_or_twin_replay_for_saved_cells=True)
        if kind=='hidden chaos':
            phases=R.phase_fits(pmax[:,index],q);write(out/f'cell-{index:02d}-all-phases.json',phases)
            row['summary']=R.aggregate(phases,q,float(np.mean(info['lyapunov'])))
            fitted=[p for p in phases if p['disposition']=='fitted']
            if fitted:
                best=min(fitted,key=lambda p:p['original']['residual_relative']);f=best['original'];s=row['summary']['original']
                row['producer_form_original']=dict(best_phase=best['phase'],degree=f['degree'],points=f['points'],
                    residual_rel=f['residual_relative'],exponent_per_repetition=f['producer_floor_exponent'],
                    critical_points_inside=f['stationary_count'],exponent_1d_per_step=s['estimate'],
                    exponent_full_per_step=s['full_twin_mean'],relative_discrepancy=s['relative_discrepancy'])
            selected=[p for p in phases if p['disposition']=='fitted' and p['amended_random']['producer_value_fit_qualified']]
            if selected:
                s=row['summary']['amended_random']
                row['producer_form_amended']=dict(single_valued_phases=[p['phase'] for p in selected],
                    degrees=[p['amended_random']['selected_degree'] for p in selected],
                    cv_residuals=[p['amended_random']['cv_residual_relative'] for p in selected],
                    exponents_per_step=s['exponents_per_step'],exponent_1d_per_step=s['estimate'],spread_per_step=s['spread'],
                    unimodal_phases=sum(p['amended_random']['fit']['producer_stationary_count']==1 for p in selected),
                    exponent_full_per_step=s['full_twin_mean'],relative_discrepancy=s['relative_discrepancy'])
        write(out/f'cell-{index:02d}.json',row);rows.append(row)
    if len(comparison['rows'])!=65:raise ValueError('producer complete65 comparison required')
    differences=[]
    for row,old in zip(rows,comparison['rows']):
        if row['cell']!=[old['log10_temperature'],old['alpha']]:raise ValueError('producer comparison row order mismatch')
        if 'summary' in row:
            for mode,key in [('original',None),('amended_random','amended')]:
                oldfit=old if key is None else old.get(key)
                own=row['summary'][mode]
                if oldfit is None:
                    differences.append(dict(index=row['index'],method=mode,disposition='producer_missing',own=own));continue
                form=row.get('producer_form_'+('original' if key is None else 'amended'))
                if form is None: differences.append(dict(index=row['index'],method=mode,disposition='own_unqualified',producer=oldfit))
                else: differences.extend(dict(index=row['index'],method=mode,**d) for d in compare_values(form,{k:oldfit.get(k) for k in form}))
    summaries={}
    for mode in ('original','amended_random','purged_contiguous'):
        eligible=[r['summary'][mode] for r in rows if 'summary' in r]
        included=[r for r in eligible if r['relative_discrepancy'] is not None and (mode!='original' or r['original_value_fit_qualified'])]
        summaries[mode]=dict(eligible=len(eligible),fitted=len(included),
            within_10_percent=sum(abs(r['relative_discrepancy'])<=.10 for r in included),
            median_absolute_discrepancy=float(np.median([abs(r['relative_discrepancy']) for r in included])) if included else None,
            stationary_point_cells=sum(r['has_one_stationary_point'] for r in included),
            quadratic_maximum_cells=sum(r['has_one_quadratic_maximum'] for r in included))
    write(out/'producer-comparison.json',dict(rows=differences,original_summary=comparison['summary'],every_delta_retained=True))
    return dict(rows=rows,summary=summaries,all65_admitted=True,
        classes={c:sum(r['class_']==c for r in rows) for c in sorted({r['class_'] for r in rows})},
        exploratory_same_sequences=True,conjugacy_proved=False,original_amendment_outcome_informed=True)


def main():
    began=time.monotonic();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    plan=validate(json.loads(a.plan.read_text()));mode=plan['mode'];caps=CAPS[mode]
    if a.output.resolve()!=a.output or a.output!=OWNED/(mode+'-001')/'run':raise ValueError('one exclusive literal attempt required')
    a.output.mkdir(exist_ok=False);stage='admission'
    try:
        if os.getpriority(os.PRIO_PROCESS,0)!=10:raise ValueError('exact external nice10 required')
        limit=caps['memory_gib']*1024**3;hard=resource.getrlimit(resource.RLIMIT_AS)[1]
        if hard!=resource.RLIM_INFINITY and hard<limit:raise ValueError('inherited AS below approved envelope')
        resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
        for k,v in dict(OMP_NUM_THREADS=str(caps['cpu_threads']),MKL_NUM_THREADS=str(caps['cpu_threads']),
            OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1').items():os.environ[k]=v
        if available()<caps['minimum_start_available_gib']*1024**3:raise RuntimeError('startup headroom')
        sys.path[:0]=[str(ROOT),str(HERE)]
        revision=os.environ.get('TASK19_SOURCE_COMMIT','')
        if re.fullmatch('[0-9a-f]{40}',revision)is None or sha(a.plan)!=os.environ.get('TASK19_PLAN_SHA256'):raise ValueError('external pinned source/plan required')
        for s,pin in plan['source_sha256'].items():
            if sha(ROOT/s)!=pin:raise ValueError('captured source differs: '+s)
        binding=authenticate();write(a.output/'input-binding.json',dict(source_commit=revision,plan=plan,input_manifest=binding))
        (a.output/'original-plan.json').write_bytes(a.plan.read_bytes())
        def check():
            if time.monotonic()-began>=caps['timeout_seconds']:raise TimeoutError('worker elapsed cap')
            if available()<caps['minimum_runtime_available_gib']*1024**3:raise RuntimeError('runtime headroom')
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>limit:raise MemoryError('RSS watchdog')
            if sum(f.stat().st_size for f in a.output.rglob('*') if f.is_file())>(caps['maximum_output_mib']-4)*1024**2:raise RuntimeError('raw cap with4MiB closing reserve')
        import numpy as np
        if np.__version__!='2.5.3':raise ValueError('reviewed NumPy backend changed')
        if mode=='trajectory':
            import torch
            if str(torch.__version__)!='2.14.1+cu130':raise ValueError('reviewed Torch backend changed')
            torch.set_num_threads(caps['cpu_threads']);torch.set_num_interop_threads(1)
        stage='own_'+mode;check();result=(run_trajectory if mode=='trajectory' else run_saved)(a.output,check)
        write(a.output/'result.json',result);check();authenticate()
        for s,pin in plan['source_sha256'].items():
            if sha(ROOT/s)!=pin:raise ValueError('source changed during execution')
        check();write(a.output/'status.json',dict(status='completed',computational_completed=True,
            experimental_claims=False,certificate_passed=False,mode=mode,source_commit=revision,
            seconds=time.monotonic()-began,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            backend=dict(python=sys.version,numpy=np.__version__,torch=str(torch.__version__) if mode=='trajectory' else None,
                         cpu_threads=caps['cpu_threads'],omp=os.environ['OMP_NUM_THREADS'],openblas=os.environ['OPENBLAS_NUM_THREADS'])))
    except BaseException as e:
        write(a.output/'status.json',dict(status='failed',computational_completed=False,stage=stage,
            error=repr(e),traceback=traceback.format_exc(),experimental_claims=False,certificate_passed=False,
            seconds=time.monotonic()-began,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));raise
    finally:
        write(a.output/'manifest.json',dict(files=[dict(path=str(f.relative_to(a.output)),bytes=f.stat().st_size,sha256=sha(f))
            for f in sorted(a.output.rglob('*')) if f.is_file()]))


if __name__=='__main__':main()
