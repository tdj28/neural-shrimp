"""Two fixed optional Task19 stress cells; unchanged finite-map experiment.

This new adapter never mutates the mandatory runtime or its original attempts.
Expected producer periods are identity/comparison metadata, never forced clocks.
The same fixed-start native map, twin evolution and independent reducer are used.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[2]; HERE=Path(__file__).resolve().parent
OWNED=Path('/data2/neural-shrimp-data/codex/llm-return-map-crosscheck')
AUTHORITY_SHA='d3014365b8445f8508271959137b922f694f0519665e9dfcb8f459e49517d2e8'
ALPHA=.0655
CELLS={4:(.1799135096497498,10,'stress-cell04-001'),
       13:(.18478127233738384,28,'stress-cell13-001')}
BASE_PINS={
    'codex/return-map-crosscheck/reduce.py':'d87ec765cde3c0cc1145e6225b9b98b9a9c85468df85368c938c571aebda098b',
    'codex/return-map-crosscheck/trajectory.py':'8da808f5413d4f7caca49b6c0cba4c8182ff4322c91e123513e9ccb3e9737524',
    'codex/return-map-crosscheck/runtime.py':'41f0f19f023f2bbd5f6842929bb07cad70b86ea8d0399eec717a22ae478d468a',
    'codex/hostile-fold-crosscheck/assets.py':'06477ee36a78f15ee8f1e8ae196dc53ae230ff19368fbda638eb6a6e0ef758e0',
    'codex/crosscheck/qwen_map.py':'493a3d1d57fec227f5ba2e3f8f9cfcfdd02d22495d1d44e6449ede901310ae6e'}
SELF='codex/return-map-crosscheck/stress_runtime.py'
SOURCES=tuple(BASE_PINS)+(SELF,)
CAPS=dict(cpu_threads=2,nice=10,memory_gib=12,timeout_seconds=4200,
    minimum_start_available_gib=36,minimum_runtime_available_gib=16,maximum_output_mib=512)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def cell(index):
    if type(index)is not int or index not in CELLS:
        raise ValueError('only exact optional cell indices4 and13 are permitted')
    return CELLS[index]


def validate(plan):
    if set(plan)!={'schema','admission','mode','cell_index','envelope','source_sha256',
                  'input_manifest_sha256','experiment','scope'}:
        raise ValueError('literal complete stress plan keys required')
    if plan['schema']!='codex-return-map-stress-v1' or plan['admission']!='reviewed' or plan['mode']!='stress':
        raise ValueError('reviewed fixed stress mode required')
    log_t,expected_q,_=cell(plan['cell_index'])
    if plan['envelope']!=CAPS or any(type(v)is not int for v in plan['envelope'].values()):
        raise ValueError('unchanged trajectory resource envelope required')
    if plan['experiment']!=dict(alpha=ALPHA,log10_temperature=log_t,transient=1000,record=2000,
        offsets=[1e-6,1e-9],seed=0,expected_producer_lead_period=expected_q,
        maximum_period=64,threshold=1e-8):
        raise ValueError('exact optional parameters and comparison clock required')
    if plan['input_manifest_sha256']!=AUTHORITY_SHA or set(plan['source_sha256'])!=set(SOURCES):
        raise ValueError('exact original input and six-source inventory required')
    if any(plan['source_sha256'][p]!=h for p,h in BASE_PINS.items()):
        raise ValueError('mandatory scientific source bytes must remain unchanged')
    if re.fullmatch('[0-9a-f]{64}',str(plan['source_sha256'][SELF])) is None:
        raise ValueError('full adapter source hash required')
    return plan


def output_path(index):
    return OWNED/cell(index)[2]/'run'


def load_base():
    for path,pin in BASE_PINS.items():
        if sha(ROOT/path)!=pin:raise ValueError('original source mismatch: '+path)
    spec=importlib.util.spec_from_file_location('task19_stress_base',HERE/'runtime.py')
    base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
    return base


def bind_cell(document,producer,index):
    """Bind only the frozen indexed parameter/clock identity; no score selection."""
    log_t,expected_q,attempt=cell(index)
    if not isinstance(document.get('cells'),list) or not isinstance(producer.get('cells'),list) or len(document['cells'])!=65 or len(producer['cells'])!=65:
        raise ValueError('missing_or_incomplete_original65_cell_inventory')
    args=producer.get('args',{})
    for key,value in dict(transient=1000,record=2000,eps=[1e-6,1e-9],model='qwen0.5b',
                          concept='hostility-vs-kindness',layer=12).items():
        if args.get(key)!=value:raise ValueError('invalid_original_trajectory_recipe: '+key)
    original=producer['cells'][index]
    if not isinstance(original,dict) or document['cells'][index]!=[log_t,ALPHA] or original.get('alpha')!=ALPHA or original.get('log10_temperature')!=log_t or type(original.get('lead_period'))is not int or original['lead_period']!=expected_q:
        raise ValueError('invalid_frozen_optional_cell_identity')
    if type(original.get('period'))is not int or not 0<=original['period']<=64 or original.get('eps')!=[1e-6,1e-9] or not isinstance(original.get('lyapunov'),list) or len(original['lyapunov'])!=2 or any(type(v)not in (int,float) or not math.isfinite(v) for v in original['lyapunov']):
        raise ValueError('invalid_original_optional_twin_comparison')
    return dict(index=index,alpha=ALPHA,log10_temperature=log_t,
        expected_producer_lead_period=expected_q,attempt=attempt,
        selection='fixed disclosed stress allowlist before own optional outcomes',
        expected_period_is_comparison_only=True)


def fit_rows(base,reducer,probabilities,q,full,out):
    """All observed phases, with every unchanged candidate/fold disposition."""
    if not q:return [],None
    rows=reducer.phase_fits(probabilities,q)
    base.write(out/'all-phase-fits.json',rows)
    return rows,reducer.aggregate(rows,q,full)


def producer_forms(phases,summary):
    """Literal comparison projections from the already accepted saved65 runtime."""
    forms={}
    fitted=[p for p in phases if p['disposition']=='fitted']
    if fitted:
        best=min(fitted,key=lambda p:p['original']['residual_relative']);f=best['original'];s=summary['original']
        forms['original']=dict(best_phase=best['phase'],degree=f['degree'],points=f['points'],
            residual_rel=f['residual_relative'],exponent_per_repetition=f['producer_floor_exponent'],
            critical_points_inside=f['stationary_count'],exponent_1d_per_step=s['estimate'],
            exponent_full_per_step=s['full_twin_mean'],relative_discrepancy=s['relative_discrepancy'])
    selected=[p for p in phases if p['disposition']=='fitted' and p['amended_random']['producer_value_fit_qualified']]
    if selected:
        s=summary['amended_random']
        forms['amended']=dict(single_valued_phases=[p['phase'] for p in selected],
            degrees=[p['amended_random']['selected_degree'] for p in selected],
            cv_residuals=[p['amended_random']['cv_residual_relative'] for p in selected],
            exponents_per_step=s['exponents_per_step'],exponent_1d_per_step=s['estimate'],spread_per_step=s['spread'],
            unimodal_phases=sum(p['amended_random']['fit']['producer_stationary_count']==1 for p in selected),
            exponent_full_per_step=s['full_twin_mean'],relative_discrepancy=s['relative_discrepancy'])
    return forms


def compare_row(base,comparison,index,forms):
    if not isinstance(comparison.get('rows'),list) or len(comparison['rows'])!=65:
        raise ValueError('missing_or_incomplete_original65_fit_rows')
    old=comparison['rows'][index];log_t,_,_=cell(index)
    if not isinstance(old,dict) or (old.get('log10_temperature'),old.get('alpha'))!=(log_t,ALPHA):
        raise ValueError('invalid_original_optional_fit_row_identity')
    differences=[]
    for method,producer in [('original',old),('amended',old.get('amended'))]:
        form=forms.get(method)
        if producer is None:
            differences.append(dict(method=method,disposition='producer_missing',own=form))
        elif form is None:
            differences.append(dict(method=method,disposition='own_unqualified',producer=producer))
        elif not isinstance(producer,dict):
            raise ValueError('invalid_original_optional_fit_schema')
        else:
            differences.extend(dict(method=method,**d) for d in base.compare_values(form,{k:producer.get(k) for k in form}))
    return dict(original_row=old,rows=differences,every_delta_retained=True,
        finite_horizons_are_different=True,late_state_equality_required=False,
        original_amendment_outcome_informed=True)

def run_trajectory(base,index,out,check):
    LOG_T,expected_q,_=cell(index)
    module=base.module;write=base.write;Recording=base.Recording
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
        optional_cell_index=index,expected_producer_lead_period=expected_q,
        expected_producer_clock_matches=(q==expected_q),
        token_counts=[dict(token=int(t),count=int((lead==t).sum())) for t in np.unique(lead)],
        full_offsets_both_positive=all(v>1e-3 for v in result['lyapunov']),
        numerical_repeating_text_chaos=bool(q and not diag['period'] and all(v>1e-3 for v in result['lyapunov'])),
        seconds_native_calls=timings,readout_clock='p(input_t); state=F(input_t) on same recorded row')
    # Optional comparisons use only the actual observed symbolic clock.
    phases,summary=fit_rows(base,R,pmax,q,float(np.mean(result['lyapunov'])),out)
    result['all_phase_disposition']='observed_clock_fits_retained' if q else 'no_observed_symbolic_period'
    if q:
        result['all_phase_summary']=summary
        result['bands']=R.sample_bands(s,q)
    result['probability_summary']=dict(mean=float(pmax.mean()),sd=float(pmax.std()),
        entropy_mean=float(record.arrays['record','entropy'][:,0].mean()),
        entropy_sd=float(record.arrays['record','entropy'][:,0].std()))
    # Comparison values are not used until the own trajectory and all fits finish.
    producer=json.loads((base.AUTHORITY/'h3-cells/hidden-chaos.json').read_text())
    document=json.loads((base.AUTHORITY/'docs/h3-repeating-text-cells.json').read_text())
    result['optional_identity']=bind_cell(document,producer,index);original=producer['cells'][index]
    result['producer_comparison']=dict(original=original,
        exponent_differences=(np.asarray(result['lyapunov'])-np.asarray(original['lyapunov'])).tolist(),
        lead_period_equal=q==original['lead_period'],state_period_equal=diag['period']==original['period'],
        late_state_equality_required=False,disagreement_is_retained=True)
    comparison=json.loads((base.AUTHORITY/'h3-cells/return-maps.json').read_text())
    result['producer_fit_comparison']=compare_row(base,comparison,index,producer_forms(phases,summary))
    for name in ('model.safetensors','config.json'):
        if sha(SNAPSHOT/name)!=PINNED_SHA256[name]:raise ValueError('admitted model asset changed: '+name)
    if sha(A.VECTOR)!=A.VECTOR_SHA256:raise ValueError('admitted hostile vector changed')
    result['post_asset_hashes_equal']=True
    return result


def main():
    began=time.monotonic();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    plan=validate(json.loads(a.plan.read_text()));index=plan['cell_index'];caps=CAPS
    if a.output.resolve()!=a.output or a.output!=output_path(index):raise ValueError('one exclusive literal optional attempt required')
    base=load_base();a.output.mkdir(exist_ok=False);stage='admission'
    try:
        if os.getpriority(os.PRIO_PROCESS,0)!=10:raise ValueError('exact external nice10 required')
        limit=caps['memory_gib']*1024**3;hard=resource.getrlimit(resource.RLIMIT_AS)[1]
        if hard!=resource.RLIM_INFINITY and hard<limit:raise ValueError('inherited AS below approved envelope')
        resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
        for k,v in dict(OMP_NUM_THREADS=str(caps['cpu_threads']),MKL_NUM_THREADS=str(caps['cpu_threads']),
            OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1').items():os.environ[k]=v
        if base.available()<caps['minimum_start_available_gib']*1024**3:raise RuntimeError('startup headroom')
        sys.path[:0]=[str(ROOT),str(HERE)]
        revision=os.environ.get('TASK19_SOURCE_COMMIT','')
        if re.fullmatch('[0-9a-f]{40}',revision)is None or sha(a.plan)!=os.environ.get('TASK19_PLAN_SHA256'):raise ValueError('external pinned source/plan required')
        for path,pin in plan['source_sha256'].items():
            if sha(ROOT/path)!=pin:raise ValueError('captured source differs: '+path)
        binding=base.authenticate()
        identity=bind_cell(json.loads((base.AUTHORITY/'docs/h3-repeating-text-cells.json').read_text()),
            json.loads((base.AUTHORITY/'h3-cells/hidden-chaos.json').read_text()),index)
        base.write(a.output/'input-binding.json',dict(source_commit=revision,plan=plan,input_manifest=binding,optional_identity=identity))
        (a.output/'original-plan.json').write_bytes(a.plan.read_bytes())
        def check():
            if time.monotonic()-began>=caps['timeout_seconds']:raise TimeoutError('worker elapsed cap')
            if base.available()<caps['minimum_runtime_available_gib']*1024**3:raise RuntimeError('runtime headroom')
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>limit:raise MemoryError('RSS watchdog')
            if sum(f.stat().st_size for f in a.output.rglob('*') if f.is_file())>(caps['maximum_output_mib']-4)*1024**2:raise RuntimeError('raw cap with4MiB closing reserve')
        import numpy as np
        import torch
        if np.__version__!='2.5.3' or str(torch.__version__)!='2.14.1+cu130':raise ValueError('reviewed backend changed')
        torch.set_num_threads(caps['cpu_threads']);torch.set_num_interop_threads(1)
        stage='own_optional_trajectory';check();result=run_trajectory(base,index,a.output,check)
        base.write(a.output/'result.json',result);check();base.authenticate()
        for path,pin in plan['source_sha256'].items():
            if sha(ROOT/path)!=pin:raise ValueError('source changed during execution')
        check();base.write(a.output/'status.json',dict(status='completed',computational_completed=True,
            experimental_claims=False,certificate_passed=False,mode='stress',cell_index=index,source_commit=revision,
            seconds=time.monotonic()-began,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            backend=dict(python=sys.version,numpy=np.__version__,torch=str(torch.__version__),
                         cpu_threads=caps['cpu_threads'],omp=os.environ['OMP_NUM_THREADS'],openblas=os.environ['OPENBLAS_NUM_THREADS'])))
    except BaseException as e:
        base.write(a.output/'status.json',dict(status='failed',computational_completed=False,stage=stage,
            error=repr(e),traceback=traceback.format_exc(),experimental_claims=False,certificate_passed=False,
            cell_index=index,seconds=time.monotonic()-began,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));raise
    finally:
        base.write(a.output/'manifest.json',dict(files=[dict(path=str(f.relative_to(a.output)),bytes=f.stat().st_size,sha256=sha(f))
            for f in sorted(a.output.rglob('*')) if f.is_file()]))


if __name__=='__main__':main()
