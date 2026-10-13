"""Task14 period7 dense896 timing and derivative-cost probe; no fold/cusp claim.

Requires reviewed external launcher: hard wall deadline and owned cleanup,
RSS/headroom watchdog. Worker checks between calls are not hard native deadlines.
"""
import argparse
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

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OWNED=Path('/data2/neural-shrimp-data/codex/llm-cusp-crosscheck')
CAPS=dict(cpu_threads=2,nice=10,memory_gib=12,timeout_seconds=2400,
          minimum_start_available_gib=28,minimum_runtime_available_gib=16,
          maximum_output_mib=64)
SOURCES=('codex/cusp-crosscheck/inputs.py','codex/cusp-crosscheck/numerics.py',
         'codex/cusp-crosscheck/dense_profile.py','codex/hostile-fold-crosscheck/assets.py',
         'codex/crosscheck/qwen_map.py')


def available():
    return int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
    return result


def validate_plan(plan):
    wanted={'schema','envelope','source_sha256','candidate_authority_sha256','profile_index',
            'period','dimension','batch_directions','vector_sha256','scope','admission'}
    if set(plan)!=wanted or plan['schema']!='codex-cusp-profile-v1':raise ValueError('exact Task14 profile schema required')
    if plan['envelope']!=CAPS or any(type(v)is not int for v in plan['envelope'].values()):raise ValueError('exact bounded resource envelope required')
    if set(plan['source_sha256'])!=set(SOURCES) or any(re.fullmatch('[0-9a-f]{64}',v)is None for v in plan['source_sha256'].values()):raise ValueError('complete literal source pins required')
    if (plan['profile_index'],plan['period'],plan['dimension'],plan['batch_directions'])!=(13,7,896,16) or any(type(plan[k])is not int for k in ['profile_index','period','dimension','batch_directions']):raise ValueError('fixed first candidate/full896/period7 profile required')
    if plan['admission']!='reviewed':raise ValueError('prepared source does not authorize model execution')
    return plan


class Store:
    def __init__(self,path,check):self.path=path;self.check=check;self.index=0
    def emit(self,name,payload):
        """Save completed arrays, including nonfinite failures, before checking them."""
        import numpy as np
        number=self.index;self.index+=1
        def convert(value,key):
            if hasattr(value,'detach'):value=value.detach().cpu().numpy()
            if isinstance(value,np.ndarray):
                p=self.path/f'{number:04d}-{name}-{key}.npy'
                if p.exists():raise ValueError('exclusive artifact required')
                with p.open('xb') as stream:np.save(stream,value,allow_pickle=False)
                return dict(path=p.name,shape=list(value.shape),dtype=str(value.dtype))
            if isinstance(value,dict):return {k:convert(v,key+'-'+k) for k,v in value.items()}
            if isinstance(value,(tuple,list)):return [convert(v,key+'-'+str(i)) for i,v in enumerate(value)]
            if isinstance(value,np.generic):return value.item()
            return value
        record=convert(payload,'data')
        with (self.path/f'{number:04d}-{name}.json').open('x') as stream:json.dump(record,stream,indent=2,allow_nan=False)
        self.check()


def measured(emit,check,name,inputs,operation,rows):
    check();emit(name+'-input',inputs);began=time.monotonic()
    result=operation();seconds=time.monotonic()-began
    # A completed native return is persisted even if nonfinite or now out of time.
    emit(name+'-return',dict(output=result,seconds=seconds))
    import torch
    if isinstance(result,torch.Tensor) and not bool(torch.isfinite(result).all()):raise ValueError('nonfinite completed '+name)
    rows.append(dict(operation=name,seconds=seconds,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    check();return result


def profile(model,state,alpha,log_temperature,emit,check,*,dimension=896,batch=16,period=7):
    import torch
    import numerics
    if state.shape!=(dimension,) or dimension%batch or model.d!=dimension:raise ValueError('complete fixed coordinate basis required')
    rows=[]
    # Cache tensors are retained explicitly; do not try to pickle an owner object.
    check();emit('prefix-cache-input',dict(alpha=alpha));began=time.monotonic();cache=model.prefix_cache(alpha)
    emit('prefix-cache-return',dict(alpha=cache.alpha,layers=cache.layers,seconds=time.monotonic()-began))
    rows.append(dict(operation='prefix-cache',seconds=time.monotonic()-began,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    function=lambda x:model.period_map(x,alpha,log_temperature,period,cache=cache)
    common=dict(alpha=alpha,log10_temperature=log_temperature,period=period)
    value=measured(emit,check,'period-map',dict(state=state,**common),lambda:function(state),rows)
    eye=torch.eye(dimension,dtype=torch.float64);returned=[]
    dense_started=time.monotonic()
    for start in range(0,dimension,batch):
        directions=eye[start:start+batch];points=state[None,:].expand(batch,-1).clone()
        result=measured(emit,check,f'basis-{start:03d}',dict(state=points,directions=directions,start=start,**common),
            lambda:model.jvp(points,directions,alpha,log_temperature,period=period,cache=cache),rows)
        if result.shape!=(batch,dimension):raise ValueError('full batch derivative return required')
        returned.append(result)
    matrix=torch.cat(returned,dim=0).T.contiguous()
    emit('dense896',dict(matrix=matrix,seconds=time.monotonic()-dense_started,columns=dimension))
    direction=torch.arange(1,dimension+1,dtype=torch.float64);direction/=torch.linalg.vector_norm(direction)
    other=torch.flip(direction,dims=(0,));other/=torch.linalg.vector_norm(other)
    single=measured(emit,check,'single-direction',dict(state=state,direction=direction,**common),
        lambda:numerics.directional(function,state,(direction,)),rows)
    mismatch=float(torch.linalg.vector_norm(matrix@direction-single)/max(float(torch.linalg.vector_norm(single)),1e-30))
    emit('dense-direction-comparison',dict(relative_error=mismatch,tolerance=1e-10))
    if not math.isfinite(mismatch) or mismatch>1e-10:raise ValueError('dense/single composed derivative mismatch')
    # Probe vectors are deterministic, not computed critical modes. No coefficients.
    for name,directions in [('second-direction',(direction,direction)),('third-direction',(direction,direction,direction)),('mixed-second',(direction,other))]:
        measured(emit,check,name,dict(state=state,directions=directions,**common),
            lambda directions=directions:numerics.directional(function,state,directions),rows)
    scalar=torch.tensor(alpha,dtype=torch.float64)
    measured(emit,check,'physical-alpha',dict(state=state,**common),
        lambda:torch.func.jvp(lambda p:model.period_map(state,p,log_temperature,period),
                             (scalar,),(torch.ones_like(scalar),))[1],rows)
    return dict(status='resource_profile_completed',resource_probe_only=True,certificate_passed=False,
                fold_qualified=False,cusp_qualified=False,normal_form_coefficients_computed=False,
                computational_completed=True,dense_columns_completed=dimension,dense_shape=list(matrix.shape),period=period,
                operation_counts=dict(basis_batches=dimension//batch,basis_directions=dimension,
                    period_maps=1,single_jvp=1,second_ad=1,third_ad=1,mixed_second_ad=1,physical_alpha_ad=1),
                dense_jvp_relative=mismatch,rows=rows)


def main():
    began=time.monotonic();p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.resolve()!=args.output or args.output!=OWNED/'profile-001'/'run':raise ValueError('exclusive Task14 profile001/run required')
    args.output.mkdir(exist_ok=False)
    def write(name,value):
        with (args.output/(name+'.json')).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    stage='admission'
    try:
        if os.getpriority(os.PRIO_PROCESS,0)!=10:raise ValueError('external exact nice10 required')
        hard=resource.getrlimit(resource.RLIMIT_AS)[1]
        if hard!=resource.RLIM_INFINITY and hard<12*1024**3:raise ValueError('insufficient inherited hard AS')
        resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,)*2)
        for k,v in dict(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1').items():os.environ[k]=v
        if available()<28*1024**3:raise RuntimeError('profile startup headroom')
        sys.path[:0]=[str(ROOT),str(HERE)]
        import inputs
        original=args.plan.read_bytes();plan=validate_plan(json.loads(original))
        revision=os.environ.get('TASK14_SOURCE_COMMIT','')
        if re.fullmatch('[0-9a-f]{40}',revision)is None or inputs.sha(args.plan)!=os.environ.get('TASK14_PLAN_SHA256'):raise ValueError('external captured source/card pins required')
        if plan['candidate_authority_sha256']!=inputs.AUTHORITY_SHA:raise ValueError('candidate manifest differs')
        for name,pin in plan['source_sha256'].items():
            if inputs.sha(ROOT/name)!=pin:raise ValueError('captured source differs: '+name)
        binding=inputs.authenticate()
        assets=module(ROOT/'codex/hostile-fold-crosscheck/assets.py','task14_old_assets')
        if plan['vector_sha256']!=assets.VECTOR_SHA256 or assets.sha(assets.VECTOR)!=assets.VECTOR_SHA256:raise ValueError('exact hostility vector required')
        write('input-binding',dict(source_commit=revision,plan_sha256=inputs.sha(args.plan),plan=plan,**binding))
        (args.output/'original-plan.json').write_bytes(original)
        def check():
            if time.monotonic()-began>=CAPS['timeout_seconds']:raise TimeoutError('profile hard envelope between calls')
            if available()<16*1024**3:raise RuntimeError('profile yields to Task1 headroom')
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>12*1024**3:raise MemoryError('profile RSS watchdog')
            if sum(f.stat().st_size for f in args.output.rglob('*') if f.is_file())>60*1024**2:raise RuntimeError('profile output plus4MiB closing reserve')
        import numpy as np
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        # Store saves before its post-write guard, retaining completed failed returns.
        store=Store(args.output,check);stage='candidate-decode';check()
        states=np.load(inputs.AUTHORITY/'states.npy',allow_pickle=False,mmap_mode='r')
        if states.shape!=(40,896) or states.dtype!=np.float64:raise ValueError('exact40x896 float64 candidate container required')
        state=torch.tensor(states[13].copy(),dtype=torch.float64)
        if not bool(torch.isfinite(state).all()):raise ValueError('finite selected candidate required')
        with np.load(assets.VECTOR,allow_pickle=False) as vectors:vector=np.asarray(vectors[assets.VECTOR_KEY],dtype=np.float64).copy()
        if vector.shape!=(896,) or not np.isfinite(vector).all():raise ValueError('full finite steering vector required')
        selected=inputs.CANDIDATES[13]
        store.emit('candidate-vector',dict(index=13,state=state,vector=vector,**selected))
        stage='model-load';begin=time.monotonic();model=assets.load_model(vector)
        if model.prefix_ids.tolist()!=binding['ancestry']['prefix_ids'] or model.steering_layer!=12 or model.d!=896 or model.layers!=24 or model.vocab!=151936:
            raise ValueError('actual independent map recipe differs')
        store.emit('model-loaded',dict(seconds=time.monotonic()-begin,dimension=model.d,layers=model.layers,
            vocabulary=model.vocab,prefix_ids=model.prefix_ids,steering_layer=model.steering_layer,
            dtype=str(model.embedding.dtype),weight_bytes=sum(v.numel()*v.element_size() for v in model.weights.values())))
        stage='dense-and-ad-profile';result=profile(model,state,selected['alpha'],selected['log10_temperature'],store.emit,check)
        check();inputs.authenticate()
        if assets.sha(assets.VECTOR)!=assets.VECTOR_SHA256:raise ValueError('steering input changed')
        result.update(seconds=time.monotonic()-began,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            backend=dict(torch=str(torch.__version__),numpy=np.__version__,python=sys.version,threads=torch.get_num_threads()),
            source_commit=revision)
        write('status',result)
    except BaseException as exc:
        write('status',dict(status='resource_profile_failed',stage=stage,error=repr(exc),traceback=traceback.format_exc(),
            resource_probe_only=True,certificate_passed=False,fold_qualified=False,cusp_qualified=False,
            seconds=time.monotonic()-began,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        raise

if __name__=='__main__':main()
