"""Task7 resource-only profile. No fold result or passage claim is produced.

The separately reviewed outer launcher must enforce the absolute 300s deadline,
owned-process cleanup and sampled16GiB MemAvailable floor. This worker checks
headroom between operations; those checks are not a hard physical-memory cap.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import resource
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
OWNED = Path('/data2/neural-shrimp-data/codex/llm-hostile-fold-crosscheck')
SOURCES = ('codex/hostile-fold-crosscheck/assets.py', 'codex/hostile-fold-crosscheck/resource_pilot.py',
           'codex/confirm-fold/fold_setup.py', 'codex/crosscheck/qwen_map.py', 'codex/crosscheck/numerics.py')
ENVELOPE = dict(cpu_threads=2, nice=10, memory_gib=12, timeout_seconds=300,
                minimum_start_available_gib=28, minimum_runtime_available_gib=16)


def available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise RuntimeError('MemAvailable absent')


def validate_plan(plan):
    if type(plan)is not dict or set(plan)!={'schema','envelope','source_sha256','seed_sha256','vector_sha256','batch_directions','purpose'}:
        raise ValueError('exact resource-only pilot plan required')
    if plan['schema']!='codex-hostile-fold-profile-v1' or type(plan['envelope'])is not dict:
        raise ValueError('Task7 profile schema/envelope required')
    if set(plan['envelope'])!=set(ENVELOPE) or any(type(plan['envelope'][k])is not int or plan['envelope'][k]!=v for k,v in ENVELOPE.items()):
        raise ValueError('exact CPU2/nice10/12GiB/300s and28/16GiB headroom envelope required')
    if type(plan['batch_directions'])is not int or plan['batch_directions']!=16:
        raise ValueError('fixed16-direction representative profile required')
    if set(plan['source_sha256'])!=set(SOURCES):raise ValueError('complete independent profile source bundle required')
    for pin in list(plan['source_sha256'].values())+[plan['seed_sha256'],plan['vector_sha256']]:
        if type(pin)is not str or re.fullmatch('[0-9a-f]{64}',pin)is None:raise ValueError('literal source/asset SHA256 required')
    if type(plan['purpose'])is not str or not plan['purpose'].strip():raise ValueError('profile purpose required')
    return plan


def setup_module():
    # This is our already reviewed independent numerical code, not producer tools.
    name='task7_independent_fold_operations'
    spec=importlib.util.spec_from_file_location(name,ROOT/'codex/confirm-fold/fold_setup.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def profile(model, seed, emit, check, *, batch=16):
    """Exercise the expensive derivative graph once; retain every full output."""
    check()
    import numpy as np
    import torch
    import assets
    setup=setup_module()
    target=setup.Target('hostile',2,'log10T','alpha',assets.ALPHA,assets.PARAMETER,(-.05,-.043))
    ops=setup.Operations(model,target,chunk=batch,save=lambda n,p:emit(n,p),check=check)
    x=setup.tensor(seed);p=assets.PARAMETER
    # Deterministic nonzero probe directions are cost probes, not critical modes.
    r=np.arange(1,model.d+1,dtype=np.float64);r/=np.linalg.norm(r)
    l=r.copy()
    rows=[]
    def measured(name,fn):
        check();begin=time.monotonic();value=fn()
        row=dict(operation=name,seconds=time.monotonic()-begin,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 available_bytes=available())
        emit(name,dict(measurement=row,output=value));rows.append(row);check();return value
    measured('period-map',lambda:ops.period_map(seed,p))
    alpha,t,cache=ops.fixed(p)
    measured('composed-jvp',lambda:setup.array(ops.model.jvp(x,setup.tensor(r),alpha,t,period=2,cache=cache)))
    directions=torch.eye(model.d,dtype=torch.float64)[:batch]
    points=x[None,:].expand(len(directions),-1).clone()
    measured('batched-composed-jvp',lambda:setup.array(ops.model.jvp(points,directions,alpha,t,period=2,cache=cache)))
    measured('parameter-ad',lambda:ops.parameter_derivative(seed,p))
    measured('nested-state-ad',lambda:ops.second(seed,r,p))
    measured('critical-gradient-ad',lambda:ops.critical_grad(seed,p,l,r))
    return dict(status='resource_profile_completed',certificate_passed=False,fold_qualified=False,
        resource_probe_only=True,normal_form_coefficients_computed=False,rows=rows,operation_counts=dict(ops.counts),
        direction_scope='Deterministic cost probes; not an admitted fold eigenspace; no passages.')


def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.resolve()!=args.output or args.output.name!='run' or args.output.parent.parent!=OWNED:
        raise ValueError('fresh owned attempt/run required')
    args.output.mkdir(exist_ok=False)
    def write(name,value):
        path=args.output/(name+'.json');temp=path.with_suffix('.json.partial')
        temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)
    stage='resource-admission'
    try:
        if os.getpriority(os.PRIO_PROCESS,0)!=10:raise ValueError('launcher must establish exact nice10')
        if available()<28*1024**3:raise RuntimeError('insufficient start headroom for certificate and Task6')
        hard=resource.getrlimit(resource.RLIMIT_AS)[1]
        if hard!=resource.RLIM_INFINITY and hard<12*1024**3:raise RuntimeError('inherited hard AS below12GiB')
        resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,)*2)
        for name,value in {'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':'','HF_HUB_OFFLINE':'1','PYTHONDONTWRITEBYTECODE':'1'}.items():os.environ[name]=value
        sys.path[:0]=[str(ROOT),str(Path(__file__).resolve().parent)]
        import assets
        original=args.plan.read_bytes();plan=validate_plan(json.loads(original))
        if assets.sha(args.plan)!=os.environ.get('TASK7_PLAN_SHA256'):raise ValueError('external plan hash required')
        revision=os.environ.get('TASK7_SOURCE_COMMIT','')
        if re.fullmatch('[0-9a-f]{40}',revision)is None:raise ValueError('pinned captured source commit required')
        for name,pin in plan['source_sha256'].items():
            if assets.sha(ROOT/name)!=pin:raise ValueError('captured independent source differs: '+name)
        if plan['seed_sha256']!=assets.SEED_SHA256 or plan['vector_sha256']!=assets.VECTOR_SHA256:raise ValueError('wrong hostile assets')
        (args.output/'original-plan.json').write_bytes(original)
        write('inputs',dict(plan=plan,source_commit=revision,plan_sha256=assets.sha(args.plan),seed_path=str(assets.SEED),seed_index=0,
                           vector_path=str(assets.VECTOR),vector_key=assets.VECTOR_KEY,reference_parameter=assets.PARAMETER,alpha=assets.ALPHA,
                           certificate_passed=False,producer_seed_only=True,pid=os.getpid(),started_monotonic=started))
        import torch
        import numpy as np
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        def check():
            if time.monotonic()-started>=300:raise TimeoutError('profile wall ceiling')
            if available()<16*1024**3:raise RuntimeError('profile yields to certificate headroom')
        setup=setup_module()
        def emit(name,payload):
            setup.persist(args.output,name,payload)
            with (args.output/'events.jsonl').open('a') as f:f.write(json.dumps(dict(stage=name,elapsed_seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))+'\n')
        check();stage='load-seed-vector';seed,vector=assets.seed_and_vector();emit('seed-vector',dict(seed=seed,vector=vector))
        stage='load-model';begin=time.monotonic();model=assets.load_model(vector)
        emit('model-loaded',dict(seconds=time.monotonic()-begin,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             weight_bytes=sum(v.numel()*v.element_size()for v in model.weights.values()),dimension=model.d,layers=model.layers,vocabulary=model.vocab,
             dtype=str(model.embedding.dtype),prefix_ids=model.prefix_ids.tolist(),steering_layer=model.steering_layer))
        check();stage='profile-operations';result=profile(model,seed,emit,check)
        result.update(seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      backend=dict(torch=str(torch.__version__),numpy=np.__version__,python=sys.version,threads=torch.get_num_threads()))
        write('status',result)
    except BaseException as exc:
        write('status',dict(status='resource_profile_failed',stage=stage,error=repr(exc),traceback=traceback.format_exc(),
              certificate_passed=False,fold_qualified=False,seconds=time.monotonic()-started,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        raise

if __name__=='__main__':main()
