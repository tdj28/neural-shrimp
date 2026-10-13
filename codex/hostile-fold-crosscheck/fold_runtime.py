"""Measured Task7 fold-only execution; no passages or interval proof.

Only the separately reviewed launcher may admit this worker, after Task6 releases
its allocation and the source/card/plan are privately pushed. The parent enforces
an absolute 1800s deadline and sampled headroom during long native operations.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import sys
import time
import traceback

STARTED = time.monotonic()
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OWNED = Path('/data2/neural-shrimp-data/codex/llm-hostile-fold-crosscheck')
PROFILE = OWNED/'profile-001'
PROFILE_SHA = 'e4d054140f960cd6e56d6958d061a69aa662f86f5f079f0a1ff68d1bed36dfa1'
PROFILE_SOURCE = '83b0e153694b20e7e059baf70bd34a0d3ce8c0a2'
HISTORICAL = Path('/data2/neural-shrimp-data/codex/task2-independent-crosscheck/throughput-001')
HISTORICAL_SHA = '02b8704da65722673d259f612266bab07e90a590a1951c517baa65d16927d860'
SOURCES = ('codex/hostile-fold-crosscheck/fold_runtime.py',
    'codex/hostile-fold-crosscheck/fold_worker.py',
    'codex/hostile-fold-crosscheck/assets.py',
    'codex/hostile-fold-crosscheck/resource_pilot.py',
    'codex/confirm-fold/fold_setup.py','codex/crosscheck/qwen_map.py',
    'codex/crosscheck/numerics.py')
CAPS = dict(cpu_threads=2,nice=10,memory_gib=12,timeout_seconds=1800,
    minimum_start_available_gib=28,minimum_runtime_available_gib=16)
NUMERICS = dict(chunk=128,max_newton=4,max_backtracks=2)
OPERATIONS = ('period-map','composed-jvp','batched-composed-jvp','parameter-ad',
              'nested-state-ad','critical-gradient-ad')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


def exact_ints(actual,expected,label):
    if type(actual)is not dict or set(actual)!=set(expected) or any(type(actual[k])is not int or actual[k]!=v for k,v in expected.items()):
        raise ValueError('exact '+label+' required')


def validate_plan(plan):
    keys={'schema','envelope','numerics','source_sha256','seed_sha256','vector_sha256',
          'profile_authority','historical_memory_authority','target','gates','basis','scope'}
    if type(plan)is not dict or set(plan)!=keys or plan['schema']!='codex-hostile-fold-run-v1':
        raise ValueError('exact executable Task7 fold-only schema required')
    exact_ints(plan['envelope'],CAPS,'CPU2/nice10/12GiB/1800s envelope')
    exact_ints(plan['numerics'],NUMERICS,'bounded chunk128/Newton4/backtracks2')
    if type(plan['source_sha256'])is not dict or set(plan['source_sha256'])!=set(SOURCES):
        raise ValueError('complete seven-file scientific source bundle required')
    for pin in [*plan['source_sha256'].values(),plan['seed_sha256'],plan['vector_sha256']]:
        if type(pin)is not str or re.fullmatch('[0-9a-f]{64}',pin)is None:raise ValueError('literal SHA256 required')
    if plan['profile_authority']!=dict(path=str(PROFILE),receipt_sha256=PROFILE_SHA,source_commit=PROFILE_SOURCE):
        raise ValueError('completed immutable hostile profile001 authority required')
    if plan['historical_memory_authority']!=dict(path=str(HISTORICAL),receipt_sha256=HISTORICAL_SHA,batch=128):
        raise ValueError('unchanged-architecture historical batch128 authority required')
    if plan['scope']!='fold_only_no_passages_rung2' or type(plan['basis'])is not dict:
        raise ValueError('numerical fold-only scope and disclosed resource basis required')
    return plan


def inventory(attempt,expected):
    """Authenticate immutable bytes, not evaluator correctness or Git ancestry."""
    if sha(attempt/'receipt.json')!=expected:raise ValueError('external receipt hash differs')
    receipt=json.loads((attempt/'receipt.json').read_text())
    if type(receipt.get('exit_code'))is not int or receipt['exit_code']!=0 or receipt.get('timed_out')is not False:
        raise ValueError('successful untimedout completed authority required')
    files={}
    for row in receipt['artifacts']:
        name=row['path'];relative=Path(name)
        if not isinstance(name,str) or relative.is_absolute() or '..'in relative.parts or relative.as_posix()!=name or name in files:
            raise ValueError('canonical unique inventory required')
        path=attempt/relative
        if path.is_symlink() or not path.is_file() or path.resolve()!=path or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:
            raise ValueError('immutable inventory differs: '+name)
        files[name]=row
    actual={str(p.relative_to(attempt)) for p in attempt.rglob('*') if p.is_file() and p!=attempt/'receipt.json'}
    if actual!=set(files):raise ValueError('complete exact inventory required')
    return receipt,files


def admit_resource_evidence(plan):
    receipt,files=inventory(PROFILE,PROFILE_SHA)
    if (receipt.get('source_commit')!=PROFILE_SOURCE or receipt.get('status')!='resource_pilot_completed'
        or receipt.get('worker_termination_verified')is not True or receipt.get('uncertain_ownership')!=[]
        or receipt.get('scientific_fold_result')is not False or receipt.get('resource_probe_only')is not True):
        raise ValueError('completed owned resource-only profile required')
    status=json.loads((PROFILE/'run/status.json').read_text())
    if (status.get('status')!='resource_profile_completed' or status.get('fold_qualified')is not False
        or status.get('certificate_passed')is not False or status.get('normal_form_coefficients_computed')is not False
        or status.get('resource_probe_only')is not True):raise ValueError('profile scope/status differs')
    rows=status['rows']
    if [row['operation'] for row in rows]!=list(OPERATIONS):raise ValueError('all six profile operations required')
    for row in rows:
        if (type(row['seconds']) not in (int,float) or not math.isfinite(row['seconds']) or row['seconds']<=0
            or type(row['max_rss_kib'])is not int or not 0<row['max_rss_kib']<12*1024**2):
            raise ValueError('finite measured cost below resource ceiling required')
    original=json.loads((PROFILE/'run/original-plan.json').read_text())
    for name,pin in original['source_sha256'].items():
        if plan['source_sha256'].get(name)!=pin or files['source/'+name]['sha256']!=pin:
            raise ValueError('profile graph/loader source differs: '+name)
    if (original['seed_sha256']!=plan['seed_sha256'] or original['vector_sha256']!=plan['vector_sha256']):
        raise ValueError('profile seed/vector differs')
    model=json.loads((PROFILE/'run/model-loaded.json').read_text())
    if (model['dimension'],model['layers'],model['vocabulary'],model['dtype'],model['steering_layer'])!=(896,24,151936,'torch.float64',12):
        raise ValueError('profile architecture differs')
    # Full arrays are small resource-probe outputs, not new model computations.
    import numpy as np
    for name in OPERATIONS:
        metadata=json.loads((PROFILE/'run'/f'{name}.json').read_text())
        expected_shape=(16,896) if name=='batched-composed-jvp' else (896,)
        desc=metadata['output'][0] if name=='critical-gradient-ad' else metadata['output']
        if desc['file']!=name+'.npz' or desc['shape']!=list(expected_shape) or desc['dtype']!='float64':
            raise ValueError('profile full output descriptor differs')
        if metadata['array_file_sha256']!=files['run/'+desc['file']]['sha256']:
            raise ValueError('profile output hashlink differs')
        with np.load(PROFILE/'run'/desc['file'],allow_pickle=False) as data:
            value=data[desc['key']]
            if value.shape!=expected_shape or value.dtype!=np.float64 or not np.isfinite(value).all():
                raise ValueError('profile output must be full finite float64')
        if name=='critical-gradient-ad' and not math.isfinite(metadata['output'][1]):
            raise ValueError('nonfinite mixed parameter derivative')
    historical,hfiles=inventory(HISTORICAL,HISTORICAL_SHA)
    mapname='codex/crosscheck/qwen_map.py'
    if hfiles['source/'+mapname]['sha256']!=plan['source_sha256'][mapname]:
        raise ValueError('historical architecture source differs')
    measurements=json.loads((HISTORICAL/'run/throughput.json').read_text())
    candidates=[row for row in measurements if row['batch']==128]
    if len(candidates)!=1 or candidates[0]['peak_rss_kib']!=6274700:
        raise ValueError('exact historical batch128 memory measurement required')
    return dict(profile_receipt_sha256=PROFILE_SHA,profile_source=PROFILE_SOURCE,
        profile_artifact_count=len(files),profile=status,profile_model=model,
        historical_receipt_sha256=HISTORICAL_SHA,historical_artifact_count=len(hfiles),
        historical_batch128=candidates[0],
        historical_caveat='Historical CPU4 timing is not a CPU2 speed prediction or a memory upper bound.',
        current_batch16_linear_dense_seconds=56*rows[2]['seconds'],
        captured_source_bytes_authenticated=True,git_commit_blob_verification_performed=False,
        certificate_passed=False)


def available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise RuntimeError('MemAvailable absent')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.resolve()!=args.output or args.output.name!='run' or args.output.parent.parent!=OWNED:
        raise ValueError('fresh owned attempt/run required')
    args.output.mkdir(exist_ok=False)
    stage='resource-admission'
    write(args.output/'status.json',dict(status='fold_running',stage=stage,passed=False,
        computational_completed=False,solver_history_complete=False,certificate_passed=False,
        resource_probe_only=False,passage_trajectories_evaluated=False))
    try:
        if os.getpriority(os.PRIO_PROCESS,0)!=10:raise ValueError('launcher must establish exact nice10')
        if available()<CAPS['minimum_start_available_gib']*1024**3:raise RuntimeError('start headroom below28GiB')
        hard=resource.getrlimit(resource.RLIMIT_AS)[1]
        if hard!=resource.RLIM_INFINITY and hard<12*1024**3:raise RuntimeError('inherited hard AS below12GiB')
        resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,)*2)
        expected_env=dict(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
            NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1')
        if any(os.environ.get(k)!=v for k,v in expected_env.items()):raise ValueError('launcher thread/device environment differs')
        original=args.plan.read_bytes()
        if hashlib.sha256(original).hexdigest()!=os.environ.get('TASK7_PLAN_SHA256'):raise ValueError('external original plan hash required')
        plan=validate_plan(json.loads(original))
        revision=os.environ.get('TASK7_SOURCE_COMMIT','')
        if re.fullmatch('[0-9a-f]{40}',revision)is None:raise ValueError('external captured source commit required')
        for name,pin in plan['source_sha256'].items():
            if sha(ROOT/name)!=pin:raise ValueError('captured scientific source differs: '+name)
        sys.path[:0]=[str(ROOT),str(HERE)]
        import assets
        import fold_worker
        import resource_pilot
        if plan['seed_sha256']!=assets.SEED_SHA256 or plan['vector_sha256']!=assets.VECTOR_SHA256:
            raise ValueError('wrong hostile assets')
        # Match declared numerical gates without constructing a model.
        expected_target=dict(period=2,alpha=assets.ALPHA,active_parameter='log10T',initial_parameter=assets.PARAMETER,bounds=[-.05,-.043],vector_key=assets.VECTOR_KEY,seed_index=0)
        expected_gates=dict(tolerances=fold_worker.TOLERANCES,
            parameter_fd_steps=list(fold_worker.PARAMETER_FD_STEPS),state_fd_steps=list(fold_worker.STATE_FD_STEPS))
        if any(json.dumps(actual,sort_keys=True)!=json.dumps(expected,sort_keys=True) for actual,expected in ((plan['target'],expected_target),(plan['gates'],expected_gates))):raise ValueError('scientific target/gates differ')
        (args.output/'original-plan.json').write_bytes(original)
        def check():
            if time.monotonic()-STARTED>=CAPS['timeout_seconds']:raise TimeoutError('absolute worker1800s ceiling')
            if available()<CAPS['minimum_runtime_available_gib']*1024**3:raise RuntimeError('yield to certificate headroom')
        check();stage='completed-profile-admission'
        admission=admit_resource_evidence(plan);write(args.output/'resource-admission.json',admission)
        import torch
        import numpy as np
        from threadpoolctl import threadpool_limits
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        backend=dict(torch=str(torch.__version__),numpy=np.__version__,python=sys.version,threads=torch.get_num_threads())
        if backend!=admission['profile']['backend']:raise ValueError('measured backend differs')
        setup=resource_pilot.setup_module()
        write(args.output/'inputs.json',dict(plan=plan,source_commit=revision,plan_sha256=hashlib.sha256(original).hexdigest(),
            resource_admission_sha256=sha(args.output/'resource-admission.json'),backend=backend,pid=os.getpid(),
            prefix_ids=setup.PREFIX_IDS,
            checkpoint_assets={name:setup.PINNED_SHA256[name] for name in ('model.safetensors','config.json')},
            steering_artifact=dict(path=str(assets.VECTOR),key=assets.VECTOR_KEY,sha256=assets.VECTOR_SHA256),
            seed_path=str(assets.SEED),seed_index=0,vector_path=str(assets.VECTOR),vector_key=assets.VECTOR_KEY,
            producer_seed_only=True,certificate_passed=False,passage_trajectories_evaluated=False))
        def emit(name,payload):
            setup.persist(args.output,name,payload)
            with (args.output/'events.jsonl').open('a') as f:
                f.write(json.dumps(dict(stage=name,elapsed_seconds=time.monotonic()-STARTED,
                    max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    operation_counts=payload.get('operation_counts',{})))+'\n')
        with threadpool_limits(limits=2):
            check();stage='load-seed-vector';seed,vector=assets.seed_and_vector();emit('seed-vector',dict(seed=seed,vector=vector))
            check();stage='load-model';model=assets.load_model(vector)
            emit('model-loaded',dict(dimension=model.d,layers=model.layers,vocabulary=model.vocab,dtype=str(model.embedding.dtype)))
            check();stage='fold-only';result=fold_worker.qualify_fold(model,seed,save=emit,check=check,**NUMERICS)
            setup.persist(args.output,'fold',result)
        write(args.output/'status.json',dict(status='fold_qualified' if result['passed'] else 'fold_gate_failed',
            stage=result['stage'],passed=result['passed'],computational_completed=True,
            solver_history_complete=result['solver_history_complete'],certificate_passed=False,resource_probe_only=False,
            passage_trajectories_evaluated=False,elapsed_seconds=time.monotonic()-STARTED,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        return 0 if result['passed'] else 2
    except BaseException as exc:
        write(args.output/'status.json',dict(status='fold_failed',stage=stage,error=repr(exc),traceback=traceback.format_exc(),
            passed=False,computational_completed=False,solver_history_complete=False,
            certificate_passed=False,resource_probe_only=False,passage_trajectories_evaluated=False,
            elapsed_seconds=time.monotonic()-STARTED,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        raise


if __name__=='__main__':sys.exit(main())
