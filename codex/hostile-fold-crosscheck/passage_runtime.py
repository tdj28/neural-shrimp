"""Bounded Task7 controlled-passage worker. Prepared source is not launch approval.

CPU2/nice10/12GiB/1800s, no GPU. The owning launcher must verify the private
checkpoint, Task1 campaign health, <=4 extra CPUs and 28/16GiB available-memory
headroom, and terminate only its owned process group. Runtime guards supplement
that external watchdog. A completed mismatch is a result, not a failed job.
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

STARTED=time.monotonic()
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OWNED=Path('/data2/neural-shrimp-data/codex/llm-hostile-fold-crosscheck')
SOURCES=('codex/hostile-fold-crosscheck/passage_runtime.py',
 'codex/hostile-fold-crosscheck/passage_worker.py','codex/hostile-fold-crosscheck/assets.py',
 'codex/hostile-fold-crosscheck/saved_audit.py','codex/confirm-fold/build_design.py',
 'codex/crosscheck/qwen_map.py')
CAPS=dict(cpu_threads=2,nice=10,memory_gib=12,timeout_seconds=1800,
 minimum_start_available_gib=28,minimum_runtime_available_gib=16)
AUTHORITY=dict(attempt=str(OWNED/'fold-001'),
 receipt_sha256='f562321e4170b396a5cd7b4e70a28df5843039a1dbfaa206dba2b96aa923a92a',
 source_commit='6271ca3bb70f859a916abf2e17ba3c2a9ae748e9',
 plan_sha256='53fcb608cd4dcaf8aa3437fffe13c5fe38e3873e6a5d31a8071ea8b91bfa4cd7',
 audit_path=str(OWNED/'fold-001-saved-audit-001/audit.json'),
 audit_sha256='01582d1fa67d26060c59beb339eaabecbf828f02c689435d9e78b2cf97dbccfa')
SCOPE='six_controlled_local_passages_no_natural_reinjection'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def same(a,b):return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)


def validate_plan(plan):
    if type(plan)is not dict or set(plan)!={'schema','envelope','source_sha256','fold_authority','design','basis','scope'}:
        raise ValueError('exact controlled-passage plan required')
    if plan['schema']!='codex-hostile-passage-v1' or plan['scope']!=SCOPE:
        raise ValueError('Task7 controlled-passage scope differs')
    caps=plan['envelope']
    if type(caps)is not dict or set(caps)!=set(CAPS) or any(type(caps[k])is not int or caps[k]!=v for k,v in CAPS.items()):
        raise ValueError('exact CPU2/nice10/12GiB/1800s envelope required')
    pins=plan['source_sha256']
    if type(pins)is not dict or set(pins)!=set(SOURCES) or any(type(p)is not str or re.fullmatch('[0-9a-f]{64}',p)is None for p in pins.values()):
        raise ValueError('complete six-file scientific source bundle required')
    if not same(plan['fold_authority'],AUTHORITY):raise ValueError('audited fold001 authority differs')
    if type(plan['basis'])is not dict or type(plan['design'])is not dict:raise ValueError('frozen design and resource basis required')
    return plan


def admit_fold(plan):
    """Original completed bytes and independent audit, without another eigensolve."""
    import saved_audit
    import passage_worker
    authority=plan['fold_authority'];attempt=Path(authority['attempt'])
    provenance,files=saved_audit.inventory(attempt,authority['receipt_sha256'],authority['source_commit'],authority['plan_sha256'])
    if sha(authority['audit_path'])!=authority['audit_sha256']:raise ValueError('independent saved audit changed')
    audit=json.loads(Path(authority['audit_path']).read_text())
    if (audit.get('numerical_audit_passed')is not True or audit.get('certificate_passed')is not False
        or audit.get('model_replayed')is not False or audit.get('saved_metadata_disagreements')!=[]
        or not audit.get('gates') or any(v is not True for v in audit['gates'].values())
        or not audit.get('independent_gates') or any(v is not True for v in audit['independent_gates'].values())):
        raise ValueError('every independent numerical fold gate must pass')
    for k in ('receipt_sha256','source_commit','plan_sha256'):
        if audit['provenance'][k]!=authority[k]:raise ValueError('audit authority differs')
    for key,name in (('fold_json_sha256','run/fold.json'),('fold_array_sha256','run/fold.npz')):
        if audit[key]!=files[name]['sha256']:raise ValueError('audit full-state hashlink differs')
    for name in ('codex/hostile-fold-crosscheck/assets.py','codex/crosscheck/qwen_map.py'):
        if plan['source_sha256'][name]!=provenance['source_sha256'][name]:raise ValueError('audited model/loader graph changed')
    if audit['audit_source_sha256']!=plan['source_sha256']['codex/hostile-fold-crosscheck/saved_audit.py']:
        raise ValueError('independent audit reader source changed')
    record=saved_audit.load_record(attempt,'fold',files)
    design=passage_worker.design(record)
    if not same(design,plan['design']):raise ValueError('frozen six-row prediction/modes differ')
    if record['x_c'].shape!=(896,):raise ValueError('full896-state original fold required')
    # The resource authority was already included in the authenticated fold
    # inventory; no new profile, producer model, or eigensolve is performed.
    measured=json.loads((attempt/'run/resource-admission.json').read_text())['profile']['rows'][0]
    if measured['operation']!='period-map' or not math.isfinite(measured['seconds']) or measured['seconds']<=0:
        raise ValueError('measured single-state F2 cost missing')
    expected_basis=dict(measured_single_G_seconds=measured['seconds'],
        fold_peak_rss_kib=provenance['worker_status']['max_rss_kib'],
        forecast_G=design['total_forecast_G'],cap_G=design['total_cap_G'],
        nominal_forecast_seconds=measured['seconds']*design['total_forecast_G'],
        nominal_cap_seconds=measured['seconds']*design['total_cap_G'],
        interpretation='Measured single-state cost extrapolation only; not a wall-time or memory upper bound. Hard1800s and headroom guards dominate.')
    if not same(plan['basis'],expected_basis):raise ValueError('frozen measured resource basis differs')
    return record,dict(authority=authority,original_artifact_count=len(files),
        captured_source_bytes_authenticated=True,git_commit_blob_verification_performed=False,
        independent_audit_passed=True,backend=provenance['inputs']['backend'],
        source_sha256=provenance['source_sha256'],checkpoint_assets=provenance['inputs']['checkpoint_assets'],
        prefix_ids=provenance['inputs']['prefix_ids'],steering_artifact=provenance['inputs']['steering_artifact'],
        basis=expected_basis,certificate_passed=False)


def available():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):return int(line.split()[1])*1024
    raise RuntimeError('MemAvailable absent')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.resolve()!=args.output or args.output.name!='run' or args.output.parent.parent!=OWNED:
        raise ValueError('exclusive owned attempt/run required')
    args.output.mkdir(exist_ok=False)
    stage='admission';rows=[];calls=0
    def status(label,**extra):
        write(args.output/'status.json',dict(status=label,stage=stage,computational_completed=False,
            certificate_passed=False,natural_reinjection_measured=False,resource_probe_only=False,
            passage_trajectories_evaluated=calls>0,model_G_calls=calls,rows=rows,
            elapsed_seconds=time.monotonic()-STARTED,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,**extra))
    status('passage_running')
    try:
        if os.getpriority(os.PRIO_PROCESS,0)!=10:raise ValueError('launcher must establish exact nice10')
        if available()<28*1024**3:raise RuntimeError('start headroom below28GiB')
        soft,hard=resource.getrlimit(resource.RLIMIT_AS)
        if hard!=resource.RLIM_INFINITY and hard<12*1024**3:raise RuntimeError('inherited hardAS below12GiB')
        resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,)*2)
        expected_env=dict(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',
            NUMEXPR_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1')
        if any(os.environ.get(k)!=v for k,v in expected_env.items()):raise ValueError('launcher CPU/thread environment differs')
        original=args.plan.read_bytes();plan_sha=hashlib.sha256(original).hexdigest()
        if plan_sha!=os.environ.get('TASK7_PLAN_SHA256'):raise ValueError('external plan hash required')
        plan=validate_plan(json.loads(original));revision=os.environ.get('TASK7_SOURCE_COMMIT','')
        if re.fullmatch('[0-9a-f]{40}',revision)is None:raise ValueError('captured source commit required')
        for name,pin in plan['source_sha256'].items():
            if sha(ROOT/name)!=pin:raise ValueError('captured source differs: '+name)
        (args.output/'original-plan.json').write_bytes(original)
        sys.path[:0]=[str(ROOT),str(HERE)]
        import assets
        import passage_worker
        record,admission=admit_fold(plan);write(args.output/'fold-admission.json',admission)
        rows=[dict(**r,status='not_run',primary_pass=False) for r in plan['design']['rows']];status('passage_running')
        def check():
            if time.monotonic()-STARTED>=1800:raise TimeoutError('absolute worker1800s ceiling')
            if available()<16*1024**3:raise RuntimeError('yield to certificate headroom')
        check()
        import numpy as np
        import torch
        from threadpoolctl import threadpool_limits
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        backend=dict(torch=str(torch.__version__),numpy=np.__version__,python=sys.version,threads=torch.get_num_threads())
        if backend!=admission['backend']:raise ValueError('audited effective-float64 backend differs')
        def save_arrays(path,**values):
            temp=path.with_suffix('.partial.npz');np.savez(temp,**values);temp.replace(path)
        arrays={k:record[k].copy() for k in ('x_c','r','l')}
        save_arrays(args.output/'frozen-modes.npz',**arrays)
        write(args.output/'inputs.json',dict(source_commit=revision,plan_sha256=plan_sha,backend=backend,
            fold_admission_sha256=sha(args.output/'fold-admission.json'),frozen_modes_sha256=sha(args.output/'frozen-modes.npz'),
            checkpoint_assets=admission['checkpoint_assets'],prefix_ids=admission['prefix_ids'],steering_artifact=admission['steering_artifact'],
            scope=SCOPE,certificate_passed=False,natural_reinjection_measured=False,pid=os.getpid()))
        with threadpool_limits(limits=2),torch.no_grad():
            check();stage='load-vector'
            if sha(assets.VECTOR)!=assets.VECTOR_SHA256:raise ValueError('hostility-vector hash changed')
            with np.load(assets.VECTOR,allow_pickle=False) as archive:vector=archive[assets.VECTOR_KEY].copy()
            save_arrays(args.output/'steering-vector.npz',vector=vector)
            check();stage='load-model';model=assets.load_model(vector)
            if (model.d,model.layers,model.vocab,str(model.embedding.dtype))!=(896,24,151936,'torch.float64'):
                raise ValueError('admitted architecture differs')
            cache=model.prefix_cache(.083)
            for i,row in enumerate(plan['design']['rows']):
                stage='row-'+row['id'];rows[i]['status']='running';status('passage_running')
                directory=args.output/row['id'];directory.mkdir(exist_ok=False);step=0
                def evaluate(state,p):
                    nonlocal calls,step
                    check();step+=1
                    save_arrays(directory/f'step-{step:04d}-input.npz',state=state,parameter=np.array(p))
                    started=time.monotonic();calls+=1
                    result=model.period_map(torch.tensor(state,dtype=torch.float64),.083,p,2,cache=cache)
                    value=result.detach().cpu().numpy().copy()
                    save_arrays(directory/f'step-{step:04d}-output.npz',state=value)
                    with (args.output/'events.jsonl').open('a') as f:
                        f.write(json.dumps(dict(row=row['id'],G=step,F=2*step,seconds=time.monotonic()-started,
                            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,available_bytes=available()))+'\n')
                    if result.dtype!=torch.float64 or result.device.type!='cpu':raise ValueError('float64 CPU output required')
                    return value
                def save_trace(data):save_arrays(directory/'trace.npz',**data)
                rows[i]=passage_worker.run_row(evaluate,row,arrays,save=save_trace,check=check)
                rows[i]['trace_sha256']=sha(directory/'trace.npz')
                write(directory/'result.json',rows[i]);status('passage_running');check()
        stage='completed'
        result=dict(status='passage_completed',stage=stage,computational_completed=True,
            all_rows_observed=all(r['status']=='observed' for r in rows),primary_all_pass=all(r['primary_pass'] for r in rows),
            certificate_passed=False,natural_reinjection_measured=False,resource_probe_only=False,
            passage_trajectories_evaluated=True,model_G_calls=calls,rows=rows,
            elapsed_seconds=time.monotonic()-STARTED,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        write(args.output/'status.json',result);return 0
    except BaseException as exc:
        for row in rows:
            if row['status']=='running':row.update(status='failed',primary_pass=False,error=repr(exc))
        status('passage_failed',error=repr(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':sys.exit(main())
