"""Task14 exact two-endpoint caller. Source preparation; no target plan yet.

An independently reviewed external owned launcher is mandatory. The actual
resource card is admitted only after complete profile and measured-cost review.
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

ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OWNED=Path('/data2/neural-shrimp-data/codex/llm-cusp-crosscheck')
SOURCES=('codex/cusp-crosscheck/inputs.py','codex/cusp-crosscheck/numerics.py',
    'codex/cusp-crosscheck/dense_profile.py','codex/cusp-crosscheck/endpoint_check.py',
    'codex/cusp-crosscheck/endpoint_runtime.py','codex/hostile-fold-crosscheck/assets.py','codex/crosscheck/qwen_map.py')


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module;spec.loader.exec_module(module);return module


def validate_plan(plan):
    required={'schema','admission','envelope','source_sha256','profile','indices','tolerances','fd_steps','parameter_steps','scope'}
    if set(plan)!=required or plan['schema']!='codex-cusp-endpoints-v1' or plan['admission']!='reviewed':raise ValueError('exact independently reviewed target plan required')
    if plan['indices']!=[13,14] or set(plan['source_sha256'])!=set(SOURCES):raise ValueError('two fixed endpoints and full source closure required')
    caps=plan['envelope']
    fixed=dict(cpu_threads=2,nice=10,memory_gib=12,minimum_start_available_gib=28,minimum_runtime_available_gib=16)
    if set(caps)!=set(fixed)|{'timeout_seconds','maximum_output_mib'} or any(type(v)is not int for v in caps.values()) or any(caps[k]!=v for k,v in fixed.items()):raise ValueError('fixed CPU2/12GiB/headroom envelope required')
    if not 1<=caps['timeout_seconds']<=7200 or not 8<=caps['maximum_output_mib']<=128:raise ValueError('source ceiling only: actual measured target card still required')
    if set(plan['profile'])!={'root','receipt_sha256','source_commit','plan_sha256'} or Path(plan['profile']['root'])!=OWNED/'profile-001':raise ValueError('only completed original profile001 may supply matrix')
    if any(re.fullmatch('[0-9a-f]{64}',v)is None for v in plan['source_sha256'].values()):raise ValueError('full source hashes required')
    if re.fullmatch('[0-9a-f]{64}',plan['profile']['receipt_sha256'])is None or re.fullmatch('[0-9a-f]{64}',plan['profile']['plan_sha256'])is None or re.fullmatch('[0-9a-f]{40}',plan['profile']['source_commit'])is None:raise ValueError('literal completed profile identity required')
    return caps


def authenticated_matrix(profile,state,alpha,temperature,source_pins,backend):
    """Hash all completed profile records, then bind every896 direction/return."""
    import numpy as np
    import inputs
    if state.shape!=(896,) or state.dtype!=np.float64 or not np.isfinite(state).all():raise ValueError('literal full float64 reused state required')
    base=Path(profile['root']);receipt=base/'receipt.json'
    if receipt.stat().st_mode&0o222 or inputs.sha(receipt)!=profile['receipt_sha256']:raise ValueError('immutable completed profile receipt required')
    rec=json.loads(receipt.read_text())
    if rec['source_commit']!=profile['source_commit'] or rec['plan_sha256']!=profile['plan_sha256'] or rec['exit_code']!=0 or rec['timed_out'] is not False or rec['worker_termination_verified'] is not True or rec['uncertain_ownership'] or rec['status']!='resource_pilot_completed':raise ValueError('profile not completed/reaped under exact source')
    inventory={}
    for row in rec['artifacts']:
        path=base/row['path']
        if row['path'] in inventory or path.resolve().parent!=base.resolve() and base.resolve() not in path.resolve().parents:raise ValueError('duplicate/outside profile artifact')
        if path.stat().st_mode&0o222 or path.stat().st_size!=row['bytes'] or inputs.sha(path)!=row['sha256']:raise ValueError('profile artifact mutated')
        inventory[row['path']]=row
    needed={'run/status.json','run/original-plan.json','run/input-binding.json'}
    if not needed.issubset(inventory):raise ValueError('complete inventoried profile status/domain records required')
    run=base/'run'
    status=json.loads((run/'status.json').read_text());old_plan=json.loads((run/'original-plan.json').read_text())
    if inputs.sha(run/'original-plan.json')!=profile['plan_sha256'] or old_plan['profile_index']!=13 or old_plan['period']!=7 or old_plan['dimension']!=896 or status['dense_columns_completed']!=896 or status['computational_completed'] is not True or status['status']!='resource_profile_completed':raise ValueError('profile domain/count differs')
    if status['backend']!=backend or status['resource_probe_only'] is not True or status['fold_qualified'] is not False or status['cusp_qualified'] is not False or status['certificate_passed'] is not False:raise ValueError('same backend and original resource-only profile required')
    for name,pin in old_plan['source_sha256'].items():
        if 'source/'+name not in inventory or source_pins[name]!=pin or inputs.sha(base/'source'/name)!=pin:raise ValueError('same evaluator/profile source required for matrix reuse')
    if set(old_plan['source_sha256'])!=set(SOURCES)-{'codex/cusp-crosscheck/endpoint_check.py','codex/cusp-crosscheck/endpoint_runtime.py'}:raise ValueError('original profile source closure differs')
    def one(pattern):
        matches=list(run.glob(pattern))
        if len(matches)!=1 or str(matches[0].relative_to(base)) not in inventory:raise ValueError('one inventoried native record required: '+pattern)
        return json.loads(matches[0].read_text())
    def arr(description):
        path=run/description['path']
        if path.parent!=run or str(path.relative_to(base)) not in inventory:raise ValueError('inventoried direct array required')
        value=np.load(path,allow_pickle=False)
        if list(value.shape)!=description['shape'] or str(value.dtype)!=description['dtype'] or value.dtype!=np.float64 or not np.isfinite(value).all():raise ValueError('exact finite float64 profile array required')
        return value
    candidate=one('*-candidate-vector.json')
    if candidate['index']!=13 or candidate['alpha']!=alpha or candidate['log10_temperature']!=temperature or not np.array_equal(arr(candidate['state']),state):raise ValueError('matrix original full state/rate identity differs')
    dense=one('*-dense896.json');matrix=arr(dense['matrix'])
    if matrix.shape!=(896,896) or dense['columns']!=896:raise ValueError('full dense profile matrix required')
    basis=np.eye(896,dtype=np.float64)
    for start in range(0,896,16):
        request=one(f'*-basis-{start:03d}-input.json');returned=one(f'*-basis-{start:03d}-return.json')
        if request['start']!=start or request['alpha']!=alpha or request['log10_temperature']!=temperature or request['period']!=7:raise ValueError('basis request state/parameter source differs')
        if not np.array_equal(arr(request['state']),np.broadcast_to(state,(16,896))) or not np.array_equal(arr(request['directions']),basis[start:start+16]):raise ValueError('full coordinate-basis identity differs')
        if not np.array_equal(matrix[:,start:start+16],arr(returned['output']).T):raise ValueError('assembled matrix differs from native columns')
    binding=json.loads((run/'input-binding.json').read_text())
    if binding['authority_sha256']!=inputs.AUTHORITY_SHA or binding['ancestry_manifest_sha256']!=inputs.ANCESTRY_SHA:raise ValueError('profile map/state authority differs')
    return matrix,dict(source_commit=profile['source_commit'],receipt_sha256=profile['receipt_sha256'],matrix_reused=True,
        all_896_column_inputs_returns_verified=True,original_matrix_shape=list(matrix.shape))


def main():
    began=time.monotonic();parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.resolve()!=args.output or args.output!=OWNED/'endpoints-001'/'run':raise ValueError('fresh owned endpoints001/run required')
    args.output.mkdir(exist_ok=False)
    def save(name,value):
        with (args.output/(name+'.json')).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    stage='admission';results=[]
    try:
        original=args.plan.read_bytes();plan=json.loads(original);caps=validate_plan(plan)
        if os.getpriority(os.PRIO_PROCESS,0)!=10:raise ValueError('external nice10 required')
        hard=resource.getrlimit(resource.RLIMIT_AS)[1]
        if hard!=resource.RLIM_INFINITY and hard<12*1024**3:raise ValueError('inherited hard AS below12GiB')
        resource.setrlimit(resource.RLIMIT_AS,(12*1024**3,)*2)
        for k,v in dict(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1').items():os.environ[k]=v
        sys.path[:0]=[str(ROOT),str(HERE)]
        import inputs
        revision=os.environ.get('TASK14_SOURCE_COMMIT','')
        if re.fullmatch('[0-9a-f]{40}',revision)is None or inputs.sha(args.plan)!=os.environ.get('TASK14_PLAN_SHA256'):raise ValueError('external source/card authority required')
        for name,pin in plan['source_sha256'].items():
            if inputs.sha(ROOT/name)!=pin:raise ValueError('captured scientific source mismatch')
        import dense_profile as D
        if D.available()<28*1024**3:raise RuntimeError('endpoint startup floor')
        binding=inputs.authenticate();assets=load(ROOT/'codex/hostile-fold-crosscheck/assets.py','task14_endpoint_assets')
        if inputs.sha(assets.VECTOR)!=assets.VECTOR_SHA256:raise ValueError('literal vector hash differs')
        save('input-binding',dict(source_commit=revision,plan_sha256=inputs.sha(args.plan),plan=plan,**binding))
        (args.output/'original-plan.json').write_bytes(original)
        def check():
            if time.monotonic()-began>=caps['timeout_seconds']:raise TimeoutError('endpoint worker wall envelope')
            if D.available()<16*1024**3:raise MemoryError('yield to Task1 headroom')
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>12*1024**3:raise MemoryError('endpoint RSS watchdog')
            if sum(f.stat().st_size for f in args.output.rglob('*') if f.is_file())>(caps['maximum_output_mib']-4)*1024**2:raise RuntimeError('endpoint raw cap/closing reserve')
        import numpy as np
        import torch
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        import endpoint_check as E
        if plan['tolerances']!=E.TOLERANCES or plan['fd_steps']!=list(E.FD_STEPS) or plan['parameter_steps']!=list(E.PARAMETER_STEPS):raise ValueError('no silent threshold or FD-step changes')
        states=np.load(inputs.AUTHORITY/'states.npy',allow_pickle=False,mmap_mode='r')
        if states.shape!=(40,896) or states.dtype!=np.float64:raise ValueError('literal candidate container shape')
        selected={i:states[i].copy() for i in [13,14]}
        if not all(np.isfinite(x).all() for x in selected.values()):raise ValueError('finite full selected states required')
        stage='profile-authentication';check()
        matrix,reuse=authenticated_matrix(plan['profile'],selected[13],**dict(alpha=inputs.CANDIDATES[13]['alpha'],temperature=inputs.CANDIDATES[13]['log10_temperature']),source_pins=plan['source_sha256'],backend=dict(torch=str(torch.__version__),numpy=np.__version__,python=sys.version,threads=torch.get_num_threads()))
        save('profile-reuse',reuse);check()
        with np.load(assets.VECTOR,allow_pickle=False) as vectors:vector=np.asarray(vectors[assets.VECTOR_KEY],dtype=np.float64).copy()
        stage='model-load';model=assets.load_model(vector)
        if model.prefix_ids.tolist()!=binding['ancestry']['prefix_ids'] or model.steering_layer!=12 or model.d!=896 or model.layers!=24 or model.vocab!=151936:raise ValueError('independent map recipe differs')
        save('backend',dict(torch=str(torch.__version__),numpy=np.__version__,python=sys.version,threads=torch.get_num_threads()))
        for i in [13,14]:
            stage='endpoint-'+str(i);check();directory=args.output/str(i);directory.mkdir()
            store=D.Store(directory,check);state=torch.tensor(selected[i],dtype=torch.float64);params=inputs.CANDIDATES[i]
            store.emit('original-input',dict(index=i,state=state,vector=vector,**params))
            ops=E.Operations(model,params['alpha'],params['log10_temperature'],7,store.emit,check)
            try:
                row=E.check_endpoint(ops,state,saved_matrix=matrix if i==13 else None)
            except E.GateFailure as exc:
                row=dict(status='numerical_endpoint_unqualified',numerical_qualified=False,stage=exc.stage,gates=exc.checks,
                    operation_counts=dict(ops.counts),certificate_passed=False,cusp_qualified=False)
            row['index']=i;store.emit('endpoint-result',row);results.append(row)
            save('endpoint-'+str(i),row)
        check();inputs.authenticate()
        if inputs.sha(assets.VECTOR)!=assets.VECTOR_SHA256:raise ValueError('input vector mutated')
        save('status',dict(status='endpoint_check_completed',computational_completed=True,
            all_endpoints_qualified=all(r['numerical_qualified'] for r in results),rows=results,
            certificate_passed=False,cusp_qualified=False,seconds=time.monotonic()-began,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_commit=revision))
    except BaseException as exc:
        save('status',dict(status='endpoint_check_failed',computational_completed=False,stage=stage,error=repr(exc),
            traceback=traceback.format_exc(),completed_rows=results,certificate_passed=False,cusp_qualified=False,
            seconds=time.monotonic()-began,max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        raise

if __name__=='__main__':main()
