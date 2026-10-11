#!/usr/bin/env python3
"""EXP-012: bounded engineering parity and matched end-to-end CUDA timings."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import importlib.metadata
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import resource
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
import run_hopfield_refinement as runtime  # thread caps before NumPy
import numpy as np
import run_preview_refinement as preview
from hopfield_drift_probe import integrate_trace
import hopfield_cuda as cuda

PLAN = ROOT / 'experiments/hopfield/EXP-012-plan.json'
LABEL_ROOT = Path('/data2/neural-shrimp-data/raw/EXP-010/attempt-001')
LABEL_RECEIPT_SHA = 'f07aa1cfbb1c4dd5fcaed57d5262ff67f4141fff55adad0a4e690b835d4b651f'
SOURCE_PATHS = list(dict.fromkeys(['tools/run_gpu_preview_qualification.py',
    'tools/hopfield_cuda.py', 'native/hopfield_cuda.cu', 'tests/test_hopfield_cuda.py',
    'tests/test_gpu_preview_qualification.py', 'tests/test_gpu_preview_lifecycle.py',
    'docs/EXP-012-protocol.md', 'docs/server-playbook-provenance-2026-09-15.json', *preview.SOURCE_PATHS]))
CONTRACT = dict(schema='EXP-012-v1', duration=75., sample_dt=.1, dts=[.0025, .00125],
    batch_sizes=[18, 144, 576, 3456], warm_repetitions=3, workers=4,
    chunk_size=257, max_device_bytes=1024**3, ram_reserve_bytes=2*1024**3,
    disk_reserve_bytes=2*1024**3, batch_ceiling_seconds=180, wall_ceiling_seconds=3600,
    state_atol=1e-10, state_rtol=1e-10, activation_atol=1e-10,
    score_atol=1e-10, phase_atol=1e-8, deployment_median_speedup=1.25,
    required_packages={'numpy': '2.5.3', 'scipy': '1.18.1'},
    deployment_size=3456, deployment_dt=.00125, paid_compute=False)
CONTROLS = dict(axes=[0,4,8,12], diagonal=[2,6,10,14], local_coverage=[0,8,16,17])


def read(path):
    return json.loads(Path(path).read_text())


def binding(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=runtime.sha(path), bytes=path.stat().st_size)


def jobs():
    result = [dict(id=f'cold-{arm}', arm=arm, dt=.0025, size=18, kind='cold', rep=None,
                   layout='canonical') for arm in ('cpu', 'gpu')]
    for dt in CONTRACT['dts']:
        for size in CONTRACT['batch_sizes']:
            for rep in range(3):
                for arm in (('cpu','gpu') if rep % 2 == 0 else ('gpu','cpu')):
                    result.append(dict(id=f'{arm}-dt{dt:g}-n{size}-r{rep}', arm=arm, dt=dt,
                        size=size, rep=rep, kind='warm', layout='canonical'))
        for layout in ('reverse', 'chunked'):
            result.append(dict(id=f'gpu-dt{dt:g}-{layout}', arm='gpu', dt=dt, size=3456,
                               rep=None, kind='layout', layout=layout))
    return result


def build_plan():
    old_plan = read(preview.PLAN)
    preview.validate(old_plan, check_bindings=True, reconstruct=True)
    if runtime.sha(LABEL_ROOT/'receipt.json') != LABEL_RECEIPT_SHA:
        raise ValueError('EXP-010 receipt changed')
    receipt = read(LABEL_ROOT/'receipt.json')
    inventory = {r['path']: r for r in receipt['artifacts']}
    # Bind the immutable inventory once; verify every consumed member at run time.
    for name in ('manifest.json','summary.json','terminal-records.json'):
        item = binding(LABEL_ROOT/name)
        if (item['sha256'], item['bytes']) != (inventory[name]['sha256'], inventory[name]['bytes']):
            raise ValueError(f'Qualified evidence changed: {name}')
    subsets={str(n):[c['id'] for c in old_plan['census'][:n]] for n in CONTRACT['batch_sizes']}
    return dict(experiment_id='EXP-012', date='2026-09-15', status='Exploratory engineering qualification on known data',
        **CONTRACT, controls=CONTROLS, jobs=jobs(), census=old_plan['census'],
        subsets=subsets, subset_hashes={n:hashlib.sha256(runtime.canonical(ids)).hexdigest() for n,ids in subsets.items()},
        old_plan=runtime.binding(preview.PLAN), label_receipt=binding(LABEL_ROOT/'receipt.json'),
        source_files=[runtime.binding(ROOT/name) for name in SOURCE_PATHS],
        scope='Only fixed-step 75-unit previews on the full known EXP-010 census; no fresh utility, '
              '300-unit, sensitivity, adaptive-solver or geometry claim.')


def validate(plan, *, check_bindings=False):
    if plan['experiment_id'] != 'EXP-012' or plan['jobs'] != jobs() or plan['controls'] != CONTROLS:
        raise ValueError('Wrong experiment/schedule')
    for name, value in CONTRACT.items():
        if plan[name] != value:
            raise ValueError(f'Changed contract: {name}')
    if plan['census'] != read(preview.PLAN)['census']:
        raise ValueError('Census changed')
    expected={str(n):[c['id'] for c in plan['census'][:n]] for n in CONTRACT['batch_sizes']}
    if (plan['subsets']!=expected or plan['subset_hashes']!=
        {n:hashlib.sha256(runtime.canonical(ids)).hexdigest() for n,ids in expected.items()}):
        raise ValueError('Subset identity changed')
    if [i['path'] for i in plan['source_files']] != SOURCE_PATHS:
        raise ValueError('Incomplete source dependency binding')
    if check_bindings:
        for item in [*plan['source_files'], plan['old_plan']]:
            if runtime.binding(ROOT/item['path']) != item:
                raise ValueError(f'Source changed: {item["path"]}')
        if binding(plan['label_receipt']['path']) != plan['label_receipt']:
            raise ValueError('Qualified input receipt changed')
        preview.validate(read(preview.PLAN), check_bindings=True)
        verify_old_members(plan)


def verify_old_members(plan):
    inventory = {r['path']: r for r in read(LABEL_ROOT/'receipt.json')['artifacts']}
    for case in plan['census']:
        for dt in CONTRACT['dts']:
            stem = f'{case["id"]}--rk4-{dt:g}'
            for name in (f'records/{stem}.json', f'traces/{stem}.npz'):
                item = binding(LABEL_ROOT/name)
                if (item['sha256'], item['bytes']) != (inventory[name]['sha256'], inventory[name]['bytes']):
                    raise ValueError(f'Qualified member changed: {name}')


def resources():
    fields = dict(line.split(':',1) for line in Path('/proc/meminfo').read_text().splitlines())
    result = dict(at=runtime.now(), load=list(os.getloadavg()),
        available_ram_bytes=int(fields['MemAvailable'].split()[0])*1024)
    query = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True, capture_output=True, timeout=10)
    result.update(nvidia_smi=query.stdout.strip(), nvidia_smi_error=query.stderr.strip(),
                  nvidia_smi_status=query.returncode)
    processes = subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader,nounits'],text=True,capture_output=True,timeout=10)
    result['gpu_processes'] = processes.stdout.strip()
    result['gpu_process_query_status'] = processes.returncode
    result['gpu_process_query_error'] = processes.stderr.strip()
    return result


def require_headroom(directory):
    snapshot = resources()
    if snapshot['nvidia_smi_status'] or snapshot['gpu_process_query_status']:
        raise RuntimeError(f'GPU resource inventory unavailable: {snapshot}')
    snapshot['free_disk_bytes'] = shutil.disk_usage(directory).free
    # Two workers retain contexts, batch outputs, compression and replay copies.
    snapshot['estimated_task_peak_bytes'] = 2*1024**3
    if snapshot['available_ram_bytes'] < CONTRACT['ram_reserve_bytes']+snapshot['estimated_task_peak_bytes']:
        raise RuntimeError('Shared host RAM reserve unavailable')
    if snapshot['free_disk_bytes'] < CONTRACT['disk_reserve_bytes']:
        raise RuntimeError('Output disk reserve unavailable')
    return snapshot


def validate_arrays(arrays, ids, contexts, dt):
    n=len(ids)
    if len(set(ids))!=n or len(arrays)!=9:
        raise ValueError('Duplicate IDs or changed output fields')
    shapes=dict(times=(751,),states=(n,751,3),statuses=(n,),final=(n,3),progress=(n,2),
                alpha=(n,),gamma=(n,),initial=(n,3),delta=(n,2))
    for name,shape in shapes.items():
        a=arrays[name]
        dtype=np.dtype('int32') if name=='statuses' else np.dtype('int64') if name=='progress' else np.dtype('float64')
        if a.shape!=shape or a.dtype!=dtype or not np.isfinite(a).all():
            raise ValueError(f'Invalid output dtype/shape/values: {name}')
    for name in ('alpha','gamma','initial','delta'):
        if not bitwise(arrays[name],np.asarray([contexts[i][name] for i in ids],dtype=np.float64)):
            raise ValueError(f'Realized input differs: {name}')
    if (not bitwise(arrays['times'],np.arange(751)*.1)
        or not bitwise(arrays['states'][:,0],arrays['initial'])
        or not bitwise(arrays['states'][:,-1],arrays['final'])
        or not np.all(arrays['statuses']==751)
        or not np.all(arrays['progress']==[round(75/dt),751])):
        raise ValueError('Invalid sample/status/progress/endpoint identity')


def write_batch(directory, arrays, scores, solver, contexts, ids, pool, timings=None):
    """Identical retained-output, hash and readback work for both arms."""
    begin=time.perf_counter()
    runtime.atomic_npz(directory/'traces.npz', **arrays)
    raw = runtime.binding(directory/'traces.npz', root=directory)
    payload = dict(scores=scores, solver=solver, raw=raw)
    runtime.write(directory/'scores.json', payload)
    score_binding = runtime.binding(directory/'scores.json', root=directory)
    after_write=time.perf_counter()
    with np.load(directory/'traces.npz', allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], a, equal_nan=True)
                                                for k,a in arrays.items()):
            raise ValueError('Raw write/readback differs')
        times,states=saved['times'].copy(),saved['states'].copy()
    saved_payload=read(directory/'scores.json')
    if saved_payload != payload or runtime.sha(directory/'traces.npz') != raw['sha256']:
        raise ValueError('JSON/hash verification differs')
    after_read=time.perf_counter()
    replay=drained_map(pool,lambda i:preview.score(contexts[ids[i]],times,states[i]),range(len(ids)))
    runtime.compare_scoring(saved_payload['scores'],replay)
    if timings is not None:
        timings.update(write_and_hash_seconds=after_write-begin,readback_seconds=after_read-after_write,
                       score_replay_seconds=time.perf_counter()-after_read)
    return dict(raw=raw, scores=score_binding)


class BatchFailure(RuntimeError):
    def __init__(self, results, errors):
        super().__init__(str(errors))
        self.results,self.errors=results,errors


def drained_map(pool, function, values):
    """Do not allow a failed future to leak ongoing work into the next arm."""
    futures=[pool.submit(function,value) for value in values]
    results,errors=[],[]
    for index,future in enumerate(futures):
        try:
            results.append(future.result())
        except BaseException as error:
            results.append(None)
            errors.append(dict(index=index,error=repr(error)))
    if errors:
        raise BatchFailure(results,errors)
    return results


def produce(job, contexts, ids, library, pool, directory):
    started = time.perf_counter()
    ids = ids[:job['size']]
    alpha = np.array([contexts[i]['alpha'] for i in ids], dtype=np.float64)
    gamma = np.array([contexts[i]['gamma'] for i in ids], dtype=np.float64)
    initial = np.array([contexts[i]['initial'] for i in ids], dtype=np.float64)
    delta = np.array([contexts[i]['delta'] for i in ids], dtype=np.float64)
    dt, n = job['dt'], len(ids)
    input_preparation_seconds=time.perf_counter()-started
    if job['arm'] == 'cpu':
        def one(i):
            begin=time.perf_counter()
            t,s = integrate_trace(library, alpha[i], gamma[i], initial[i], dt,
                                  duration=75., sample_dt=.1,
                                  failure_path=directory/f'partial-{i}.npz')
            integrated=time.perf_counter()
            try:
                score=preview.score(contexts[ids[i]],t,s)
            except BaseException:
                runtime.atomic_npz(directory/f'unscored-{i}.npz',times=t,states=s)
                raise
            return t,s,score,dict(integration_seconds=integrated-begin,scoring_seconds=time.perf_counter()-integrated)
        began = time.perf_counter()
        try:
            outputs = drained_map(pool,one,range(n))
        except BatchFailure as error:
            for i,item in enumerate(error.results):
                if item is not None:
                    runtime.atomic_npz(directory/f'completed-{i}.npz',times=item[0],states=item[1])
                    runtime.write(directory/f'completed-{i}.json',item[2])
            runtime.write(directory/'case-failures.json',error.errors)
            raise
        states = np.stack([x[1] for x in outputs])
        times = outputs[0][0]
        if any(not np.array_equal(times, x[0]) for x in outputs):
            raise ValueError('CPU time grids differ')
        scores = [x[2] for x in outputs]
        statuses = np.full(n,len(times),dtype=np.int32)
        progress = np.tile([round(75/dt),len(times)],(n,1)).astype(np.int64)
        final = states[:,-1].copy()
        solver = dict(integration_and_scoring_seconds=time.perf_counter()-began)
        solver['sum_case_seconds']={k:sum(x[3][k] for x in outputs) for k in ('integration_seconds','scoring_seconds')}
    else:
        order = np.arange(n)[::-1] if job['layout']=='reverse' else np.arange(n)
        chunk = CONTRACT['chunk_size'] if job['layout']=='chunked' else n
        parts = []
        try:
            for start in range(0,n,chunk):
                ind = order[start:start+chunk]
                result = library.integrate_batch(alpha[ind],gamma[ind],initial[ind],dt,
                    duration=75.,sample_dt=.1,max_device_bytes=CONTRACT['max_device_bytes'])
                parts.append(result)
        except BaseException as error:
            for i,part in enumerate(parts):
                runtime.atomic_npz(directory/f'completed-chunk-{i}.npz',
                    indices=order[i*chunk:(i+1)*chunk],
                    **{k:v for k,v in part.items() if isinstance(v,np.ndarray)})
            if isinstance(error,cuda.CudaError) and error.payload:
                error.payload.update(requested_indices=ind.copy(),alpha=alpha[ind],gamma=gamma[ind],initial=initial[ind])
            raise
        restore = np.argsort(order)
        states,statuses,final,progress = [np.concatenate([p[k] for p in parts])[restore]
                                        for k in ('states','statuses','final','progress')]
        times = parts[0]['times']
        solver = dict(chunks=[dict(timings=p['timings'],memory=p['memory'],device=p['device']) for p in parts])
        if any(not np.array_equal(times,p['times']) for p in parts) or not np.all(statuses == 751):
            runtime.atomic_npz(directory/'partial.npz',times=times,states=states,statuses=statuses,
                               final=final,progress=progress)
            raise ValueError('CUDA integration incomplete')
        began = time.perf_counter()
        try:
            scores = drained_map(pool,lambda i:preview.score(contexts[ids[i]],times,states[i]),range(n))
        except BaseException as error:
            runtime.atomic_npz(directory/'unscored.npz',times=times,states=states,statuses=statuses,
                               final=final,progress=progress,alpha=alpha,gamma=gamma,initial=initial,delta=delta)
            if isinstance(error,BatchFailure):
                runtime.write(directory/'case-failures.json',error.errors)
                runtime.write(directory/'partial-scores.json',error.results)
            raise
        solver['scoring_seconds'] = time.perf_counter()-began
    arrays = dict(times=times,states=states,statuses=statuses,final=final,progress=progress,
                  alpha=alpha,gamma=gamma,initial=initial,delta=delta)
    validate_arrays(arrays,ids,contexts,dt)
    integration_scoring = time.perf_counter()-started
    phases={}
    artifacts = write_batch(directory,arrays,scores,solver,contexts,ids,pool,timings=phases)
    elapsed = time.perf_counter()-started
    return dict(status='complete',ids=ids,artifacts=artifacts,end_to_end_seconds=elapsed,
        input_preparation_seconds=input_preparation_seconds,phase_seconds=phases,
        integration_scoring_seconds=integration_scoring, storage_verification_seconds=elapsed-integration_scoring,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def worker_entry(connection, arm, library_path, old_plan, parent_pid):
    signal.signal(signal.SIGTERM,signal.SIG_DFL)
    if ctypes.CDLL(None).prctl(1, signal.SIGTERM, 0,0,0) != 0 or os.getppid()!=parent_pid:
        os._exit(125)
    signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGINT,signal.SIGTERM})
    try:
        began = time.perf_counter()
        contexts = preview.load_contexts(old_plan)
        ids = [c['id'] for c in old_plan['census']]
        library = runtime.load_native(library_path) if arm=='cpu' else cuda.load(library_path)
        device = None if arm=='cpu' else library.device_info()
        pool = ThreadPoolExecutor(max_workers=CONTRACT['workers'])
        connection.send(dict(status='ready',pid=os.getpid(),device=device,
                             startup_seconds=time.perf_counter()-began))
        while True:
            command = connection.recv()
            if command is None:
                break
            job,directory = command
            directory = Path(directory)
            directory.mkdir()
            record = dict(**job,pid=os.getpid(),started_at=runtime.now(),status='failed',ids=ids[:job['size']])
            try:
                record.update(produce(job,contexts,ids,library,pool,directory))
            except BaseException as error:
                record['error'] = repr(error)
                if isinstance(error,cuda.CudaError) and error.payload:
                    arrays = {k:v for k,v in error.payload.items() if isinstance(v,np.ndarray)}
                    runtime.atomic_npz(directory/'cuda-failure.npz',**arrays)
                    record['cuda_error'] = {k:v for k,v in error.payload.items() if not isinstance(v,np.ndarray)}
            record['ended_at'] = runtime.now()
            runtime.write(directory/'record.json',record)
            connection.send(record)
        pool.shutdown(wait=True)
    except BaseException as error:
        try:
            connection.send(dict(status='worker_failed',pid=os.getpid(),error=repr(error)))
        except (BrokenPipeError,EOFError,OSError):
            pass
    finally:
        connection.close()


def receive(process, connection, deadline):
    while time.monotonic() < deadline:
        if connection.poll(min(.1,max(0,deadline-time.monotonic()))):
            return connection.recv()
        if not process.is_alive():
            raise RuntimeError(f'Owned worker {process.pid} exited {process.exitcode}')
    raise TimeoutError('Owned batch/startup deadline expired')


def metric_differences(a,b,path='score'):
    """Report every changed field; fixed metadata exact, phase circular, RMSE absolute."""
    differences=[]
    if isinstance(a,dict):
        if not isinstance(b,dict) or set(a)!=set(b):
            return [dict(path=path,passed=False,reason='keys')]
        for k in a:
            differences.extend(metric_differences(a[k],b[k],f'{path}.{k}'))
    elif isinstance(a,list):
        if not isinstance(b,list) or len(a)!=len(b):
            return [dict(path=path,passed=False,reason='length')]
        for i,(x,y) in enumerate(zip(a,b)):
            differences.extend(metric_differences(x,y,f'{path}[{i}]'))
    elif isinstance(a,(int,float)) and not isinstance(a,bool) and (
            isinstance(b,bool) or not isinstance(b,(int,float)) or not np.isfinite(a) or not np.isfinite(b)):
        differences.append(dict(path=path,passed=False,reason='invalid numerical leaf'))
    elif type(a) is not type(b) and not (isinstance(a,float) and type(b) in (int,float)):
        differences.append(dict(path=path,passed=False,reason='category type'))
    elif path.endswith('phase_search.refinement_evaluations') and (
            type(a) is not int or type(b) is not int or a<0 or b<0):
        differences.append(dict(path=path,passed=False,reason='invalid evaluation count'))
    elif a != b:
        if path.endswith('phase_search.refinement_evaluations'):
            differences.append(dict(path=path,a=a,b=b,passed=True,diagnostic=True))
        elif isinstance(a,float) and isinstance(b,(int,float)) and not isinstance(b,bool):
            difference=abs(a-b)
            if path.endswith('phase_offset_cycles'):
                difference=abs((a-b+.5)%1-.5)
                tolerance=CONTRACT['phase_atol']
            elif ('rms' in path or path.endswith('.rmse')):
                tolerance=CONTRACT['score_atol']
            else:
                tolerance=0.
            differences.append(dict(path=path,a=a,b=b,difference=difference,tolerance=tolerance,
                                    passed=bool(np.isfinite(difference) and difference<=tolerance)))
        else:
            differences.append(dict(path=path,a=a,b=b,passed=False,reason='category'))
    return differences


def rank_choices(census,scores):
    if not census or len(scores)!=len(census) or len({c['id'] for c in census})!=len(census):
        raise ValueError('Ranking requires the exact nonempty case/score census')
    groups=defaultdict(list)
    for case,score in zip(census,scores,strict=True):
        key=case['key']
        if len(key)!=5 or type(key[-1]) is not int or not 0<=key[-1]<18:
            raise ValueError('Invalid ranking context/candidate identity')
        value=score['rmse']
        if type(value) not in (int,float) or not np.isfinite(value) or not 0<=value<=2:
            raise ValueError('Ranking requires finite nonboolean preview scores')
        groups[tuple(key[:-1])].append((float(value),key[-1],case['id']))
    answer=[]
    for key,values in sorted(groups.items()):
        if len(values)!=18 or {v[1] for v in values}!=set(range(18)):
            raise ValueError('Every ranking context requires all 18 distinct candidate IDs')
        ordered=sorted(values,key=lambda v:(v[0],v[1]))
        ranked=[v[1] for v in ordered]
        by_candidate={v[1]:v[0] for v in values}
        answer.append(dict(key=list(key),ranked=ranked,top4=ranked[:4],minimum=ranked[0],
            controls={name:next(i for i in ranked if i in candidates) for name,candidates in CONTROLS.items()},
            scores=[by_candidate[i] for i in range(18)],minimum_rmse=ordered[0][0],
            best_second_gap=ordered[1][0]-ordered[0][0],fourth_fifth_gap=ordered[4][0]-ordered[3][0]))
    return answer


def bitwise(a,b):
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def compare_batches(cpu,gpu,cpu_scores,gpu_scores,census):
    ids=[c['id'] for c in census]
    if not ids or len(cpu_scores)!=len(ids) or len(gpu_scores)!=len(ids):
        raise ValueError('CPU/GPU score census differs from the declared case census')
    expected_fields={'times','states','statuses','final','progress','alpha','gamma','initial','delta'}
    if set(cpu)!=expected_fields or set(gpu)!=expected_fields:
        raise ValueError('CPU/GPU array key census differs')
    contexts={ident:{k:cpu[k][i] for k in ('alpha','gamma','initial','delta')} for i,ident in enumerate(ids)}
    if cpu['progress'].shape!=(len(ids),2) or len(set(cpu['progress'][:,0].tolist()))!=1:
        raise ValueError('CPU reference progress cannot identify one planned step size')
    steps=int(cpu['progress'][0,0])
    dts=[dt for dt in CONTRACT['dts'] if round(75/dt)==steps]
    if len(dts)!=1:
        raise ValueError('Unplanned CPU reference step count')
    validate_arrays(cpu,ids,contexts,dts[0])
    validate_arrays(gpu,ids,contexts,dts[0])
    for scores in (cpu_scores,gpu_scores):
        for score in scores:
            if set(score)!={'rmse','metrics','old_fixed_phase_rmse'}:
                raise ValueError('Incomplete score packet')
            if score['rmse']!=score['metrics']['aligned_rms']['aggregate']:
                raise ValueError('Aggregate label binding differs')
    metadata={k:bitwise(cpu[k],gpu[k]) for k in cpu if k not in ('states','final')}
    with np.errstate(over='ignore',invalid='ignore'):
        signed=cpu['states']-gpu['states']
        diff=np.abs(signed)
        tolerance=CONTRACT['state_atol']+CONTRACT['state_rtol']*np.maximum(np.abs(cpu['states']),np.abs(gpu['states']))
    activation_signed=np.tanh(cpu['states'])-np.tanh(gpu['states'])
    def statistics(delta):
        absolute=np.abs(delta)
        location=tuple(int(i) for i in np.unravel_index(int(np.argmax(absolute)),absolute.shape))
        def rms(values):
            maximum=float(np.max(np.abs(values)))
            return maximum*float(np.sqrt(np.mean((values/maximum)**2))) if maximum else 0.
        finite=bool(np.isfinite(delta).all())
        return dict(maximum_absolute_difference=float(np.max(absolute)) if finite else None,
            rms_difference=rms(delta) if finite else None,
            per_component_rms=[rms(delta[:,:,j]) for j in range(3)] if finite else None,
            per_component_maximum=[float(np.max(absolute[:,:,j])) for j in range(3)] if finite else None,
            maximum_location=dict(case_id=ids[location[0]],sample_index=location[1],
                                  time=float(cpu['times'][location[1]]),component=location[2]),
            nonfinite_difference_count=int(np.count_nonzero(~np.isfinite(delta))))
    state_diagnostic=statistics(signed);activation_diagnostic=statistics(activation_signed)
    metrics=[dict(id=c['id'],differences=metric_differences(a,b))
             for c,a,b in zip(census,cpu_scores,gpu_scores,strict=True)]
    metric_pass=all(d['passed'] for m in metrics for d in m['differences'])
    cpu_rank,gpu_rank=rank_choices(census,cpu_scores),rank_choices(census,gpu_scores)
    ranking=[]
    choices=('key','ranked','top4','minimum','controls')
    for a,b in zip(cpu_rank,gpu_rank,strict=True):
        equal_choices={name:a[name]==b[name] for name in choices}
        changes=(np.asarray(b['scores'])-np.asarray(a['scores'])).tolist()
        ranking.append(dict(key=a['key'],cpu=a,gpu=b,exact_choices=equal_choices,
            signed_gpu_minus_cpu=changes,maximum_absolute_score_change=max(abs(v) for v in changes),
            minimum_rmse_signed_change=b['minimum_rmse']-a['minimum_rmse'],passed=all(equal_choices.values())))
    ranking_pass=all(r['passed'] for r in ranking)
    state_pass=bool(np.isfinite(diff).all() and np.all(diff<=tolerance))
    finals=bool(bitwise(cpu['final'],cpu['states'][:,-1]) and bitwise(gpu['final'],gpu['states'][:,-1]))
    activation=activation_diagnostic['maximum_absolute_difference']
    passed=all(metadata.values()) and state_pass and finals and activation<=CONTRACT['activation_atol'] and metric_pass and ranking_pass
    return dict(passed=bool(passed),cases=len(ids),contexts=len(ranking),exact_metadata=metadata,
                state_pass=state_pass,state_max_difference=state_diagnostic['maximum_absolute_difference'],
                activation_max_difference=activation,state_diagnostic=state_diagnostic,
                activation_diagnostic=activation_diagnostic,finals_match=finals,metrics_pass=metric_pass,
                ranking_pass=ranking_pass,rankings=ranking,metric_changes=metrics)


def load_batch(directory, *, expected_job=None, census=None):
    directory=Path(directory)
    record=read(directory/'record.json')
    if record.get('status')!='complete':
        raise ValueError('Cannot load an incomplete batch as complete evidence')
    if expected_job is not None and any(type(record.get(k)) is not type(v) or record.get(k)!=v
                                        for k,v in expected_job.items()):
        raise ValueError('Saved batch job identity differs from plan')
    ids=record.get('ids')
    if (not isinstance(ids,list) or not ids or any(type(i) is not str for i in ids)
            or len(set(ids))!=len(ids) or type(record.get('size')) is not int or len(ids)!=record['size']):
        raise ValueError('Saved batch case identity census differs')
    if census is not None and ids!=[c['id'] for c in census[:record['size']]]:
        raise ValueError('Saved batch IDs are not the exact canonical planned prefix')
    if record.get('dt') not in CONTRACT['dts'] or type(record.get('dt')) is not float:
        raise ValueError('Saved batch step size is unplanned')
    if (type(record.get('end_to_end_seconds')) not in (int,float)
            or not np.isfinite(record['end_to_end_seconds']) or record['end_to_end_seconds']<=0):
        raise ValueError('Invalid completed-batch wall timing')
    if set(record.get('artifacts',{}))!={'raw','scores'}:
        raise ValueError('Batch artifact census differs')
    for name,artifact in record['artifacts'].items():
        expected='traces.npz' if name=='raw' else 'scores.json'
        path=directory/artifact['path']
        if artifact['path']!=expected or path.is_symlink() or not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError('Batch artifact path differs or escapes its owned directory')
        if runtime.binding(path,root=directory)!=artifact:
            raise ValueError('Batch artifact binding changed')
    with np.load(directory/'traces.npz',allow_pickle=False) as raw:
        arrays={k:raw[k].copy() for k in raw.files}
    required={'times','states','statuses','final','progress','alpha','gamma','initial','delta'}
    if set(arrays)!=required:
        raise ValueError('Saved trace array census differs')
    contexts={ident:{k:arrays[k][i] for k in ('alpha','gamma','initial','delta')} for i,ident in enumerate(ids)}
    validate_arrays(arrays,ids,contexts,record['dt'])
    payload=read(directory/'scores.json')
    if set(payload)!={'scores','solver','raw'} or payload['raw']!=record['artifacts']['raw']:
        raise ValueError('Score packet/raw binding differs')
    scores=payload['scores']
    if not isinstance(scores,list) or len(scores)!=len(ids):
        raise ValueError('Saved score count differs from case count')
    def finite_tree(value):
        if isinstance(value,dict):return all(finite_tree(v) for v in value.values())
        if isinstance(value,list):return all(finite_tree(v) for v in value)
        if type(value) in (float,int):return bool(np.isfinite(value))
        return type(value) in (str,bool)
    for score in scores:
        if (not isinstance(score,dict) or set(score)!={'rmse','metrics','old_fixed_phase_rmse'}
                or not finite_tree(score) or type(score['rmse']) not in (int,float)
                or not 0<=score['rmse']<=2 or score['rmse']!=score['metrics']['aligned_rms']['aggregate']):
            raise ValueError('Incomplete/nonfinite/misbound saved score packet')
    if census is not None:
        rank_choices(census[:len(ids)],scores)
    return arrays,scores


def audit(plan,directory,records,*,output_path=None):
    """Complete saved-evidence audit; never integrate or fit. Keep all discrepancies."""
    directory=Path(directory)
    expected=jobs();planned={j['id']:j for j in expected}; census=plan['census']
    issues=[]; lookup={}; validation=[]; baseline=[]; details=[];determinism=[];timing=[]
    def issue(where,error):issues.append(dict(location=where,error=str(error)))
    if plan.get('jobs')!=expected:
        issue('plan','Schedule differs from the fixed executor schedule')
    if len(records)!=len(expected):issue('terminal_census','Expected exactly 54 terminal records')
    for record in records:
        ident=record.get('id')
        if ident not in planned or ident in lookup:
            issue('terminal_census',f'Unexpected/duplicate batch ID: {ident}')
            continue
        lookup[ident]=record
        if any(type(record.get(k)) is not type(v) or record.get(k)!=v for k,v in planned[ident].items()):
            issue(ident,'Terminal task fields differ from the fixed plan')
    for ident in set(planned)-set(lookup):issue(ident,'Missing planned terminal record')
    if (len(census)!=3456 or len({c['id'] for c in census})!=3456
            or [c['key'] for c in census]!=sorted(c['key'] for c in census)):
        issue('census','Expected the canonical complete 3456-case census')
    else:
        origins={(c['key'][0],c['key'][1]) for c in census}
        product={(role,anchor,phase,radius,candidate) for role,anchor in origins
                 for phase in (675,1350) for radius in (.00025,.001,.004,.016) for candidate in range(18)}
        if (len(origins)!=24 or {role for role,_ in origins}!={'development','test'}
                or any(sum(role==wanted for role,_ in origins)!=12 for wanted in ('development','test'))
                or {tuple(c['key']) for c in census}!=product):
            issue('census','Physical-origin/phase/radius/candidate product differs')
        try:rank_choices(census,[dict(rmse=0.) for _ in census])
        except (ValueError,KeyError,TypeError) as error:issue('census',error)
    checked={}
    for job in expected:
        ident=job['id'];record=lookup.get(ident,{})
        item=dict(id=ident,arm=job['arm'],dt=job['dt'],size=job['size'],kind=job['kind'],
                  terminal_status=record.get('status','missing'),passed=False)
        if record.get('status')=='complete':
            try:
                saved=read(directory/'batches'/ident/'record.json')
                # The parent adds its resource snapshot after the worker seals its row.
                if {k:v for k,v in record.items() if k!='resources_before'}!=saved:
                    raise ValueError('Terminal and worker batch records differ')
                arrays,scores=load_batch(directory/'batches'/ident,expected_job=job,census=census)
                item['passed']=True;checked[ident]=True
                del arrays,scores
            except (ValueError,KeyError,TypeError,OSError,IndexError) as error:
                item['error']=str(error);issue(ident,error)
        else:item['error']=record.get('error','Required batch did not complete')
        validation.append(item)
    def old_reference(dt):
        arrays=dict(times=np.arange(751,dtype=np.float64)*.1,states=np.empty((3456,751,3),dtype=np.float64),
            statuses=np.full(3456,751,dtype=np.int32),final=np.empty((3456,3),dtype=np.float64),
            progress=np.tile([round(75/dt),751],(3456,1)).astype(np.int64),
            alpha=np.empty(3456),gamma=np.empty(3456),initial=np.empty((3456,3)),delta=np.empty((3456,2)))
        scores=[]
        for i,case in enumerate(census):
            stem=f'{case["id"]}--rk4-{dt:g}';old=read(LABEL_ROOT/'records'/f'{stem}.json')
            if (old['id'],old['case_id'],old['method'],old['status'])!=(stem,case['id'],f'rk4-{dt:g}','complete'):
                raise ValueError(f'Old native record identity differs: {stem}')
            with np.load(LABEL_ROOT/'traces'/f'{stem}.npz',allow_pickle=False) as raw:
                if not bitwise(raw['times'],arrays['times']):raise ValueError(f'Old time grid differs: {stem}')
                for name in ('states','initial','alpha','gamma','delta'):
                    expected_shape=(751,3) if name=='states' else (3,) if name=='initial' else (2,) if name=='delta' else ()
                    if raw[name].dtype!=np.dtype('float64') or raw[name].shape!=expected_shape or not np.isfinite(raw[name]).all():
                        raise ValueError(f'Invalid old array: {stem}/{name}')
                    arrays[name][i]=raw[name]
            arrays['final'][i]=arrays['states'][i,-1]
            scores.append({k:old[k] for k in ('rmse','metrics','old_fixed_phase_rmse')})
        return arrays,scores
    def prefix(arrays,n):return {k:(v if k=='times' else v[:n]) for k,v in arrays.items()}
    for dt in CONTRACT['dts']:
        if not any(checked.get(j['id']) for j in expected if j['dt']==dt):
            continue
        try:old_arrays,old_scores=old_reference(dt)
        except (ValueError,KeyError,TypeError,OSError,IndexError) as error:
            issue(f'old_reference/{dt}',error);continue
        cpu_name=f'cpu-dt{dt:g}-n3456-r0'
        canonical_cpu=load_batch(directory/'batches'/cpu_name,expected_job=planned[cpu_name],census=census) if checked.get(cpu_name) else None
        canonical_gpu=None;repeat_one=None
        for job in expected:
            if job['dt']!=dt:continue
            ident=job['id'];n=job['size']
            if not checked.get(ident):
                target=baseline if job['arm']=='cpu' else details
                target.append(dict(id=ident,dt=dt,cases=n,passed=False,undefined=True,reason='Required batch unavailable/invalid'))
                continue
            arrays,scores=load_batch(directory/'batches'/ident,expected_job=job,census=census)
            if job['arm']=='cpu':
                failures=[]
                old_prefix=prefix(old_arrays,n)
                for name in arrays:
                    if not bitwise(arrays[name],old_prefix[name]):
                        bad=np.any(arrays[name]!=old_prefix[name],axis=tuple(range(1,arrays[name].ndim))) if arrays[name].ndim>1 else arrays[name]!=old_prefix[name]
                        failures.append(dict(field=name,changed_cases=int(np.count_nonzero(bad)),
                                             note='Time-grid coordinates counted when field is times'))
                for case,saved,actual in zip(census[:n],old_scores[:n],scores,strict=True):
                    try:runtime.compare_scoring(saved,actual)
                    except (ValueError,KeyError,TypeError) as error:failures.append(dict(id=case['id'],error=str(error)))
                baseline.append(dict(id=ident,dt=dt,cases=n,passed=not failures,failures=failures))
            else:
                if canonical_cpu is None:
                    details.append(dict(id=ident,dt=dt,cases=n,passed=False,undefined=True,reason='Canonical CPU reference unavailable'))
                else:
                    try:details.append(dict(id=ident,dt=dt,**compare_batches(prefix(canonical_cpu[0],n),arrays,canonical_cpu[1][:n],scores,census[:n])))
                    except (ValueError,KeyError,TypeError,IndexError) as error:
                        details.append(dict(id=ident,dt=dt,cases=n,passed=False,error=str(error)))
                if job['kind']=='warm' and n==3456 and job['rep']==0:
                    canonical_gpu={k:v.copy() for k,v in arrays.items()}
                if job['kind']=='warm' and n==3456 and job['rep']==1:
                    repeat_one={k:v.copy() for k,v in arrays.items()}
                comparisons=[]
                if n==3456 and ((job['kind']=='warm' and job['rep'] in (1,2)) or job['kind']=='layout'):
                    comparisons.append((f'gpu-dt{dt:g}-n3456-r0',canonical_gpu))
                    if job['kind']=='warm' and job['rep']==2:
                        comparisons.append((f'gpu-dt{dt:g}-n3456-r1',repeat_one))
                for reference_id,reference in comparisons:
                    checks={k:bitwise(reference[k],arrays[k]) for k in arrays} if reference is not None else {}
                    determinism.append(dict(id=ident,reference_id=reference_id,dt=dt,
                        passed=bool(checks and all(checks.values())),fields=checks,
                        undefined=reference is None))
            del arrays,scores
        # Emit an explicit failed entry for every unavailable required comparison.
        expected_comparisons=[(f'gpu-dt{dt:g}-n3456-r1',f'gpu-dt{dt:g}-n3456-r0'),
            (f'gpu-dt{dt:g}-n3456-r2',f'gpu-dt{dt:g}-n3456-r0'),
            (f'gpu-dt{dt:g}-n3456-r2',f'gpu-dt{dt:g}-n3456-r1'),
            (f'gpu-dt{dt:g}-reverse',f'gpu-dt{dt:g}-n3456-r0'),
            (f'gpu-dt{dt:g}-chunked',f'gpu-dt{dt:g}-n3456-r0')]
        present={(d['id'],d['reference_id']) for d in determinism}
        for ident,reference_id in expected_comparisons:
            if (ident,reference_id) not in present:
                determinism.append(dict(id=ident,reference_id=reference_id,dt=dt,passed=False,undefined=True,fields={}))
        del old_arrays,old_scores,canonical_cpu,canonical_gpu,repeat_one
    # Every required check receives an explicit terminal entry, even when input
    # validation prevents reading reference arrays or comparing any outputs.
    for arm,target in [('cpu',baseline),('gpu',details)]:
        present={r['id'] for r in target}
        for job in expected:
            if job['arm']==arm and job['id'] not in present:
                target.append(dict(id=job['id'],dt=job['dt'],cases=job['size'],passed=False,
                    undefined=True,reason='Required reference/batch evidence unavailable'))
    present={(d['id'],d['reference_id']) for d in determinism}
    for dt in CONTRACT['dts']:
        r0=f'gpu-dt{dt:g}-n3456-r0';r1=f'gpu-dt{dt:g}-n3456-r1';r2=f'gpu-dt{dt:g}-n3456-r2'
        for ident,reference_id in [(r1,r0),(r2,r0),(r2,r1),(f'gpu-dt{dt:g}-reverse',r0),(f'gpu-dt{dt:g}-chunked',r0)]:
            if (ident,reference_id) not in present:
                determinism.append(dict(id=ident,reference_id=reference_id,dt=dt,passed=False,undefined=True,fields={}))
    for dt in CONTRACT['dts']:
        for size in CONTRACT['batch_sizes']:
            pairs=[]
            for rep in range(3):
                names=[f'{arm}-dt{dt:g}-n{size}-r{rep}' for arm in ('cpu','gpu')]
                if all(checked.get(name) for name in names):
                    cpu,gpu=[lookup[name]['end_to_end_seconds'] for name in names]
                    pairs.append(dict(rep=rep,cpu_seconds=cpu,gpu_seconds=gpu,speedup=cpu/gpu))
            ratios=[p['speedup'] for p in pairs]
            timing.append(dict(dt=dt,size=size,pairs=pairs,median_speedup=float(np.median(ratios)) if ratios else None,
                cpu_median_seconds=float(np.median([p['cpu_seconds'] for p in pairs])) if pairs else None,
                gpu_median_seconds=float(np.median([p['gpu_seconds'] for p in pairs])) if pairs else None,
                speed_gate=bool(len(ratios)==3 and min(ratios)>1 and np.median(ratios)>=CONTRACT['deployment_median_speedup'])))
    complete=bool(not issues and len(validation)==54 and all(v['passed'] for v in validation))
    parity=bool(complete and len(baseline)==25 and all(b['passed'] for b in baseline)
        and len(details)==29 and all(d['passed'] for d in details)
        and len(determinism)==10 and all(d['passed'] for d in determinism))
    deployment=next(t for t in timing if t['dt']==CONTRACT['deployment_dt'] and t['size']==CONTRACT['deployment_size'])
    full=dict(complete=complete,terminal_census_issues=issues,batch_validation=validation,
        expected_batches=54,observed_terminal_records=len(records),expected_case_integrations=64188,
        valid_completed_case_integrations=sum(v['size'] for v in validation if v['passed']),
        cpu_baseline=baseline,cpu_gpu_parity=details,determinism=determinism,timing=timing,
        all_parity_pass=parity,gpu_deployment_qualified=bool(parity and deployment['speed_gate']))
    runtime.write(Path(output_path) if output_path is not None else directory/'audit.json',full)
    return dict(complete=complete,all_parity_pass=parity,gpu_deployment_qualified=full['gpu_deployment_qualified'],
        terminal_census_issues=issues,expected_batches=54,validated_complete_batches=sum(v['passed'] for v in validation),
        expected_case_integrations=64188,valid_completed_case_integrations=full['valid_completed_case_integrations'],
        cpu_baseline=[{k:v for k,v in b.items() if k!='failures'} for b in baseline],
        parity=[{k:v for k,v in d.items() if k not in ('metric_changes','rankings')} for d in details],
        determinism_passes=sum(d['passed'] for d in determinism),determinism_required=10,timing=timing)


def source_guard(plan,plan_path):
    guard=runtime.source_guard(plan,plan_path)
    for relative in SOURCE_PATHS:
        content=subprocess.check_output(['git','-C',str(ROOT),'show',f'{guard["source_commit"]}:{relative}'])
        if hashlib.sha256(content).hexdigest()!=runtime.sha(ROOT/relative):
            raise ValueError(f'Uncommitted dependency: {relative}')
    return guard


def create_attempt(directory):
    directory=Path(directory).resolve()
    if directory.is_relative_to(ROOT):
        raise ValueError('Raw evidence must be external')
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'batches').mkdir()
    return directory


def process_identity(pid):
    if pid is None:
        return None
    try:
        stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,start_ticks=stat[19],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except FileNotFoundError:
        return None


def journal(directory,event):
    with (directory/'events.jsonl').open('ab') as stream:
        stream.write(runtime.canonical(dict(at=runtime.now(),**event))+b'\n')
        stream.flush()
        os.fsync(stream.fileno())


def interrupted(signum,frame):
    raise TimeoutError('Whole attempt deadline') if signum==signal.SIGALRM else KeyboardInterrupt(f'Signal {signum}')


def cost_summary(directory,records,startup):
    builds=read(directory/'builds.json') if (directory/'builds.json').exists() else {}
    cold={}
    for arm in ('cpu','gpu'):
        row=next((r for r in records if r['id']==f'cold-{arm}' and r['status']=='complete'),None)
        if row is not None and arm in startup:
            cold[arm]=row['end_to_end_seconds']+startup[arm]['parent_observed_startup_seconds']
    medians={}
    for arm in ('cpu','gpu'):
        values=[r['end_to_end_seconds'] for r in records if r['arm']==arm and r['kind']=='warm'
                and r['size']==3456 and r['dt']==.00125 and r['status']=='complete']
        if len(values)==3:
            medians[arm]=float(np.median(values))
    extra=None
    saving=None
    if set(cold)==set(medians)=={'cpu','gpu'} and builds:
        extra=max(0.,cold['gpu']+builds['gpu']['wall_seconds']-cold['cpu']-builds['cpu']['wall_seconds'])
        saving=medians['cpu']-medians['gpu']
    return dict(cold_pipeline_seconds=cold,warm_full_batch_median_seconds=medians,
        additional_gpu_setup_seconds=extra,median_full_batch_saving_seconds=saving,
        break_even_full_batches=math.ceil(extra/saving) if saving is not None and saving>0 else None,
        note='Cold18 startup plus build difference amortized over full warm batches; shared machine observation only.')


def run(plan,directory,plan_path):
    validate(plan,check_bindings=True)
    guard=source_guard(plan,plan_path)
    directory=create_attempt(directory)
    started=time.monotonic()
    deadline=started+CONTRACT['wall_ceiling_seconds']
    manifest=dict(**guard,plan_sha256=runtime.sha(plan_path),started_at=runtime.now(),pid=os.getpid(),
        plan=plan,python=sys.version,
        packages=None,
        thread_env={n:os.environ[n] for n in runtime.THREAD_ENV})
    runtime.write(directory/'manifest.json',manifest)
    records=[]
    workers={}
    startup={}
    identities={}
    error=None
    active_job=None
    old_signals={s:signal.signal(s,interrupted) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM)}
    signal.setitimer(signal.ITIMER_REAL,CONTRACT['wall_ceiling_seconds'])
    try:
        manifest['packages']={n:importlib.metadata.version(n) for n in CONTRACT['required_packages']}
        manifest['initial_resources']=require_headroom(directory)
        if manifest['initial_resources']['nvidia_smi_status']:
            raise RuntimeError('GPU availability unknown; cannot run the frozen GPU qualification')
        runtime.write(directory/'manifest.json',manifest)
        if manifest['packages']!=CONTRACT['required_packages']:
            raise RuntimeError('Runtime package versions changed')
        build_start=time.monotonic()
        cpu=runtime.build_native(directory/'cpu-build',deadline)
        cpu['wall_seconds']=time.monotonic()-build_start
        gpu=cuda.build(directory/'gpu-build',host_compiler='g++-12',timeout=120)
        runtime.write(directory/'builds.json',dict(cpu=cpu,gpu=gpu))
        for arm,library in [('cpu',cpu['path']),('gpu',gpu['library'])]:
            began=time.monotonic()
            parent,child=mp.get_context('spawn').Pipe()
            process=mp.get_context('spawn').Process(target=worker_entry,
                args=(child,arm,library,read(preview.PLAN),os.getpid()))
            workers[arm]=(process,parent)
            previous_mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM,signal.SIGALRM})
            try:
                process.start()
                identities[arm]=process_identity(process.pid)
                journal(directory,dict(event='worker_started',arm=arm,identity=identities[arm]))
            finally:
                child.close()
                signal.pthread_sigmask(signal.SIG_SETMASK,previous_mask)
            record=receive(process,parent,min(deadline,time.monotonic()+180))
            record['parent_observed_startup_seconds']=time.monotonic()-began
            startup[arm]=record
            runtime.write(directory/'startup.json',startup)
            if record['status']!='ready':
                raise RuntimeError(f'{arm} worker startup failed: {record}')
        for job in jobs():
            process,connection=workers[job['arm']]
            snapshot=require_headroom(directory)
            if time.monotonic()>=deadline:
                raise TimeoutError('Whole attempt deadline')
            runtime.write(directory/'pending.json',dict(job=job,resources=snapshot))
            active_job=job['id']
            journal(directory,dict(event='dispatch',job=job,resources=snapshot))
            connection.send((job,str(directory/'batches'/job['id'])))
            record=receive(process,connection,min(deadline,time.monotonic()+180))
            if record.get('id')!=job['id']:
                raise RuntimeError(f'Worker returned wrong task: {record}')
            record['resources_before']=snapshot
            records.append(record)
            journal(directory,dict(event='terminal',id=job['id'],status=record['status']))
            active_job=None
            runtime.write(directory/'terminal-records.json',records)
            print(json.dumps({k:record.get(k) for k in ('id','status','end_to_end_seconds','error')}),flush=True)
    except BaseException as exc:
        error=repr(exc)
    finally:
        # Cleanup/sealing may outlive the execution deadline; no further ODE work.
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig in (signal.SIGINT,signal.SIGTERM):
            signal.signal(sig,signal.SIG_IGN)
        cleanup=[]
        for arm,(process,connection) in workers.items():
            try:
                if process.pid is not None and process.is_alive():
                    connection.send(None)
                    process.join(timeout=2.)
            except (BrokenPipeError,EOFError,OSError):
                pass
            runtime.terminate_owned(process)
            connection.close()
            current=process_identity(process.pid)
            cleanup.append(dict(arm=arm,pid=process.pid,exitcode=process.exitcode,
                alive=process.is_alive(),initial_identity=identities.get(arm),current_identity=current,
                original_identity_absent=current is None or current!=identities.get(arm)))
        observed={r['id'] for r in records}
        for job in jobs():
            if job['id'] not in observed:
                path=directory/'batches'/job['id']/'record.json'
                records.append(read(path) if path.exists() else dict(**job,
                    status='interrupted' if job['id']==active_job else 'not_started',error=error,
                    ids=[c['id'] for c in plan['census'][:job['size']]]))
        runtime.write(directory/'terminal-records.json',records)
        runtime.write(directory/'cleanup.json',dict(workers=cleanup,
            all_owned_workers_stopped=all(not c['alive'] and c['original_identity_absent'] for c in cleanup)))
    analysis_start=time.monotonic()
    summary=dict(complete=False,all_parity_pass=False,gpu_deployment_qualified=False)
    try:
        for sig in (signal.SIGINT,signal.SIGTERM):
            signal.signal(sig,interrupted)
        remaining=deadline-time.monotonic()
        if remaining<=0:
            raise TimeoutError('No time remaining for required qualification analysis')
        signal.setitimer(signal.ITIMER_REAL,remaining)
        summary=audit(plan,directory,records)
        validate(plan,check_bindings=True)
        if runtime.git('rev-parse','HEAD')!=guard['source_commit']:
            raise ValueError('Source HEAD changed during collection')
        runtime.write(directory/'postflight.json',dict(source_and_inputs_unchanged=True,
            resources=resources(),checked_at=runtime.now()))
        summary['cost']=cost_summary(directory,records,startup)
    except BaseException as exc:
        error=f'{error}; {repr(exc)}' if error else repr(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        for sig in (signal.SIGINT,signal.SIGTERM):
            signal.signal(sig,signal.SIG_IGN)
    if error or any(c['alive'] or not c['original_identity_absent'] for c in cleanup):
        summary.update(complete=False,all_parity_pass=False,gpu_deployment_qualified=False)
    summary.update(error=error,wall_seconds=time.monotonic()-started,ended_at=runtime.now(),startup=startup,
                   analysis_seconds=time.monotonic()-analysis_start)
    runtime.write(directory/'summary.json',summary)
    sealing_start=time.monotonic()
    artifacts=[runtime.binding(p,root=directory) for p in sorted(directory.rglob('*')) if p.is_file()]
    receipt=dict(experiment_id='EXP-012',**guard,plan_sha256=runtime.sha(plan_path),
        status='complete' if summary['complete'] else 'incomplete',summary=summary,
        directory=str(directory),artifacts=artifacts,
        inventory_sealed_at=runtime.now(),sealing_seconds=time.monotonic()-sealing_start)
    runtime.write(directory/'receipt.json',receipt)
    for sig,handler in old_signals.items():
        signal.signal(sig,handler)
    print(json.dumps(summary,indent=2),flush=True)


def verify_attempt(plan,directory,output):
    directory=Path(directory).resolve()
    output=Path(output).resolve()
    if output.is_relative_to(directory) or output.is_relative_to(ROOT):
        raise ValueError('Audit output must be new and outside raw evidence/source')
    output.mkdir(parents=True,exist_ok=False)
    receipt=read(directory/'receipt.json')
    inventory=receipt['artifacts']
    if len({i['path'] for i in inventory})!=len(inventory):
        raise ValueError('Duplicate receipt members')
    actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}
    if actual!={i['path'] for i in inventory}|{'receipt.json'}:
        raise ValueError('Receipt file census changed')
    for item in inventory:
        path=(directory/item['path']).resolve()
        if not path.is_relative_to(directory) or runtime.binding(path,root=directory)!=item:
            raise ValueError(f'Receipt member changed: {item["path"]}')
    frozen=subprocess.check_output(['git','-C',str(ROOT),'show',
        f'{receipt["source_commit"]}:experiments/hopfield/EXP-012-plan.json'])
    if hashlib.sha256(frozen).hexdigest()!=receipt['plan_sha256'] or json.loads(frozen)!=plan:
        raise ValueError('Exact frozen plan differs')
    validate(plan,check_bindings=True)
    result=audit(plan,directory,read(directory/'terminal-records.json'),output_path=output/'audit.json')
    if runtime.sha(output/'audit.json')!=runtime.sha(directory/'audit.json'):
        if read(output/'audit.json')!=read(directory/'audit.json'):
            raise ValueError('Reconstructed qualification audit differs from frozen raw result')
    saved_summary=read(directory/'summary.json')
    if any(saved_summary.get(k)!=v for k,v in result.items()):
        raise ValueError('Reconstructed gates/timings differ from the saved summary')
    result['receipt']=binding(directory/'receipt.json')
    result['cost']=cost_summary(directory,read(directory/'terminal-records.json'),read(directory/'startup.json'))
    if result['cost']!=saved_summary['cost']:
        raise ValueError('Reconstructed startup/break-even accounting differs')
    result['original_audit_and_summary_reproduced']=True
    runtime.write(output/'receipt.json',dict(verified_at=runtime.now(),result=result,
        artifacts=[runtime.binding(p,root=output) for p in sorted(output.rglob('*')) if p.is_file()]))
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','validate','run','audit'])
    parser.add_argument('--plan',type=Path,default=PLAN)
    parser.add_argument('--directory',type=Path,default=Path('/data2/neural-shrimp-data/raw/EXP-012/attempt-001'))
    parser.add_argument('--audit-output',type=Path)
    args=parser.parse_args()
    if args.command=='prepare':
        runtime.write(args.plan,build_plan())
    else:
        plan=read(args.plan)
        if args.command=='validate':
            validate(plan,check_bindings=True)
            print('Frozen inputs and schedule valid; no integrations performed.')
        elif args.command=='run':
            run(plan,args.directory,args.plan)
        else:
            if args.audit_output is None:
                raise ValueError('--audit-output requires a new external directory')
            verify_attempt(plan,args.directory,args.audit_output)


if __name__=='__main__':
    main()
