"""EXP-013 owned CPU stages, durable outputs, and unchanged numerical primitives."""
from __future__ import annotations
import ctypes
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import resource
import signal
import subprocess
import time

import run_hopfield_refinement as legacy  # fixes numerical thread limits before NumPy
import numpy as np
from scipy.integrate import solve_ivp
import run_hopfield as baseline_engine
import run_hopfield_drift as baseline_recipe
from hopfield_diagnostics import hopfield_rhs
from hopfield_drift_probe import integrate_trace, local_parameter_sensitivity, phase_projected_gram
from hopfield_drift_metrics import waveform_errors, periodic_interpolate

ROOT = Path(__file__).resolve().parents[1]
BASELINE = dict(initial_state=[.1,.1,.1], dt=[.01,.005], burn_time=10000, sample_time=2000,
    event_capacity=128, max_peak_period=32, minimum_repeats=4, recurrence_atol=2e-6, recurrence_rtol=2e-5)
METHODS = [dict(id=f'rk4-{dt:g}', kind='RK4', dt=dt) for dt in (.005,.0025,.00125)] + [
    dict(id='dop853-1e-11', kind='DOP853', rtol=1e-11, atol=1e-13, max_step=.05),
    dict(id='dop853-1e-12', kind='DOP853', rtol=1e-12, atol=1e-14, max_step=.025)]
TIMES = np.arange(751, dtype=np.float64)*.1
TIMES[-1] = 75.
SENSITIVITY_TIMES = np.linspace(0,300,3001)


def write(path, value):
    """Write a new record exclusively; nonfinite diagnostics must be explicit nulls."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(baseline_engine.safe(value),f,indent=2,allow_nan=False); f.write('\n'); f.flush();os.fsync(f.fileno())


def append(path, value):
    with Path(path).open('a') as f:
        f.write(json.dumps(baseline_engine.safe(value),sort_keys=True,allow_nan=False)+'\n');f.flush();os.fsync(f.fileno())


def identity(pid):
    try:
        content=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),start_ticks=content[19],boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    except (OSError,IndexError):return None


def inventory(directory):
    directory=Path(directory)
    return [dict(path=str(p.relative_to(directory)),sha256=legacy.sha(p),bytes=p.stat().st_size)
            for p in sorted(directory.rglob('*')) if p.is_file() and p.name!='receipt.json']


def checked_command(command, timeout):
    """Owned compiler group with interruption masked until its PID is registered."""
    process=None
    try:
        mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM,signal.SIGALRM})
        try:process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,start_new_session=True)
        finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
        out,err=process.communicate(timeout=timeout)
        if process.returncode:raise RuntimeError(f'Compiler command failed ({process.returncode}): {err}')
        return out
    except BaseException:
        if process is not None:
            try:os.killpg(process.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            process.communicate()
        raise


def build_libraries(directory, deadline):
    directory=Path(directory);directory.mkdir()
    result={}
    for kind,source in [('baseline','native/hopfield.cpp'),('trace','native/hopfield_drift.cpp')]:
        compiler=os.environ.get('CXX','c++'); flags=['-O3','-std=c++17','-fPIC','-shared'];dest=directory/f'{kind}.so'
        start=time.monotonic();command=[compiler,*flags,str(ROOT/source),'-o',str(dest)]
        record=dict(kind=kind,command=command,source=legacy.binding(ROOT/source))
        try:
            record['stdout']=checked_command(command,max(.01,min(60.,deadline-time.monotonic())))
            record.update(status='complete',path=str(dest),sha256=legacy.sha(dest),bytes=dest.stat().st_size,
                compiler=checked_command([compiler,'--version'],5.).splitlines()[0])
        except BaseException as e:
            record.update(status='failed',error=repr(e));raise
        finally:
            record['wall_seconds']=time.monotonic()-start;write(directory/f'{kind}-build.json',record)
        result[kind]=record
    return result


def load_baseline(path):
    fn=ctypes.CDLL(str(path)).hopfield_integrate;ptr=ctypes.POINTER(ctypes.c_double)
    fn.argtypes=[ctypes.c_double,ctypes.c_double,ptr,ctypes.c_double,ctypes.c_int,ctypes.c_int,ctypes.c_int,ptr,ptr,ptr]
    fn.restype=ctypes.c_int;return fn


def certificate_ok(cert, anchor=None):
    canonical_gates={"closure","phase_condition","section_is_maximum","nonstationary","expected_peak_count","primitive","neutral_mode","transverse_stability"}
    if not (cert.get('verified') is True and cert.get('attracting') is True and
            cert.get('primitive') is True and cert.get('peak_count')==9 and cert.get('expected_peak_count')==9 and
            isinstance(cert.get('gates'),dict) and set(cert['gates'])==canonical_gates and all(v is True for v in cert['gates'].values())):
        return False
    try:
        t=np.asarray(cert['orbit_times']);x=np.asarray(cert['orbit_states']);u=np.asarray(cert['orbit_activations'])
        return bool(t.shape==(2701,) and np.isfinite(cert['period']) and cert['period']>0 and x.shape==u.shape==(2701,3) and np.isfinite(t).all()
            and np.isfinite(x).all() and np.isfinite(u).all() and t[0]==0 and t[-1]==cert['period']
            and np.array_equal(t,np.linspace(0.,cert['period'],2701)) and np.array_equal(np.tanh(x),u)
            and (anchor is None or (cert['alpha']==anchor['alpha'] and cert['gamma']==anchor['gamma']))
            and 0<cert['transverse_spectral_radius']<1 and np.array_equal(x[0],cert['section_state']))
    except (KeyError,TypeError,ValueError):return False


def score(context, times, states):
    if (np.asarray(times).dtype!=np.float64 or np.asarray(states).dtype!=np.float64 or not np.array_equal(times,TIMES) or np.shape(states)!=(751,3) or not np.isfinite(states).all()
            or not np.array_equal(states[0],context['initial'])):
        raise ValueError('Complete unchanged 751-sample preview required')
    u=np.tanh(states)
    metrics=waveform_errors(context['reference'],times+context['epoch'],u,context['period'],phase_grid=128)
    fixed=periodic_interpolate(context['reference'],(times+context['epoch'])/context['period'])
    return dict(rmse=metrics['aligned_rms']['aggregate'],metrics=metrics,
                reference_fixed_phase_rmse=float(np.sqrt(np.mean((u-fixed)**2))))


def worker(task, context, directory, libraries, parent_pid):
    """One bounded CPU process writes all available evidence before its terminal row."""
    start=time.monotonic();cpu_start=time.process_time();directory=Path(directory)
    ident=task['id'];kind=task['kind'];dest=directory/'payloads'/ident;dest.mkdir()
    row=dict(id=ident,kind=kind,case_id=task['case']['id'],method=task['method']['id'],status='failed',
             started_at=legacy.now(),pid=os.getpid(),identity=identity(os.getpid()),rmse=None)
    try:
        if kind=='models':
            import preview_final_model as final
            import preview_shortlist_features as features
            import run_preview_shortlist as old
            old_plan=context['old_plan']; bundle=features.load_features()
            if old.feature_identity(bundle)!=old_plan['feature_identity']:raise ValueError('Old feature identity drift')
            store=old.LabelStore(bundle['rows'],old_plan['label_records'],allowed_training_groups=context['groups'],log_path=dest/'access.jsonl')
            labels={resolution:store.training([r['id'] for r in bundle['rows']],resolution) for resolution in final.RESOLUTIONS}
            packet=final.fit_four_recipes(bundle,labels,old_origin_groups=context['groups'])
            write(dest/'models.json',packet)
            row.update(status='complete',models_complete=packet['complete'],qualified=packet['complete'] and packet['all_model_checks_pass'],
                models_path=str((dest/'models.json').relative_to(directory)),old_feature_identity=old.feature_identity(bundle))
        elif kind=='baseline':
            anchor=context['anchor']; result=baseline_recipe.baseline(load_baseline(libraries['baseline']['path']),anchor,BASELINE,dest)
            row.update(status='complete',qualified=result['eligible'] and certificate_ok(result['certificate'],anchor),
                result_path=str((dest/'baselines'/anchor['id']/'result.json').relative_to(directory)))
        elif kind=='sensitivity':
            anchor=context['anchor'];initial=np.asarray(context['initial'])
            result=local_parameter_sensitivity(anchor['alpha'],anchor['gamma'],initial,SENSITIVITY_TIMES)
            tail=result['times']>75
            result['tail_direction']=np.asarray(phase_projected_gram(result['activation_sensitivity'][tail],result['activation_velocity'][tail])['minimum_eigenvector'])
            arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
            arrays.update(initial=initial,alpha=anchor['alpha'],gamma=anchor['gamma'],phase=context['phase'],epoch=context['epoch'])
            path=dest/'sensitivity.npz';legacy.atomic_npz(path,**arrays)
            write(dest/'sensitivity.json',{k:v for k,v in result.items() if not isinstance(v,np.ndarray)})
            row.update(status='complete',qualified=True,raw=str(path.relative_to(directory)),raw_sha256=legacy.sha(path),raw_bytes=path.stat().st_size,
                       metadata_path=str((dest/'sensitivity.json').relative_to(directory)))
        elif kind=='preview':
            method=task['method'];path=dest/'trace.npz';integration_start=time.monotonic()
            if method['kind']=='RK4':
                times,states=integrate_trace(legacy.load_native(libraries['trace']['path']),context['alpha'],context['gamma'],
                    context['initial'],method['dt'],duration=75.,sample_dt=.1,failure_path=dest/'native-failure.npz')
                row['solver']=dict(**method,steps=round(75/method['dt']))
            else:
                result=solve_ivp(lambda t,x:hopfield_rhs(t,x,context['alpha'],context['gamma']),(0.,75.),context['initial'],
                    method='DOP853',t_eval=TIMES,rtol=method['rtol'],atol=method['atol'],max_step=method['max_step'])
                times,states=result.t,result.y.T
                row['solver']=dict(**method,success=bool(result.success),message=result.message,nfev=result.nfev)
                if not result.success or states.shape!=(751,3) or not np.isfinite(states).all():
                    legacy.atomic_npz(dest/'adaptive-failure.npz',times=times,states=states,requested_times=TIMES,initial=context['initial'])
                    raise ValueError('Independent preview solver failed')
            row['integration_seconds']=time.monotonic()-integration_start;storage_start=time.monotonic()
            legacy.atomic_npz(path,times=times,states=states,initial=context['initial'],alpha=context['alpha'],gamma=context['gamma'],delta=context['delta'])
            row.update(raw=str(path.relative_to(directory)),raw_sha256=legacy.sha(path),raw_bytes=path.stat().st_size)
            row['storage_hash_seconds']=time.monotonic()-storage_start;score_start=time.monotonic()
            row.update(score(context,times,states),status='complete');row['score_seconds']=time.monotonic()-score_start
        else:raise ValueError('Unplanned worker kind')
    except BaseException as error:
        row['error']=repr(error)
        if hasattr(error,'payload'):legacy.atomic_npz(dest/'partial-failure.npz',**error.payload)
        if hasattr(error,'packet'):write(dest/'partial-packet.json',error.packet)
    row.update(ended_at=legacy.now(),wall_seconds=time.monotonic()-start,cpu_seconds=time.process_time()-cpu_start,
        max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(directory/'records'/f'{ident}.json',row)


def execute(specs, contexts, directory, libraries, *, workers, deadline, per_task, worker_fn=worker):
    """Reuse reviewed EXP009 owned-entry lifecycle; retain identity and every status."""
    if not 1<=workers<=4:raise ValueError('Worker cap 1..4')
    directory=Path(directory);directory.mkdir();(directory/'payloads').mkdir();(directory/'records').mkdir()
    if len({t['id'] for t in specs})!=len(specs):raise ValueError('Duplicate tasks')
    pending=iter(specs);active={};results=[];exhausted=False;process_context=mp.get_context('fork')
    def record(task,row,expected_pid=None):
        if (not isinstance(row,dict) or row.get('id')!=task['id'] or row.get('case_id')!=task['case']['id']
                or row.get('method')!=task['method']['id'] or row.get('kind')!=task['kind']
                or (expected_pid is not None and row.get('pid')!=expected_pid)):
            row=dict(id=task['id'],case_id=task['case']['id'],method=task['method']['id'],kind=task['kind'],status='invalid_worker_identity',rmse=None)
        append(directory/'journal.jsonl',row);results.append(row)
        if len(results)%100==0:
            try:print(json.dumps(dict(stage=directory.name,terminal=len(results),required=len(specs))),flush=True)
            except OSError:pass  # A closed progress consumer must not alter research execution.
    try:
        while active or not exhausted:
            expired=time.monotonic()>=deadline
            while not expired and not exhausted and len(active)<workers:
                task=next(pending,None)
                if task is None:exhausted=True;break
                process=process_context.Process(target=legacy.owned_entry,args=(worker_fn,task,contexts[task['case']['id']],directory,libraries,os.getpid()))
                active[task['id']]=(process,task,time.monotonic());mask=signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGINT,signal.SIGTERM,signal.SIGALRM})
                try:
                    try:process.start()
                    finally:
                        if process.pid is not None:append(directory/'processes.jsonl',dict(task_id=task['id'],pid=process.pid,identity=identity(process.pid),started_at=legacy.now()))
                finally:signal.pthread_sigmask(signal.SIG_SETMASK,mask)
            for ident,(process,task,started) in list(active.items()):
                timed_out=expired or time.monotonic()-started>=per_task
                if process.is_alive() and not timed_out:continue
                if timed_out:legacy.terminate_owned(process)
                else:process.join()
                p=directory/'records'/f'{ident}.json'
                if p.exists() and not timed_out and process.exitcode==0:
                    try:row=json.loads(p.read_text())
                    except (ValueError,OSError):row=dict(id=ident,case_id=task['case']['id'],method=task['method']['id'],kind=task['kind'],pid=process.pid,status='invalid_worker_record',rmse=None)
                else:
                    row=dict(id=ident,case_id=task['case']['id'],method=task['method']['id'],kind=task['kind'],pid=process.pid,
                        status='whole_attempt_timeout' if expired else 'task_timeout' if timed_out else 'worker_failed',rmse=None,exitcode=process.exitcode)
                    write(directory/'records'/f'{ident}-termination.json',row)
                record(task,row,process.pid);del active[ident]
            if expired and not exhausted:
                for task in pending:record(task,dict(id=task['id'],case_id=task['case']['id'],method=task['method']['id'],kind=task['kind'],status='not_started_deadline',rmse=None))
                exhausted=True
            if active:time.sleep(.01)
    except BaseException:
        recorded={r['id'] for r in results}
        for process,task,_ in active.values():
            legacy.terminate_owned(process)
            if task['id'] not in recorded:record(task,dict(id=task['id'],case_id=task['case']['id'],method=task['method']['id'],kind=task['kind'],status='interrupted',pid=process.pid,rmse=None))
        for task in pending:record(task,dict(id=task['id'],case_id=task['case']['id'],method=task['method']['id'],kind=task['kind'],status='not_started_interruption',rmse=None))
        raise
    finally:
        for process,_,_ in active.values():legacy.terminate_owned(process)
        write(directory/'terminal-records.json',results)
        identities=[json.loads(line) for line in (directory/'processes.jsonl').read_text().splitlines()] if (directory/'processes.jsonl').exists() else []
        checks=[]
        for row in identities:
            current=identity(row['pid'])
            checks.append(dict(**row,current_identity=current,original_identity_absent=current is None or (row['identity'] is not None and current!=row['identity'])))
        write(directory/'cleanup.json',dict(workers=checks,all_registered_workers_joined=not any(p.is_alive() for p,_,_ in active.values()),all_original_identities_absent=all(c['original_identity_absent'] for c in checks)))
    return results
