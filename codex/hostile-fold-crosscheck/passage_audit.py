"""Narrow saved six-row passage audit: numerical evidence, no model replay.

The caller first admits original source/Git/lifecycle authority. This checker
independently hashes the external completed receipt/plan and exact inventory,
then verifies all stored trajectory links and recomputes the scalar comparisons.
A faithfully recorded forecast failure can pass this evidence audit.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import resource
import signal
import sys
import time
import traceback

OWNED=Path('/data2/neural-shrimp-data/codex/llm-hostile-fold-crosscheck')
ENVELOPE=dict(cpu_threads=1,nice=10,memory_gib=4,timeout_seconds=180)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,data):
    tmp=path.with_suffix('.json.partial');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def predictions(u,a,b,epsilon,U):
    initial=u;drift=a*epsilon
    if not drift>0 or not b>0:raise ValueError('wrong physical ghost-side sign')
    n=None
    for g in range(1,100001):
        u=u+drift+b*u*u
        if u>U:n=g;break
    if n is None:raise ValueError('scalar prediction did not exit')
    scale=math.sqrt(b/drift)
    return n,(math.atan(U*scale)-math.atan(initial*scale))/math.sqrt(b*drift)


def numeric(plan,status,modes,load,events,*,dimension=896):
    import numpy as np
    errors=[];rows=[];d=plan['design'];x,r,l=(modes[k] for k in ('x_c','r','l'))
    def require(condition,label):
        if not condition:errors.append(label)
    def finite(value,shape,name):
        value=np.asarray(value)
        if value.shape!=shape or value.dtype!=np.float64 or not np.isfinite(value).all():raise ValueError('finite float64 shape: '+name)
        return value
    for key,value in zip(('x_c','r','l'),(x,r,l)):
        finite(value,(dimension,),key)
        binding=dict(shape=list(value.shape),dtype=str(value.dtype),sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())
        require(d['modes'][key]==binding,'mode_hash_'+key)
    require(abs(np.linalg.norm(r)-1)<=1e-10 and abs(l@r-1)<=1e-10,'gauge')
    require(d['period']==2 and d['coordinate']=='log10T' and d['alpha']==.083,'clock_parameter')
    a,b,U,p=(d[k] for k in ('a','b','U','parameter_c'))
    if not all(math.isfinite(v) for v in (a,b,U,p)) or not a<0<b or U!=.1/b:raise ValueError('fold scalar gauge differs')
    frozen=d['rows'];reported=status['rows']
    if len(frozen)!=6 or len(reported)!=6:raise ValueError('all six frozen and reported rows required')
    expected_events=[];total=0
    for i,(row,observed) in enumerate(zip(frozen,reported)):
        label=f'e{i//2}-'+('full' if i%2==0 else 'midpoint')
        eps=(-1e-5,-3e-5,-1e-4)[i//2];entrance='full' if i%2==0 else 'midpoint'
        require(row['id']==label and row['epsilon']==eps and row['entrance']==entrance,label+'_identity')
        u0=-U if entrance=='full' else 0.
        require(row['u_in']==u0 and row['u_out']==U and row['parameter']==p+eps,label+'_endpoints_parameter')
        forecast,arc=predictions(u0,a,b,eps,U)
        asym=math.pi/math.sqrt(abs(a*b)*abs(eps))*(1. if entrance=='full' else .5)
        require(row['quadratic_G']==forecast and row['arctan_G']==arc and row['asymptotic_G']==asym,label+'_frozen_forecasts')
        require(row['max_G']==2*forecast+20,label+'_cap')
        for k,v in row.items():require(observed.get(k)==v,label+'_reported_'+k)
        trace=load(label+'/trace.npz')
        if set(trace)!={'state_trace','u','transverse','recorded_G','initial_states','states_at_stop'}:raise ValueError('exact full trace fields required')
        states=np.asarray(trace['state_trace']);n=states.shape[0]-1 if states.ndim==2 else -1
        finite(states,(n+1,dimension),'state_trace');finite(trace['u'],(n+1,),'u');finite(trace['transverse'],(n+1,),'transverse')
        require(0<n<=row['max_G'] and int(trace['recorded_G'])==n,label+'_length')
        initial=x+u0*r
        require(np.array_equal(states[0],initial) and np.array_equal(trace['initial_states'],initial),label+'_initial')
        require(row['initial_state']==dict(shape=list(initial.shape),dtype='float64',sha256=hashlib.sha256(initial.tobytes()).hexdigest()),label+'_initial_hash')
        require(np.array_equal(trace['states_at_stop'],states[-1]),label+'_stop')
        u=np.asarray([float((v-x)@l) for v in states]);trans=np.asarray([np.linalg.norm((v-x)-z*r) for v,z in zip(states,u)])
        require(row['actual_initial_u']==float(u[0]),label+'_actual_initial_u')
        require(np.allclose(trace['u'],u,rtol=1e-12,atol=1e-14),label+'_projections')
        require(np.allclose(trace['transverse'],trans,rtol=1e-12,atol=1e-14),label+'_transverse')
        crossings=np.flatnonzero(u[1:]>U)+1
        count=int(crossings[0]) if len(crossings) else None
        require(count==n if count is not None else n==row['max_G'],label+'_first_strict_exit_or_cap')
        for g in range(1,n+1):
            before=load(f'{label}/step-{g:04d}-input.npz');after=load(f'{label}/step-{g:04d}-output.npz')
            if set(before)!={'state','parameter'} or set(after)!={'state'}:raise ValueError('exact saved step fields required')
            finite(before['state'],(dimension,),'step input');finite(after['state'],(dimension,),'step output');finite(before['parameter'],(),'parameter')
            require(np.array_equal(before['state'],states[g-1]) and np.array_equal(after['state'],states[g]),label+f'_chain_{g}')
            require(float(before['parameter'])==row['parameter'],label+f'_parameter_{g}')
            expected_events.append((label,g,2*g))
        error=None if count is None else count-forecast
        expected=dict(status='censored' if count is None else 'observed',measured_G=count,
            measured_F=None if count is None else 2*count,recorded_G=n,quadratic_error=error,
            arctan_error=None if count is None else count-arc,asymptotic_error=None if count is None else count-asym,
            primary_pass=count is not None and abs(error)<=1,
            pre_crossing_u=None if count is None else float(u[-2]),crossing_u=None if count is None else float(u[-1]))
        for k,v in expected.items():require(observed.get(k)==v,label+'_summary_'+k)
        rows.append(dict(id=label,epsilon=eps,entrance=entrance,**expected,quadratic_G=forecast,arctan_G=arc,asymptotic_G=asym,
            maximum_projection_discrepancy=float(np.max(abs(trace['u']-u)))))
        total+=n
    require([(e['row'],e['G'],e['F']) for e in events]==expected_events,'complete_ordered_events')
    require(all(type(e['seconds'])in(int,float) and math.isfinite(e['seconds']) and e['seconds']>=0 for e in events),'finite_step_times')
    primary=all(row['primary_pass'] for row in rows);all_observed=all(row['status']=='observed' for row in rows)
    require(status['model_G_calls']==total,'total_model_calls')
    require(status['primary_all_pass'] is primary and status['all_rows_observed'] is all_observed,'aggregate_scientific_gates')
    require(d['total_forecast_G']==sum(row['quadratic_G'] for row in rows),'total_prediction')
    require(d['total_cap_G']==sum(row['max_G'] for row in frozen),'total_cap')
    return dict(audit_passed=not errors,errors=errors,rows=rows,scientific_primary_all_pass=primary,
        all_rows_observed=all_observed,total_G=total,total_F=2*total,certificate_passed=False,
        natural_reinjection_measured=False,model_replayed=False,
        scope='Independent saved numerical trajectories and scalar forecasts; no model replay, natural reinjection inference or interval proof.')


def audit(attempt,receipt_sha,source,plan_sha):
    import numpy as np
    if sha(attempt/'receipt.json')!=receipt_sha:raise ValueError('external receipt changed')
    receipt=json.loads((attempt/'receipt.json').read_text())
    if (receipt['source_commit']!=source or receipt['plan_sha256']!=plan_sha or type(receipt['exit_code'])is not int
        or receipt['exit_code']!=0 or receipt['timed_out']is not False or receipt['worker_termination_verified']is not True
        or receipt['uncertain_ownership']!=[] or receipt['status']!='passage_attempt_completed'
        or receipt['controlled_passage_result_available']is not True or receipt['certificate_passed']is not False):
        raise ValueError('successful closed external authority required')
    files={}
    for row in receipt['artifacts']:
        name=row['path'];path=attempt/name
        if (type(name)is not str or Path(name).is_absolute() or '..'in Path(name).parts or Path(name).as_posix()!=name
            or name in files or path.resolve()!=path or path.is_symlink() or not path.is_file()
            or path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']):raise ValueError('original inventory differs: '+str(name))
        files[name]=row
    if {str(p.relative_to(attempt)) for p in attempt.rglob('*') if p.is_file() and p!=attempt/'receipt.json'}!=set(files):raise ValueError('complete original inventory required')
    if sha(attempt/'run/original-plan.json')!=plan_sha:raise ValueError('original prospective plan differs')
    plan=json.loads((attempt/'run/original-plan.json').read_text());status=json.loads((attempt/'run/status.json').read_text())
    if (plan['schema']!='codex-hostile-passage-v1' or status['status']!='passage_completed'
        or status['computational_completed']is not True or status['certificate_passed']is not False
        or status['passage_trajectories_evaluated']is not True or status['natural_reinjection_measured']is not False
        or status['resource_probe_only']is not False):raise ValueError('complete numerical controlled-passage scope required')
    for name,pin in plan['source_sha256'].items():
        if files['source/'+name]['sha256']!=pin:raise ValueError('captured scientific source differs')
    inputs=json.loads((attempt/'run/inputs.json').read_text())
    if inputs['source_commit']!=source or inputs['plan_sha256']!=plan_sha:raise ValueError('worker identity differs')
    for key,name in (('fold_admission_sha256','run/fold-admission.json'),('frozen_modes_sha256','run/frozen-modes.npz')):
        if inputs[key]!=files[name]['sha256']:raise ValueError('input hashlink differs')
    admission=json.loads((attempt/'run/fold-admission.json').read_text())
    if admission['authority']!=plan['fold_authority'] or admission['independent_audit_passed']is not True:raise ValueError('fold authority differs')
    def load(name):
        with np.load(attempt/'run'/name,allow_pickle=False) as values:return {k:values[k].copy() for k in values.files}
    events=[json.loads(line) for line in (attempt/'run/events.jsonl').read_text().splitlines()]
    result=numeric(plan,status,load('frozen-modes.npz'),load,events)
    # Refuse hidden extra step outputs rather than checking only observed prefixes.
    for row in result['rows']:
        label=row['id'];expected={f'run/{label}/trace.npz',f'run/{label}/result.json'}
        for g in range(1,row['recorded_G']+1):
            expected.update(f'run/{label}/step-{g:04d}-{kind}.npz' for kind in ('input','output'))
        if {name for name in files if name.startswith(f'run/{label}/')}!=expected:raise ValueError('complete exact row files required')
        observed=next(r for r in status['rows'] if r['id']==label)
        if json.loads((attempt/'run'/label/'result.json').read_text())!=observed or observed['trace_sha256']!=files[f'run/{label}/trace.npz']['sha256']:
            raise ValueError('row summary/trace hashlink differs')
    result.update(receipt_sha256=receipt_sha,source_commit=source,plan_sha256=plan_sha,
        original_artifact_count=len(files),source_sha256=plan['source_sha256'],audit_source_sha256=sha(Path(__file__)),
        git_blob_and_ownership_audit='Externally admitted by the lifecycle owner; this checker authenticates captured bytes and termination receipt, not live process history.')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('attempt','output'):parser.add_argument('--'+name,type=Path,required=True)
    for name in ('receipt-sha256','source','plan-sha256'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args();started=time.monotonic()
    if args.output.resolve()!=args.output or args.output.parent!=OWNED or args.output.exists():raise ValueError('new exclusive own audit directory required')
    args.output.mkdir();result=None
    try:
        if args.attempt.resolve()!=args.attempt or args.attempt.parent!=OWNED:raise ValueError('owned complete target attempt required')
        if re.fullmatch('[0-9a-f]{40}',args.source)is None or any(re.fullmatch('[0-9a-f]{64}',s)is None for s in (args.receipt_sha256,args.plan_sha256)):
            raise ValueError('full external source/receipt/plan pins required')
        if os.getpriority(os.PRIO_PROCESS,0)!=10:raise ValueError('exact nice10 required')
        if any(os.environ.get(k)!='1' for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')):raise ValueError('one-thread environment required')
        soft,hard=resource.getrlimit(resource.RLIMIT_AS);cap=4*1024**3
        if hard!=resource.RLIM_INFINITY and hard<cap:cap=hard
        if soft!=resource.RLIM_INFINITY:cap=min(cap,soft)
        resource.setrlimit(resource.RLIMIT_AS,(cap,cap))
        def timeout(*args):raise TimeoutError('180s saved-audit deadline')
        signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,180)
        result=audit(args.attempt,args.receipt_sha256,args.source,args.plan_sha256)
    except BaseException as exc:
        result=dict(audit_passed=False,error=repr(exc),traceback=traceback.format_exc(),certificate_passed=False,model_replayed=False)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        result.update(seconds=time.monotonic()-started,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,envelope=ENVELOPE)
        write(args.output/'audit.json',result)
    return 0 if result['audit_passed'] else 2


if __name__=='__main__':sys.exit(main())
