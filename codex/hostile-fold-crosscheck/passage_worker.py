"""Six prospective controlled passages, preserving the Task5 counting rule.

The original Task5 run_group has three fixed entrances. This small serial adapter
uses exactly the two requested entrances, reuses its unchanged scalar predictor,
and stops each trajectory at the first strict crossing after a whole G=F^2 step.
It measures controlled local transits, never natural reinjection or token loops.
"""
import hashlib
import importlib.util
import math
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
EPSILONS=(-1e-5,-3e-5,-1e-4)
ENTRANCES=('full','midpoint')
CRITERION=dict(primary='all six observed |measured_G-quadratic_G|<=1',
 secondary='report signed arctan and asymptotic errors; no one-iterate gate',
 endpoint='first strict u>+U after a whole G=F^2 step; no burn-in',
 missing='censored, failed, partial and not-run rows never pass',
 reinjection='controlled starts only; natural reinjection is not measured')


def predictor():
    spec=importlib.util.spec_from_file_location('task7_task5_prediction',ROOT/'codex/confirm-fold/build_design.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.predictions


def digest(value):
    value=np.asarray(value)
    return dict(shape=list(value.shape),dtype=str(value.dtype),
                sha256=hashlib.sha256(value.tobytes(order='C')).hexdigest())


def vectors(arrays):
    values=[np.asarray(arrays[k]) for k in ('x_c','r','l')]
    x,r,l=values
    if x.ndim!=1 or x.size==0 or any(v.shape!=x.shape or v.dtype!=np.float64 or not np.isfinite(v).all() for v in values):
        raise ValueError('matching full finite float64 fold modes required')
    if abs(np.linalg.norm(r)-1)>1e-10 or abs(l@r-1)>1e-10:
        raise ValueError('unit right mode and paired left mode required')
    return x,r,l


def design(record):
    if (record.get('passed')is not True or record.get('stage')!='qualified'
        or record.get('certificate_passed')is not False or record.get('passage_trajectories_evaluated')is not False
        or record.get('solver_history_complete')is not True or record.get('return_map')!='G=F^2'
        or record.get('active_parameter_units')!='log10T' or record.get('alpha')!=.083
        or not record.get('gates') or any(v is not True for v in record['gates'].values())):
        raise ValueError('qualified period-two log10T fold required before passage design')
    x,r,l=vectors(record)
    p,a,b,U=(float(record[k]) for k in ('parameter_c','a','b','U'))
    if not all(math.isfinite(v) for v in (p,a,b,U)) or not(a<0<b) or U!=.1/b or record['ghost_parameter_direction']!=-1:
        raise ValueError('audited b-positive gauge and negative ghost side required')
    prediction=predictor();rows=[]
    for i,eps in enumerate(EPSILONS):
        for name in ENTRANCES:
            u=-U if name=='full' else 0.
            start=x+u*r
            forecast=prediction(u,abs(a),b,abs(eps),U)
            if type(forecast['quadratic_G'])is not int or forecast['quadratic_G']<=0:
                raise ValueError('finite strict scalar exit required')
            rows.append(dict(id=f'e{i}-{name}',epsilon=eps,entrance=name,parameter=p+eps,
                u_in=u,u_out=U,**forecast,
                asymptotic_G=math.pi/math.sqrt(abs(a*b)*abs(eps))*(1. if name=='full' else .5),
                max_G=2*forecast['quadratic_G']+20,initial_state=digest(start),
                actual_initial_u=float((start-x)@l)))
    return dict(period=2,alpha=.083,coordinate='log10T',parameter_c=p,a=a,b=b,ab=a*b,U=U,
        gauge='u=l dot(x-x_c); ||r||2=1; l dot r=1; b>0; negative physical log10T offsets',
        modes={k:digest(record[k]) for k in ('x_c','r','l')},rows=rows,
        total_forecast_G=sum(r['quadratic_G'] for r in rows),total_cap_G=sum(r['max_G'] for r in rows),
        criterion=CRITERION,serial_execution=True)


def run_row(period_map,row,arrays,*,save,check=lambda:None):
    """Run one frozen row; callback returns the complete float64 G-state.

    save is called before the first model call and after every completed G step.
    The final crossing state and preceding state are included in state_trace.
    Invalid output and interruption retain the previous full valid prefix; the
    caller also saves the failed raw step when possible. No row is retried here.
    """
    x,r,l=vectors(arrays)
    U=float(row['u_out']);u0=float(row['u_in']);p=float(row['parameter'])
    if (not all(math.isfinite(v) for v in (U,u0,p)) or not U>0 or u0>U
        or type(row['max_G'])is not int or row['max_G']<=0
        or type(row['quadratic_G'])is not int or row['quadratic_G']<=0):
        raise ValueError('frozen finite row and positive integer cap/prediction required')
    state=x+u0*r
    if row['initial_state']!=digest(state):raise ValueError('frozen initial state differs')
    states=[state.copy()];us=[float((state-x)@l)]
    trans=[float(np.linalg.norm((state-x)-us[-1]*r))]
    def snapshot():
        return dict(state_trace=np.asarray(states),u=np.asarray(us),transverse=np.asarray(trans),
            recorded_G=len(states)-1,initial_states=states[0].copy(),states_at_stop=states[-1].copy())
    save(snapshot());count=None
    for g in range(1,row['max_G']+1):
        check();candidate=period_map(state.copy(),p)
        candidate=np.asarray(candidate)
        if candidate.shape!=x.shape or candidate.dtype!=np.float64 or not np.isfinite(candidate).all():
            raise ValueError('invalid full-state map output')
        state=candidate.copy();diff=state-x;u=float(diff@l)
        transverse=float(np.linalg.norm(diff-u*r))
        if not math.isfinite(u) or not math.isfinite(transverse):raise ValueError('nonfinite projection')
        states.append(state.copy());us.append(u);trans.append(transverse);save(snapshot())
        if u>U:
            count=g;break
    error=None if count is None else count-row['quadratic_G']
    result=dict(**row,status='censored' if count is None else 'observed',measured_G=count,
        measured_F=None if count is None else 2*count,recorded_G=len(states)-1,
        quadratic_error=error,arctan_error=None if count is None else count-row['arctan_G'],
        asymptotic_error=None if count is None else count-row['asymptotic_G'],
        primary_pass=count is not None and abs(error)<=1,
        pre_crossing_u=None if count is None else us[-2],
        crossing_u=None if count is None else us[-1])
    return result
