"""One bounded inexact-Newton direction diagnostic, never an orbit acceptor."""
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres

ALPHAS=(1.,.5,.25,.125)
CLOSURE_TOLERANCE=1e-8
PRIMITIVE_SEPARATION=1e-7


def finite_vector(value,shape=None):
    if np.iscomplexobj(value):raise ValueError('real vectors required')
    a=np.asarray(value,dtype=np.float64)
    if a.ndim!=1 or (shape is not None and a.shape!=shape) or not np.isfinite(a).all():
        raise ValueError('expected finite real vector of matching shape')
    return a


def positive_integer(value,name):
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,np.integer)) or value<=0:
        raise ValueError(name+' must be a positive integer')


def check_deadline(deadline):
    if deadline is not None and time.monotonic()>=deadline:
        raise TimeoutError('linear diagnostic wall budget exhausted')


def solve_direction(action,residual,*,restart=50,cycles=4,rtol=1e-10,forcing_limit=.5,
                    max_matvec=205,deadline=None,event=None,save_direction=None):
    """Save estimated inner history and independently evaluate A*p+R.

    SciPy callback_type='pr_norm' keeps maxiter counting restart cycles. The
    return code is retained even if a separately declared forcing test passes.
    No preconditioner, additional solve, correction application or eigensolver.
    """
    for name,value in [('restart',restart),('cycles',cycles),('max_matvec',max_matvec)]:positive_integer(value,name)
    if not np.isfinite(rtol) or not 0<rtol<1 or not np.isfinite(forcing_limit) or not 0<forcing_limit<1:
        raise ValueError('relative tolerances must be finite and lie in (0,1)')
    residual=finite_vector(residual);denominator=float(np.linalg.norm(residual))
    if not np.isfinite(denominator) or denominator<=0:raise ValueError('finite nonzero residual required')
    calls=0;history=[];started=time.monotonic()
    def emit(row):
        if event is not None:event(row)
    def bounded(v):
        nonlocal calls
        check_deadline(deadline)
        if calls>=max_matvec:raise RuntimeError('linear diagnostic matvec budget exhausted')
        vector=finite_vector(v,residual.shape);calls+=1;t0=time.monotonic()
        result=finite_vector(action(vector),residual.shape)
        emit(dict(kind='matvec',index=calls,seconds=time.monotonic()-t0))
        return result
    def callback(value):
        value=float(value)
        if not np.isfinite(value) or value<0:raise ValueError('nonfinite/negative GMRES callback residual')
        row=dict(kind='gmres_inner',iteration=len(history)+1,estimated_relative_residual=value)
        history.append(value);emit(row)
    operator=LinearOperator((len(residual),len(residual)),matvec=bounded,dtype=np.float64)
    check_deadline(deadline)
    step,info=gmres(operator,-residual,rtol=rtol,atol=0.,restart=restart,maxiter=cycles,
                    callback=callback,callback_type='pr_norm')
    if save_direction is not None:save_direction(np.asarray(step).copy(),int(info))
    step=finite_vector(step,residual.shape)
    if not np.isfinite(np.linalg.norm(step)):raise ValueError('nonfinite correction norm')
    applied=bounded(step)  # Independent true, unpreconditioned residual check.
    true_residual=finite_vector(applied+residual,residual.shape)
    true_norm=float(np.linalg.norm(true_residual));ratio=true_norm/denominator
    slope=float(residual@applied)
    if not all(np.isfinite(v) for v in (true_norm,ratio,slope)):
        raise ValueError('nonfinite true residual or merit derivative')
    result=dict(gmres_info=int(info),gmres_reported_converged=bool(info==0),
        requested_linear_tolerance=rtol,requested_tolerance_met=bool(true_norm<=rtol*denominator),
        estimated_relative_residual_history=history,inner_iterations=len(history),
        matvec_calls=calls,max_matvec=max_matvec,restart=restart,cycles=cycles,
        true_linear_residual_norm=true_norm,rhs_norm=denominator,true_forcing_ratio=ratio,
        forcing_limit=forcing_limit,forcing_test_pass=bool(ratio<=forcing_limit),
        merit_directional_derivative=slope,descent_direction=bool(slope<0),
        seconds=time.monotonic()-started,
        scope='diagnostic direction only; GMRES status retained; no Newton step applied or orbit accepted')
    return step,applied,true_residual,result


def evaluate_trials(theta,residual,step,linear_result,evaluate,*,alphas=ALPHAS,armijo=1e-4,
                    deadline=None,save_state=None,event=None):
    """Four fixed independent trials from the same original state.

    Merit phi=||R||^2/2 uses the absolute residual, avoiding denominator changes
    masquerading as improvement. Candidate qualification requires true forcing,
    descent and Armijo. Even passing closure/primitive screens accepts no orbit.
    """
    theta=finite_vector(theta);residual=finite_vector(residual,theta.shape);step=finite_vector(step,theta.shape)
    if not all(np.isfinite(np.linalg.norm(v)) for v in (theta,step)):
        raise ValueError('nonfinite state/correction norm')
    if tuple(alphas)!=ALPHAS:raise ValueError('only the four declared damping factors are permitted')
    if not np.isfinite(armijo) or not 0<armijo<1:raise ValueError('Armijo constant must lie in (0,1)')
    slope=float(linear_result['merit_directional_derivative'])
    if not np.isfinite(slope):raise ValueError('finite merit derivative required')
    phi0=.5*float(residual@residual)
    if not np.isfinite(phi0) or phi0<=0:raise ValueError('finite positive initial merit required')
    rows=[]
    for index,alpha in enumerate(alphas):
        check_deadline(deadline)
        with np.errstate(over='ignore',invalid='ignore'):trial=theta+alpha*step
        if save_state is not None:save_state(index,trial)
        row=dict(index=index,alpha=alpha,accepted_orbit=False,applied_to_continuation=False,
                 relative_step_norm=float(alpha*np.linalg.norm(step)/max(1.,np.linalg.norm(theta))),
                 armijo_upper=phi0+armijo*alpha*slope)
        if np.isfinite(trial).all():
            details=evaluate(trial,index)
            r=np.asarray(details['residual'],dtype=np.float64)
            if r.shape!=theta.shape:raise ValueError('trial residual shape mismatch')
            separation=float(details['relative_separation'])
            finite=bool(np.isfinite(r).all() and np.isfinite(separation))
            if finite:
                norm=float(np.linalg.norm(r));phi=.5*float(r@r)
                trial_norm=float(np.linalg.norm(trial));relative=norm/max(1.,trial_norm)
                finite=bool(np.isfinite(norm) and np.isfinite(phi) and np.isfinite(relative) and np.isfinite(trial_norm))
            if finite:
                row.update(residual_norm=norm,relative_residual=relative,merit=phi,
                    merit_ratio=phi/phi0,relative_separation=separation,
                    closure_screen=bool(relative<CLOSURE_TOLERANCE),
                    primitive_screen=bool(separation>=PRIMITIVE_SEPARATION),
                    merit_decreased=bool(phi<phi0),armijo_pass=bool(phi<=row['armijo_upper']))
        else:finite=False
        row['finite']=finite
        row['inexact_step_candidate']=bool(finite and linear_result['forcing_test_pass'] and slope<0 and row['armijo_pass'])
        row['scope']='local trial only; closure and primitive screens unchanged; no orbit or stability acceptance'
        rows.append(row)
        if event is not None:event(dict(kind='trial_result',**row))
    return rows
