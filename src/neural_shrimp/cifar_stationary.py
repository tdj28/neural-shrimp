"""Bounded stationary Newton proposal and separately qualified Hessian mode.

Numerical seeding only. Neither a stationary certificate nor an optimizer
cycle is implied. The caller authenticates the objective and meters every
native gradient/Hessian action and retains all attempted inputs/returns.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator,minres
from .cifar_amplitude import vector

SETTINGS=dict(max_steps=4,maxiter=256,rtol=1e-12,forcing_limit=.5,
              stationarity=1e-10,armijo=1e-4,trust_relative=.05,
              power_steps=32,eigen_residual=1e-8)


def qualified_vector(value,n):
    v=vector(value)
    if v.shape!=(n,):raise ValueError('full stationary coordinate vector')
    return v


def stationary(initial,initial_gradient,gradient,hvp,*,event=lambda x:None):
    """Fixed four Newton units; terminate at first unqualified linear/trial step."""
    x=vector(initial).copy();n=len(x);origin=x.copy();scale=max(1.,float(np.linalg.norm(origin)))
    g=qualified_vector(initial_gradient,n).copy();steps=[];status='maximum_steps'
    for k in range(SETTINGS['max_steps']):
        relative=float(np.linalg.norm(g))/max(1.,float(np.linalg.norm(x)))
        if relative<=SETTINGS['stationarity']:status='stationarity_qualified';break
        iterations=[];calls=0
        def action(v):
            nonlocal calls
            calls+=1
            if calls>SETTINGS['maxiter']:raise ValueError('fixed stationary linear action cap')
            return qualified_vector(hvp(x,qualified_vector(v,n)),n)
        def callback(v):
            row=dict(kind='stationary_iteration',step=k,iteration=len(iterations)+1,norm=float(np.linalg.norm(v)))
            iterations.append(row);event(row)
        A=LinearOperator((n,n),matvec=action,dtype=np.float64)
        # SciPy x0=None is the exact zero initial vector; no old-state warm start.
        p,info=minres(A,-g,x0=None,rtol=SETTINGS['rtol'],maxiter=SETTINGS['maxiter'],callback=callback,check=False)
        p=qualified_vector(p,n);hp=qualified_vector(hvp(x,p),n)
        norm=float(np.linalg.norm(g));forcing=float(np.linalg.norm(g+hp))/norm
        slope=float(np.dot(g,hp));phi=.5*norm*norm
        row=dict(step=k,iterations=len(iterations),linear_actions=calls,info=int(info),forcing=forcing,slope=slope,trials=[],accepted_index=None)
        steps.append(row);event(dict(kind='stationary_linear',**{a:b for a,b in row.items()if a!='trials'}))
        if not(np.isfinite([forcing,slope]).all()and forcing<=SETTINGS['forcing_limit']and slope<0):status='true_forcing_failed';break
        accepted=False
        for j in range(13):
            alpha=2.**(-j);proposal=x+alpha*p
            distance=float(np.linalg.norm(proposal-origin))/scale
            trial=dict(index=j,alpha=alpha,relative_origin_distance=distance,within_trust=distance<=SETTINGS['trust_relative'])
            if trial['within_trust']:
                newg=qualified_vector(gradient(proposal),n);merit=.5*float(np.dot(newg,newg))
                bound=phi+SETTINGS['armijo']*alpha*slope
                trial.update(merit=merit,armijo_bound=bound,accepted=bool(np.isfinite(merit)and merit<=bound))
            else:trial['accepted']=False
            row['trials'].append(trial);event(dict(kind='stationary_trial',step=k,**trial))
            if trial['accepted']:
                x,g=proposal,newg;row['accepted_index']=j;accepted=True;break
        if not accepted:status='backtracking_failed';break
    relative=float(np.linalg.norm(g))/max(1.,float(np.linalg.norm(x)))
    passed=relative<=SETTINGS['stationarity']
    if passed:status='stationarity_qualified'
    return x,g,dict(status=status,passed=passed,relative_gradient=relative,steps=steps,
                    relative_origin_distance=float(np.linalg.norm(x-origin))/scale,
                    scope='Finite precision stationary candidate only; no exact root, same-branch or cycle claim.')


def positive_mode(point,seed,hvp,*,event=lambda x:None):
    x=vector(point);n=len(x);q=qualified_vector(seed,n).copy();norm=float(np.linalg.norm(q))
    if not np.isfinite(norm)or norm==0:raise ValueError('nonzero fixed phase-mode proposal')
    q/=norm
    for k in range(SETTINGS['power_steps']):
        hq=qualified_vector(hvp(x,q),n);norm=float(np.linalg.norm(hq))
        if not np.isfinite(norm)or norm==0:raise ValueError('nonzero finite Hessian power return')
        q=hq/norm;event(dict(kind='stationary_power',iteration=k+1,action_norm=norm))
    hq=qualified_vector(hvp(x,q),n);tau=float(np.dot(q,hq))
    residual=float(np.linalg.norm(hq-tau*q))/max(1.,abs(tau))
    passed=bool(np.isfinite([tau,residual]).all()and tau>0 and residual<=SETTINGS['eigen_residual'])
    return q,hq,dict(passed=passed,tau=tau,eta_candidate=2/tau if tau>0 else None,relative_eigen_residual=residual,
                     scope='Numerical positive Hessian eigenpair at the candidate only; new eta varies the optimizer and no cycle follows automatically.')
