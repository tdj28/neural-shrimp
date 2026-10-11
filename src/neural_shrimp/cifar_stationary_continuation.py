"""Four fresh inexact Newton units after an audited numerical merit step.

Source-derived from cifar_stationary.stationary at f967ff94. Deliberate changes:
forcing limit .9 and caller-authenticated global origin retained across runs.
Every actual trial still requires measured gradient-squared Armijo decrease.
No Hessian action from the old state is transferred to the new state.
"""
import numpy as np
from scipy.sparse.linalg import LinearOperator,minres
from .cifar_amplitude import vector
from .cifar_stationary import qualified_vector
SETTINGS=dict(max_steps=4,maxiter=256,rtol=1e-12,forcing_limit=.9,
              stationarity=1e-10,armijo=1e-4,trust_relative=.05)


def stationary_from_seed(initial,initial_gradient,origin,gradient,hvp,*,event=lambda x:None):
    """Fixed four Newton units; terminate at first unqualified linear/trial step."""
    x=vector(initial).copy();n=len(x);origin=qualified_vector(origin,n).copy();scale=max(1.,float(np.linalg.norm(origin)))
    if float(np.linalg.norm(x-origin))/scale>SETTINGS['trust_relative']:raise ValueError('initial point within original global trust ball')
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
                    scope='Finite precision continuation within the original midpoint trust ball only. No exact root, same-branch or cycle claim.')
