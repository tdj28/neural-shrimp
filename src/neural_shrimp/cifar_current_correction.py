"""One true-action-qualified numerical Newton correction in the original trust ball."""
import numpy as np
from .cifar_reorthogonalized import vector,SETTINGS as KRYLOV_SETTINGS
SETTINGS=dict(columns=512,forcing_limit=.5,stationarity=1e-10,armijo=1e-4,
              trust_relative=.05,backtracks=13)


def direction_gate(g,predicted,actual,report):
    g=vector(g);predicted=vector(predicted,len(g));actual=vector(actual,len(g))
    ng=float(np.linalg.norm(g))
    if not np.isfinite(ng) or ng<=0:raise ValueError('positive finite current gradient')
    forcing=float(np.linalg.norm(g+actual))/ng
    discrepancy=float(np.linalg.norm(predicted-actual))/max(ng,float(np.linalg.norm(actual)),1e-12)
    slope=float(np.dot(g,actual))
    full=bool(report['completed_fixed_extension']and report['total_columns']==SETTINGS['columns']and report['status']=='maximum_columns'and report['orthogonality_qualified'])
    passed=bool(full and np.isfinite([forcing,discrepancy,slope]).all()and discrepancy<=KRYLOV_SETTINGS['linear_action_relative']and forcing<=SETTINGS['forcing_limit']and slope<0)
    return dict(passed=passed,full_basis_qualified=full,true_forcing=forcing,
                physical_action_relative_error=discrepancy,slope=slope)


def correct(initial,g,origin,p,predicted,actual,report,gradient,*,event=lambda r:None):
    x=vector(initial).copy();n=len(x);g=vector(g,n).copy();origin=vector(origin,n)
    p=vector(p,n);scale=max(1.,float(np.linalg.norm(origin)))
    if float(np.linalg.norm(x-origin))/scale>SETTINGS['trust_relative']:raise ValueError('current point outside original global trust ball')
    gate=direction_gate(g,predicted,actual,report);trials=[];accepted=None
    before=float(np.linalg.norm(g))/max(1.,float(np.linalg.norm(x)))
    status='direction_unqualified';event(dict(kind='current_direction_gate',**gate))
    if gate['passed']:
        status='backtracking_failed';phi=.5*float(np.dot(g,g))
        for j in range(SETTINGS['backtracks']):
            alpha=2.**(-j);proposal=x+alpha*p
            distance=float(np.linalg.norm(proposal-origin))/scale
            row=dict(index=j,alpha=alpha,relative_origin_distance=distance,within_trust=distance<=SETTINGS['trust_relative'],accepted=False)
            if row['within_trust']:
                newg=vector(gradient(proposal.copy()),n);merit=.5*float(np.dot(newg,newg));bound=phi+SETTINGS['armijo']*alpha*gate['slope']
                row.update(merit=merit,armijo_bound=bound,accepted=bool(np.isfinite(merit)and merit<=bound))
            trials.append(row);event(dict(kind='current_merit_trial',**row))
            if row['accepted']:
                x,g=proposal,newg;accepted=j;status='numerical_correction_accepted';break
    relative=float(np.linalg.norm(g))/max(1.,float(np.linalg.norm(x)))
    return x,g,dict(status=status,direction_gate=gate,trials=trials,accepted_index=accepted,
        direction_applied=accepted is not None,relative_gradient_before=before,relative_gradient=relative,
        gradient_norm=float(np.linalg.norm(g)),relative_origin_distance=float(np.linalg.norm(x-origin))/scale,
        stationarity_qualified=relative<=SETTINGS['stationarity'],
        scope='One finite-precision correction, fresh current-state Hessian, original global trust ball. No exact stationary root, cycle or training benefit follows.')
