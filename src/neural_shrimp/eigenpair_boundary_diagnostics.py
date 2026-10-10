"""Floating saved-shooting fold indicators; not certified normal-form data."""
import math
import numpy as np
from .rigorous_eigenpair_march import decode
from .rigorous_eigenpair_locus import require


def fold_indicator(chart,observe=lambda k,v:None):
    require(chart['sigma']==1,'fold indicator only for plus-one locus')
    n=chart['dimension'];p=chart['period'];m=2*n;size=m*p+2
    J=np.array([[float(decode(v).mid()) for v in row] for row in chart['center_jacobian']])
    z=np.array([float(decode(v).mid()) for v in chart['center']])
    require(J.shape==(size,size) and z.shape==(size,) and np.isfinite(J).all() and np.isfinite(z).all(),'finite native saved arrays')
    M=np.eye(n);second=np.zeros(n);rates=np.zeros((n,2));mode_errors=[]
    for k in range(p):
        a=k*m;local=J[a:a+n,a:a+n].copy()
        if p==1:local+=np.eye(n)
        qi=z[a+n:a+m];qnext=z[((k+1)%p)*m+n:((k+1)%p)*m+m]
        mode_errors.append(float(np.max(abs(local@qi-qnext))))
        # This block is D_x(DF*q_i), so contraction uses the retained q_i.
        second=local@second+J[a+n:a+m,a:a+n]@qi
        rates=local@rates+J[a:a+n,-2:]
        M=local@M
    q=z[n:2*n];A=M-np.eye(n);border=np.block([[A.T,q[:,None]],[q[None,:],np.zeros((1,1))]])
    inverse=np.linalg.solve(border,np.eye(n+1))
    finite=lambda a:[[float(v) if math.isfinite(v) else None for v in row] for row in a]
    observe('fold_indicator_linear_return',{'monodromy':M.tolist(),'border':border.tolist(),'inverse':finite(inverse),'mode0':q.tolist(),'second_lifted_contraction':second.tolist(),'rate_derivatives':rates.tolist(),'phase_mode_residuals':mode_errors,'scope':'Floating returned inverse retained before checks; saved shooting phases, not freshly acquired actual orbit derivatives.'})
    require(np.isfinite(inverse).all(),'finite diagnostic inverse')
    w=inverse[:n,-1];alpha=float(inverse[-1,-1]);c=float(w@second);a=w@rates
    require(math.isfinite(c) and np.isfinite(a).all(),'finite coefficient proxies')
    fnorm=lambda x:float(np.sqrt(np.sum(abs(x)**2)))
    residual=float(np.max(abs(border@inverse-np.eye(n+1))))
    center_res=max(abs(float(decode(v[0]).mid())) for v in chart['center_residual'])
    return {'a_actual_rates':a.tolist(),'second_derivative_contraction':c,'quadratic_coefficient':c/2,'left':w.tolist(),'left_border_alpha':alpha,'left_mode_pairing':float(w@q),'critical_mode_residual':float(np.max(abs(A@q))),'phase_mode_residual_max':max(mode_errors),'shooting_residual_max':center_res,'inverse_residual_max':residual,'border_frobenius_condition_proxy':fnorm(border)*fnorm(inverse),'diagnostic_residuals_small':bool(max(center_res,max(mode_errors),residual)<1e-8),'scope':'Floating saved-shooting-point coefficient proxy only. c=w^T D²R[q,q], quadratic coefficient c/2. Approximate native closure/mode consistency and border residuals are exposed. No validated coefficient sign, cusp, critical simplicity, generic fold, actual finite-orbit derivative or parameter-curve association claim.'}
