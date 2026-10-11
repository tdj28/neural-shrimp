"""Reversal-odd/even response decomposition; retrospective, not a forecast.

For symmetric GD Jacobians, the reversed product is P.T. Reversal preserves
singular values; its even part may still depend on other order permutations.
"""
import math
import numpy as np
HORIZONS=(4,16)

def finite(x):
    a=np.asarray(x,dtype=np.float64)
    if a.ndim!=1 or not np.isfinite(a).all():raise ValueError('finite flat vector')
    return a

def reverse_product(states,u,eta,h,hvp,save=lambda *args:None):
    states=np.asarray(states,dtype=np.float64);u=finite(u)
    if h not in HORIZONS or states.ndim!=2 or states.shape[0]<h or states.shape[1]!=len(u)or not np.isfinite(states).all()or not math.isfinite(eta)or eta<=0:raise ValueError('fixed actual reverse product')
    trace=[u.copy()];indices=[]
    try:
        for index in range(h-1,-1,-1):
            action=finite(hvp(states[index],trace[-1]))
            if action.shape!=u.shape:raise ValueError('same action dimension')
            trace.append(finite(trace[-1]-eta*action));indices.append(index)
    finally:save(np.asarray(trace),indices)
    return trace[-1]

def decompose(u,ordered,reversed_,frozen,ce_gradient):
    u,p,r,f,g=[finite(x)for x in(u,ordered,reversed_,frozen,ce_gradient)]
    if not all(x.shape==u.shape for x in(p,r,f,g)):raise ValueError('same actual vector space')
    a=(p-r)/2;b=(p+r)/2-f;e=p-f
    na,nb,ne=[float(np.linalg.norm(x))for x in(a,b,e)];inner=float(np.dot(a,b));rhs=na*na+nb*nb+2*inner
    base=max(float(np.linalg.norm(p)),float(np.linalg.norm(r)),float(np.linalg.norm(f)),1e-12)
    residual=float(np.linalg.norm(e-a-b));squared=abs(ne*ne-rhs)/max(ne*ne,na*na+nb*nb,base*base)
    scale=max(float(np.linalg.norm(u))*max(float(np.linalg.norm(p)),float(np.linalg.norm(r))),1e-12)
    skew=abs(float(np.dot(u,a)))/scale
    out=dict(odd_norm=na,even_remainder_norm=nb,frozen_error_norm=ne,odd_to_error=na/max(ne,1e-12),even_to_error=nb/max(ne,1e-12),odd_even_inner_product=inner,squared_error=ne*ne,squared_odd=na*na,squared_even=nb*nb,twice_cross_term=2*inner,squared_identity_scaled_error=squared,identity_scale=base,vector_identity_residual=residual,odd_even_cosine=None if na*nb==0 else inner/(na*nb),skew_quadratic_relative_error=skew,algebra_qualified=bool(squared<=1e-12 and skew<=1e-8 and residual/base<=1e-12),ce_projections={k:float(np.dot(g,x))for k,x in [('ordered',p),('reversed',r),('frozen',f),('odd',a),('even_remainder',b),('error',e)]},scope='Reversal-odd contribution and reversal-even remainder. Neither ratio is a percentage; even part can depend on order. No operator-norm or training-utility claim.')
    return out,a,b

def reciprocal(directions,ordered,reversed_):
    u,p,r=[np.asarray(x,dtype=np.float64)for x in(directions,ordered,reversed_)]
    if not(u.shape==p.shape==r.shape and u.ndim==2 and u.shape[0]==2 and all(np.isfinite(x).all()for x in(u,p,r))):raise ValueError('two finite fixed directions')
    rows=[]
    for i in range(2):
        for j in range(2):
            lhs=float(np.dot(u[i],p[j]));rhs=float(np.dot(r[i],u[j]));scale=max(float(np.linalg.norm(u[i])*np.linalg.norm(p[j])),float(np.linalg.norm(r[i])*np.linalg.norm(u[j])),1e-12)
            rows.append(dict(i=i,j=j,lhs=lhs,rhs=rhs,relative_error=abs(lhs-rhs)/scale))
    return dict(passed=all(x['relative_error']<=1e-8 for x in rows),rows=rows)

def profile_forecast(rows):
    if len(rows)!=2 or [r['arm']for r in rows]!=[8,256]:raise ValueError('two fixed profile arms')
    total=20.
    for row in rows:
        v=row['times']
        if len(v)!=2 or not all(type(x)is float and math.isfinite(x)and x>0 for x in v):raise ValueError('two finite measured HVP costs')
        total+=4*40*max(v)
    return total
