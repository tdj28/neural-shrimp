"""Numerical minimum-norm projected Newton proposal at one fixed state.

Whitening uses the actual V.T V rather than assuming exact orthogonality.
The physical gradient/action and actual nonlinear merit remain caller gates.
"""
import math
import numpy as np
from scipy.linalg import solve_triangular


def need(condition,message):
    if not condition:raise ValueError(message)


def norm(x):return math.sqrt(math.fsum(float(v)*float(v)for v in np.asarray(x).ravel()))


def propose(V,W,g,*,target=.49,rank_rtol=1e-12,tick=lambda:None):
    V=np.asarray(V,dtype=float);W=np.asarray(W,dtype=float);g=np.asarray(g,dtype=float)
    need(V.ndim==2 and W.shape==V.shape and g.shape==(V.shape[0],)and 0<V.shape[1]<=V.shape[0],'same-state complete basis/actions/gradient')
    need(all(np.isfinite(a).all()for a in(V,W,g))and 0<target<.5 and 0<rank_rtol<1,'finite declared target and rank threshold')
    ng=norm(g);need(ng>0,'nonzero measured gradient')
    tick();G=V.T@V;R=np.linalg.cholesky(G).T;need(np.linalg.cond(R)<1e6,'bounded full-rank physical basis metric')
    tick();A=solve_triangular(R.T,W.T,lower=True,check_finite=True).T
    tick();U,s,Zt=np.linalg.svd(A,full_matrices=False);tick();need(np.all(s>s[0]*rank_rtol),'complete numerical action rank after physical whitening')
    ug=U.T@g;perp=g-U@ug;perp2=norm(perp)**2;s2=s*s
    def forcing(mu):return math.sqrt(max(0.,perp2+math.fsum(float(v)**2 for v in (mu/(s2+mu))*ug)))/ng
    minimum=forcing(0.);need(minimum<target,'requested forcing achievable by saved action span')
    high=float(s2[0]);need(high>0 and math.isfinite(high),'finite positive regularization bracket')
    expansions=[]
    for k in range(64):
        f=forcing(high);expansions.append(dict(index=k,ridge=high,forcing=f));tick()
        if f>target:break
        high*=2;need(math.isfinite(high),'finite fixed ridge bracket')
    else:raise ValueError('fixed64 regularization bracket exhausted')
    low=0.;rows=[]
    for k in range(64):
        mid=low+(high-low)/2;f=forcing(mid);rows.append(dict(index=k,ridge=mid,forcing=f))
        if f<=target:low=mid
        else:high=mid
        tick()
    def coefficients(mu):return solve_triangular(R,-Zt.T@((s/(s2+mu))*ug),lower=False,check_finite=True)
    c=coefficients(low);c0=coefficients(0.);p=V@c;hp=W@c;p0=V@c0;physical=norm(g+hp)/ng;slope=math.fsum(float(x)*float(y)for x,y in zip(g,hp,strict=True));np0=norm(p0);nprop=norm(p)
    need(all(np.isfinite(a).all()for a in(c,p,hp))and physical<=target+1e-11 and slope<0,'finite physically recomputed predicted forcing/descent')
    need(nprop<=np0*(1+1e-10),'regularized physical norm no larger than unregularized least squares')
    return dict(direction=p,predicted_action=hp,coefficients=c,summary=dict(target=target,ridge=low,upper_ridge=high,predicted_forcing=physical,minimum_span_forcing=minimum,gradient_merit_slope=slope,direction_norm=nprop,unregularized_norm=np0,norm_ratio=nprop/np0,rank=len(s),smallest_singular=float(s[-1]),largest_singular=float(s[0]),gram_identity_error=float(np.max(np.abs(G-np.eye(G.shape[0])))),bracket=expansions,bisection=rows,scope='Numerical saved-span minimum-norm proposal at the original state. No fresh physical action, nonlinear acceptance or transferred-state authority follows.'))
