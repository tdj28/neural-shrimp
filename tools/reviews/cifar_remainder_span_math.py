"""Numerical same-start Hessian-image diagnostics; no new derivatives or steps.

For saved gradient trials b(alpha)=(g_alpha-g-alpha Hp)/alpha**2, fit Wc=-b
in the declared numerical action image. Vc is a proposed coefficient for an
alpha**2 path correction; it is not a validated accelerated gradient response.
"""
import math
import numpy as np


def need(ok,message):
 if not ok:raise ValueError(message)

def dot(a,b):return math.fsum(float(x)*float(y)for x,y in zip(a,b,strict=True))
def norm(a):return math.sqrt(max(0.,dot(a,a)))


def diagnose(V,W,x,g,p,hp,origin,trials,*,rank_rtol=1e-12,tick=lambda:None):
 V,W=np.asarray(V,dtype=float),np.asarray(W,dtype=float);x,g,p,hp,origin=[np.asarray(a,dtype=float)for a in(x,g,p,hp,origin)]
 need(V.ndim==2 and W.shape==V.shape and 0<V.shape[1]<=V.shape[0]and all(a.shape==(V.shape[0],)for a in(x,g,p,hp,origin)),'same-state complete dimensions')
 need(all(np.isfinite(a).all()for a in(V,W,x,g,p,hp,origin))and 0<rank_rtol<1 and norm(g)>0 and norm(p)>0,'finite nonzero same-state data')
 need(1<=len(trials)<=13 and all(0<t['alpha']<=1 for t in trials)and all(trials[i]['alpha']>trials[i+1]['alpha']for i in range(len(trials)-1)),'fixed complete descending saved alphas')
 tick();U,s,Zt=np.linalg.svd(W,full_matrices=False);tick();need(np.isfinite(s).all()and s[0]>0,'finite nonzero action image');keep=s>s[0]*rank_rtol;rank=int(sum(keep));need(rank>0,'nonempty declared numerical image');U=U[:,keep];Zt=Zt[keep];sk=s[keep]
 scale=max(1.,norm(origin));ng=norm(g);np_=norm(p);rows=[];vectors=[];remainders=[]
 for j,t in enumerate(trials):
  alpha=float(t['alpha']);state=np.asarray(t['state'],dtype=float);actual=np.asarray(t['gradient'],dtype=float)
  need(state.shape==actual.shape==g.shape and np.isfinite(actual).all()and np.array_equal(state,x+alpha*p),'literal saved current-point trial coordinates')
  b=(actual-g-alpha*hp)/(alpha*alpha);ub=U.T@b;c=-Zt.T@(ub/sk);action=W@c;acceleration=V@c;residual=b+action;nb=norm(b);nr=norm(residual);tick()
  need(all(np.isfinite(a).all()for a in(b,c,action,acceleration,residual)),'finite complete fitted vectors')
  projection=-U@ub;err=norm(action-projection)/max(nb,norm(action),1e-300);orth=norm(U.T@residual)/max(nb,1e-300);need(err<1e-9 and orth<1e-9,'actual assembled image projection and orthogonality')
  curved=state+alpha*alpha*acceleration;formal=actual+alpha*alpha*action
  row=dict(index=j,alpha=alpha,remainder_coefficient_norm=nb,irreducible_norm=nr,irreducible_fraction=None if nb==0 else nr/nb,represented_squared_fraction=None if nb==0 else 1-nr*nr/(nb*nb),coefficient_norm=norm(c),acceleration_norm=norm(acceleration),path_correction_to_primary_step=alpha*norm(acceleration)/np_,curved_path_relative_origin_distance=norm(curved-origin)/scale,curved_path_inside_original_trust=norm(curved-origin)/scale<=.05,assembled_projection_error=err,projection_orthogonality_error=orth,formal_gradient_merit_ratio=dot(formal,formal)/(ng*ng),actual_straight_gradient_merit_ratio=dot(actual,actual)/(ng*ng))
  rows.append(row);vectors.append(dict(coefficients=c,acceleration=acceleration,predicted_action=action));remainders.append(b)
 reference=len(rows)-1;ref=remainders[reference];refaction=vectors[reference]['predicted_action'];refacc=vectors[reference]['acceleration'];cross=[]
 for row,t,b in zip(rows,trials,remainders,strict=True):
  a=row['alpha'];actual=np.asarray(t['gradient']);formal=actual+a*a*refaction;cross.append(dict(alpha=a,reference_alpha=rows[reference]['alpha'],coefficient_relative_drift=norm(b-ref)/max(norm(ref),1e-300),residual_fraction=None if norm(b)==0 else norm(b+refaction)/norm(b),formal_gradient_merit_ratio=dot(formal,formal)/(ng*ng),curved_path_relative_origin_distance=norm(x+a*p+a*a*refacc-origin)/scale))
 return dict(summary=dict(numerical_rank=rank,columns=V.shape[1],rank_rtol=rank_rtol,largest_singular=float(s[0]),smallest_retained_singular=float(sk[-1]),rows=rows,fixed_reference_index=reference,fixed_reference_alpha=rows[reference]['alpha'],reference_cross_checks=cross,scope='Consumed same-current022 binary64 Hessian-image representability. Formal gradient values omit change of Hessian along the curved path; no actual accelerated merit, descent, trust acceptance, stationarity or new action authority.'),vectors=vectors)
