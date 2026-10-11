"""Saved gradient-ray quadratic sufficiency, all points consumed development."""
import math
import numpy as np

ALPHAS=(1/8,1/16,1/32,1/64,1/128)
FIT=(1,2)
CHECK=(0,3,4)
SETTINGS=dict(alphas=list(ALPHAS),fit_indices=list(FIT),check_indices=list(CHECK),gradient_error_limit=.01,remainder_error_limit=.1,remainder_relative_floor=1e-12)

def need(ok,message):
 if not ok:raise AssertionError(message)

def dot(a,b):return math.fsum(float(x)*float(y)for x,y in zip(a,b))
def norm(a):return math.sqrt(dot(a,a))
def summation(*vectors):return np.fromiter((math.fsum(float(v[j])for v in vectors)for j in range(len(vectors[0]))),dtype=np.float64,count=len(vectors[0]))

def diagnose(g,hp,gradients,tick=lambda:None):
 g=np.asarray(g);hp=np.asarray(hp);ys=[np.asarray(v)for v in gradients]
 need(g.ndim==1 and g.size>0 and hp.shape==g.shape and len(ys)==5 and all(v.shape==g.shape and v.dtype==np.float64 and np.isfinite(v).all()for v in [g,hp,*ys]),'complete finite binary64 five-ray gradients')
 ng=norm(g);need(ng>0,'nonzero original gradient');phi=dot(g,g)/2
 residuals=[];linear=[]
 for a,y in zip(ALPHAS,ys):
  linear.append(summation(g,a*hp));residuals.append(summation(y,-g,-a*hp));tick()
 coefficient=summation(residuals[FIT[0]]/(ALPHAS[FIT[0]]**2),residuals[FIT[1]]/(ALPHAS[FIT[1]]**2))*.5
 nc=norm(coefficient);rows=[];predictions=[]
 for i,(a,y,r,l)in enumerate(zip(ALPHAS,ys,residuals,linear)):
  pred=summation(g,a*hp,(a*a)*coefficient);predictions.append(pred);rem_model=(a*a)*coefficient
  nr=norm(r);ny=norm(y);den=max(ny,np.finfo(float).tiny);eligible=bool(nr>1e-12*ng)
  change=norm(summation(y,-g));change_eligible=bool(change>1e-12*ng)
  re=norm(summation(r,-rem_model))/nr if eligible else None
  le=norm(summation(y,-l))/den;qe=norm(summation(y,-pred))/den
  cosine=dot(r,coefficient)/(nr*nc) if nr>0 and nc>0 else None
  # ||g+aHp+r||² is reported with all cross terms; no cancellation term omitted.
  components=dict(base=dot(g,g),linear_cross=2*a*dot(g,hp),linear_square=a*a*dot(hp,hp),remainder_cross=2*dot(l,r),remainder_square=dot(r,r))
  reconstructed=math.fsum(components.values());actual=dot(y,y)
  need(abs(reconstructed-actual)<=1e-11*max(phi,actual,np.finfo(float).tiny),'complete squared norm decomposition')
  rows.append(dict(index=i,alpha=a,role='fit'if i in FIT else'check',gradient_norm=ny,actual_gradient_change_norm=change,gradient_change_eligible=change_eligible,quadratic_error_over_actual_gradient_change=norm(summation(y,-pred))/change if change_eligible else None,linear_error_over_actual_gradient_change=norm(summation(y,-l))/change if change_eligible else None,persistence_gradient_relative_error=change/den,remainder_norm=nr,remainder_over_initial_gradient=nr/ng,scaled_coefficient_norm=nr/(a*a),remainder_eligible=eligible,remainder_relative_error=re,gradient_relative_error=qe,linear_gradient_relative_error=le,quadratic_improves_full_gradient=bool(qe<le),remainder_coefficient_cosine=cosine,actual_merit_ratio=actual/(2*phi),linear_merit_ratio=dot(l,l)/(2*phi),quadratic_merit_ratio=dot(pred,pred)/(2*phi),squared_norm_components=components))
  tick()
 qualified=all(rows[i]['remainder_eligible']and rows[i]['gradient_relative_error']<=.01 and rows[i]['remainder_relative_error']<=.1 for i in CHECK)
 report=dict(settings=SETTINGS,rows=rows,initial_gradient_norm=ng,physical_action_norm=norm(hp),quadratic_coefficient_norm=nc,quadratic_sufficiency_qualified=bool(qualified),all_five_consumed_development=True,new_native_calls=0,scope='Fixed same-ray retrospective quadratic remainder sufficiency; no new state, step choice, holdout prediction, stationarity or utility claim.')
 return report,dict(coefficient=coefficient,residuals=np.stack(residuals),linear_gradients=np.stack(linear),quadratic_gradients=np.stack(predictions))
