"""Saved same-state gradient-merit nonlinearity diagnostics, no native calls.

All quantities are binary64 descriptive diagnostics. No derivative bound,
new trust radius, stationarity, or correction acceptance is inferred here.
"""
import math
import numpy as np


def need(x,s):
 if not x:raise ValueError(s)

def dot(a,b):return math.fsum(float(x)*float(y)for x,y in zip(a,b,strict=True))
def norm(a):return math.sqrt(max(0.,dot(a,a)))
def same(a,b):return a.shape==b.shape and np.array_equal(a,b)


def analyze(origin,x,g,p,hp,trials):
 need(x.ndim==1 and all(a.shape==x.shape and np.isfinite(a).all()for a in(origin,g,p,hp)),'all complete same-state finite vectors')
 need(1<=len(trials)<=13,'bounded actual dyadic trial history')
 scale=max(1.,norm(origin));radius=.05*scale;offset=x-origin;gap=radius*radius-dot(offset,offset);pp=dot(p,p);bp=dot(offset,p)
 need(gap>=0 and pp>0 and norm(g)>0,'positive direction and original global trust authority')
 disc=math.sqrt(bp*bp+pp*gap)
 # Stable positive root of ||x-origin+alpha p||^2=radius^2.
 trust_alpha=(gap/(disc+bp)if bp>=0 and disc+bp>0 else(-bp+disc)/pp)
 phi=dot(g,g)/2;gh=dot(g,hp);hh=dot(hp,hp);need(hh>0 and gh<0,'actual descent action')
 rows=[];chosen=None;previous=None
 for i,item in enumerate(trials):
  need(chosen is None,'no evaluation after first acceptance')
  alpha=2.**(-i);state=x+alpha*p;distance=norm(state-origin)/scale;inside=distance<=.05;linear=g+alpha*hp;phi_linear=dot(linear,linear)/2;predicted=phi-phi_linear
  row=dict(index=i,alpha=alpha,within_trust=inside,relative_origin_distance=distance,relative_step=alpha*norm(p)/scale,linear_merit_ratio=phi_linear/phi,predicted_merit_decrease=predicted)
  if not inside:need(item is None,'trust-skipped trial must not have a native gradient');row.update(accepted=False,nonlinear_observed=False)
  else:
   need(item is not None,'every admitted trial has a saved actual gradient');actual_state,newg=item;need(same(actual_state,state)and newg.shape==x.shape and np.isfinite(newg).all(),'literal actual candidate and gradient')
   error=newg-linear;ne=norm(error);cross=dot(linear,error);square=dot(error,error)/2;actual=dot(newg,newg)/2;decomposition=phi_linear+cross+square
   need(abs(decomposition-actual)<=1e-12*max(phi,phi_linear,actual,abs(cross),square,1e-300),'actual merit decomposition on original magnitude scale')
   accepted=actual<=phi+1e-4*alpha*gh
   row.update(nonlinear_observed=True,actual_merit_ratio=actual/phi,actual_merit_decrease=phi-actual,actual_to_predicted_decrease=(phi-actual)/predicted if predicted>0 else None,nonlinear_norm=ne,nonlinear_relative_to_initial_gradient=ne/norm(g),nonlinear_relative_to_linear_change=ne/(alpha*math.sqrt(hh)),quadratic_remainder_coefficient=ne/(alpha*alpha),linear_error_cross=cross,nonlinear_square_merit=square,merit_decomposition_error=decomposition-actual,nonlinear_action_projection=dot(error,hp)/hh,accepted=accepted)
   row['successive_remainder_order']=math.log(ne/previous[1])/math.log(alpha/previous[0])if previous is not None and ne>0 and previous[1]>0 else None;previous=(alpha,ne)
   if accepted:chosen=i
  rows.append(row)
 need(chosen is not None or len(trials)==13,'complete actual acceptance history')
 return dict(rows=rows,initial_merit=phi,initial_gradient_norm=norm(g),action_norm=math.sqrt(hh),direction_relative_norm=norm(p)/scale,actual_slope=gh,true_forcing=norm(g+hp)/norm(g),numerical_trust_alpha=trust_alpha,linear_line_optimum=-gh/hh,accepted_index=chosen,accepted_alpha=None if chosen is None else rows[chosen]['alpha'],scope='Saved binary64 same-state nonlinear gradient remainder and original trust/Armijo decomposition. No new derivative or step; empirical remainder orders are descriptive, not rigorous curvature bounds.')
