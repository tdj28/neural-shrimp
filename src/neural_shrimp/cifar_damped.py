"""Two bounded inexact-Newton directions with retained fixed backtracking."""
import numpy as np
from .cifar_linear_diagnostic import solve_direction,finite_vector
ALPHAS=tuple(2.**(-k)for k in range(13))
def trial_metrics(theta,one,two,alpha,phi0,slope):
 theta=finite_vector(theta);one=finite_vector(one,theta.shape);two=finite_vector(two,theta.shape)
 if alpha not in ALPHAS or not np.isfinite(phi0)or phi0<=0 or not np.isfinite(slope)or slope>=0:raise ValueError('fixed descent trial')
 r=two-theta;scale=max(1.,float(np.linalg.norm(theta)));merit=.5*float(r@r);relative=float(np.linalg.norm(r))/scale;sep=float(np.linalg.norm(one-theta))/scale;upper=phi0+1e-4*alpha*slope
 if not all(np.isfinite(x)for x in (merit,relative,sep,upper)):raise ValueError('finite trial metrics')
 return dict(alpha=alpha,merit=merit,merit_ratio=merit/phi0,armijo_upper=upper,armijo_pass=merit<=upper,strict_merit_decrease=merit<phi0,relative_residual=relative,relative_separation=sep,closure_screen=relative<1e-8,primitive_screen=sep>=1e-7,accepted_local_step=bool(merit<phi0 and merit<=upper and sep>=1e-7),accepted_orbit=False)

def correct(initial,initial_one,initial_two,map_one,action,budget,store,result,solver=solve_direction):
 """Callbacks evaluate only one map/action; every returned output is retained.

 result is caller-owned so the last durable accepted state survives exceptions.
 No accepted local step is an orbit acceptance. No old-state callback matching.
 """
 theta=finite_vector(initial).copy();one=finite_vector(initial_one,theta.shape);two=finite_vector(initial_two,theta.shape)
 result.update(steps=[],accepted_steps=0,last_accepted_state='initial-theta.npy',stop='two_direction_cap')
 for iteration in range(2):
  budget.check();r=two-theta;phi0=.5*float(r@r);relative=float(np.linalg.norm(r))/max(1.,float(np.linalg.norm(theta)))
  if relative<1e-8:result['stop']='closure_screen_candidate';break
  tag=f'newton-{iteration}';final=[False]
  def a(v):
   value=action(theta,one,v)
   if final[0]:store.array(tag+'-applied-direction',value);store.array(tag+'-linear-error',value+r)
   return value
  def save_direction(p,info):
   store.array(tag+'-direction',p);store.json(tag+'-returned-direction.json',dict(info=info));final[0]=True
  row=dict(iteration=iteration,base_state=result['last_accepted_state'],trials=[],accepted_index=None);result['steps'].append(row)
  p,ap,error,linear=solver(a,r,restart=400,cycles=1,rtol=1e-10,forcing_limit=.5,max_matvec=402,explicit_zero_start=True,deadline=budget.deadline,event=lambda event:store.event(dict(event,newton_iteration=iteration)),save_direction=save_direction)
  row['linear']=linear;store.json(tag+'-linear.json',linear)
  if not linear['forcing_test_pass']or not linear['descent_direction']:result['stop']='true_forcing_or_descent_failed';break
  slope=linear['merit_directional_derivative'];accepted=False
  for i,alpha in enumerate(ALPHAS):
   budget.check();trial=theta+alpha*p;prefix=f'{tag}-trial-{i}';store.array(prefix+'-theta',trial);finite_vector(trial,theta.shape)
   first=map_one(trial);store.array(prefix+'-map1',first);second=map_one(first);store.array(prefix+'-map2',second);store.array(prefix+'-residual',second-trial)
   metric=trial_metrics(trial,first,second,alpha,phi0,slope);metric['index']=i;row['trials'].append(metric);store.event(dict(metric,kind='trial_result',newton_iteration=iteration))
   if metric['accepted_local_step']:
    # The marker is durable before mutating the currently accepted state.
    store.json(tag+'-accepted.json',dict(index=i,alpha=alpha,state=prefix+'-theta.npy',metrics=metric));theta,one,two=trial,first,second;row['accepted_index']=i;result['accepted_steps']+=1;result['last_accepted_state']=prefix+'-theta.npy';accepted=True;break
  if not accepted:result['stop']='all_thirteen_trials_failed';break
 budget.check();result['final_relative_residual']=float(np.linalg.norm(two-theta))/max(1.,float(np.linalg.norm(theta)));result['final_relative_separation']=float(np.linalg.norm(one-theta))/max(1.,float(np.linalg.norm(theta)))
 result['closure_screen']=result['final_relative_residual']<1e-8;result['primitive_screen']=result['final_relative_separation']>=1e-7;return result
