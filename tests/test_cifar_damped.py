import importlib.util,time
from pathlib import Path
import numpy as np
import pytest
from neural_shrimp.cifar_damped import correct,trial_metrics
class Store:
 def __init__(self):self.records={}
 def array(self,n,v):self.records[n+'.npy']=np.array(v).copy()
 def json(self,n,v):self.records[n]=v
 def event(self,v):pass
class Budget:
 deadline=None
 def check(self):pass
def solver(action,r,save_direction,**kw):
 p=r/.75;save_direction(p,1);ap=action(p);e=ap+r
 return p,ap,e,dict(forcing_test_pass=True,descent_direction=True,merit_directional_derivative=float(r@ap),gmres_info=1,requested_tolerance_met=False)
def run(mapper=lambda x:x/2,solve=solver):
 s=Store();r={};correct(np.array([1.]),np.array([.5]),np.array([.25]),mapper,lambda t,f,v:-.75*v,Budget(),s,r,solver=solve);return s,r
def test_two_steps_first_passing_and_primitive_guard():
 s,r=run();assert r['accepted_steps']==2 and [x['accepted_index']for x in r['steps']]==[1,1]
 assert r['steps'][0]['trials'][0]['primitive_screen'] is False
 assert np.array_equal(s.records[r['last_accepted_state']],np.array([.25]))
 assert all(not t['accepted_orbit']for step in r['steps']for t in step['trials'])
def test_failed_direction_no_trial_or_step():
 calls=[]
 def bad(*args,**kw):
  x=solver(*args,**kw);x[-1]['forcing_test_pass']=False;return x
 s,r=run(lambda x:(calls.append(x),x/2)[1],bad);assert r['accepted_steps']==0 and r['stop']=='true_forcing_or_descent_failed'and not calls
 assert 'newton-0-direction.npy'in s.records and 'newton-0-linear-error.npy'in s.records
def test_interruption_preserves_last_durable_acceptance():
 s=Store();r={};calls=[0]
 def interrupted(x):
  calls[0]+=1
  if calls[0]==5:raise InterruptedError('synthetic boundary')
  return x/2
 with pytest.raises(InterruptedError):correct(np.array([1.]),np.array([.5]),np.array([.25]),interrupted,lambda t,f,v:-.75*v,Budget(),s,r,solver=solver)
 assert r['accepted_steps']==1 and r['last_accepted_state']=='newton-0-trial-1-theta.npy'
 assert 'newton-0-accepted.json'in s.records and np.array_equal(s.records[r['last_accepted_state']],[.5])
def test_strict_merit_and_primitive_both_required():
 r=trial_metrics(np.array([1000.]),np.array([1001.]),np.array([1001.1]),1/32,.5,-1.)
 assert r['relative_residual']<.002 and not r['accepted_local_step']
def test_all_thirteen_trials_failure_keeps_original():
 s,r=run(lambda x:x+10);assert r['stop']=='all_thirteen_trials_failed'and r['accepted_steps']==0
 assert len(r['steps'][0]['trials'])==13 and r['last_accepted_state']=='initial-theta.npy'
def module():
 p=Path(__file__).parents[1]/'tools/explore_cifar_damped.py';s=importlib.util.spec_from_file_location('damped_driver',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def test_parent_gate_rejects_wrong_saved_state():
 m=module();data={'initial-theta':np.array([1.]),'initial-residual':np.array([-1.]),'direction':np.array([-.5]),'applied-direction':np.array([1.]),'trial-1-theta':np.array([.99]),'trial-1-map1':np.array([1.]),'trial-1-map2':np.array([.1])}
 with pytest.raises(ValueError,match='exact declared parent'):m.parent_gate({},data.__getitem__)
def test_global_caps_and_failed_call_consumes_attempt():
 m=module();b=m.Budget(lambda _:None);assert 149<b.deadline-time.monotonic()<=150
 for kind,cap in m.CAPS.items():
  b.attempted[kind]=cap-1
  with pytest.raises(ArithmeticError):b.call(kind,lambda:(_ for _ in()).throw(ArithmeticError()))
  with pytest.raises(ValueError):b.call(kind,lambda:0)
def test_measured_cost_finite_and_bounded():
 m=module();events=[dict(kind='operation_returned',operation='direct_gradient',seconds=.01)]*20
 assert m.measured_gate({'elapsed_seconds':29.},events)['projected_seconds']<150
 for value in(float('nan'),float('inf'),100.):
  with pytest.raises(ValueError):m.measured_gate({'elapsed_seconds':value},events)
