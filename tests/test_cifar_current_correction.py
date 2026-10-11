import numpy as np
import pytest
from neural_shrimp.cifar_current_correction import correct,direction_gate
REPORT=dict(completed_fixed_extension=True,total_columns=512,status='maximum_columns',orthogonality_qualified=True)

def test_indefinite_quadratic_actual_merit_acceptance():
 H=np.diag([2.,-3.]);x=np.array([.01,.01]);g=H@x;p=-x;hp=H@p
 y,ng,r=correct(x,g,np.zeros(2),p,hp,hp,REPORT,lambda z:H@z)
 assert r['direction_applied']and r['accepted_index']==0 and r['stationarity_qualified']
 assert np.array_equal(y,np.zeros(2))and np.array_equal(ng,np.zeros(2))

def test_bad_true_action_rejects_without_evaluating_trial():
 x=np.array([.01]);g=x.copy();p=-x;hp=-g;calls=[]
 y,ng,r=correct(x,g,np.zeros(1),p,hp,g,REPORT,lambda z:calls.append(z))
 assert not r['direction_applied']and not calls and np.array_equal(y,x)and np.array_equal(ng,g)
 assert not direction_gate(g,hp,hp,{**REPORT,'total_columns':511})['passed']

def test_global_origin_trust_is_retained_and_skips_full_trial():
 x=np.array([.04]);g=np.array([1.]);p=np.array([.04]);hp=-g;calls=[]
 def gradient(z):calls.append(z.copy());return np.array([.1])
 y,ng,r=correct(x,g,np.zeros(1),p,hp,hp,REPORT,gradient)
 assert r['accepted_index']==2 and len(calls)==1 and r['relative_origin_distance']<=.05
 assert all(not v['within_trust']for v in r['trials'][:2])

def test_nonlinear_merit_failure_retains_initial_state():
 x=np.array([.01]);g=np.array([1.]);p=np.array([.001]);hp=-g;calls=[]
 def gradient(z):calls.append(z.copy());return np.array([2.])
 y,ng,r=correct(x,g,np.zeros(1),p,hp,hp,REPORT,gradient)
 assert len(calls)==13 and r['status']=='backtracking_failed'and np.array_equal(y,x)and np.array_equal(ng,g)
 assert all(not t['accepted']for t in r['trials'])
 with pytest.raises(ValueError,match='original global trust'):correct(np.array([.06]),g,np.zeros(1),p,hp,hp,REPORT,gradient)
