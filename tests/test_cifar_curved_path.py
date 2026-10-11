import numpy as np
import pytest
from neural_shrimp.cifar_curved_path import evaluate
REPORT=dict(completed_fixed_extension=True,total_columns=512,status='maximum_columns',orthogonality_qualified=True)
def setup():
 x=np.array([1.,0.]);g=np.array([.1,0.]);p=-g;a=np.array([.01,0.]);return x,g,p,a
def test_both_actual_points_and_correct_initial_curve_derivative():
 x,g,p,a=setup();seen=[]
 def grad(y):seen.append(y.copy());return g+(y-x)
 r,v=evaluate(x,g,x,p,p,p,REPORT,a,a,a,grad)
 assert r['accepted_index']==0 and len(seen)==2 and all(q['accepted']for q in r['trials'])
 assert r['tangent_gate']['slope']==pytest.approx(-.01)
 for alpha,y in zip([1/32,1/64],seen):assert np.array_equal(y,x+alpha*p+alpha*alpha*a)
 assert len(v)==2

def test_invalid_acceleration_does_not_spend_gradient_calls():
 x,g,p,a=setup()
 def forbidden(y):raise AssertionError('unqualified action')
 r,v=evaluate(x,g,x,p,p,p,REPORT,a,a,2*a,forbidden)
 assert not r['acceleration_action_qualified']and r['accepted_index']is None and v==[None,None]

def test_failed_first_merit_kept_and_smaller_point_selected():
 x,g,p,a=setup();calls=[]
 def grad(y):calls.append(y);return 2*g if len(calls)==1 else .99*g
 r,v=evaluate(x,g,x,p,p,p,REPORT,a,a,a,grad)
 assert r['accepted_index']==1 and r['trials'][0]['merit_ratio']==pytest.approx(4)and len(calls)==2

def test_reject_wrong_tangent_and_excessive_curvature():
 x,g,p,a=setup()
 with pytest.raises(ValueError):evaluate(x,g,x,p,g,g,REPORT,a,a,a,lambda y:g)
 a*=1000
 r,v=evaluate(x,g,x,p,p,p,REPORT,a,a,a,lambda y:(_ for _ in()).throw(AssertionError()))
 assert not any(q['admitted']for q in r['trials'])
