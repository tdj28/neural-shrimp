import numpy as np
import pytest
from neural_shrimp.cifar_merit_lsmr import direction,trial

def test_nonsymmetric_damped_solution_matches_augmented_dense_lstsq():
 a=np.array([[.2,5.,1.],[-1.,.02,3.],[0.,2.,.3]]);r=np.array([.2,-.4,.1]);damp=.01
 p,ap,info=direction(lambda v:a@v,lambda v:a.T@v,r,damp,iterations=12,radius=100.)
 expected=np.linalg.lstsq(np.vstack([a,damp*np.eye(3)]),np.r_[-r,np.zeros(3)],rcond=None)[0]
 assert np.allclose(p,expected,rtol=1e-8,atol=1e-10)and np.allclose(ap,a@p)
 assert info['usable_descent']and info['old_newton_forcing_pass']

def test_clipped_step_uses_fresh_physical_action():
 a=np.diag([1e-3,2.]);r=np.array([1.,.1]);p,ap,info=direction(lambda v:a@v,lambda v:a.T@v,r,.00001,iterations=10,radius=.25)
 assert np.isclose(np.linalg.norm(p),.25)and np.allclose(ap,a@p)and info['clip']<1
 assert info['usable_descent']and not info['old_newton_forcing_pass']

def test_actual_increase_fixed_point_and_bad_prediction_cannot_accept():
 r=np.array([1.,0.]);ap=np.array([-.5,0.]);theta=np.array([1.,2.]);first=theta+.1
 good=trial(theta,first,theta+r+ap,r,ap,1.);assert good['accepted_local_step']and not good['accepted_orbit']
 assert not trial(theta,first,theta+2*r,r,ap,1.)['accepted_local_step']
 assert not trial(theta,theta,theta+r+ap,r,ap,1.)['accepted_local_step']
 assert not trial(theta,first,theta+r/2,r,r,1.)['accepted_local_step']

def test_deadline_and_invalid_damping():
 with pytest.raises(TimeoutError):direction(lambda v:v,lambda v:v,np.ones(2),.1,deadline=0)
 with pytest.raises(ValueError):direction(lambda v:v,lambda v:v,np.ones(2),0)
