import numpy as np
import pytest
from neural_shrimp.cifar_symmetric_shooting import shooting_action,direction

def test_nonsymmetric_eliminated_system_solved_by_symmetric_shooting():
 j0=np.array([[.2,1.3],[1.3,-.4]]);j1=np.array([[2.,.1],[.1,.3]])
 a=j1@j0-np.eye(2);assert not np.allclose(a,a.T)
 r=np.array([.3,-.7]);p,ap,z,az,d=direction(lambda v:j0@v,lambda v:j1@v,r,iterations=12)
 expected=np.linalg.solve(a,-r)
 assert np.allclose(p,expected,atol=1e-11)and np.allclose(ap,a@p)
 assert np.allclose(z[2:],j0@p)and d['forcing_pass']and d['true_forcing']<1e-10
 block=np.block([[j0,-np.eye(2)],[-np.eye(2),j1]])
 assert np.allclose(az,block@z)and np.allclose(block,block.T)

def test_true_forcing_is_not_shooting_stopping_residual():
 j0=np.diag([.99,10.,-.1]);j1=np.array([[.2,.7,0],[.7,2.,.1],[0,.1,4.]])
 r=np.array([1.,2.,3.]);p,ap,z,az,d=direction(lambda v:j0@v,lambda v:j1@v,r,iterations=1)
 assert np.isclose(d['true_forcing'],np.linalg.norm(r+(j1@j0-np.eye(3))@p)/np.linalg.norm(r))
 assert not np.isclose(d['true_forcing'],d['shooting_relative_residual'])
 assert d['iterations']==1 and d['shooting_actions']==3

def test_singular_inconsistent_system_does_not_become_a_pass():
 p,ap,z,az,d=direction(lambda v:v,lambda v:v,np.array([1.,2.]),iterations=5)
 assert not d['forcing_pass']and np.isclose(d['true_forcing'],1)

def test_action_and_deadline_gates():
 with pytest.raises(ValueError):shooting_action(lambda v:v,lambda v:v,np.ones(3))
 with pytest.raises(TimeoutError):direction(lambda v:v,lambda v:v,np.ones(2),deadline=0)
