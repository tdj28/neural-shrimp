import importlib.util
from pathlib import Path
import numpy as np
import pytest

def module():
 p=Path(__file__).parents[1]/'tools/qualify_clark_rnn_n12_native12_flip.py';s=importlib.util.spec_from_file_location('daughter_flip',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_full_native_minus_augmented_and_third_profile():
 m=module();j=np.eye(12)*.2+np.ones((12,12))*.025;x=np.arange(1,13)/20;q=np.eye(12)[0]
 h,b,a=m.M.augmented(np.r_[x,.8,q],j,2.,12,-1);v=np.linalg.solve(b,-h);assert max(abs(b@v+h))<1e-8
 c,_=m.third_directional(x,q,j,2.,.8,12);report,w,rhs,sol,_=m.flip_coefficients(a,q,c);assert np.isfinite(sol).all();assert max(abs((a['monodromy']-np.eye(12))@sol-rhs).ravel())<1e-8

def test_native_flip_full_fixed_point_and_quadratic_corrections():
 m=module();q=np.eye(12)[0];a=dict(monodromy=np.diag([-1]+[.2]*11),first=np.zeros((12,14)),second=np.zeros((12,14,14)));a['first'][0,13]=.3;a['second'][0,0,0]=1.4;a['second'][0,0,13]=2.
 report,w,rhs,sol,bqq=m.flip_coefficients(a,q,-2.4*q)
 assert report['passed']and report['parameter']==pytest.approx(2.21)and report['cubic']==pytest.approx(.09)
 a['second'][0,0,0]=0;report,*_=m.flip_coefficients(a,q,np.zeros(12));assert not report['passed']

def test_full_native_directional_third_matches_differentiated_second():
 m=module();rng=np.random.default_rng(675);j=rng.normal(size=(12,12))*.08;x=rng.normal(size=12)*.2;q=rng.normal(size=12);q/=np.linalg.norm(q);eps=1e-5
 c,_=m.third_directional(x,q,j,2.1,.73,12)
 def second(y):
  a=m.M.augmented(np.r_[y,.73,q],j,2.1,12,-1)[2];return np.einsum('ijk,j,k->i',a['second'][:,:12,:12],q,q)
 np.testing.assert_allclose(c,(second(x+eps*q)-second(x-eps*q))/(2*eps),rtol=2e-7,atol=1e-10)

def test_native_augmented_and_third_caps_precede_calls(tmp_path):
 m=module();w=m.Work(tmp_path);w.native_aug_calls=32
 with pytest.raises(ValueError,match='32 full-native'):w.aug(np.zeros(25),np.eye(12),2.,'synthetic')
 w.third_calls=16
 with pytest.raises(ValueError,match='16 full-native'):w.third(np.ones(12),np.eye(12)[0],np.eye(12),2.,.8,'synthetic')
 assert w.calls==0


def test_full_native12_augmented_jacobian_physical_leak():
 m=module();rng=np.random.default_rng(900);j=rng.normal(size=(12,12))*.09;z=np.r_[rng.normal(size=12)*.2,.74,np.eye(12)[0]];eps=1e-5
 h,b,a=m.M.augmented(z,j,2.1,12,-1)
 fd=np.column_stack([(m.M.augmented(z+eps*np.eye(25)[k],j,2.1,12,-1)[0]-m.M.augmented(z-eps*np.eye(25)[k],j,2.1,12,-1)[0])/(2*eps)for k in range(25)])
 np.testing.assert_allclose(b,fd,rtol=2e-6,atol=1e-9)
 np.testing.assert_allclose(h[:12],a['endpoint']-z[:12]);np.testing.assert_allclose(h[12:24],a['monodromy']@z[13:]+z[13:])
