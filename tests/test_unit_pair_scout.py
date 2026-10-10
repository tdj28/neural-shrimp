from fractions import Fraction
import numpy as np
import pytest
from flint import ctx,arb
from neural_shrimp.unit_pair_scout import Jet,TanhMap,evaluate,coefficients,native_flat,qualify,continue_branch
from neural_shrimp.rigorous_generalized_chain import augmented_chain
from neural_shrimp.rigorous_cusps import TwoRateCuspMap
from neural_shrimp.rigorous_orbits import exact,dyadic

class Normal:
 n=3
 def cusp_step(self,z,h,o):
  x,y,w=z;return[x+y,y+h-1+x*x*.5+(o-1)*y+2*x*y,w*.2]

def inp():return {'schema':'neural-shrimp-c1-input-v1','network':{'activation':'tanh','d':1,'width':1,'bias':True,'in_scale':'3/2','out_scale':'2/3','loss_scale':'1/2','ridge':'1/100'},'x':[['-1/4'],['1/3']],'y':['1/7','-1/9'],'lr_hidden':'1/100','lr_output':'1/100'}
def arr(z):return[exact(dyadic(v))for v in z]

@pytest.mark.parametrize('t',[0.,1e-5,.2,2.])
def test_full_float_jacobian_and_parameter(t):
 m=Normal();z=np.array([.03,-.01,.02,1.001,.97,.8,.2,.1,.1,.7,-.2]);v,J,ht,d=evaluate(m,z,2,t)
 eps=1e-6
 for k in range(len(z)):
  dz=np.eye(len(z))[k]*eps
  fd=(evaluate(m,z+dz,2,t)[0]-evaluate(m,z-dz,2,t)[0])/(2*eps)
  np.testing.assert_allclose(J[:,k],fd,atol=1e-8,rtol=1e-6)
 if t>eps:
  fd=(evaluate(m,z,2,t+eps)[0]-evaluate(m,z,2,t-eps)[0])/(2*eps)
  np.testing.assert_allclose(ht,fd,atol=1e-8,rtol=1e-6)

def test_float_tanh_matches_independent_arb_chain_all_blocks():
 m=TanhMap(inp());z=np.linspace(-.1,.2,3*m.n+2);z[m.n:m.n+2]=.01
 v,J,_,d=evaluate(m,z,3,0)
 with ctx.workprec(256):a,b,c=augmented_chain(TwoRateCuspMap(inp()),arr(z),3)
 np.testing.assert_allclose(v,np.array([float(a[i,0])for i in range(len(z))]),atol=2e-14)
 np.testing.assert_allclose(J,np.array([[float(b[i,j])for j in range(len(z))]for i in range(len(z))]),atol=2e-14)
 np.testing.assert_allclose(d['monodromy'],np.array([[float(c['monodromy'][i,j])for j in range(m.n)]for i in range(m.n)]),atol=2e-14)

def test_exact_normal_form_branch_and_native_pack():
 t=.02;c,e,d,*_=coefficients(t)
 # Exact rotation-form synthetic map has native unit pair and a stable complement.
 class Rotation:
  n=3
  def cusp_step(self,z,h,o):
   x,y,w=z;return[c*x+e*y,-d*x+c*y,.2*w]
 z=np.array([0.,0.,0.,1.,1.,1.,0.,0.,0.,1.,0.]);v,J,ht,data=evaluate(Rotation(),z,1,t)
 assert max(abs(v))<1e-14
 assert qualify(z,t,v,data)['passed']
 flat=native_flat(z,data);np.testing.assert_array_equal(flat,np.r_[z[:3],z[5:8],z[8:],z[3:5]])

def test_repeated_cycle_unstable_complement_and_wrong_pair_exposed():
 t=.1;c,e,d,*_=coefficients(t)
 class Rotation:
  n=3
  def cusp_step(self,z,h,o):
   x,y,w=z;return[c*x+e*y,-d*x+c*y,1.2*w]
 z=np.array([0.,0.,0.,1.,1.,1.,0.,0.,0.,1.,0.]);v,_,_,data=evaluate(Rotation(),z,1,t)
 q=qualify(z,t,v,data);assert q['passed']and q['complement_max_modulus']>1
 wrong=qualify(z,t+.01,v,data);assert not wrong['gates']['actual_pair']
 v,_,_,data=evaluate(Rotation(),z,2,4*t);assert not qualify(z,4*t,v,data)['gates']['native_primitive']

def test_entire_zero_and_callback_cap():
 assert coefficients(0)==(1.,1.,0.,-.5,-1/6,1.)
 class Never:
  n=3
  def cusp_step(self,*_):raise AssertionError('should not evaluate')
 def stop():raise TimeoutError('owned cap')
 with pytest.raises(TimeoutError):evaluate(Never(),np.ones(11),18,.1,stop)
