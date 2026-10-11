from fractions import Fraction as F
from flint import arb,arb_mat,ctx
import numpy as np
import pytest
from neural_shrimp import rigorous_clark_generalized_flip as A
from neural_shrimp.clark_rnn_symmetry import g_jets
from neural_shrimp.clark_rnn_boundaries import third_directional


def test_dual_polynomial_tanh_derivative_includes_fourth_order():
 with ctx.workprec(256):
  x=A.Dual.variable(arb('1/4'),1,0);out=A.poly_tanh([x,x*0+1,x*0,x*0]);t=x.f.tanh();s=1-t*t
  assert out[3].f.contains(s*(6*t*t-2)/6)
  assert out[3].g[0].overlaps((16*t-40*t**3+24*t**5)/6)


def test_validated_dual_solve_implicit_derivative():
 with ctx.workprec(256):
  x=A.Dual.variable(arb(2),1,0);z=x*0;out=A.solve_dual([[x,z],[z,x+1]],[x+1,x*x],'fixture')
  assert out[0].f.contains(arb(3)/2)and out[0].g[0].contains(-arb(1)/4)
  assert out[1].f.contains(arb(4)/3)and out[1].g[0].contains(arb(8)/9)
  with pytest.raises((ValueError,ZeroDivisionError)):A.solve_dual([[z]],[x],'singular')


def test_scalar_three_tanh_cubic_and_all_four_unknown_derivatives():
 with ctx.workprec(256):
  h,j,data=A.augmented([[arb(1)]],[arb(0),arb(1),arb(1),arb(1)])
  assert h[0,0].contains(0)and h[1,0].contains(0)and h[2,0].contains(0)and h[3,0].contains(1)
  assert data['left'][0].contains(1)and data['left_auxiliary'].contains(0)
  # At x=0: c3=q²*d*g*a²*(1+a²+a⁴)/3, a=1-d+d*g.
  # c3=1, derivative in leak=1, gain=5, q=2 at d=g=q=1.
  for k,want in enumerate([0,1,2,5]):assert j[3,k].contains(want)
  assert j[1,1].contains(0)and j[1,3].contains(-3)


def test_all_augmented_derivatives_against_unrelated_centered_finite_difference():
 with ctx.workprec(256):
  mat=np.array([[.4,-.15],[.2,.3]]);z=np.array([.2,-.3,.7,.8,.6,1.2]);j0=[[arb(str(F(float(x))))for x in row]for row in mat]
  def run(a):return A.augmented(j0,[arb(str(F(float(x))))for x in a])
  h,j,d=run(z);step=2.**-18
  for k in range(6):
   plus=z.copy();minus=z.copy();plus[k]+=step;minus[k]-=step;hp,_,_=run(plus);hm,_,_=run(minus)
   approx=np.array([float((hp[i,0]-hm[i,0])/(2*step))for i in range(6)]);actual=np.array([float(j[i,k])for i in range(6)]);assert np.max(abs(approx-actual))<2e-9
  end,u,v,_,_=g_jets(z[:2],mat,z[-1],z[2],3);q=z[3:5];third,_=third_directional(z[:2],q,mat,z[-1],z[2],3)
  assert np.allclose([float(h[i,0])for i in range(2)],end-z[:2],atol=1e-14)
  assert np.allclose([float(d['third_coefficient'][i])for i in range(2)],-third/6,atol=1e-14)
  assert np.allclose([float(d['second_directional'][i])for i in range(2)],np.einsum('ijk,j,k->i',v[:,:2,:2],q,q),atol=1e-14)


def test_whole_interval_contains_dense_scalar_values_and_gradients():
 with ctx.workprec(256):
  b=[arb(0,arb('1/1024')),arb('3/4',arb('1/4096')),arb(1,arb('1/4096')),arb(1,arb('1/4096'))];h,j,_=A.augmented([[arb(1)]],b)
  for sign in [-1,0,1]:
   point=[v.mid()+sign*v.rad()/2 for v in b];v,k,_=A.augmented([[arb(1)]],point)
   assert all(h[i,0].contains(v[i,0])and all(j[i,l].contains(k[i,l])for l in range(4))for i in range(4))
