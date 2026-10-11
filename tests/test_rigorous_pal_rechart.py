from flint import arb,arb_mat,ctx
import pytest
from neural_shrimp.rigorous_pal_rechart import *

def test_known_linear_endpoint_and_nonlinear_residual_localization():
 with ctx.workprec(256):
  endpoint,p=localize_endpoint(arb_mat([[1]]),arb_mat([[1]]),arb_mat([[arb('1/16'),0,0,0]]),arb_mat(1,4),[arb(0)],[arb(1)],[arb(0)],[arb(1)],arb(1),arb('1/8'))
  assert p['passed']and endpoint[0].contains(arb('1/16'))
  with pytest.raises(ValueError):localize_endpoint(arb_mat([[1]]),arb_mat([[3]]),arb_mat(1,4),arb_mat(1,4),[arb(0)],[arb(1)],[arb(0)],[arb(1)],arb(1),arb('1/8'))

def test_reparametrized_quadratic_tangent_and_curvature():
 with ctx.workprec(256):
  h=arb('1/8');v=[arb(1),arb(2)];a=[arb(3),arb(-1)];ep=[arb(0),arb(0)];p=rotation_proposal(ep,v,a,h)
  n=[exact(x)for x in p['normal']];s=[exact(x)for x in p['slope']];k=[exact(x)for x in p['curvature']]
  assert abs(sum(n[i]*s[i]for i in range(2))-1)<arb((1,-245))
  assert abs(sum(n[i]*k[i]for i in range(2)))<arb((1,-245))
  gamma=sum(n[i]*(v[i]+a[i]*h)for i in range(2));g2=sum(n[i]*a[i]for i in range(2))
  assert all(abs(k[i]-(a[i]-(v[i]+a[i]*h)*g2/gamma)/(gamma*gamma))<arb((1,-245))for i in range(2))
  assert len(rechart_seed(p,'1/64'))==2

def test_rotation_zero_tangent_and_unknown_dimension_fail():
 with pytest.raises(ValueError):rotation_proposal([arb(0)],[arb(0)],[arb(0)],arb(1))
 with pytest.raises(ValueError):rotation_proposal([arb(0)],[arb(1),arb(2)],[arb(0)],arb(1))
