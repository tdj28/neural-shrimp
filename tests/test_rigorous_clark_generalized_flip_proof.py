from flint import arb,arb_mat,ctx
import pytest
from neural_shrimp import rigorous_clark_generalized_flip_proof as P


def decode(v):return arb(arb(tuple(v['mid'])),arb(tuple(v['rad'])))


def test_strict_full_root_and_failed_zero_newton_are_retained():
 with ctx.workprec(256):
  def evaluate(x):return arb_mat([[x[0]*x[0]-2],[x[1]-3]]),arb_mat([[2*x[0],0],[0,1]]),{}
  rows=[];b,j,d,r=P.root_enclosure(evaluate,[arb('7/5'),arb(3)],observe=lambda k,v:rows.append((k,v)))
  assert r['passed']and all(r['coordinate_inclusions'])and b[0].contains(arb(2).sqrt())and b[1].contains(3)
  assert sum(k=='newton_proposal'for k,v in rows)==4 and len(r['whole_jacobian'])==2
  _,_,_,r=P.root_enclosure(evaluate,[arb(1),arb(3)],newton_steps=0)
  assert not r['passed']and not r['coordinate_inclusions'][0]


def test_validated_inverse_and_singular_rejection():
 with ctx.workprec(192):
  a=arb_mat([[2,1],[1,3]]);q=P.inverse(a,'fixture');assert all((q*a)[i,j].contains(int(i==j))for i in range(2)for j in range(2))
  with pytest.raises((ValueError,ZeroDivisionError)):P.inverse(arb_mat([[1,1],[1,1]]),'singular')


def scalar_path(h,cubic=0,quintic=2):
 x=[v[0]for v in h];p2=P.multiply(x,x);p3=P.multiply(p2,x);p5=P.multiply(p3,p2)
 return [[-x[i]+cubic*p3[i]+quintic*p5[i]]for i in range(6)]


def test_conditional_exact_quintic_and_nonzero_cubic_rejection():
 with ctx.workprec(256):
  out=P.conditional_normal_form([arb(0)],arb_mat([[-1]]),[arb(1)],[arb(1)],scalar_path)
  assert out['passed']and decode(out['quintic']).contains(2)and decode(out['cubic_consistency']).contains(0)
  with pytest.raises(ValueError):P.conditional_normal_form([arb(0)],arb_mat([[-1]]),[arb(1)],[arb(1)],lambda h:scalar_path(h,1))
  out=P.conditional_normal_form([arb(0)],arb_mat([[-1]]),[arb(1)],[arb(1)],lambda h:scalar_path(h,0,0));assert not out['passed']


def test_quintic_with_nonlinearly_curved_stable_coordinate():
 with ctx.workprec(256):
  beta,gamma,rate=arb(1)/3,arb(2)/5,arb(1)/4
  def evaluate(h):
   x,y=[v[0]for v in h],[v[1]for v in h];fx=[v[0]for v in scalar_path([[v]for v in x])];x2=P.multiply(x,x);x3=P.multiply(x2,x);f2=P.multiply(fx,fx);f3=P.multiply(f2,fx)
   return [[fx[i],rate*(y[i]-beta*x2[i]-gamma*x3[i])+beta*f2[i]+gamma*f3[i]]for i in range(6)]
  out=P.conditional_normal_form([arb(0),arb(0)],arb_mat([[-1,0],[0,rate]]),[arb(1),arb(0)],[arb(1),arb(0)],evaluate)
  assert out['passed']and decode(out['quintic']).contains(2)
  assert decode(out['parameterization'][2][1]).contains(beta)and decode(out['parameterization'][3][1]).contains(gamma)
  assert all(decode(v).contains(0)for row in out['invariance_defect']for v in row)


def test_physical_unfolding_determinant_and_degenerate_rejection():
 with ctx.workprec(256):
  J=arb_mat([[-2,0,0,0],[0,-3,0,-2],[0,0,1,0],[0,5,0,7]]);r=P.conditional_unfolding(J,arb_mat([[-1]]),[arb(1)])
  assert r['passed']and decode(r['multiplier_leak_derivative']).contains(-3)and (3*decode(r['critical_cubic_gain_derivative'])).contains(11)and decode(r['physical_determinant']).contains(11)
  J[3,3]=arb(10)/3;r=P.conditional_unfolding(J,arb_mat([[-1]]),[arb(1)]);assert not r['passed']and decode(r['physical_determinant']).contains(0)


def test_native_fixed_point_not_falsely_primitive6_and_square_context():
 with ctx.workprec(256):
  j=[[arb(1),arb(0)],[arb(0),arb(0)]];z=[arb(0),arb(0),arb(1)/2,arb(1),arb(0),arb(1)]
  out=P.conditional_native_context(j,z,arb_mat([[-1,0],[0,-arb(1)/8]]),[arb(1),arb(0)])
  assert not out['passed']and not any(v['excluded']for v in out['divisors'])and out['witness_power']==1 and out['complement_dimension']==1
  assert all(decode(v).contains(0)for row in out['square_consistency']for v in row)
