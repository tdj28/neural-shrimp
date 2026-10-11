import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp.rigorous_correlated_complement import total_two_rate_jacobian,quotient_derivative,correlated_complement
from neural_shrimp.rigorous_orbits import eye

class Nonlinear:
 n=2
 def cusp_step(self,x,h,o):return [h*x[0]*x[0]+o*x[1],o*x[0]*x[1]+h*x[1]*x[1]]

def test_actual_two_rate_mixed_direction_includes_both_rates():
 with ctx.workprec(256):
  x=list(map(arb,[2,3]));rates=list(map(arb,[5,7]));v=list(map(arb,[11,13]));vp=list(map(arb,[17,19]))
  A,Ap=total_two_rate_jacobian(Nonlinear(),x,rates,v,vp)
  assert A==arb_mat([[20,7],[21,44]])
  assert Ap==arb_mat([[178,19],[148,347]])
  # Changing only output-rate tangent must alter the corresponding blocks.
  _,without=total_two_rate_jacobian(Nonlinear(),x,rates,v,[arb(17),arb(0)])
  assert Ap-without==arb_mat([[0,19],[57,38]])

def test_variable_frame_inverse_derivative_cancels_known_similarity():
 with ctx.workprec(256):
  t=arb('1/16');K=arb_mat(4,4);K[2,0]=1;K[3,1]=1;F=eye(4)+K*t;Finv=eye(4)-K*t
  D=arb_mat([[1,0,0,0],[0,1,0,0],[0,0,arb('1/4'),0],[0,0,0,arb('1/2')]])
  A=F*D*Finv;Ap=K*D-D*K
  T,Tp=quotient_derivative(A,Ap,F,K,Finv,K)
  assert T==D and all(v==0 for row in Tp.tolist()for v in row)

def fixture(center_error=0):
 n=4;h=arb('1/1024');d=arb(0,h);E=arb_mat([[0,0],[0,0],[1,0],[0,1]])
 c=[arb(0)]*4+[arb(1),arb(0),arb(0),arb(0)]+[arb(0),arb(1),arb(0),arb(0)]+[arb(1),arb(1)]
 w=list(c);w[6]=d;w[11]=d;v=[arb(0)]*14;v[6]=1;v[11]=1;v=list(map(arb,v))
 D=arb_mat([[1,0,0,0],[0,1,0,0],[0,0,arb('1/4'),0],[0,0,0,arb('1/2')]])
 K=arb_mat(4,4);K[2,0]=1;K[3,1]=1;Ap=K*D-D*K;A0=arb_mat(D)
 A0[2,2]+=arb(0,arb(center_error))
 return dict(rows=[{'phase':0,'center_A':A0,'whole_A':D+Ap*d,'whole_Aprime':Ap}],center=c,whole=w,tangent=v,complements=[E],metrics=[eye(2)],halfwidth=h,terminal=eye(2))

def test_complete_common_parameter_complement_retains_all_modes():
 with ctx.workprec(256):
  r=correlated_complement(**fixture());assert r['passed']and r['complement_dimension']==2

def test_center_root_uncertainty_is_not_discarded():
 with ctx.workprec(256):
  r=correlated_complement(**fixture(2));assert not r['passed']

def test_inexact_fixed_complement_and_missing_rate_tangent_rejected():
 with ctx.workprec(256):
  a=fixture();a['complements'][0][2,0]=arb(1,arb('1/100'))
  with pytest.raises(ValueError,match='fixed exact'):correlated_complement(**a)
  with pytest.raises(ValueError,match='both rate'):total_two_rate_jacobian(Nonlinear(),[arb(1)]*2,[arb(1)]*2,[arb(1)]*2,[arb(1)])
