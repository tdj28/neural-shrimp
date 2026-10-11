from fractions import Fraction as F
from flint import arb,arb_mat,ctx
import pytest
from neural_shrimp.rigorous_polynomial_localization import localize_polynomial_root,refine_polynomial_tangent,preconditioned_derivative_coefficients


def test_nonlinear_square_root_family_preserves_exact_parameter_correlation():
 # H(x,t)=x²-t, t=1+d, p=1+d/2. C=.5-.25d, error domain±1/32.
 # J(segment)=2+d+2E; D=-E+.5E*d+.25d²; R=d²/4.
 with ctx.workprec(384):
  h=F(1,8);E=arb(0,arb(1)/32);A=lambda x:arb_mat([[x]])
  C=[A(arb(1)/2),A(-arb(1)/4)];D=[A(-E),A(E/2),A(arb(1)/4)]
  rhs=[A(0),A(0),A(-arb(1)/8),A(arb(1)/16)];rp=[A(0),A(arb(1)/2)]
  for lo,hi in[(-h,h),(-h,F(0)),(F(0),h),(F(1,16),F(1,16))]:
   err,p=localize_polynomial_root(D,rhs,[arb(1)],A(E),lo,hi,h)
   tangent,q=refine_polynomial_tangent(C,D,rp,err,A(arb(1)/2),[arb(1)],lo,hi,h)
   for val in [lo,(lo+hi)/2,hi]:
    d=arb(val.numerator)/val.denominator;x=(1+d).sqrt();e=x-(1+d/2)
    assert err[0,0].contains(e) and tangent[0,0].contains(1/(2*x))
   assert p['passed']and q['passed']


def test_convolution_retains_cancellation_before_interval_evaluation():
 with ctx.workprec(192):
  C=[arb_mat([[1,2],[0,1]]),arb_mat([[2,-1],[1,0]])]
  R=[arb_mat([[1],[3]]),arb_mat([[-2],[4]]),arb_mat([[5],[-1]])]
  f=preconditioned_derivative_coefficients(C,R)
  assert f[0]==-C[0]*R[0] and f[1]==-C[0]*R[1]-C[1]*R[0]
  assert f[2]==-C[0]*R[2]-C[1]*R[1] and f[3]==-C[1]*R[2]


def test_full_weighted_nonnormal_error_and_known_image_are_retained():
 with ctx.workprec(192):
  D=[arb_mat([[0,4],[0,0]])];forcing=[arb_mat([[1],[arb(1)/8]])]
  old=arb_mat([[arb(0,4)],[arb(0,arb(1)/4)]]);known=arb_mat([[arb(3)/2],[arb(1)/8]])
  box,proof=localize_polynomial_root(D,forcing,[arb(16),arb(1)],old,-F(1,8),F(1,8),F(1,4),known_image=known)
  assert box[0,0].contains(arb(3)/2) and box[1,0].contains(arb(1)/8)and proof['dimension']==2


def test_reject_wrong_domains_noncontracting_or_uncertified_inverse():
 A=lambda x:arb_mat([[x]])
 with pytest.raises(ValueError,match='original closed'):localize_polynomial_root([A(0)],[A(0)],[arb(1)],A(arb(0,1)),-1,1,F(1,2))
 with pytest.raises(ValueError,match='strict correlated'):localize_polynomial_root([A(1)],[A(0)],[arb(1)],A(arb(0,1)),-1,1,1)
 with pytest.raises(ValueError,match='exact inverse'):preconditioned_derivative_coefficients([A(arb(1,arb(1)/10))],[A(1)])
 with pytest.raises(ValueError,match='finite'):localize_polynomial_root([A(0)],[A(arb('nan'))],[arb(1)],A(arb(0,1)),-1,1,1)
