import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/reviews'))
from flint import arb,arb_mat,ctx
from fractions import Fraction
import review_eigenpair_path_math as E

def check(a=1,h=0,w=1):
 with ctx.workprec(256):
  H=arb_mat([[E.R.exact(h),0,0,0]]);W=arb_mat([[0,0,0,0]])
  return E.reduction(arb_mat([[1]]),arb_mat([[E.R.exact(a)]]),H,W,[Fraction(w)],Fraction(1,16),Fraction(1,32))
def test_valid_strict_contraction():assert check()['root']
def test_residual_must_fit():assert not check(h=Fraction(1,8))['root']
def test_boundary_is_not_strict():assert not check(h=Fraction(1,16))['root']
def test_zero_derivative_not_contracting():assert not check(a=0)['root']
def test_radius_weights_are_used():assert check(h=Fraction(1,32))['root'] and not check(h=Fraction(1,32),w=Fraction(1,4))['root']
def test_defect_and_residual_combined():assert not check(a=Fraction(1,2),h=Fraction(1,32))['root']
def test_small_nondiagonal_weighted_case():
 with ctx.workprec(256):
  a=arb_mat([[1,E.R.exact(Fraction(1,4))],[0,1]]);h=arb_mat(2,4)
  result=E.reduction(arb_mat([[1,0],[0,1]]),a,h,h,[Fraction(1,8),Fraction(1)],Fraction(1,16),Fraction(1,32))
  assert not result['root'] and result['norm']>=2
