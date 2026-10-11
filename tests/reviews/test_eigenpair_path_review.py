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

def test_state_subtraction_does_not_insert_radius_rounding():
 from neural_shrimp.rigorous_orbits import vector_json,ball_json
 with ctx.workprec(256):
  c={'period':1,'dimension':1,'sigma':1,'center':vector_json([arb(1)]*4),'slope':vector_json([arb(1)]*4),'curvature':vector_json([arb(0)]*4),'normal':['0','0','0','1'],'origin':['1']*4,'parameter_center':'0'}
  d=arb(0,E.R.exact('1/32768'));z=E.path_inputs(c,d)
  inputs=[[vector_json(z[0]),vector_json(z[1])]]+[[vector_json(v),vector_json([arb(0)]*4)] for v in z[-2:]]
  event={'phase':0,'d':ball_json(d),'inputs':inputs,'returns':[[vector_json(z[0]),vector_json(z[1])]]}
  result=E.rebuild_coefficients(c,[event],d)
  for j in range(4):
   expected=ball_json(z[0][j]-z[0][j]);actual=ball_json(result[0,j])
   assert (actual['mid'],actual['rad'])==(expected['mid'],expected['rad'])
