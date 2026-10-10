from flint import arb,arb_mat,ctx
import pytest
from neural_shrimp.rigorous_adjoint_refinement import refine,coefficients


def test_affine_nonzero_proposal_and_alpha():
 with ctx.workprec(128):
  s=[arb(0,10),arb(0,10)];y0=arb_mat([[2],[3]]);f=arb_mat([[-1],[-3]]);d=arb_mat([[0,0],[0,0]])
  y=refine(s,y0,f,d)
  assert y==[arb(1),arb(0)]


def test_conditional_restriction_improves_coupled_system():
 with ctx.workprec(128):
  # At alpha0, first equation has unique w2; unconditional alpha width pollutes it.
  s=[arb(2,5),arb(0,2)];y0=arb_mat([[2],[0]]);f=arb_mat([[0],[0]]);d=arb_mat([[arb(1)/4,2],[0,arb(1)/2]])
  y=refine(s,y0,f,d)
  assert y[0].contains(2) and y[0].rad()<arb(1)/100 and y[-1]==0


def test_empty_intersection_rejected():
 with pytest.raises(ValueError):refine([arb(1),arb(0)],arb_mat([[0],[0]]),arb_mat([[3],[0]]),arb_mat([[0,0],[0,0]]))


def test_known_alpha_must_be_in_initial():
 with pytest.raises(ValueError):refine([arb(1),arb(2)],arb_mat([[0],[0]]),arb_mat([[0],[0]]),arb_mat([[0,0],[0,0]]))


@pytest.mark.parametrize('bad',['nan','inf','-inf'])
def test_nonfinite_inputs(bad):
 with pytest.raises(ValueError):refine([arb(1),arb(0)],arb_mat([[0],[0]]),arb_mat([[arb(bad)],[0]]),arb_mat([[0,0],[0,0]]))


def test_return_saved_before_interrupt():
 seen=[]
 def obs(kind,value):
  seen.append(kind)
  if kind=='image_return':raise TimeoutError('after durable return')
 with pytest.raises(TimeoutError):refine([arb(1),arb(0)],arb_mat([[0],[0]]),arb_mat([[1],[0]]),arb_mat([[0,0],[0,0]]),obs)
 assert seen==['conditional_start','image_return']


def test_nonnormal_coefficients():
 r=coefficients([arb(1),arb(4),arb(0)],arb_mat([[1,0],[0,1]]),[arb(2),arb(3)],2)
 assert r['passed'] and r['second_derivative_contraction']['mid']==[7,1]


def test_zero_coefficient_abstains():
 r=coefficients([arb(1),arb(0)],arb_mat([[0,0]]),[arb(0)],1)
 assert not r['passed']
