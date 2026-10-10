from flint import arb,arb_mat,ctx
import pytest
from neural_shrimp.rigorous_adjoint_functional import functional


def test_cancellation_in_exact_identity():
 with ctx.workprec(128):
  # B y=e3, actual y=[-1,1,0]. b^T y=0 despite loose independent Y.
  b=arb_mat([[1,1,0],[0,0,1],[0,1,0]]);c=b.inv()
  r,v=functional(b,c,[arb(-1,arb(1)/10),arb(1,arb(1)/10),arb(0)],[arb(1),arb(1),arb(0)])
  assert v==0 and r['direct']['rad'][0]>0


def test_nonnormal_and_nonzero_scalar():
 with ctx.workprec(128):
  b=arb_mat([[1,2,0],[0,0,1],[0,1,0]]);c=b.inv()
  r,v=functional(b,c,[arb(-2,arb(1)/10),arb(1,arb(1)/10),arb(0)],[arb(3),arb(1),arb(0)])
  assert v==-5 and r['excludes_zero']


def test_interval_parameter_exact_samples():
 with ctx.workprec(128):
  b=arb_mat([[1,arb(2,arb(1)/100),0],[0,0,1],[0,1,0]])
  c=arb_mat([[1,-2,0],[0,0,1],[0,1,0]])
  r,v=functional(b,c,[arb(-2,arb(1)/100),arb(1),arb(0)],[arb(3),arb(1),arb(0)])
  assert v.contains(arb(-497)/100) and v.contains(arb(-503)/100)


@pytest.mark.parametrize('bad',['nan','inf','-inf'])
def test_nonfinite(bad):
 with pytest.raises(ValueError):functional(arb_mat([[1,0],[0,1]]),arb_mat([[1,0],[0,1]]),[arb(bad),arb(0)],[arb(1),arb(0)])


def test_auxiliary_restrictions():
 with pytest.raises(ValueError):functional(arb_mat([[1,0],[0,1]]),arb_mat([[1,0],[0,1]]),[arb(1),arb(1)],[arb(1),arb(0)])


def test_exact_preconditioner_required():
 with pytest.raises(ValueError):functional(arb_mat([[1,0],[0,1]]),arb_mat([[arb(1,arb(1)/100),0],[0,1]]),[arb(1),arb(0)],[arb(1),arb(0)])


def test_return_before_observer_failure():
 seen=[]
 def obs(k,v):
  seen.append(k)
  if k=='dual_residual_return':raise OSError('durability')
 with pytest.raises(OSError):functional(arb_mat([[0,1],[1,0]]),arb_mat([[0,1],[1,0]]),[arb(1),arb(0)],[arb(1),arb(0)],obs)
 assert seen==['dual_proposal_return','dual_residual_return']
