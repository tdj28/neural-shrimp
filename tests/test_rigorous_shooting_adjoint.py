from flint import arb,arb_mat,ctx
import pytest
from neural_shrimp.rigorous_shooting_adjoint import qualify,bounded_solve,shooting_blocks


def fixture(js,qs,hs,bs):
 n=len(qs[0]);p=len(js);m=2*n*p+2;J=arb_mat(m,m);box=[arb(0)]*m
 for k in range(p):
  z=2*n*k;after=2*n*((k+1)%p)
  for i in range(n):
   box[z+n+i]=arb(qs[k][i])
   for j in range(n):
    J[z+i,z+j]+=js[k][i][j];J[z+n+i,z+n+j]+=js[k][i][j]
    J[z+n+i,z+j]=hs[k][i][j]
   J[z+i,after+i]-=1;J[z+n+i,after+n+i]-=1
   for r in range(2):J[z+i,m-2+r]=bs[k][i][r]
 return J,box


def test_nonnormal_left_not_right():
 with ctx.workprec(128):
  # M=[[1,2],[0,1/2]], q=e1, normalized left=[1,4].
  J,x=fixture([[[1,2],[0,arb(1)/2]]],[[1,0]],[[[2,0],[3,0]]],[[[1,0],[0,1]]])
  r=qualify(J,x,2,1)
  assert r['passed'];assert '4.000' in r['left_native'][1]['display'] or r['left_native'][1]['mid'][0]==1
  from neural_shrimp.rigorous_orbits import ball_json
  assert r['actual_rate_coefficients']==[ball_json(arb(1)),ball_json(arb(4))]
  assert r['second_derivative_contraction']==ball_json(arb(14))


def test_two_phase_composition_and_last_left():
 with ctx.workprec(128):
  # q1=2 q0; M=1. F0''=3,F1''=5 -> G''=.5*3+5*4=21.5.
  J,x=fixture([[[2]],[[arb(1)/2]]],[[1],[2]],[[[3]],[[10]]],[[[1,0]],[[0,1]]])
  r=qualify(J,x,1,2)
  assert r['passed']
  from neural_shrimp.rigorous_orbits import ball_json
  assert r['actual_rate_coefficients']==[ball_json(arb(1)/2),ball_json(arb(1))]
  assert r['second_derivative_contraction']==ball_json(arb(43)/2)


def test_c_zero_is_not_fold():
 J,x=fixture([[[1]]],[[1]],[[[0]]],[[[1,0]]]);r=qualify(J,x,1,1)
 assert not r['passed'] and r['gates']['simple_native_plus_one'] and not r['gates']['nonzero_quadratic']


def test_no_transversality():
 J,x=fixture([[[1]]],[[1]],[[[2]]],[[[0,0]]]);r=qualify(J,x,1,1)
 assert not r['passed'] and not r['gates']['nonzero_rate_transversality']


def test_multiple_critical_rejected():
 J,x=fixture([[[1,0],[0,1]]],[[1,0]],[[[2,0],[0,0]]],[[[1,0],[0,1]]])
 with pytest.raises((ValueError,ZeroDivisionError)):qualify(J,x,2,1)


def test_defective_critical_rejected():
 J,x=fixture([[[1,1],[0,1]]],[[1,0]],[[[2,0],[0,0]]],[[[1,0],[0,1]]])
 with pytest.raises((ValueError,ZeroDivisionError)):qualify(J,x,2,1)


def test_interval_solution_contains_exact_samples():
 with ctx.workprec(128):
  b=arb_mat([[arb(2,arb(1)/100),arb(1)],[arb(0),arb(3,arb(1)/100)]])
  r,y=bounded_solve(b);assert r['passed']
  for a in [arb(199)/100,arb(201)/100]:
   for d in [arb(299)/100,arb(301)/100]:
    assert y[0].contains(-1/(a*d)) and y[1].contains(1/d)


def test_wide_interval_abstains():
 r,y=bounded_solve(arb_mat([[arb(1,2),0],[0,1]]))
 assert not r['passed'] and y is None


@pytest.mark.parametrize('bad',['nan','inf','-inf'])
def test_nonfinite_rejected(bad):
 with pytest.raises(ValueError):bounded_solve(arb_mat([[arb(bad),0],[0,1]]))


def test_observer_errors_propagate_after_return():
 seen=[]
 def obs(kind,value):
  seen.append(kind)
  if kind=='preconditioner_return':raise OSError('write failure')
 with pytest.raises(OSError):bounded_solve(arb_mat([[2,1],[0,3]]),obs)
 assert seen==['border','preconditioner_return']


def test_phase_orientation_changes_coeff_sign_not_quadratic_magnitude():
 J,x=fixture([[[1]]],[[-1]],[[[-2]]],[[[1,0]]]);r=qualify(J,x,1,1)
 assert r['passed'] and r['second_derivative_contraction']['mid'][0]<0
