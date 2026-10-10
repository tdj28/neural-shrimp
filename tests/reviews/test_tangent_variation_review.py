import sys
from pathlib import Path
from fractions import Fraction as Q
import pytest
from flint import ctx,arb,arb_mat
sys.path[:0]=[str(Path(__file__).parents[2]/'tools/reviews')]
import review_tangent_variation_math as T
from neural_shrimp.rigorous_orbits import matrix_json,ball_json

def a(x):
 q=Q(x);return arb(q.numerator)/q.denominator

def test_exact_coupled_two_cycle_in_scalar_quartic_gradient_map():
 # F(x,h)=(1-h)x+h*x^3; cycle +/-sqrt(1-2/h).
 with ctx.workprec(256):
  h=a('8/3');v=arb_mat([[a('9/32')],[-a('9/32')],[1]])
  J=[arb_mat([[a('1/3')]])]*2;K=[arb_mat([[2]])]*2
  b=T.forcing(J,K,v,h,1)
  assert b[0,0].contains(a('63/128')) and b[1,0].contains(-a('63/128')) and b[2,0].is_zero()
  A=arb_mat([[a('1/3'),-1,-a('3/8')],[-1,a('1/3'),a('3/8')],[0,0,1]])
  C=A.inv().mid();z,_=T.solve(C,A,-b,['1']*3)
  for i,x in enumerate(['-189/512','189/512','0']):assert z[i,0].contains(a(x)) and z[i,0].rad()<a('1e-70')

def test_total_jprime_needs_second_parameter_cross_term():
 with ctx.workprec(256):
  b=T.forcing([arb_mat([[-35]])],[arb_mat([[-264]])],arb_mat([[5],[7]]),arb(3),1)
  assert b[0,0]==-1740 and not b[0,0].contains(-1320)

def test_interval_rhs_and_coupled_inverse_contains_all_corners():
 with ctx.workprec(256):
  A=arb_mat([[2,a('1/4')],[a('1/8'),1]])
  C=arb_mat([[a('1/2'),0],[0,1]])
  b=arb_mat([[arb(1,a('1/4'))],[arb(-2,a('1/8'))]])
  z,_=T.solve(C,A,b,['1','2'])
  for x in [a('3/4'),a('5/4')]:
   for y in [-a('17/8'),-a('15/8')]:
    exact=A.solve(arb_mat([[x],[y]]))
    assert all(z[i,0].contains(exact[i,0]) for i in range(2))

def test_output_rows_have_no_extra_hidden_rate_cross_term():
 b=T.forcing([arb_mat([[-2,0],[0,4]])],[arb_mat([[7,0],[0,11]])],arb_mat([[3],[5],[2]]),arb(3),1)
 assert b[0,0]==15 and b[1,0]==55 and b[2,0]==0

@pytest.mark.parametrize('bad',['q','tail','rhs','initial','image','retained','box'])
def test_saved_corruption_is_rejected(bad):
 from neural_shrimp.rigorous_tangent_variation import implicit_solve
 from copy import deepcopy
 with ctx.workprec(256):
  C=arb_mat([[a('1/2')]]);A=arb_mat([[a('3/2')]]);b=arb_mat([[arb(1,a('1/8'))]])
  rec,_=implicit_solve(C,A,b,[arb(1)]);T.solve(C,A,b,['1'],rec);rec=deepcopy(rec)
  if bad in ('q','tail'):rec[{'q':'weighted_neumann_bound','tail':'weighted_tail_radius'}[bad]]=ball_json(arb(0))
  elif bad in ('image','retained'):rec['refinements'][0][bad]=matrix_json(arb_mat([[99]]))
  else:rec[{'rhs':'preconditioned_rhs','initial':'initial_box','box':'box'}[bad]]=matrix_json(arb_mat([[99]]))
  with pytest.raises(ValueError):T.solve(C,A,b,['1'],rec)

@pytest.mark.parametrize('A,C,weights',[(0,1,['1']),(1,1,['0']),(1,1,['-1'])])
def test_singular_or_bad_weights_fail(A,C,weights):
 with pytest.raises(ValueError):T.solve(arb_mat([[C]]),arb_mat([[A]]),arb_mat([[1]]),weights)
