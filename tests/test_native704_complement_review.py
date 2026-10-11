import importlib.util
from pathlib import Path
from flint import arb,arb_mat,ctx
import pytest
p=Path(__file__).parents[1]/'tools/reviews/review_native704_complement_math.py';sp=importlib.util.spec_from_file_location('native704_audit',p);A=importlib.util.module_from_spec(sp);sp.loader.exec_module(A)

def test_known_root_enclosure_and_noncontracting_rejection():
 with ctx.workprec(480):
  D=arb_mat([[arb('1/8'),arb(0)],[arb(0),arb('1/16')]]);v=arb_mat([[arb(1)],[arb(-1)]]);x=A.enclosure(D,v,[arb(1),arb(1)],arb('1/8'))
  assert x[0,0].contains(arb(8)/7)and x[1,0].contains(-arb(16)/15)
  with pytest.raises(ValueError):A.enclosure(D,v,[arb(1),arb(1)],arb(1))

def test_common_parameter_polynomial_order_and_even_powers():
 x=[arb_mat([[arb(1),arb(2)],[arb(0),arb(1)]]),arb_mat([[arb(0),arb(1)],[arb(1),arb(0)]])];y=[x[0].transpose(),-x[1]];out=A.convolution(x,y)
 for i in range(2):
  for j in range(2):assert out[1][i,j]==(x[0]*y[1]+x[1]*y[0])[i,j]
 z=A.polynomial([arb_mat([[arb(0)]]),arb_mat([[arb(0)]]),arb_mat([[arb(1)]])]);assert z[0,0].contains(0)and z[0,0].contains(1)and not z[0,0].contains(-1)

def fixture(multiplier):
 flat=[]
 for _ in range(18):flat += [arb(0)]*13+[arb(int(i==0))for i in range(13)]+[arb(int(i==1))for i in range(13)]
 flat += [arb(1),arb(1)];ee=arb_mat([[arb(int(i==k+2))for k in range(11)]for i in range(13)]);native=[];phases=[];h=arb((1,-16));term=A.U.terminal(2,h+arb(0,h));tc=A.U.terminal(2,h)
 for j in range(18):
  mat=A.M.ident(13)
  for k in range(2,13):mat[k,k]=multiplier
  whole=mat+arb_mat(13,13)
  if j==17:
   for r,k,aa,bb in [(0,0,2,2),(0,1,4,2),(1,0,2,4),(1,1,4,4)]:mat[r,k]=tc[aa,bb];whole[r,k]=term[aa,bb]
  native.append((mat,whole,arb_mat(13,13)));phases.append(dict(phase=j,coefficients=[A.matrix_json(A.mid(A.M.ident(11)*multiplier)),A.matrix_json(arb_mat(11,11)),A.matrix_json(arb_mat(11,11))],operator_bound=dict(gamma=A.ball_json(arb(2)),positive=dict(basis=A.matrix_json(A.M.ident(11))))))
 metric=dict(P=A.matrix_json(A.M.ident(11)),P_positive=dict(basis=A.matrix_json(A.M.ident(11))),complete_D_positive=dict(basis=A.matrix_json(A.M.ident(11))))
 return flat,flat,[arb(0)]*704,[ee]*18,[A.M.ident(11)]*18,native,phases,metric

def test_full18phase11mode_strict_decrease_and_modulus():
 with ctx.workprec(480):
  r=A.reduction(*fixture(arb('9/10')));assert r['passed']and r['complement_dimension']==11 and A.R.original_ball(r['complementary_modulus_upper'])<1

def test_unstable_complement_not_promoted_by_neutral_critical_plane():
 with ctx.workprec(480):
  with pytest.raises(ValueError,match='Lyapunov decrease'):A.reduction(*fixture(arb('101/100')))

def test_six_own_intersections_keep_known_solution_and_tighten():
 with ctx.workprec(480):
  d=arb_mat([[arb('1/2')]]);v=arb_mat([[arb(1)]]);w=[arb(1)];k=arb('1/2')
  a=A.enclosure(d,v,w,k,iterations=3);b=A.enclosure(d,v,w,k,iterations=6)
  assert a[0,0].contains(2)and b[0,0].contains(2)and b[0,0].lower()>a[0,0].lower()
  with pytest.raises(ValueError):A.enclosure(d,v,w,k,iterations=7)
