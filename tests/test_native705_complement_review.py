import hashlib,importlib.util,sys
from pathlib import Path
from fractions import Fraction as F
from flint import arb,arb_mat,ctx
import pytest
ROOT=Path(__file__).parents[1];sys.path.insert(0,str(ROOT/'tools/reviews'))
import review_native705_complement_math as A
from neural_shrimp.rigorous_orbits import ball_json

def test_exact_reviewed_pal_helper_copy():
 assert hashlib.sha256((ROOT/'tools/reviews/native704_aux/review_unit_pair_pal_math.py').read_bytes()).hexdigest()=='d9808644ffc23b7da7d49bc0efd546a902aeb5e7a682d42e3c629991df9b9377'

@pytest.mark.parametrize('power',[1,3])
def test_independent_full_augmented_polarization_all_modes_rates_angle(power):
 with ctx.workprec(480):
  z=list(map(arb,[2,3,1,2,3,4,5,6,0]));e=[arb(str(F(i+1,32)))for i in range(9)];v=[arb(str(F(10-i,64)))for i in range(9)];b=arb('1/256');plus=[e[i]/b+b*v[i]for i in range(9)];minus=[e[i]/b-b*v[i]for i in range(9)];events=[]
  def mul(x,y):return [sum((x[k]*y[j-k]for k in range(j+1)),arb(0))for j in range(4)]
  def power_poly(x,k):
   out=[arb(1),arb(0),arb(0),arb(0)]
   for _ in range(k):out=mul(out,x)
   return out
  for slope in (plus,minus):
   coeff=[[z[i],slope[i],arb(0),arb(0)]for i in range(9)]
   for j,name in [(1,'q'),(2,'r')]:
    inputs=[[coeff[i],coeff[2*j+i]]for i in range(2)]+[[coeff[6+i],[arb(0)]*4]for i in range(2)]
    returns=[]
    for i in range(2):returns.append([mul(coeff[6+i],power_poly(coeff[i],power)),[power*x for x in mul(mul(coeff[6+i],power_poly(coeff[i],power-1)),coeff[2*j+i])]])
    events.append(dict(phase=0,direction=name,d=ball_json(arb(0)),inputs=[[[ball_json(t)for t in half]for half in item]for item in inputs],returns=[[[ball_json(t)for t in half]for half in item]for item in returns]))
  normal=[arb(0)]*8+[arb(1)];got=A.polarized_paths(z,plus,minus,events,normal,[arb(0)]*9,arb(0),2,1);want=[];m=power
  for i in range(2):want.append(m*z[i]**(m-1)*(e[6+i]*v[i]+v[6+i]*e[i])+z[6+i]*m*(m-1)*z[i]**(m-2)*e[i]*v[i])
  def mode(i,k):
   x=z[i];a=z[6+i];q=z[k+i]
   return m*((m-1)*x**(m-2)*q*(e[6+i]*v[i]+v[6+i]*e[i])+x**(m-1)*(e[6+i]*v[k+i]+v[6+i]*e[k+i])+a*(m-1)*(m-2)*x**(m-3)*e[i]*v[i]*q+a*(m-1)*x**(m-2)*(e[i]*v[k+i]+v[i]*e[k+i]))
  for i in range(2):want.append(mode(i,2)+(e[8]*v[2+i]+v[8]*e[2+i])/2-e[8]*v[8]*z[2+i]/12+e[8]*v[4+i]+v[8]*e[4+i]-e[8]*v[8]*z[4+i]/3)
  for i in range(2):want.append(mode(i,4)+(e[8]*v[4+i]+v[8]*e[4+i])/2-e[8]*v[8]*z[4+i]/12+(e[8]*v[2+i]+v[8]*e[2+i])/6-e[8]*v[8]*z[2+i]/60)
  want += [sum(e[2+i]*v[2+i]for i in range(2)),sum(e[2+i]*v[4+i]+e[4+i]*v[2+i]for i in range(2)),arb(0)]
  assert all(got[i,0].contains(x)for i,x in enumerate(want))
  events[0]['inputs'][0][0][0]=ball_json(arb(9))
  with pytest.raises(ValueError):A.polarized_paths(z,plus,minus,events,normal,[arb(0)]*9,arb(0),2,1)

def fixture(multiplier):
 flat=[]
 for _ in range(18):flat += [arb(0)]*13+[arb(int(i==0))for i in range(13)]+[arb(int(i==1))for i in range(13)]
 flat += [arb(1),arb(1)];ee=arb_mat([[arb(int(i==k+2))for k in range(11)]for i in range(13)]);native=[];phases=[];angle=arb('1/64','1/256');term=A.U.terminal(2,angle);tc=A.U.terminal(2,arb('1/64'))
 for j in range(18):
  mat=A.M.ident(13)
  for k in range(2,13):mat[k,k]=multiplier
  whole=mat+arb_mat(13,13)
  if j==17:
   for r,k,aa,bb in [(0,0,2,2),(0,1,4,2),(1,0,2,4),(1,1,4,4)]:mat[r,k]=tc[aa,bb];whole[r,k]=term[aa,bb]
  native.append((mat,whole,arb_mat(13,13)));phases.append(dict(phase=j,coefficients=[A.matrix_json(A.H.mid(A.M.ident(11)*multiplier)),A.matrix_json(arb_mat(11,11)),A.matrix_json(arb_mat(11,11))],operator_bound=dict(gamma=A.ball_json(arb(2)),positive=dict(basis=A.matrix_json(A.M.ident(11))))))
 metric=dict(P=A.matrix_json(A.M.ident(11)),P_positive=dict(basis=A.matrix_json(A.M.ident(11))),complete_D_positive=dict(basis=A.matrix_json(A.M.ident(11))))
 return flat,flat,[arb(0)]*704,[ee]*18,[A.M.ident(11)]*18,native,phases,metric,arb('1/128'),angle

def test_full18phase_positive_angle_with_independent_gauge_halfwidth():
 with ctx.workprec(480):
  r=A.reduction(*fixture(arb('9/10')));assert r['passed']and A.R.original_ball(r['complementary_modulus_upper'])<1
  with pytest.raises(ValueError,match='Lyapunov decrease'):A.reduction(*fixture(arb('101/100')))
