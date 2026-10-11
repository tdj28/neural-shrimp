import sys
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/reviews'))
import review_clark_organizer_math as M


def test_inverse_and_strict_root_negative():
 with ctx.workprec(160):
  A=arb_mat([[2,1],[1,3]]);I=M.inverse(A)*A;assert all(I[i,j].contains(int(i==j))for i in range(2)for j in range(2))
  with pytest.raises((ValueError,ZeroDivisionError)):M.inverse(arb_mat([[1,1],[1,1]]))
  r=M.krawczyk([arb(0)]*2,[arb(0,arb('0.001'))]*2,M.col([0,0]),A);assert all(r['coordinate_inclusions'])
  with pytest.raises(ValueError):M.krawczyk([arb(0)]*2,[arb(0,arb('0.001'))]*2,M.col([1,0]),A)


def test_exp_tanh_coefficients_and_whole_interval():
 with ctx.workprec(200):
  got=M.tanh_series([arb(0),arb(1),arb(0),arb(0),arb(0),arb(0)])
  for v,e in zip(got,[0,1,0,-arb(1)/3,0,arb(2)/15]):assert(v-e).contains(0)
  x=[arb(arb('.2'),arb('.01')),arb('.7'),arb('.03'),arb('.01'),arb(0),arb(0)]
  whole=M.tanh_series(x)
  for point in('.19','.2','.21'):
   y=M.tanh_series([arb(point)]+x[1:]);assert all(a.contains(b)for a,b in zip(whole,y))


def test_signed_path_independent_composition():
 with ctx.workprec(200):
  h=[[arb(0)],[arb(1)]]+[[arb(0)]for _ in range(4)];out=M.signed_path(h,[[arb(1)]],arb(1),arb(1))
  # -tanh(tanh(tanh u)) = -u+u^3-7u^5/5.
  assert(out[3][0]-1).contains(0);assert(out[5][0]+arb(7)/5).contains(0)


def test_odd_gauge_quintic_and_degenerate_rejection():
 with ctx.workprec(180):
  def run(c):
   def poly(h):
    u=[r[0]for r in h];f=[-v for v in u];f[5]+=c*u[1]**5;return [[v]for v in f]
   return M.normal_form([arb(0)],arb_mat([[-1]]),[arb(1)],[arb(1)],poly)
  assert(run(arb(3))['quintic']-3).contains(0)
  with pytest.raises(ValueError):run(arb(0))


def test_physical_determinant_and_degenerate():
 with ctx.workprec(200):
  J=arb_mat([[-2,0,0,0],[0,-3,0,-2],[0,0,1,0],[0,5,0,7]]);out=M.unfolding(J,arb_mat([[-1]]),[arb(1)])
  assert(out['physical_determinant']-11).contains(0)
  J[3,3]=arb(10)/3
  with pytest.raises(ValueError):M.unfolding(J,arb_mat([[-1]]),[arb(1)])


def test_fixedpoint_cannot_pass_native_primitive6():
 with ctx.workprec(160):
  with pytest.raises(ValueError,match='proper divisors'):M.native([[arb(1),arb(0)],[arb(0),arb('.5')]],[arb(0),arb(0),arb('.5'),arb(1),arb(0),arb(1)])


def test_curved_stable_center_manifold_reduction():
 with ctx.workprec(220):
  def mul(a,b):return [sum((a[i]*b[k-i]for i in range(k+1)),arb(0))for k in range(6)]
  beta,gamma,rate=arb(1)/4,arb(3)/8,arb(1)/4
  def path(h):
   u,v=[r[0]for r in h],[r[1]for r in h];u2=mul(u,u);u3=mul(u2,u);u5=mul(u3,u2);f=[-u[k]+3*u5[k]for k in range(6)];f2=mul(f,f);f3=mul(f2,f)
   return [[f[k],rate*(v[k]-beta*u2[k]-gamma*u3[k])+beta*f2[k]+gamma*f3[k]]for k in range(6)]
  r=M.normal_form([arb(0)]*2,arb_mat([[-1,0],[0,rate]]),[arb(1),arb(0)],[arb(1),arb(0)],path)
  assert(r['quintic']-3).contains(0);assert(r['parameterization'][2][1]-beta).contains(0);assert(r['parameterization'][3][1]-gamma).contains(0)
