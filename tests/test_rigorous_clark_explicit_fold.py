from flint import arb,arb_mat,ctx
import pytest
from neural_shrimp import rigorous_clark_explicit_fold as E
from neural_shrimp import rigorous_clark_amplitude_fold as A
from test_rigorous_clark_amplitude_fold import polynomial_family

@pytest.mark.parametrize('b',[arb(0),arb('1/64')])
def test_full_polynomial_fold_and_derivative_through_zero(b):
 with ctx.workprec(384):
  z=[arb(0),-2*b*b,arb(1),-4*b];u=z+[arb(0)]*3;f,J,h,data=E.augment(polynomial_family,u,b,1)
  assert all(x.is_zero()for row in f.tolist()for x in row)and not J.det().contains(0)
  t=-J.inv()*h;assert t[1,0].contains(-4*b)and t[3,0].contains(-4)and t[5,0].contains(0)
  assert data['paired_tangent'].nrows()==3

def test_all_explicit_base_tangent_and_parameter_columns():
 with ctx.workprec(384):
  u=list(map(arb,['1/128','-1/512','1','-1/16','1/64','1/128','-1/32']));b=arb('1/128');_,J,h,_=E.augment(polynomial_family,u,b,1);eps=arb((1,-100))
  for k in range(8):
   plus=u.copy();minus=u.copy();bp=b;bm=b
   if k<7:plus[k]+=eps;minus[k]-=eps
   else:bp+=eps;bm-=eps
   fp=E.augment(polynomial_family,plus,bp,1)[0];fm=E.augment(polynomial_family,minus,bm,1)[0]
   assert all(abs((fp[i,0]-fm[i,0])/(2*eps)-(J[i,k]if k<7 else h[i,0])).upper()<arb((1,-150))for i in range(7))

def test_exact_b0_organizer_anchor_and_authority_guard():
 with ctx.workprec(384):
  z=list(map(arb,[0,0,1,0]));known,report=E.conditional_organizer_anchor(polynomial_family,z,1,arb(-1));f,_,_,_=E.augment(polynomial_family,known,arb(0),1)
  assert all(v.contains(0)for row in f.tolist()for v in row)and known[5].contains(0)
  with pytest.raises(ValueError):E.conditional_organizer_anchor(polynomial_family,z,1,arb(0))
  z[-1]=1
  with pytest.raises(ValueError):E.conditional_organizer_anchor(polynomial_family,z,1,arb(-1))

def test_one_actual_native_pair_pass_matches_old_elimination_at_point():
 with ctx.workprec(384):
  j=[[arb('1/4'),arb('-1/8')],[arb('3/8'),arb('1/8')]];z=list(map(arb,['1/128','-1/128','3/4','1','0','2']));b=arb('1/256');old,_,_,v=A.augmented(j,z,b);u=z+[v[i,0]for i in range(5)];events=[];f,J,h,d=E.augmented(j,u,b,observe=lambda k,v:events.append(k))
  assert all((f[i,0]-old[i,0]).contains(0)for i in range(5))and all(f[i,0].contains(0)for i in range(5,10))and(f[10,0]-old[5,0]).contains(0)
  assert events.count('native_even_odd_phase')==3 and J.nrows()==11

def test_no_internal_inverse_required_to_evaluate_wide_or_singular_family(monkeypatch):
 with ctx.workprec(256):
  def family(z,b):return [z[0]*z[0]+b,z[1],z[2]-1]
  u=list(map(arb,[0,0,1,1,0,0,0]));f,J,h,_=E.augment(family,u,arb(0),1)
  assert all(x.is_finite()for row in J.tolist()for x in row)and f[3,0].contains(1)
  # A finite family value is explicitly not a successful root certificate.
  assert J.det().contains(0)
