from fractions import Fraction as F
from flint import arb,arb_mat,ctx
import pytest
from neural_shrimp import rigorous_clark_amplitude_fold as A
J=A.MixedJet

def test_entire_series_values_and_derivatives_with_tails():
 with ctx.workprec(384):
  z=arb('1/4');w=z.sqrt();sh=w.sinh();ch=w.cosh()
  expected=[[ch,sh/(2*w),(w*ch-sh)/(4*w**3)],[sh/w,(w*ch-sh)/(2*w**3),((w*w+3)*sh-3*w*ch)/(4*w**5)]]
  for o in(0,1):
   for d in(0,1,2):
    assert A.entire(z,o,d).overlaps(expected[o][d])
    assert (A.entire(arb(0),o,d)*A.factorial(2*d+o)).contains(A.factorial(d))
  with pytest.raises(ValueError):A.entire(arb(17),0)


def test_tanh_pair_has_exact_analytic_zero_amplitude_limit():
 with ctx.workprec(384):
  m=J.variable(arb('1/3'),3,0);q=J.variable(arb('2/5'),3,1);b=J.variable(arb(0),3,2)
  even,odd=A.tanh_pair(m,q,b);t=m.f.tanh();sech=1-t*t
  assert even.f.overlaps(t)and odd.f.overlaps(q.f*sech)
  assert even.g[2].overlaps(-t*sech*q.f*q.f)
  assert odd.g[2].overlaps((4*t*t*sech-2*sech*sech)*q.f**3/6)


def test_literal_native_three_phase_pair_matches_direct_positive_b():
 with ctx.workprec(384):
  j=[[arb('1/3'),arb('-1/4')],[arb('2/5'),arb('1/7')]];z=list(map(arb,['1/8','-1/5','3/4','3/5','4/5','2']));b=arb('1/64');v=[J.variable(x,7,i)for i,x in enumerate(z+[b])];got=A.paired_family(j,v[:-1],v[-1]);amp=b.sqrt();m=z[:2];q=z[3:5]
  paths=[]
  for sign in(1,-1):
   x=[m[i]+sign*amp*q[i]for i in range(2)]
   for _ in range(3):x=[(1-z[2])*x[i]+z[2]*z[-1]*sum(j[i][k]*x[k].tanh()for k in range(2))for i in range(2)]
   paths.append([-s for s in x])
  want=[(paths[0][i]+paths[1][i])/2-m[i]for i in range(2)]+[(paths[0][i]-paths[1][i])/(2*amp)+q[i]for i in range(2)]
  assert all(got[i].f.overlaps(want[i])for i in range(4))and got[4].f.contains(0)


def polynomial_family(z,b):
 # G(x)=-(1+mu)x+a*x³+2*x⁵; exactfold a=-4b,mu=-2b².
 m,mu,q,a=z;m2=m*m;q2=q*q
 even=-(2+mu)*m+a*(m*m2+3*m*b*q2)+2*(m*m2*m2+10*m*m2*b*q2+5*m*b*b*q2*q2)
 odd=-mu*q+a*(3*m2*q+b*q*q2)+2*(5*m2*m2*q+10*m2*b*q*q2+b*b*q*q2*q2)
 return [even,odd,(q2-1)*arb('1/2')]

@pytest.mark.parametrize('b',[arb(0),arb('1/64')])
def test_full_augmented_fold_and_external_amplitude_derivative(b):
 with ctx.workprec(384):
  z=[arb(0),-2*b*b,arb(1),-4*b];f,j,h,v=A.augment(polynomial_family,z,b,1)
  assert all(x.is_zero()for row in f.tolist()for x in row)
  assert j[3,3].contains(1)and h[3,0].contains(4)
  assert j.det().abs_lower()>0
  tangent=-(j.inv()*h);assert tangent[1,0].contains(-4*b)and tangent[3,0].contains(-4)
  assert v[1,0].is_zero()


def test_mixed_fold_gradient_includes_all_base_and_b_columns():
 with ctx.workprec(384):
  z=list(map(arb,['1/128','-1/512','1','-1/16']));b=arb('1/128');_,jac,h,_=A.augment(polynomial_family,z,b,1);eps=arb((1,-100))
  for k in range(5):
   plus=list(z);minus=list(z);bp=b;bm=b
   if k<4:plus[k]+=eps;minus[k]-=eps
   else:bp+=eps;bm-=eps
   fp=A.augment(polynomial_family,plus,bp,1)[0];fm=A.augment(polynomial_family,minus,bm,1)[0]
   for i in range(4):assert abs((fp[i,0]-fm[i,0])/(2*eps)-(jac[i,k]if k<4 else h[i,0])).upper()<arb((1,-150))


def test_entire_polynomial_stays_finite_through_negative_and_zero_containing_z():
 with ctx.workprec(384):
  for z in (arb(0,arb('1/8')),arb('-1/4')):
   for offset in (0,1):
    for derivative in (0,1,2):assert A.entire(z,offset,derivative).is_finite()
  z=arb('-1/4');assert A.entire(z,0).overlaps(arb('1/2').cos())and A.entire(z,1).overlaps(2*arb('1/2').sin())
  interval=arb(0,arb('1/8'));assert A.entire(interval,0).contains(1)
