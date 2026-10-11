from fractions import Fraction
from math import factorial
import pytest
from flint import arb,arb_mat,ctx
from neural_shrimp import rigorous_clark_correlated_path as P
from neural_shrimp import rigorous_clark_explicit_fold as E

def setup():
 j=[[arb(1)/4,arb(1)/16],[arb(-1)/8,arb(1)/4]];c=[arb(1)/8,arb(-1)/4,arb(5)/8,arb(3)/4,arb(5)/8,arb(11)/8]+[arb(i-2)/32 for i in range(5)];s=[arb(i+1)/64 for i in range(11)];a=[arb((-1)**i)/128 for i in range(11)];return j,c,s,a,arb(1)/512

def test_entire_tail_derivatives_through_five_at_zero_and_negative_support():
 with ctx.workprec(384):
  for kind,offset in [('C',0),('S',1)]:
   ds=P.scalar_derivatives(arb(0),kind)
   for k,x in enumerate(ds):assert (x-arb(str(Fraction(factorial(k),factorial(2*k+offset))))).contains(0)
   assert all(x.is_finite()for x in P.scalar_derivatives(arb(0,arb(1)/8),kind))
  with pytest.raises(ValueError):P.scalar_derivatives(arb(17),'C')
  with pytest.raises(ValueError):P.scalar_derivatives(arb(0,1),'reciprocal')

def test_nested_path_mode_base_unary_has_required_fifth_derivative():
 with ctx.workprec(384):
  x=P.Gradient(P.Path([arb(1),1,0,0],[1,0,0,0]),[1]);y=P.unary(x,'sinh')+P.unary(x,'cosh')
  for k in range(4):
   expected=arb(1).exp()/factorial(k)
   for value in [y.v.a[k],y.v.b[k],y.g[0].a[k],y.g[0].b[k]]:assert (value-expected).contains(0)

def test_point_value_full_J_and_external_b_chain_against_original_mixedjet():
 with ctx.workprec(384):
  j,c,s,a,b=setup();f,J=P.coefficients(j,c,s,a,[arb(0)]*11,b,arb(0));value,old,fb,_=E.augmented(j,c,b)
  for i in range(11):
   assert (f[i,0]-value[i,0]).contains(0)
   assert (f[i,1]-sum((old[i,k]*s[k]for k in range(11)),arb(0))-fb[i,0]).contains(0)
   for k in range(11):assert (J[0][i,k]-old[i,k]).contains(0)
  assert J[0][10,8]==1 and all(J[k][10,l]==0 for k in (1,2)for l in range(11))

def test_whole_interval_Taylor_encloses_independent_endpoint_values_and_J():
 with ctx.workprec(384):
  j,c,s,a,b=setup();h=arb(1)/512;zero=[arb(0)]*11;f0,j0=P.coefficients(j,c,s,a,zero,b,arb(0));fw,jw=P.coefficients(j,c,s,a,zero,b,arb(0,h))
  for d in [-h,h]:
   z=[c[i]+s[i]*d+a[i]*d*d/2 for i in range(11)];value,J,_,_=E.augmented(j,z,b+d)
   for i in range(11):
    enclosure=f0[i,0]+f0[i,1]*d+f0[i,2]*d*d+fw[i,3]*d*d*d
    assert (enclosure-value[i,0]).contains(0)
    for k in range(11):assert (j0[0][i,k]+j0[1][i,k]*d+jw[2][i,k]*d*d-J[i,k]).contains(0)

def test_full_error_including_v_is_retained_at_d0_and_whole_d():
 with ctx.workprec(384):
  j,c,s,a,b=setup();err=[arb(0,arb((1,-30)))]*11;events=[];f,J=P.coefficients(j,c,s,a,err,b,arb(0),observe=lambda k,v:events.append((k,v)))
  old,Jo,_,_=E.augmented(j,[c[i]+err[i]for i in range(11)],b)
  for i in range(11):
   assert (f[i,0]-old[i,0]).contains(0)
   for k in range(11):assert (J[0][i,k]-Jo[i,k]).contains(0)
  assert sum(k=='clark_gradient_path_input'for k,_ in events)==3 and events[-1][1]['error']==P.vector_json(err)
  with pytest.raises(ValueError):P.coefficients(j,c,s,a,err[:-1],b,arb(0))
