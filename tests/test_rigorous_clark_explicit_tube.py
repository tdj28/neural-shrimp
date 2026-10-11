from flint import arb,arb_mat,ctx
import pytest
from neural_shrimp import rigorous_clark_explicit_fold as E
from neural_shrimp import rigorous_clark_explicit_tube as T
from test_rigorous_clark_amplitude_fold import polynomial_family


def family(u,b):return E.augment(polynomial_family,u,b,1)


def test_explicit_critical_tube_and_all_coordinate_join():
 with ctx.workprec(384):
  b=arb(1)/64;h=arb(1)/128;c=[arb(0),-2*b*b,arb(1),-4*b]+[arb(0)]*3;r=[arb(1)/256]*7
  whole,v,_,J,hb,ends,proof=T.tube(family,c,b,h,r)
  assert proof['passed']and len(proof['coordinate_inclusions'])==7
  for x in [b-h,b,b+h]:
   exact=[arb(0),-2*x*x,arb(1),-4*x]+[arb(0)]*3
   assert T.join(exact,x,c,v,b,h,r)['passed']
  wrong=ends[0].copy();wrong[-1]+=1;assert not T.join(wrong,b-h,c,v,b,h,r)['passed']


def test_full51_dimension_and_negative_radius():
 with ctx.workprec(192):
  def linear(z,b):return T.col([x-b for x in z]),T.eye(51),T.col([-arb(1)]*51),None
  c=[arb(1)]*51;h=arb(1)/128;r=[arb(1)/1024]*51
  got=T.tube(linear,c,arb(1),h,r);assert got[-1]['passed']and got[-1]['dimension']==51
  assert T.join([arb(1)-h]*51,arb(1)-h,c,got[1],arb(1),h,r)['coordinates']==51
  _,_,_,root=T.root_enclosure(lambda z:(T.col([x-1 for x in z]),T.eye(51),None),c,newton_steps=0,radius_exponent=-80);assert root['passed']and root['dimension']==51
  bad=r.copy();bad[50]=arb(-1)
  with pytest.raises(ValueError):T.tube(linear,c,arb(1),h,bad)


def test_fixed_gain_columns_and_full_root():
 with ctx.workprec(384):
  b=arb(1)/64;g=-4*b;u=[arb(0),-2*b*b,arb(1)]+[arb(0)]*3+[b];f,J,data=T.fixed_gain(family,u,g,1);assert all(v.is_zero()for row in f.tolist()for v in row)and not J.det().contains(0)
  e=arb((1,-100))
  for k in range(7):
   plus=u.copy();minus=u.copy();plus[k]+=e;minus[k]-=e
   a=T.fixed_gain(family,plus,g,1)[0];z=T.fixed_gain(family,minus,g,1)[0]
   assert all(abs((a[i,0]-z[i,0])/(2*e)-J[i,k]).upper()<arb((1,-150))for i in range(7))
  box,_,_,proof=T.root_enclosure(lambda z:T.fixed_gain(family,z,g,1),u,newton_steps=0,radius_exponent=-80);assert proof['passed']and all(proof['coordinate_inclusions'])


def test_endpoint_gate_rejects_zero_before_native_calls():
 with pytest.raises(ValueError):T.conditional_endpoint_fold([[arb(1)]],[arb(0)]*7,arb(0))
