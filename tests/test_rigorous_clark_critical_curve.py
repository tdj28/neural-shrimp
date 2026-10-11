from flint import arb,arb_mat,ctx
import pytest
from neural_shrimp import rigorous_clark_critical_curve as C
from neural_shrimp import rigorous_clark_generalized_flip as G


def test_native_critical_family_matches_full_cubic_equations_and_derivatives():
 with ctx.workprec(256):
  j=[[arb('2/5'),arb('-3/20')],[arb('1/5'),arb('3/10')]];z=[arb('1/5'),arb('-3/10'),arb('7/10'),arb('4/5'),arb('3/5'),arb('6/5')]
  h,J,_=C.critical(j,z,z[-1]);old,A,_=G.augmented(j,z)
  assert all(h[i,0].overlaps(old[i,0])and all(J[i,k].overlaps(A[i,k])for k in range(6))for i in range(5))
  assert h[5,0].contains(0) and all(J[5,k]==int(k==5)for k in range(6))
  delta=arb((1,-20))
  for k in range(6):
   plus=z.copy();minus=z.copy();plus[k]+=delta;minus[k]-=delta
   hp,_,_=C.critical(j,plus,z[-1]);hm,_,_=C.critical(j,minus,z[-1])
   assert max(float(abs((hp[i,0]-hm[i,0])/(2*delta)-J[i,k]))for i in range(6))<2e-11


def quadratic(z,s):
 x,g=z;return arb_mat([[x-g*g],[g-s]]),arb_mat([[1,-2*g],[0,1]]),{}


def test_uniform_mean_value_tube_and_full_known_root_join():
 with ctx.workprec(256):
  c=[arb(1),arb(1)];r=[arb(1)/64]*2;whole,v,_,report=C.tube(quadratic,c,arb(1),arb(1)/16,r)
  assert report['passed']and all(report['coordinate_inclusions'])and v==[arb(2),arb(1)]
  exact=[arb('1089/1024'),arb('33/32')]
  assert C.anchor(exact,c,v,arb(1),arb(1)/16,r)['passed']
  assert not C.anchor([arb(2),arb('33/32')],c,v,arb(1),arb(1)/16,r)['passed']
  for k in range(-16,17):
   g=arb(1)+arb(k)/256;assert whole[0].contains(g*g)and whole[1].contains(g)


def test_too_small_tube_is_retained_as_failed_without_radius_relaxation():
 with ctx.workprec(256):
  _,_,_,report=C.tube(quadratic,[arb(1)]*2,arb(1),arb(1)/16,[arb(1)/65536]*2)
  assert not report['passed']and not all(report['coordinate_inclusions'])
  with pytest.raises(ValueError):C.tube(quadratic,[arb(1)]*2,arb(2),arb(1)/16,[arb(1)/64]*2)


def test_literal_native_transport_retains_six_phases_and_mode_normalization():
 with ctx.workprec(256):
  j=[[arb(1),arb(0)],[arb(0),arb('1/2')]];box=[arb(0),arb(0),arb('3/4'),arb(1),arb(0),arb(1)];events=[]
  transformed,r=C.native_transport(j,box,observe=lambda k,v:events.append((k,v)))
  assert transformed==box and len(r['orbit'])==len(r['modes'])==7 and len(r['native_jacobians'])==6
  assert sum(k=='native_transport_phase'for k,v in events)==6 and r['phase']==6
  with pytest.raises(ValueError):C.native_transport(j,box,steps=4)
  bad=box.copy();bad[3]=arb(0)
  with pytest.raises(ValueError):C.native_transport(j,bad)
