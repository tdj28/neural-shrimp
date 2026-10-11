from flint import arb,arb_mat,ctx
from neural_shrimp import rigorous_clark_amplitude_tube as T
from neural_shrimp import rigorous_clark_amplitude_fold as A
from test_rigorous_clark_amplitude_fold import polynomial_family


def family(z,b):return A.augment(polynomial_family,z,b,1)


def test_general_parameter_derivative_and_closed_endpoint_root_join():
 with ctx.workprec(384):
  s=arb('1/64');h=arb('1/128');c=[arb(0),-2*s*s,arb(1),-4*s];r=[arb('1/256')]*4
  whole,v,_,J,hb,ends,proof=T.tube(family,c,s,h,r)
  assert proof['passed']and v[1].contains(-4*s)and v[-1].contains(-4)
  for b in(s-h,s,s+h):
   exact=[arb(0),-2*b*b,arb(1),-4*b]
   assert T.join(exact,b,c,v,s,h,r)['passed']
   assert all(whole[i].contains(exact[i])for i in range(4))
  assert T.join(ends[0],s-h,c,v,s,h,r)['passed']
  assert not T.join(ends[0],s-2*h,c,v,s,h,r)['passed']
  wrong=list(ends[0]);wrong[-1]+=1
  assert not T.join(wrong,s-h,c,v,s,h,r)['passed']


def test_failed_tube_is_reported_without_radius_adaptation():
 with ctx.workprec(384):
  b=arb('1/64');_,_,_,_,_,_,proof=T.tube(family,[arb(0),-2*b*b,arb(1),-4*b],b,arb('1/128'),[arb((1,-40))]*4)
  assert not proof['passed']and not all(proof['coordinate_inclusions'])


def test_fixed_gain_columns_and_fold_curvature():
 with ctx.workprec(384):
  b=arb('1/64');gain=-4*b;z=[arb(0),-2*b*b,arb(1),gain];f,J,data=T.fixed_gain(family,z[:-1]+[b],gain)
  assert all(x.contains(0)for row in f.tolist()for x in row)and not J.det().contains(0)
  eps=arb((1,-100));unknown=z[:-1]+[b]
  for k in range(4):
   p=unknown.copy();m=unknown.copy();p[k]+=eps;m[k]-=eps
   plus=T.fixed_gain(family,p,gain)[0];minus=T.fixed_gain(family,m,gain)[0]
   assert all(abs((plus[i,0]-minus[i,0])/(2*eps)-J[i,k]).upper()<arb((1,-150))for i in range(4))
  _,j,h,v=family(z,b);curvature=T.fold_curvature(j,h,v,1,b)
  assert curvature['passed']and T.fold_curvature(j,h,v,1,arb(0))['passed']is False
  # mu(b)=a*b+2*b², so mu_bb=4 at the fixed gain a.
  assert T.arb(T.arb(tuple(curvature['leak_bb']['mid'])),T.arb(tuple(curvature['leak_bb']['rad']))).contains(4)


def test_conditional_native_context_never_promotes_a_fixed_point_to_period_six():
 with ctx.workprec(384):
  # At b=0, G has one -1 and one stable mode; F fixes x=0.
  j=[[arb(1),arb(0)],[arb(0),arb(0)]];z=[arb(0),arb(0),arb('1/2'),arb(1),arb(0),arb(1)];v=arb_mat(5,1);events=[]
  out=T.conditional_native(j,z,arb(0),v,observe=lambda k,v:events.append(k))
  assert not out['passed']and not out['positive_amplitude']and len(out['contexts'])==4
  assert all(c['witness_power']==1 and not any(d['excluded']for d in c['divisors'])for c in out['contexts'])
  assert events.count('paired_native_phase')==6
