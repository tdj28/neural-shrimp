import math
import mpmath as mp
import pytest
from flint import arb,ctx
from neural_shrimp.rigorous_cubic_unit_pair import QuarticPathJet as Q,cubic_path,entire_quartic,cubic_residual_coefficients


def poly(v,t):return sum(a*t**k for k,a in enumerate(v))
def mpv(v):return mp.mpf(float(v))


def test_full_quartic_tanh_value_and_mixed_direction_against_independent_derivatives():
 with ctx.workprec(256),mp.workdps(85):
  a=[arb(1)/8,arb(-1)/16,arb(1)/32,arb(3)/128,arb(-1)/256];b=[arb(1)/4,arb(1)/8,arb(-1)/16,arb(1)/32,arb(1)/64]
  j=Q(a,b).tanh();aa,bb=list(map(mpv,a)),list(map(mpv,b))
  for k in range(5):
   refs=[mp.diff(lambda d:mp.tanh(poly(aa,d)),0,k)/math.factorial(k),mp.diff(lambda d:(1-mp.tanh(poly(aa,d))**2)*poly(bb,d),0,k)/math.factorial(k)]
   assert all((got-arb(mp.nstr(ref,80))).contains(0)for got,ref in zip([j.a[k],j.b[k]],refs))


class Model:
 n=2
 def cusp_step(self,z,a,b):return [z[0]+a*z[1].tanh()+b*z[0]*z[1],z[1]+a*(z[0]*z[0]).tanh()+b*z[1]*z[1]]


def fixture():
 c=[arb(1)/8,-arb(1)/16,arb(1),arb(1)/4,-arb(1)/4,arb(1),arb(1)/16,arb(3)/32,arb(1)/64]
 s=[arb(i-4)/256 for i in range(9)];a=[arb(i%3-1)/512 for i in range(9)];b=[arb(i%4-2)/1024 for i in range(9)];n=[arb(i+1)/64 for i in range(9)]
 return c,s,a,b,n


def residual(v,n):
 x,y,q0,q1,r0,r1,a,b,t=v;c=mp.cos(mp.sqrt(t));e=mp.sin(mp.sqrt(t))/mp.sqrt(t);d=t*e
 f=[x+a*mp.tanh(y)+b*x*y,y+a*mp.tanh(x*x)+b*y*y]
 A=[[1+b*y,a*(1-mp.tanh(y)**2)+b*x],[2*a*x*(1-mp.tanh(x*x)**2),1+2*b*y]]
 q=[q0,q1];r=[r0,r1]
 return [f[0]-x,f[1]-y]+[sum(A[i][j]*q[j]for j in range(2))-c*q[i]+d*r[i]for i in range(2)]+[sum(A[i][j]*r[j]for j in range(2))-e*q[i]-c*r[i]for i in range(2)]+[(q0*q0+q1*q1-1)/2,q0*r0+q1*r1,sum(u*v for u,v in zip(n,v))]


def test_all_native_augmented_residual_rows_through_order4():
 with ctx.workprec(256),mp.workdps(85):
  c,s,a,b,n=fixture();calls=[];events=[];R=cubic_residual_coefficients(Model(),c,s,a,b,arb(0),1,n,[arb(0)]*9,arb(0),tick=lambda:calls.append(1),observe=lambda k,v:events.append((k,v)))
  cc,ss,aa,bb,nn=[list(map(mpv,row))for row in(c,s,a,b,n)]
  for row in range(9):
   for order in range(5):
    def f(d):return residual([cc[i]+ss[i]*d+aa[i]*d*d/2+bb[i]*d*d*d for i in range(9)],nn)[row]-(d if row==8 else 0)
    ref=mp.diff(f,0,order)/math.factorial(order)
    assert(R[row,order]-arb(mp.nstr(ref,80))).contains(0),(row,order)
  assert len(calls)==2 and [k for k,v in events[:4]]==['cubic_path_input','cubic_path_return']*2


def test_whole_quartic_taylor_derivative_enclosure_and_entire_at_zero():
 with ctx.workprec(256):
  c,s,a,b,n=fixture();h=arb(1)/128;whole=cubic_residual_coefficients(Model(),c,s,a,b,arb(0,h),1,n,[arb(0)]*9,arb(0))
  for d in [-h,arb(0),h]:
   narrow=cubic_residual_coefficients(Model(),c,s,a,b,d,1,n,[arb(0)]*9,arb(0))
   assert all(whole[i,4].contains(narrow[i,4])for i in range(9))
  cs,es,ds=entire_quartic(Q([0,1,0,0,0]))
  for k in range(5):
   assert(cs.a[k]-arb((-1)**k)/math.factorial(2*k)).contains(0)
   assert(es.a[k]-arb((-1)**k)/math.factorial(2*k+1)).contains(0)
  assert ds.a[0].contains(0)and all((ds.a[k]-es.a[k-1]).contains(0)for k in range(1,5))


def test_invalid_exact_predictor_or_angle_rejects_before_native():
 c,s,a,b,n=fixture();calls=[];c[0]=arb(0,1)
 with pytest.raises(ValueError):cubic_residual_coefficients(Model(),c,s,a,b,arb(0),1,n,[arb(0)]*9,arb(0),tick=lambda:calls.append(1))
 assert not calls
 with pytest.raises(ValueError):entire_quartic(Q([1,1,0,0,0]))
