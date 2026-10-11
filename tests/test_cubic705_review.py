import sys,copy,math
from pathlib import Path
import pytest,mpmath as mp
from flint import arb,arb_mat,ctx
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/reviews'))
import review_cubic705_math as H
from neural_shrimp.rigorous_orbits import ball_json
P=H.P


def scale(a,v):return [x*v for x in a]
def add(*rows):return [sum((a[i]for a in rows),arb(0))for i in range(4)]
def mul(*rows):
 out=P.const(arb(1))
 for row in rows:out=P.mul(out,row)
 return out

def encoded(rows):return [[ball_json(v)for v in row]for row in rows]

def fixture(d=arb(0),error=None):
 c=[arb(1)/8,-arb(1)/16,arb(1),arb(1)/4,-arb(1)/4,arb(1),arb(1)/16,arb(3)/32,arb(1)/64]
 s=[arb(i-4)/256 for i in range(9)];a=[arb(i%3-1)/512 for i in range(9)];normal=[arb(i+1)/64 for i in range(9)];error=error or [arb(0)]*9
 b=[arb(i%4-2)/2048 for i in range(9)];z=[[c[i]+s[i]*d+a[i]*d*d/2+b[i]*d*d*d+error[i],s[i]+a[i]*d+3*b[i]*d*d,a[i]/2+3*b[i]*d,b[i]]for i in range(9)];x,y=z[:2];aa,bb=z[-3:-1];one=P.const(arb(1));zero=P.const(arb(0));events=[]
 F=[add(x,mul(aa,y,y),mul(bb,x,y)),add(y,mul(aa,x,x,x),mul(bb,y))]
 DF=[[add(one,mul(bb,y)),add(scale(mul(aa,y),2),mul(bb,x))],[scale(mul(aa,x,x),3),add(one,bb)]]
 for label,group in [('q',2),('r',4)]:
  qx,qy=z[group:group+2];direction=[qx,qy];values=[add(mul(row[0],qx),mul(row[1],qy))for row in DF]
  mixed=[[mul(bb,qy),add(mul(bb,qx),scale(mul(aa,qy),2))],[scale(mul(aa,x,qx),6),zero]]
  rates=[[mul(y,y),mul(x,y)],[mul(x,x,x),y]];drates=[[scale(mul(y,qy),2),add(mul(y,qx),mul(x,qy))],[scale(mul(x,x,qx),3),qy]]
  inputs=[]
  for i in range(4):
   value=[z[i],direction[i]]if i<2 else[z[6+i-2],zero]
   inputs.append(dict(value=encoded(value),gradient=[encoded(H.scalar_path(arb(int(i==j))))for j in range(4)]))
  out=[]
  for i in range(2):out.append(dict(value=encoded([F[i],values[i]]),gradient=[encoded([DF[i][j],mixed[i][j]])for j in range(2)]+[encoded([rates[i][j],drates[i][j]])for j in range(2)]))
  events.append(dict(phase=0,direction=label,d=ball_json(d),inputs=inputs,returns=out))
 return c,s,a,b,error,events,normal

def residual(v,normal):
 x,y,q0,q1,r0,r1,a,b,t=v;cs=mp.cos(mp.sqrt(t));es=mp.sin(mp.sqrt(t))/mp.sqrt(t);ds=t*es
 f=[x+a*y*y+b*x*y,y+a*x*x*x+b*y];D=[[1+b*y,2*a*y+b*x],[3*a*x*x,1+b]];q=[q0,q1];r=[r0,r1]
 return [f[0]-x,f[1]-y]+[sum(D[i][j]*q[j]for j in range(2))-cs*q[i]+ds*r[i]for i in range(2)]+[sum(D[i][j]*r[j]for j in range(2))-es*q[i]-cs*r[i]for i in range(2)]+[(q0*q0+q1*q1-1)/2,q0*r0+q1*r1,sum(a*b for a,b in zip(normal,v))]


def test_cubic_native_jacobian_orders_match_separate_nested_differentiation():
 with ctx.workprec(256),mp.workdps(85):
  c,s,a,b,error,events,normal=fixture();J=H.jacobian_coefficients(c,s,a,b,error,arb(0),events,2,1,normal)
  cc,ss,aa,bb,nn=[[mp.mpf(float(v))for v in row]for row in(c,s,a,b,normal)]
  for order in range(3):
   for i in range(9):
    for j in range(9):
     def value(d):
      z=[cc[k]+ss[k]*d+aa[k]*d*d/2+bb[k]*d*d*d for k in range(9)]
      return mp.diff(lambda u:residual([z[k]+(u if k==j else 0)for k in range(9)],nn)[i],mp.mpf(0))
     x=mp.diff(value,mp.mpf(0),order)/math.factorial(order);assert(J[order][i,j]-arb(mp.nstr(x,80))).contains(0),(order,i,j)

def test_full_error_input_domain_and_basis_corruption_rejected():
 with ctx.workprec(256):
  d=arb(0,arb(1)/32);err=[arb(0,arb((1,-20)))]*9;c,s,a,b,error,e,n=fixture(d,err);J=H.jacobian_coefficients(c,s,a,b,error,d,e,2,1,n)
  for dd in (arb(1)/32,-arb(1)/32):
   for ee in (arb((1,-20)),-arb((1,-20))):
    cc=[v+ee for v in c];*_,ev,_=fixture(dd,[ee]*9);fine=H.jacobian_coefficients(cc,s,a,b,[arb(0)]*9,dd,ev,2,1,n)
    assert all(J[2][i,j].contains(fine[2][i,j])for i in range(9)for j in range(9))
  bad=copy.deepcopy(e);bad[0]['inputs'][0]['value'][0][3]=ball_json(arb(100))
  with pytest.raises(ValueError):H.jacobian_coefficients(c,s,a,b,error,d,bad,2,1,n)
  bad=copy.deepcopy(e);bad[0]['inputs'][0]['gradient'][0][0][0]=ball_json(arb(0))
  with pytest.raises(ValueError):H.jacobian_coefficients(c,s,a,b,error,d,bad,2,1,n)

def test_quartic_terminal_against_high_precision_composition():
 with ctx.workprec(256),mp.workdps(85):
  t=[arb(1)/16,arb(1)/32,-arb(1)/64,arb(1)/128,arb(0)];tt=[mp.mpf(float(v))for v in t]
  for offset in (0,1):
   out=H.terminal5(t,offset)
   def f(x):
    q=sum(tt[i]*x**i for i in range(5));return mp.cos(mp.sqrt(q))if offset==0 else mp.sin(mp.sqrt(q))/mp.sqrt(q)
   for j,v in enumerate(out):assert(v-arb(mp.nstr(mp.diff(f,mp.mpf(0),j)/math.factorial(j),80))).contains(0)

def quartic_fixture():
 # A separate hand-derived polynomial native map, including nonzero fourth
 # path coefficients from multiplication; no producer jet implementation.
 c,s,a,b,_,_,normal=fixture();z=[[c[i],s[i],a[i]/2,b[i],arb(0)]for i in range(9)];x,y=z[:2];aa,bb=z[-3:-1];zero=H.const5(arb(0));one=H.const5(arb(1));ad=H.add5;mu=H.mul5;sc=H.scale5
 Fv=[ad(x,ad(mu(aa,mu(y,y)),mu(bb,mu(x,y)))),ad(y,ad(mu(aa,mu(x,mu(x,x))),mu(bb,y)))];DF=[[ad(one,mu(bb,y)),ad(sc(mu(aa,y),2),mu(bb,x))],[sc(mu(aa,mu(x,x)),3),ad(one,bb)]];events=[]
 for name,group in [('q',2),('r',4)]:
  inputs=[encoded([z[i],z[group+i]])for i in range(2)]+[encoded([aa,zero]),encoded([bb,zero])]
  output=[encoded([Fv[i],ad(mu(DF[i][0],z[group]),mu(DF[i][1],z[group+1]))])for i in range(2)]
  events.append(dict(phase=0,direction=name,d=ball_json(arb(0)),inputs=inputs,returns=output))
 return c,s,a,b,normal,events

def test_all_quartic_residual_coordinates_and_acquired_cubic_path():
 with ctx.workprec(256),mp.workdps(85):
  c,s,a,b,n,e=quartic_fixture();out=H.residual_coefficients(c,s,a,b,arb(0),e,2,1,n,[arb(0)]*9,arb(1)/8);cc,ss,aa,bb,nn=[[mp.mpf(float(v))for v in row]for row in(c,s,a,b,n)]
  for i in range(9):
   def f(d):
    z=[cc[k]+ss[k]*d+aa[k]*d*d/2+bb[k]*d*d*d for k in range(9)];v=residual(z,nn)[i];return v-mp.mpf(1)/8-d if i==8 else v
   for order in range(5):assert(out[i,order]-arb(mp.nstr(mp.diff(f,mp.mpf(0),order)/math.factorial(order),80))).contains(0),(i,order)
  bad=copy.deepcopy(e);bad[0]['inputs'][0][0][4]=ball_json(arb(1))
  with pytest.raises(ValueError):H.residual_coefficients(c,s,a,b,arb(0),bad,2,1,n,[arb(0)]*9,arb(1)/8)

def test_degree_five_noncommuting_forcing_and_strict_failure():
 with ctx.workprec(180):
  C=arb_mat([[1,arb(1)/4],[0,1]]);J0=arb_mat([[1,-arb(1)/4],[0,1]]);J1=arb_mat([[0,arb(1)/8],[arb(1)/16,0]]);J2=arb_mat([[arb(1)/32,0],[0,arb(1)/64]]);r=[arb_mat([[arb((1,-30))],[0]])]+[arb_mat(2,1)for _ in range(3)]+[arb_mat([[arb(1)/1024],[arb(1)/512]])];seen={};h=arb(1)/16;rho=arb(1)/1024
  proof=H.polynomial_root(C,[J0,J1,J2],r,[arb(1)]*2,rho,h,[arb(0,rho)]*2,emit=lambda k,v:seen.__setitem__(k,v));assert proof['passed']
  C1=seen['inverse_linear_proposal']
  for d in(-h,h):
   direct=-(C+C1*d)*sum((r[k]*d**k for k in range(5)),arb_mat(2,1));poly=sum((seen['forcing_coefficient_'+str(k)]*d**k for k in range(6)),arb_mat(2,1))
   assert all((direct[i,0]-poly[i,0]).contains(0)for i in range(2))
  r[0]=arb_mat([[1],[1]]);assert not H.polynomial_root(C,[J0,J1,J2],r,[arb(1)]*2,rho,h,[arb(0,rho)]*2)['passed']

def test_complete_cubic_anchor_membership_and_exact_gauge_guard():
 with ctx.workprec(180):
  c=[arb(0),arb(0)];s=[arb(1),arb(0)];a=[arb(0)]*2;b=[arb(0),arb(1)];n=[arb(1),arb(0)];o=[arb(0)]*2;d=arb(1)/32;anchor=[ball_json(d),ball_json(d*d*d)]
  good=H.cubic_membership(anchor,c,s,a,b,n,o,arb(0),arb(1)/16,[arb(1)/1024]*2,known_parameter='1/32');assert good['passed']and good['coordinates']==2
  with pytest.raises(ValueError):H.cubic_membership(anchor,c,s,a,b,n,o,arb(0),arb(1)/16,[arb(1)/1024]*2,known_parameter='1/16')
