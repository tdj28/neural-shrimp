import sys,copy,math
from pathlib import Path
import pytest,mpmath as mp
from flint import arb,arb_mat,ctx
sys.path.insert(0,str(Path(__file__).parents[1]/'tools/reviews'))
import review_correlated705_math as H
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
 z=[[c[i]+s[i]*d+a[i]*d*d/2+error[i],s[i]+a[i]*d,a[i]/2,arb(0)]for i in range(9)];x,y=z[:2];aa,bb=z[-3:-1];one=P.const(arb(1));zero=P.const(arb(0));events=[]
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
 return c,s,a,error,events,normal


def residual(v,normal):
 x,y,q0,q1,r0,r1,a,b,t=v;cs=mp.cos(mp.sqrt(t));es=mp.sin(mp.sqrt(t))/mp.sqrt(t);ds=t*es
 f=[x+a*y*y+b*x*y,y+a*x*x*x+b*y];D=[[1+b*y,2*a*y+b*x],[3*a*x*x,1+b]];q=[q0,q1];r=[r0,r1]
 return [f[0]-x,f[1]-y]+[sum(D[i][j]*q[j]for j in range(2))-cs*q[i]+ds*r[i]for i in range(2)]+[sum(D[i][j]*r[j]for j in range(2))-es*q[i]-cs*r[i]for i in range(2)]+[(q0*q0+q1*q1-1)/2,q0*r0+q1*r1,sum(a*b for a,b in zip(normal,v))]


def test_all_gradient_path_columns_against_nested_multiprecision_diff():
 with ctx.workprec(256),mp.workdps(85):
  c,s,a,error,events,normal=fixture();J=H.jacobian_coefficients(c,s,a,error,arb(0),events,2,1,normal)
  cc,ss,aa,nn=[[mp.mpf(float(v))for v in row]for row in(c,s,a,normal)]
  for order in range(3):
   for i in range(9):
    for j in range(9):
     def value(d):
      z=[cc[k]+ss[k]*d+aa[k]*d*d/2 for k in range(9)]
      return mp.diff(lambda u:residual([z[k]+(u if k==j else 0)for k in range(9)],nn)[i],mp.mpf(0))
     ref=mp.diff(value,mp.mpf(0),order)/math.factorial(order);assert(J[order][i,j]-arb(mp.nstr(ref,80))).contains(0),(order,i,j)


def test_full_error_constant_path_and_exact_input_rejection():
 with ctx.workprec(256):
  d=arb(0,arb(1)/32);err=[arb(0,arb((1,-20)))]*9;c,s,a,error,e,n=fixture(d,err);J=H.jacobian_coefficients(c,s,a,error,d,e,2,1,n)
  for dd in (arb(1)/32,-arb(1)/32):
   for ee in (arb((1,-20)),-arb((1,-20))):
    cc=[v+ee for v in c];_,_,_,_,ev,_=fixture(dd,[ee]*9);fine=H.jacobian_coefficients(cc,s,a,[arb(0)]*9,dd,ev,2,1,n)
    assert all(J[2][i,j].contains(fine[2][i,j])for i in range(9)for j in range(9))
  bad=copy.deepcopy(e);bad[0]['inputs'][0]['value'][0][0]=ball_json(arb(100))
  with pytest.raises(ValueError):H.jacobian_coefficients(c,s,a,error,d,bad,2,1,n)
  bad=copy.deepcopy(e);bad[0]['inputs'][0]['gradient'][0][0][0]=ball_json(arb(0))
  with pytest.raises(ValueError):H.jacobian_coefficients(c,s,a,error,d,bad,2,1,n)


def test_noncommuting_matrix_polynomial_and_all_inclusions():
 with ctx.workprec(180):
  C=arb_mat([[1,arb(1)/4],[0,1]]);J0=arb_mat([[1,-arb(1)/4],[0,1]]);J1=arb_mat([[0,arb(1)/8],[arb(1)/16,0]]);J2=arb_mat([[arb(1)/32,0],[0,arb(1)/64]]);res=[arb_mat([[arb((1,-30))],[0]])]+[arb_mat(2,1)for _ in range(3)];seen={}
  report=H.polynomial_root(C,[J0,J1,J2],res,[arb(1)]*2,arb(1)/1024,arb(1)/16,[arb(0,arb(1)/1024)]*2,emit=lambda k,v:seen.__setitem__(k,v));assert report['passed']and all(report['coordinate_inclusions'])
  C1=seen['inverse_linear_proposal']
  for d in (arb(1)/16,-arb(1)/16):
   direct=H.M.ident(2)-(C+C1*d)*(J0+J1*d+J2*d*d);poly=sum((seen['defect_coefficient_'+str(k)]*(d**k)for k in range(4)),arb_mat(2,2))
   assert all((direct[i,j]-poly[i,j]).contains(0)for i in range(2)for j in range(2))
  with pytest.raises(ValueError):H.polynomial_root(C,[J0,J1,J2],res,[arb(1)]*2,arb(1)/1024,arb(1)/16,[arb(0)]*2)


def test_failed_inclusion_is_retained_and_not_promoted():
 with ctx.workprec(160):
  out=H.polynomial_root(arb_mat([[1]]),[arb_mat([[1]]),arb_mat(1,1),arb_mat(1,1)],[arb_mat([[1]])]+[arb_mat(1,1)for _ in range(3)],[arb(1)],arb(1)/1024,arb(1)/16,[arb(0,arb(1)/1024)])
  assert not out['passed']and out['coordinate_inclusions']==[False]
