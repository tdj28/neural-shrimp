import sys,copy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'tools/reviews'),str(ROOT/'src')]
import pytest,mpmath as mp
from flint import arb,ctx
import review_signed_correlated_math as D
from neural_shrimp.rigorous_orbits import ball_json,vector_json
P=D.P

def pair(a,b):return [vector_json(a),vector_json(b)]
def fixture(d=arb(0),error=None):
 c=list(map(arb,[1,1,2,3]));s=[arb(1)/16,arb(-1)/32,arb(1)/8,arb(1)/64];a=[arb(1)/128,arb(-1)/64,arb(1)/256,arb(-1)/128];error=error or[arb(0)]*4
 z=[[c[i]+s[i]*d+a[i]*d*d/2+error[i],s[i]+a[i]*d,a[i]/2,arb(0)]for i in range(4)];x,q,h,o=z;zero=P.const(arb(0));one=P.const(arb(1));fx=P.add(P.scale(P.mul(x,x),arb(3)),h);value=P.add(P.add(P.mul(P.mul(x,x),x),P.mul(h,x)),o);dv=P.mul(fx,q)
 g=[pair(fx,P.scale(P.mul(x,q),arb(6))),pair(x,q),pair(one,zero)]
 vals=[pair(x,q),pair(h,zero),pair(o,zero)];inputs=[dict(value=v,gradient=[pair(P.const(arb(int(i==j))),zero)for j in range(3)])for i,v in enumerate(vals)];ret=dict(value=pair(value,dv),gradient=g)
 return c,s,a,error,dict(phase=0,d=ball_json(d),inputs=inputs,returns=[ret]),dict(phase=0,d=ball_json(d),inputs=vals,returns=[pair(value,dv)])

def test_all_signed_jacobian_orders_independent_formula_and_residual():
 with ctx.workprec(256),mp.workdps(80):
  c,s,a,e,g,p=fixture();normal=list(map(arb,[0,0,1,0]));J=D.jacobian(c,s,a,e,arb(0),[g],1,1,-1,normal)
  def vals(t):return [mp.mpf(str(float(v)))+mp.mpf(str(float(u)))*t+mp.mpf(str(float(w)))*t*t/2 for v,u,w in zip(c,s,a)]
  def matrix(t):
   x,q,h,o=vals(t);return [[3*x*x+h-1,0,x,1],[6*x*q,3*x*x+h+1,q,0],[0,q,0,0],[0,0,1,0]]
  for k in range(3):
   for i in range(4):
    for j in range(4):assert abs(mp.mpf(str(float(J[k][i,j].mid())))-mp.diff(lambda t:matrix(t)[i][j],0,k)/mp.factorial(k))<mp.mpf('1e-15')
  R=D.residual(c,s,a,arb(0),[p],1,1,-1,normal,c,arb(0));assert R.nrows()==4 and R[1,0]==6 and R[2,0]==0 and R[3,1]==s[2]-1

def test_actual_correction_and_held_basis_corruption_reject():
 with ctx.workprec(256):
  e=[arb(0,arb(1)/1024)]*4;c,s,a,e,g,p=fixture(error=e);normal=list(map(arb,[0,0,1,0]));D.jacobian(c,s,a,e,arb(0),[g],1,1,-1,normal)
  bad=copy.deepcopy(g);bad['inputs'][0]['gradient'][0]=pair(P.const(arb(0)),P.const(arb(0)))
  with pytest.raises(ValueError):D.jacobian(c,s,a,e,arb(0),[bad],1,1,-1,normal)
  *_,small,_=fixture()
  with pytest.raises(ValueError):D.jacobian(c,s,a,e,arb(0),[small],1,1,-1,normal)

def test_complete_anchor_projection_and_strict_boundary():
 c=[arb(0)]*4;s=[arb(0)]*4;a=[arb(0)]*4;normal=[arb(1),arb(0),arb(0),arb(0)];r=[arb(1)]*4;box=vector_json(c)
 assert D.anchor(box,c,s,a,normal,c,arb(0),arb(1),r)['passed']
 box[3]=ball_json(arb(1));assert not D.anchor(box,c,s,a,normal,c,arb(0),arb(1),r)['passed']
