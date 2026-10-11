import sys
from pathlib import Path
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import pytest,mpmath as mp
from flint import arb,ctx
from neural_shrimp.rigorous_signed_gradient_path import signed_jacobian_path
from neural_shrimp.rigorous_orbits import exact

class Nonlinear:
 n=2
 def cusp_step(self,x,h,o):return[(h*x[0]+o*x[1]+x[0]*x[1]).tanh(),(o*x[0]-.3*x[1]+h*x[0]*x[0]).tanh()]
def native(z,p,sigma):
 rows=[]
 for k in range(p):
  a=4*k;b=4*((k+1)%p);x=z[a:a+2];q=z[a+2:a+4];h,o=z[-2:]
  f=lambda xx:[mp.tanh(h*xx[0]+o*xx[1]+xx[0]*xx[1]),mp.tanh(o*xx[0]-mp.mpf('.3')*xx[1]+h*xx[0]*xx[0])]
  y=f(x);D=mp.matrix([[mp.diff(lambda u:f([u,x[1]])[i],x[0]),mp.diff(lambda u:f([x[0],u])[i],x[1])]for i in range(2)])
  rows.extend([y[i]-z[b+i]for i in range(2)]);rows.extend([sum(D[i,j]*q[j]for j in range(2))-(sigma if k==p-1 else 1)*z[b+2+i]for i in range(2)])
 rows.extend([(z[2]*z[2]+z[3]*z[3]-1)/2,sum(mp.mpf(i+1)*v/16 for i,v in enumerate(z))]);return rows
@pytest.mark.parametrize('sigma',[-1,1])
def test_every_nonlinear_native_column_and_path_order(sigma):
 mp.mp.dps=75
 with ctx.workprec(256):
  p=2;size=10;c=[exact(str((i+1)/32))for i in range(size)];s=[exact(str((i%3-1)/64))for i in range(size)];a=[exact(str((i%4-2)/128))for i in range(size)];err=[arb(0,exact('1/1048576'))for _ in c];normal=[exact(str((i+1)/16))for i in range(size)]
  J=signed_jacobian_path(Nonlinear(),c,s,a,err,arb(0),p,sigma,normal)
  cm=[mp.mpf(str(float(v)))for v in c];sm=[mp.mpf(str(float(v)))for v in s];am=[mp.mpf(str(float(v)))for v in a]
  for i in range(size):
   for j in range(size):
    def one(t):
     z=[cm[k]+sm[k]*t+am[k]*t*t/2 for k in range(size)]
     return mp.diff(lambda u:native(z[:j]+[u]+z[j+1:],p,sigma)[i],z[j])
    for k in range(3):
     v=mp.diff(one,0,k)/mp.factorial(k);assert J[k][i,j].contains(arb(str(v))), (i,j,k,J[k][i,j],v)
  # E remains visible even at d=0, including mode held-direction derivatives.
  assert J[0][0,0].rad()>0 and J[0][2,0].rad()>0

def test_preinput_retained_on_native_failure():
 class Bad:
  n=1
  def cusp_step(self,*_):raise RuntimeError('native stop')
 ev=[]
 with pytest.raises(RuntimeError,match='native stop'):signed_jacobian_path(Bad(),[arb(1)]*4,[arb(0)]*4,[arb(0)]*4,[arb(0)]*4,arb(0),1,-1,[arb(1)]*4,observe=lambda k,v:ev.append((k,v)))
 assert [k for k,v in ev]==['signed_gradient_path_input']

def test_completed_nonfinite_return_precedes_rejection():
 from neural_shrimp.rigorous_gradient_path import GradientPathJet
 from neural_shrimp.rigorous_eigenpair_path import PathJet
 class Bad:
  n=1
  def cusp_step(self,*_):return[GradientPathJet(PathJet.constant(arb('nan')),[0,0,0])]
 ev=[]
 with pytest.raises(ValueError,match='finite signed gradient'):signed_jacobian_path(Bad(),[arb(1)]*4,[arb(0)]*4,[arb(0)]*4,[arb(0)]*4,arb(0),1,-1,[arb(1)]*4,observe=lambda k,v:ev.append((k,v)))
 assert [k for k,v in ev]==['signed_gradient_path_input','signed_gradient_path_return']
 assert ev[-1][1]['returns'][0]['value'][0][0]['finite']is False
