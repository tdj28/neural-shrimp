"""Common-parameter Taylor coefficients for the explicit Clark fold system.

R[t,e]/(t^4,e^2) keeps the path and the paired tangent direction distinct.
Full base gradients differentiate z=(m,leak,q,gain) and the external b while
holding v fixed; the independent v columns are restored from H_U. All51
correction coordinates remain constant in every Taylor path coefficient.
No root or known-root identity is asserted by this preparatory acquisition.
"""
from math import factorial
from fractions import Fraction
from flint import arb,arb_mat
from .rigorous_clark_amplitude_fold import require,integer_power,rational,finite
from .rigorous_orbits import ball_json,vector_json,matrix_json

class Path:
 """a[k]=t^k, b[k]=e*t^k, k0..3; factorial-scaled coefficients."""
 __slots__=('a','b')
 def __init__(self,a,b=None):
  require(len(a)==4 and (b is None or len(b)==4),'complete path/mode coefficients');self.a=tuple(arb(v)for v in a);self.b=tuple(arb(v)for v in([0]*4 if b is None else b))
 @classmethod
 def constant(cls,x):return cls([x,0,0,0])
 def __add__(self,v):
  if not isinstance(v,Path):v=Path.constant(v)
  return Path([x+y for x,y in zip(self.a,v.a,strict=True)],[x+y for x,y in zip(self.b,v.b,strict=True)])
 __radd__=__add__
 def __neg__(self):return self*-1
 def __sub__(self,v):return self+-v
 def __rsub__(self,v):return -self+v
 def __mul__(self,v):
  if not isinstance(v,Path):return Path([x*v for x in self.a],[x*v for x in self.b])
  return Path([sum((self.a[i]*v.a[k-i]for i in range(k+1)),arb(0))for k in range(4)],[sum((self.a[i]*v.b[k-i]+self.b[i]*v.a[k-i]for i in range(k+1)),arb(0))for k in range(4)])
 __rmul__=__mul__
 def __truediv__(self,v):
  require(not isinstance(v,Path),'scalar path division only');return Path([x/v for x in self.a],[x/v for x in self.b])
 def serialized(self):return [vector_json(self.a),vector_json(self.b)]

def path(v):return v if isinstance(v,Path)else Path.constant(v)

def compose(v,derivatives):
 """The ideal(t,e) has fifth power zero, so four derivatives suffice."""
 require(isinstance(v,Path)and len(derivatives)==5,'five scalar derivatives')
 nil=Path([0,*v.a[1:]],v.b);power=Path.constant(1);out=Path.constant(0)
 for k in range(5):out+=power*(derivatives[k]/factorial(k));power=power*nil
 return out

class Gradient:
 __slots__=('v','g')
 def __init__(self,v,g):self.v=path(v);self.g=tuple(path(x)for x in g)
 def __add__(self,v):
  if not isinstance(v,Gradient):return Gradient(self.v+v,self.g)
  require(len(v.g)==len(self.g),'full base gradients');return Gradient(self.v+v.v,[a+b for a,b in zip(self.g,v.g,strict=True)])
 __radd__=__add__
 def __neg__(self):return self*-1
 def __sub__(self,v):return self+-v
 def __rsub__(self,v):return -self+v
 def __mul__(self,v):
  if not isinstance(v,Gradient):return Gradient(self.v*v,[x*v for x in self.g])
  require(len(v.g)==len(self.g),'full base gradients');return Gradient(self.v*v.v,[x*v.v+self.v*y for x,y in zip(self.g,v.g,strict=True)])
 __rmul__=__mul__
 def __truediv__(self,v):
  require(not isinstance(v,(Path,Gradient)),'scalar gradient division only');return Gradient(self.v/v,[x/v for x in self.g])
 def serialized(self):return dict(value=self.v.serialized(),gradient=[x.serialized()for x in self.g])

def scalar_derivatives(x,kind):
 require(x.is_finite(),'finite scalar expansion domain')
 if kind=='reciprocal':
  require(not x.contains(0),'nonzero whole pair denominator')
  # Take the reciprocal of the original nonzero interval first. Repeated
  # denominator products can wrap across zero although x itself excludes it.
  inverse=1/x
  return [arb((-1)**k*factorial(k))*integer_power(inverse,k+1)for k in range(6)]
 if kind in('sinh','cosh'):
  a,b=(x.sinh(),x.cosh())if kind=='sinh'else(x.cosh(),x.sinh());return [a if k%2==0 else b for k in range(6)]
 require(kind in('C','S')and abs(x).upper()<=16,'bounded entire pair argument');offset=int(kind=='S');out=[]
 # For derivatives j<=5 and k>=25, successive absolute terms have
 # ratio <=26/(26-5)*16/(51*52)<1/2. Two first omitted terms suffice.
 for j in range(6):
  value=arb(0)
  for k in range(j,25):value+=rational(Fraction(factorial(k),factorial(k-j)*factorial(2*k+offset)))*integer_power(x,k-j)
  tail=2*rational(Fraction(factorial(25),factorial(25-j)*factorial(50+offset)))*integer_power(abs(x).upper(),25-j);out.append(value+arb(0,tail))
 return out

def unary(v,kind):
 require(isinstance(v,Gradient),'full gradient-path unary');ds=scalar_derivatives(v.v.a[0],kind);value=compose(v.v,ds[:5]);derivative=compose(v.v,ds[1:]);return Gradient(value,[derivative*x for x in v.g])

def pair(m,q,b):
 z=4*b*q*q;u=2*m;inv=unary(unary(u,'cosh')+unary(z,'C'),'reciprocal');return unary(u,'sinh')*inv,2*q*unary(z,'S')*inv

def paired(j,z,b,*,tick=lambda:None,observe=lambda *_:None):
 n=len(j);require(len(z)==2*n+2,'actual paired base dimension');m0,delta,q0,gain=z[:n],z[n],z[n+1:-1],z[-1];m=list(m0);q=list(q0)
 for phase in range(3):
  observe('clark_gradient_path_input',dict(phase=phase,even=[x.serialized()for x in m],odd=[x.serialized()for x in q],leak=delta.serialized(),gain=gain.serialized(),squared_amplitude=b.serialized()));tick();t=[pair(x,y,b)for x,y in zip(m,q,strict=True)]
  m=[(1-delta)*m[i]+delta*gain*sum((j[i][k]*t[k][0]for k in range(n)),0)for i in range(n)];q=[(1-delta)*q[i]+delta*gain*sum((j[i][k]*t[k][1]for k in range(n)),0)for i in range(n)]
  observe('clark_gradient_path_return',dict(phase=phase,even=[x.serialized()for x in m],odd=[x.serialized()for x in q]));require(all(v.is_finite()for x in m+q for p in[x.v,*x.g]for v in p.a+p.b),'all finite native coefficients');tick()
 return [-m[i]-m0[i]for i in range(n)]+[-q[i]+q0[i]for i in range(n)]+[(sum((x*x for x in q0),0)-1)/2]

def coefficients(j,center,slope,curvature,error,parameter,d,*,tick=lambda:None,observe=lambda *_:None):
 """F0..3 and J0..2 at arbitrary d, with complete constant correction E.

For residual Taylor use E=0; for J0/J1 at d=0 and J2 over whole d,
use the SAME full E. External b(d+t)=parameter+d+t remains present in
the directional and base derivative fields. No affine-gauge shortcut applies.
"""
 n=len(j);K=2*n+1;N=K+1;dim=N+K
 require(1<=n<=12 and all(len(x)==dim for x in(center,slope,curvature,error))and all(len(row)==n for row in j),'full bounded51 path dimensions')
 require(all(x.is_finite()and x.is_exact()for x in center+slope+curvature+[parameter]+[x for row in j for x in row]),'finite exact map and predictor')
 require(d.is_finite()and all(x.is_finite()and x.contains(0)for x in error),'full fixed zero-containing correction domain')
 paths=[Path([center[i]+slope[i]*d+curvature[i]*d*d/2+error[i],slope[i]+curvature[i]*d,curvature[i]/2,0])for i in range(dim)]
 require(paths[n].a[0]>0 and paths[n].a[0]<1 and paths[N-1].a[0]>0,'whole physical leak/gain')
 z=[Gradient(Path(paths[i].a,paths[N+i].a if i<K else[0]*4),[int(i==k)for k in range(N+1)])for i in range(N)];b=Gradient(Path([parameter+d,1,0,0],[1,0,0,0]),[int(k==N)for k in range(N+1)])
 h=paired(j,z,b,tick=tick,observe=observe);require(len(h)==K,'all explicit paired equations');F=arb_mat([list(x.v.a)for x in h]+[list(x.v.b)for x in h]+[list(paths[N+n].a)]);J=[arb_mat(dim,dim)for _ in range(3)]
 for k in range(3):
  for i,x in enumerate(h):
   for u in range(N):J[k][i,u]=x.g[u].a[k];J[k][K+i,u]=x.g[u].b[k]
   for u in range(K):J[k][K+i,N+u]=x.g[u].a[k]
  J[k][dim-1,N+n]=int(k==0)
 require(finite(F)and all(finite(x)for x in J),'finite full explicit Taylor arrays')
 observe('clark_correlated_path_return',dict(center=vector_json(center),slope=vector_json(slope),curvature=vector_json(curvature),error=vector_json(error),parameter=ball_json(parameter),d=ball_json(d),value_coefficients=matrix_json(F),jacobian_coefficients=[matrix_json(x)for x in J],scope='Native analytic Taylor coefficient acquisitions only. Caller binds d0/full-d correction domains and exact original map/root/phase before any correlated polynomial root inequality.'))
 return F,J
