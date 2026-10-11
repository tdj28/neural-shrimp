"""Exact-real augmented signed-half cubic-zero map and its full Jacobian.

All base variables, including gain/leak and the critical direction, are Arb
first-order jets. Third formal-path coefficients carry fourth derivatives.
This helper has no target caller and makes no root-existence assertion.
"""
from flint import arb,arb_mat
from .rigorous_orbits import eye,midpoint_matrix,norm_inf_upper,ball_json,matrix_json,vector_json


def require(ok,message):
 if not ok:raise ValueError(message)

class Dual:
 __slots__=('f','g')
 def __init__(self,f,g):self.f=arb(f);self.g=tuple(arb(v)for v in g)
 @classmethod
 def variable(cls,value,n,index):return cls(value,[int(i==index)for i in range(n)])
 def __add__(self,other):
  if not isinstance(other,Dual):return Dual(self.f+other,self.g)
  require(len(self.g)==len(other.g),'same full base dimension');return Dual(self.f+other.f,[a+b for a,b in zip(self.g,other.g)])
 __radd__=__add__
 def __neg__(self):return self*-1
 def __sub__(self,other):return self+-other
 def __rsub__(self,other):return -self+other
 def __mul__(self,other):
  if not isinstance(other,Dual):return Dual(self.f*other,[v*other for v in self.g])
  require(len(self.g)==len(other.g),'same full base dimension');return Dual(self.f*other.f,[a*other.f+self.f*b for a,b in zip(self.g,other.g)])
 __rmul__=__mul__
 def __truediv__(self,other):
  require(not isinstance(other,Dual),'only constant division');return self*(arb(1)/other)
 def tanh(self):
  t=self.f.tanh();return Dual(t,[(1-t*t)*v for v in self.g])

def value(v):return v.f if isinstance(v,Dual)else v

def serial(v):
 if isinstance(v,Dual):return dict(value=ball_json(v.f),gradient=vector_json(v.g))
 if isinstance(v,(list,tuple)):return [serial(x)for x in v]
 return ball_json(v)

def poly_tanh(coeff):
 degree=len(coeff)-1;require(0<=degree<=5,'bounded ordinary Taylor degree');zero=coeff[0]*0;out=[zero for _ in coeff];out[0]=coeff[0].tanh()
 for k in range(degree):
  square=[sum((out[i]*out[j-i]for i in range(j+1)),zero)for j in range(k+1)];s=[-v for v in square];s[0]+=1
  out[k+1]=sum(((i+1)*coeff[i+1]*s[k-i]for i in range(k+1)),zero)/(k+1)
 return out

def matmul(a,b):return [[sum((a[i][k]*b[k][j]for k in range(len(b))),0)for j in range(len(b[0]))]for i in range(len(a))]
def matvec(a,b):return [sum((v*w for v,w in zip(row,b)),0)for row in a]

def solve_dual(a,b,tag,observe=lambda kind,row:None):
 n=len(b);size=len(b[0].g);require(len(a)==n and all(len(row)==n for row in a),'complete square dual solve')
 A=arb_mat([[v.f for v in row]for row in a]);rhs=arb_mat([[v.f]for v in b]);pre=midpoint_matrix(A.inv());defect=eye(n)-pre*A;bound=norm_inf_upper(defect)
 require(bound.is_finite()and bound<1,'strict validated internal solve inverse');inverse=(pre*A).inv()*pre;x=inverse*rhs
 require(all(v.is_finite()for row in inverse.tolist()for v in row),'finite validated inverse')
 gradients=[]
 for k in range(size):
  dr=arb_mat([[b[i].g[k]-sum((a[i][j].g[k]*x[j,0]for j in range(n)),arb(0))]for i in range(n)]);gradients.append(inverse*dr)
 answer=[Dual(x[i,0],[g[i,0]for g in gradients])for i in range(n)]
 observe('dual_linear_solve',dict(tag=tag,matrix=serial(a),rhs=serial(b),preconditioner=matrix_json(pre),inverse=matrix_json(inverse),inverse_defect=matrix_json(defect),inverse_bound=ball_json(bound),solution=serial(answer)));return answer

def signed_path(coeff,j,g,d,observe=lambda kind,row:None):
 n=len(j);require(1<=n<=16 and len(coeff[0])==n and all(len(row)==n for row in j)and 1<=len(coeff)<=6,'bounded full native path');out=[list(row)for row in coeff]
 for phase in range(3):
  t=[poly_tanh([out[k][i]for k in range(len(out))])for i in range(n)]
  out=[[(1-d)*out[k][i]+d*g*sum((j[i][l]*t[l][k]for l in range(n)),0)for i in range(n)]for k in range(len(out))]
  observe('native_polynomial',dict(phase=phase,coefficients=serial(out)))
 return [[-v for v in row]for row in out]

def augmented(j,z,observe=lambda kind,row:None):
 """H=(G-x,(DG+I)q,(q.q-1)/2,c3), z=(x,leak,q,gain)."""
 n=len(j);N=2*n+2;require(1<=n<=16 and len(z)==N and all(len(row)==n for row in j),'complete bounded generalized flip')
 require(all(v.is_finite()for v in z)and all(v.is_finite()for row in j for v in row)and z[-1]>0 and 0<z[n]<=1,'finite actual physical map parameters')
 v=[Dual.variable(x,N,i)for i,x in enumerate(z)];x,d,q,g=v[:n],v[n],v[n+1:-1],v[-1];zero=v[0]*0;one=zero+1;M=[[one if i==k else zero for k in range(n)]for i in range(n)];coeff=[x,q,[zero]*n,[zero]*n];stages=[]
 for phase in range(3):
  t0=[a.tanh()for a in coeff[0]];J=[[(1-d)*int(i==k)+d*g*j[i][k]*(1-t0[k]*t0[k])for k in range(n)]for i in range(n)];second=[-2*t*(1-t*t)for t in t0];stages.append((J,second));M=matmul(J,M)
  t=[poly_tanh([coeff[k][i]for k in range(4)])for i in range(n)];coeff=[[(1-d)*coeff[k][i]+d*g*sum((j[i][l]*t[l][k]for l in range(n)),zero)for i in range(n)]for k in range(4)]
  observe('native_augmented_polynomial',dict(phase=phase,coefficients=serial(coeff),native_jacobian=serial(J),product=serial(M)))
 coeff=[[-a for a in row]for row in coeff];A=[[-a for a in row]for row in M]
 border=[[A[k][i]+int(i==k)for k in range(n)]+[q[i]]for i in range(n)]+[q+[zero]]
 left=solve_dual(border,[zero]*n+[one],'normalized-left',observe);p=left[:n]
 bqq=[2*a for a in coeff[2]];resolvent=solve_dual([[A[i][k]-int(i==k)for k in range(n)]for i in range(n)],bqq,'flip-quadratic',observe)
 qflow,vflow,bflow=q,resolvent,[zero]*n
 for J,second in stages:
  h=[d*g*sum((j[i][k]*second[k]*qflow[k]*vflow[k]for k in range(n)),zero)for i in range(n)];bflow=[a+b for a,b in zip(matvec(J,bflow),h)];qflow=matvec(J,qflow);vflow=matvec(J,vflow)
 bqv=[-a for a in bflow];cubic=sum((p[i]*(coeff[3][i]-bqv[i]/2)for i in range(n)),zero)
 equations=[coeff[0][i]-x[i]for i in range(n)]+[coeff[1][i]+q[i]for i in range(n)]+[(sum((a*a for a in q),zero)-1)/2,cubic]
 valuevec=arb_mat([[v.f]for v in equations]);jac=arb_mat([list(v.g)for v in equations]);data=dict(monodromy=arb_mat([[v.f for v in row]for row in A]),left=[v.f for v in p],left_auxiliary=left[-1].f,cubic=cubic.f,cubic_gradient=list(cubic.g),third_coefficient=[v.f for v in coeff[3]],second_directional=[v.f for v in bqq],mixed_q_resolvent=[v.f for v in bqv])
 require(all(v.is_finite()for row in valuevec.tolist()+jac.tolist()for v in row),'finite full augmented values and Jacobian');observe('generalized_flip_return',dict(value=matrix_json(valuevec),jacobian=matrix_json(jac),monodromy=matrix_json(data['monodromy']),left=vector_json(data['left']),left_auxiliary=ball_json(data['left_auxiliary']),cubic=ball_json(cubic.f)))
 return valuevec,jac,data
