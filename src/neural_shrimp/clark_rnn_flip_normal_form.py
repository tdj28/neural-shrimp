"""Numerical order-five scalar parameterization of a simple signed-map flip.

Ordinary polynomial coefficients include factorial denominators. This helper
neither finds a root nor proves its primitive native orbit or unfolding.
"""
import numpy as np

def require(ok,message):
 if not ok:raise ValueError(message)

def multiply(a,b,degree=5):
 a=np.asarray(a);b=np.asarray(b);out=np.zeros((degree+1,)+np.broadcast_shapes(a.shape[1:],b.shape[1:]),dtype=np.result_type(a,b,float))
 for k in range(degree+1):
  for i in range(max(0,k-len(b)+1),min(len(a)-1,k)+1):out[k]+=a[i]*b[k-i]
 return out

def compose(parameterization,reduced):
 h=np.asarray(parameterization);power=np.zeros(6);power[0]=1.;answer=np.zeros_like(h)
 for k in range(6):answer+=power[:,None]*h[k];power=multiply(power,reduced)
 return answer

def tanh_series5(coeff):
 """Solve t'=x'(1-t²) coefficient by coefficient through degree five."""
 x=np.asarray(coeff,dtype=float);require(x.ndim==2 and x.shape[0]==6 and np.isfinite(x).all(),'six finite vector coefficients')
 t=np.zeros_like(x);t[0]=np.tanh(x[0])
 for k in range(5):
  square=multiply(t,t);one_minus=-square;one_minus[0]+=1
  t[k+1]=sum(((i+1)*x[i+1]*one_minus[k-i]for i in range(k+1)),np.zeros(x.shape[1]))/(k+1)
 require(np.isfinite(t).all(),'finite tanh path coefficients');return t

def signed_half_series(coeff,j,gain,leak,halfperiod=3,observe=lambda kind,row:None):
 x=np.asarray(coeff,dtype=float).copy();j=np.asarray(j,dtype=float);n=x.shape[1]if x.ndim==2 else 0
 require(x.shape==(6,n)and 1<=n<=16 and j.shape==(n,n)and np.isfinite(j).all()and np.isfinite(gain)and gain>0 and np.isfinite(leak)and 0<leak<=1 and type(halfperiod)is int and halfperiod==3,'fixed finite signed-half polynomial map')
 for k in range(halfperiod):
  before=x.copy();t=tanh_series5(x);x=(1-leak)*x+leak*gain*(t@j.T)
  observe('native_path5',dict(phase=k,input=before,tanh=t,output=x.copy(),gain=float(gain),leak=float(leak)))
  require(np.isfinite(x).all(),'finite complete native fifth-order return')
 return -x

def normal_form5(x,A,q,p,evaluate,solve=lambda a,b,tag:np.linalg.solve(a,b),observe=lambda kind,row:None):
 """G(h(u))=h(-u+c3*u³+c5*u⁵)+O(u⁶), p.h3=p.h5=0.

At the degenerate flip c3=0, nonzero c5 supplies fifth-order nondegeneracy.
Away from c3=0 this is a fixed coordinate convention, not an invariant c5.
Caller authenticates the fixed root, the actual simple -1 mode and native orbit.
"""
 x=np.asarray(x,dtype=float);A=np.asarray(A,dtype=float);q=np.asarray(q,dtype=float);p=np.asarray(p,dtype=float);n=len(x)
 require(1<=n<=16 and A.shape==(n,n)and q.shape==p.shape==(n,)and all(np.isfinite(v).all()for v in(x,A,q,p)),'full finite simple-flip data')
 require(abs(q@q-1)<1e-10 and abs(p@q-1)<1e-10 and np.linalg.norm(A@q+q,np.inf)<1e-9 and np.linalg.norm(p@A+p,np.inf)<1e-9,'normalized actual flip eigenvectors')
 h=np.zeros((6,n));h[0]=x;h[1]=q;identity=np.eye(n);border=np.block([[A+identity,-q[:,None]],[p[None,:],np.zeros((1,1))]])
 def path():
  value=np.asarray(evaluate(h.copy()),dtype=float);observe('normalform_path',dict(input=h.copy(),output=value.copy()));require(value.shape==h.shape and np.isfinite(value).all(),'full retained fifth-order path');return value
 def linear(matrix,rhs,tag):
  answer=np.asarray(solve(matrix,rhs,tag),dtype=float);observe('normalform_solve',dict(tag=tag,matrix=matrix.copy(),rhs=rhs.copy(),solution=answer.copy()));require(np.isfinite(answer).all(),'finite normal-form linear solve');return answer
 h[2]=linear(identity-A,path()[2],'even2')
 third=linear(border,np.r_[-path()[3],0.],'odd3');h[3]=third[:-1];c3=float(third[-1])
 h[4]=linear(identity-A,path()[4]+2*c3*h[2],'even4')
 fifth=linear(border,np.r_[3*c3*h[3]-path()[5],0.],'odd5');h[5]=fifth[:-1];c5=float(fifth[-1])
 reduced=np.array([0.,-1.,0.,c3,0.,c5]);actual=path();expected=compose(h,reduced);defect=actual-expected
 scale=max(1.,float(np.max(abs(actual))),float(np.max(abs(expected))));residual=float(np.max(abs(defect))/scale)
 report=dict(cubic=c3,quintic=c5,invariance_residual=residual,coefficient_residuals=np.max(abs(defect),axis=1),odd_gauge_errors=np.array([abs(p@h[3]),abs(p@h[5])]),parameterization=h,reduced=reduced,actual=actual,expected=expected,defect=defect,scope='Numerical signed-map center parameterization with fixed normalized q and p.h3=p.h5=0. Nonzero c5 at an exact c3=0 simple flip would establish fifth-order nondegeneracy; root, unfolding, native primitive orbit and interval proof are separate prerequisites.')
 observe('normalform_result',report);return report
