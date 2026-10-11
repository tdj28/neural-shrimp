"""Bounded damped least-squares directions with true residual-merit acceptance.

LSMR status and actual linear forcing remain reported. A truncated direction
can be useful without satisfying the older inexact-Newton forcing condition.
Only true native residual decrease can accept a local step; no orbit acceptor.
"""
import math,time
import numpy as np
from scipy.sparse.linalg import LinearOperator,lsmr
from .cifar_linear_diagnostic import finite_vector

def dot(x,y):return math.fsum(float(a)*float(b)for a,b in zip(x,y,strict=True))
def direction(action,adjoint,residual,damp,*,iterations=200,radius=1.,deadline=None):
 r=finite_vector(residual);n=len(r)
 if not math.isfinite(damp)or damp<=0 or not math.isfinite(radius)or radius<=0 or type(iterations)is not int or iterations<=0:raise ValueError('positive solver settings')
 counters={'forward':0,'adjoint':0}
 def call(kind,fun,v):
  if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('least-squares deadline')
  cap=iterations+2 if kind=='forward'else iterations+1
  if counters[kind]>=cap:raise RuntimeError('least-squares '+kind+' cap')
  counters[kind]+=1;return finite_vector(fun(finite_vector(v,r.shape)),r.shape)
 op=LinearOperator((n,n),matvec=lambda v:call('forward',action,v),rmatvec=lambda v:call('adjoint',adjoint,v),dtype=np.float64)
 start=time.monotonic();result=lsmr(op,-r,damp=damp,atol=1e-12,btol=1e-12,conlim=1e12,maxiter=iterations,show=False,x0=np.zeros(n))
 p=finite_vector(result[0],r.shape);length=float(np.linalg.norm(p));clip=min(1.,radius/max(length,np.finfo(float).tiny));p=p*clip
 ap=call('forward',action,p);slope=dot(r,ap);phi=dot(r,r)/2;pred=phi-dot(r+ap,r+ap)/2
 ratio=math.sqrt(dot(r+ap,r+ap)/dot(r,r))
 return p,ap,{'lsmr_stop':int(result[1]),'lsmr_iterations':int(result[2]),'lsmr_normr':float(result[3]),'lsmr_normar':float(result[4]),'lsmr_norma':float(result[5]),'lsmr_conda':float(result[6]),'lsmr_normx':float(result[7]),'calls':counters,'damp':damp,'radius':radius,'unclipped_norm':length,'clip':clip,'true_forcing':ratio,'old_newton_forcing_pass':bool(ratio<=.5),'true_slope':slope,'full_predicted_reduction':pred,'seconds':time.monotonic()-start,'usable_descent':bool(math.isfinite(slope)and slope<0 and np.isfinite(p).all())}

def trial(theta,first,second,residual,ap,alpha):
 theta,first,second,residual,ap=[finite_vector(v)for v in(theta,first,second,residual,ap)]
 if not all(v.shape==theta.shape for v in(first,second,residual,ap))or not math.isfinite(alpha)or not 0<alpha<=1:raise ValueError('trial shape/step')
 phi=dot(residual,residual)/2;actual_residual=second-theta;merit=dot(actual_residual,actual_residual)/2;slope=dot(residual,ap)
 predicted=phi-dot(residual+alpha*ap,residual+alpha*ap)/2;actual=phi-merit;rho=actual/predicted if predicted>0 else -math.inf
 norm=max(1.,float(np.linalg.norm(theta)));closure=float(np.linalg.norm(actual_residual))/norm;separation=float(np.linalg.norm(first-theta))/norm
 accepted=bool(slope<0 and predicted>0 and actual>0 and rho>=.1 and merit<=phi+1e-4*alpha*slope and separation>=1e-7)
 return {'alpha':alpha,'initial_merit':phi,'merit':merit,'actual_reduction':actual,'predicted_reduction':predicted,'ratio':rho if math.isfinite(rho)else None,'slope':slope,'relative_residual':closure,'relative_separation':separation,'primitive_screen':bool(separation>=1e-7),'closure_screen':bool(closure<1e-8),'accepted_local_step':accepted,'accepted_orbit':False}
