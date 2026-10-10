"""Symmetric two-phase Newton system for scalar-step gradient descent.

For F(x)=x-eta*grad L(x), H=(F(x)-y,F(y)-x) has symmetric
Jacobian [[Jx,-I],[-I,Jy]]. This is not a symmetrization of J_y J_x-I.
Its first solution block solves the original residual Newton equation when
y=F(x). True eliminated forcing is computed separately, never inferred from
MINRES status or the shooting residual.
"""
import math,time
import numpy as np
from scipy.sparse.linalg import LinearOperator,minres
from .cifar_linear_diagnostic import finite_vector
from .cifar_merit_lsmr import dot

def shooting_action(jx,jy,vector):
 v=finite_vector(vector)
 if len(v)%2:raise ValueError('even two-phase dimension')
 n=len(v)//2;x,y=v[:n],v[n:]
 return np.r_[finite_vector(jx(x),x.shape)-y,finite_vector(jy(y),y.shape)-x]

def direction(jx,jy,residual,*,iterations=400,deadline=None):
 r=finite_vector(residual);n=len(r)
 if type(iterations)is not int or iterations<=0 or dot(r,r)<=0:raise ValueError('positive iteration budget and nonzero residual')
 calls=0;iters=0
 def action(v):
  nonlocal calls
  if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('shooting MINRES deadline')
  if calls>=iterations+2:raise RuntimeError('shooting action budget')
  calls+=1;return shooting_action(jx,jy,v)
 def callback(x):
  nonlocal iters
  iters+=1
 op=LinearOperator((2*n,2*n),matvec=action,rmatvec=action,dtype=np.float64)
 rhs=np.r_[np.zeros(n),-r]
 z,code=minres(op,rhs,rtol=1e-12,maxiter=iterations,show=False,check=False,callback=callback,x0=np.zeros(2*n))
 z=finite_vector(z,rhs.shape);az=action(z);shooting=math.sqrt(dot(az-rhs,az-rhs)/dot(r,r))
 # One fresh Jx and Jy composition evaluates the original two-step residual.
 p=z[:n].copy();ap=finite_vector(jy(finite_vector(jx(p),r.shape)),r.shape)-p
 forcing=math.sqrt(dot(r+ap,r+ap)/dot(r,r));slope=dot(r,ap)
 return p,ap,z,az,dict(minres_info=int(code),iterations=iters,shooting_actions=calls,shooting_relative_residual=shooting,true_forcing=forcing,true_slope=slope,forcing_pass=bool(forcing<=.5),usable_descent=bool(slope<0),extra_phase_actions=2)
