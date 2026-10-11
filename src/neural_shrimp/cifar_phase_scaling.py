"""Fixed ridge-scale congruence in orthogonal mean/alternating phase coordinates.

Q(x,y)=((x+y)/sqrt(2),(x-y)/sqrt(2)); Q=Q^T. If d=eta*lambda,
the ridge-only shooting matrix becomes diag(-d,2-d). P=Q*S with
S=diag(d^-1/2,(2-d)^-1/2) gives P^T*A*P=diag(-1,1).
This is an SPD scaling proposal, not an exact Hessian model or convergence claim.
"""
import math,time
from fractions import Fraction
import numpy as np
from scipy.sparse.linalg import LinearOperator,minres


def vector(v):
    a=np.asarray(v,dtype=np.float64)
    if a.ndim!=1 or len(a)==0 or len(a)%2 or not np.isfinite(a).all():raise ValueError('finite even full phase vector')
    return a

def transform(v):
    a=vector(v);n=len(a)//2;x,y=a[:n],a[n:];return np.r_[(x+y)/math.sqrt(2),(x-y)/math.sqrt(2)]


def scales(eta,ridge):
    if not all(isinstance(v,(int,float))and math.isfinite(v)for v in (eta,ridge)):raise ValueError('finite scalar map constants')
    d=Fraction.from_float(float(eta))*Fraction.from_float(float(ridge))
    if not 0<d<2:raise ValueError('positive mean/alternating ridge scales')
    return (1/math.sqrt(float(d)),1/math.sqrt(float(2-d))),dict(exact_eta_ridge=str(d),exact_alternating_scale=str(2-d),mean_sqrt_inverse=1/math.sqrt(float(d)),alternating_sqrt_inverse=1/math.sqrt(float(2-d)),scope='Fixed positive congruence based only on exact binary64 rate and ridge; data curvature and cross-phase coupling omitted from proposal.')


def diagonal(v,weights):
    a=vector(v);n=len(a)//2
    if len(weights)!=2 or not all(math.isfinite(w)and w>0 for w in weights):raise ValueError('two positive finite block scales')
    return np.r_[weights[0]*a[:n],weights[1]*a[n:]]

def physical(v,weights):return transform(diagonal(v,weights))
def dual(v,weights):return diagonal(transform(v),weights)

def dot(a,b):return math.fsum(float(x)*float(y)for x,y in zip(a,b,strict=True))

def direction(jx,jy,residual,eta,ridge,*,iterations=400,deadline=None):
    r=np.asarray(residual,dtype=np.float64)
    if r.ndim!=1 or not np.isfinite(r).all()or dot(r,r)<=0 or type(iterations)is not int or iterations<=0:raise ValueError('nonzero finite residual and fixed positive budget')
    w,record=scales(eta,ridge);n=len(r);calls=0;iters=0
    def raw(v):
        a=vector(v);x,y=a[:n],a[n:];fx=np.asarray(jx(x));fy=np.asarray(jy(y))
        if fx.shape!=x.shape or fy.shape!=y.shape or not np.isfinite(fx).all()or not np.isfinite(fy).all():raise ValueError('finite actual native phase actions')
        return np.r_[fx-y,fy-x]
    def action(v):
        nonlocal calls
        if deadline is not None and time.monotonic()>=deadline:raise TimeoutError('fixed scaled MINRES deadline')
        if calls>=iterations+2:raise RuntimeError('fixed scaled phase-action cap')
        calls+=1;return dual(raw(physical(v,w)),w)
    def callback(_):
        nonlocal iters
        iters+=1
    rhs=np.r_[np.zeros(n),-r];scaled_rhs=dual(rhs,w);op=LinearOperator((2*n,2*n),matvec=action,rmatvec=action,dtype=np.float64)
    q,code=minres(op,scaled_rhs,x0=np.zeros(2*n),rtol=1e-12,maxiter=iterations,callback=callback,show=False,check=False)
    q=vector(q);z=physical(q,w);scaled_action=action(q)
    # Fresh physical actions are the acceptance authority, separately charged.
    actual_az=raw(z);p=z[:n].copy();ap=np.asarray(jy(np.asarray(jx(p))))-p
    if ap.shape!=r.shape or not np.isfinite(ap).all():raise ValueError('finite physical eliminated action')
    forcing=math.sqrt(dot(r+ap,r+ap)/dot(r,r));slope=dot(r,ap)
    return p,ap,z,actual_az,q,dict(minres_info=int(code),iterations=iters,scaled_actions=calls,extra_phase_actions=4,scaled_relative_residual=math.sqrt(dot(scaled_action-scaled_rhs,scaled_action-scaled_rhs)/dot(scaled_rhs,scaled_rhs)),shooting_relative_residual=math.sqrt(dot(actual_az-rhs,actual_az-rhs)/dot(r,r)),true_forcing=forcing,true_slope=slope,forcing_pass=bool(forcing<=.5),usable_descent=bool(slope<0),scaling=record)
