"""Symmetry-restricted G=-F^k arithmetic; numerical, not an original-map relabeling."""
import numpy as np
from .clark_rnn_boundaries import jets

def g_jets(x,j,g,delta,halfperiod=5):
    end,u,v,cycle,phase=jets(x,j,g,delta,halfperiod)
    return -end,-u,-v,cycle,phase

def fold_family(z,j,halfperiod=5):
    n=len(j)
    if len(z)!=2*n+2:raise ValueError('family dimension')
    x=z[:n];g=float(z[n]);d=float(z[n+1]);q=z[n+2:]
    end,u,v,cycle,phase=g_jets(x,j,g,d,halfperiod);a=u[:,:n]-np.eye(n)
    h=np.r_[end-x,a@q,(q@q-1)/2];b=np.zeros((2*n+1,2*n+2));b[:n,:n+2]=u;b[:n,:n]-=np.eye(n)
    b[n:2*n,:n+2]=np.einsum('ijk,j->ik',v[:,:n,:],q);b[n:2*n,n+2:]=a;b[-1,n+2:]=q
    return h,b,dict(theta=np.array(x),gain=g,delta=d,right_vector=q.copy(),endpoint=end,first=u,second=v,half_cycle=cycle,half_phase_jacobians=phase,monodromy=u[:,:n])

def fold_flip(z,j,halfperiod=5):
    n=len(j)
    if len(z)!=3*n+2:raise ValueError('interaction dimension')
    x=z[:n];g=float(z[n]);d=float(z[n+1]);q=z[n+2:2*n+2];w=z[2*n+2:]
    end,u,v,cycle,phase=g_jets(x,j,g,d,halfperiod);plus=u[:,:n]-np.eye(n);minus=u[:,:n]+np.eye(n)
    h=np.r_[end-x,plus@q,minus@w,(q@q-1)/2,(w@w-1)/2];b=np.zeros((3*n+2,3*n+2));b[:n,:n+2]=u;b[:n,:n]-=np.eye(n)
    b[n:2*n,:n+2]=np.einsum('ijk,j->ik',v[:,:n,:],q);b[n:2*n,n+2:2*n+2]=plus
    b[2*n:3*n,:n+2]=np.einsum('ijk,j->ik',v[:,:n,:],w);b[2*n:3*n,2*n+2:]=minus;b[-2,n+2:2*n+2]=q;b[-1,2*n+2:]=w
    return h,b,dict(theta=x.copy(),gain=g,delta=d,right_plus=q.copy(),right_minus=w.copy(),endpoint=end,first=u,second=v,half_cycle=cycle,half_phase_jacobians=phase,monodromy=u[:,:n])


def critical_fixed_gain(z,j,g,sign,halfperiod=3):
    """G=-F^k fixed point and signed critical eigenpair at fixed native gain."""
    n=len(j)
    if len(z)!=2*n+1 or sign not in (-1,1):raise ValueError('fixed-gain critical dimensions/sign')
    x=z[:n];delta=float(z[n]);q=z[n+1:]
    end,u,v,cycle,phase=g_jets(x,j,g,delta,halfperiod)
    m=u[:,:n];a=m-sign*np.eye(n)
    h=np.r_[end-x,a@q,(q@q-1)/2]
    b=np.zeros((2*n+1,2*n+1));b[:n,:n]=m-np.eye(n);b[:n,n]=u[:,n+1]
    b[n:2*n,:n]=np.einsum('ijk,j->ik',v[:,:n,:n],q)
    b[n:2*n,n]=v[:,:n,n+1]@q;b[n:2*n,n+1:]=a;b[-1,n+1:]=q
    return h,b,dict(theta=np.array(x),gain=g,delta=delta,right_vector=q.copy(),endpoint=end,first=u,second=v,half_cycle=cycle,half_phase_jacobians=phase,monodromy=m)


def pair_fixed_amplitude(z,j,g,amplitude,projection,halfperiod=3):
    """G two-cycle with fixed signed amplitude; no quotient of the native state.

    z=(x,y,delta), G(x)=y, G(y)=x and projection.(x-y)=2*amplitude.
    For odd F and G=-F^3 each point is a native F^6 root, but primitive native
    period and stability still require separate full-state checks.
    """
    n=len(j);projection=np.asarray(projection)
    if len(z)!=2*n+1 or projection.shape!=(n,)or not np.isfinite(projection).all():
        raise ValueError('paired state/projection dimensions')
    if not np.isfinite(amplitude)or amplitude<=0 or abs(projection@projection-1)>1e-12:
        raise ValueError('positive amplitude and normalized fixed projection')
    x,y=z[:n],z[n:2*n];d=float(z[-1])
    ex,ux,vx,cx,px=g_jets(x,j,g,d,halfperiod)
    ey,uy,vy,cy,py=g_jets(y,j,g,d,halfperiod)
    h=np.r_[ex-y,ey-x,projection@(x-y)-2*amplitude]
    b=np.zeros((2*n+1,2*n+1));b[:n,:n]=ux[:,:n];b[:n,n:2*n]=-np.eye(n);b[:n,-1]=ux[:,n+1]
    b[n:2*n,:n]=-np.eye(n);b[n:2*n,n:2*n]=uy[:,:n];b[n:2*n,-1]=uy[:,n+1]
    b[-1,:n]=projection;b[-1,n:2*n]=-projection
    return h,b,dict(theta_x=np.array(x),theta_y=np.array(y),gain=g,delta=d,amplitude=float(amplitude),projection=projection.copy(),
        Gx=ex,Gy=ey,first_x=ux,first_y=uy,second_x=vx,second_y=vy,half_cycle_x=cx,half_cycle_y=cy,half_phase_x=px,half_phase_y=py)
