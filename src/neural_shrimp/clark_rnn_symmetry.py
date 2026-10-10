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
