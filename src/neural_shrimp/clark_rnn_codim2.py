"""Numerical native fold--flip candidates; analytic second jets, no proof claim."""
import numpy as np
from .clark_rnn_boundaries import jets

def fold_flip_augmented(z,j,period):
    n=len(j)
    if len(z)!=3*n+2:raise ValueError('fold-flip augmented dimension')
    x=z[:n];g=float(z[n]);delta=float(z[n+1]);q=z[n+2:2*n+2];r=z[2*n+2:]
    end,u,v,cycle,phase=jets(x,j,g,delta,period);m=u[:,:n]
    plus=m-np.eye(n);minus=m+np.eye(n)
    h=np.r_[end-x,plus@q,minus@r,(q@q-1)/2,(r@r-1)/2]
    dh=np.zeros((3*n+2,3*n+2));dh[:n,:n+2]=u;dh[:n,:n]-=np.eye(n)
    dh[n:2*n,:n+2]=np.einsum('ijk,j->ik',v[:,:n,:],q);dh[n:2*n,n+2:2*n+2]=plus
    dh[2*n:3*n,:n+2]=np.einsum('ijk,j->ik',v[:,:n,:],r);dh[2*n:3*n,2*n+2:]=minus
    dh[-2,n+2:2*n+2]=q;dh[-1,2*n+2:]=r
    return h,dh,{'theta':np.array(x),'gain':g,'delta':delta,'right_plus':np.array(q),'right_minus':np.array(r),'endpoint':end,'first':u,'second':v,'cycle':cycle,'phase_jacobians':phase,'monodromy':m}
