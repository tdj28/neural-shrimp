"""Amplitude-constrained period-two equations with a variable learning rate.

This defines a new optimizer-rate continuation problem, not a replacement
residual or acceptance rule for any prior fixed-rate CIFAR experiment.
"""
import numpy as np


def vector(v):
    v=np.asarray(v,dtype=np.float64)
    if v.ndim!=1 or not len(v) or not np.isfinite(v).all():raise ValueError('finite flat vector')
    return v


def equations(m,d,tau,amplitude,gradient):
    m,d=vector(m),vector(d)
    if m.shape!=d.shape or not np.isfinite([tau,amplitude]).all() or tau<=0 or amplitude<=0:
        raise ValueError('same dimension, positive reciprocal rate and amplitude')
    gp,gm=vector(gradient(m+d)),vector(gradient(m-d))
    if gp.shape!=m.shape or gm.shape!=m.shape:raise ValueError('complete gradients')
    return np.r_[(gp+gm)/2,(gp-gm)/2-tau*d,(amplitude**2-np.dot(d,d))/2]


def jacobian_action(m,d,tau,direction,hvp):
    m,d,z=vector(m),vector(d),vector(direction);n=len(m)
    if d.shape!=m.shape or z.shape!=(2*n+1,)or not np.isfinite(tau)or tau<=0:
        raise ValueError('complete augmented direction and positive tau')
    dm,dd,dt=z[:n],z[n:2*n],z[-1]
    hp,hm=vector(hvp(m+d,dm+dd)),vector(hvp(m-d,dm-dd))
    if hp.shape!=m.shape or hm.shape!=m.shape:raise ValueError('complete phase Hessian actions')
    return np.r_[(hp+hm)/2,(hp-hm)/2-tau*dd-dt*d,-np.dot(d,dd)]


def physical_residuals(residual,tau):
    v=vector(residual)
    if len(v)%2!=1 or len(v)<3 or not np.isfinite(tau)or tau<=0:raise ValueError('complete positive-rate augmented residual')
    n=(len(v)-1)//2;eta=2/tau
    return -eta*(v[:n]+v[n:2*n]),-eta*(v[:n]-v[n:2*n])
