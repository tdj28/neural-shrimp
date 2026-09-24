"""Constructive lower and mixture-TV upper bounds on contextual bit recovery.

Four equally likely means, two per binary label, common isotropic Gaussian
measurement noise. Starting context and the complete forward model are known.
No claim of learned, context-blind or noise-free computational capacity.
"""
import numpy as np
from scipy.special import ndtr

LABELS=np.array([[-1,-1,1],[-1,1,-1],[1,-1,-1],[1,1,1]])


def bounds(means, sigma, labels):
    means=np.asarray(means,dtype=float)
    labels=np.asarray(labels)
    if means.ndim!=2 or means.shape[0]!=4 or not means.shape[1] or not np.isfinite(means).all():
        raise ValueError('Four finite, nonempty observation vectors required')
    if labels.shape!=(4,) or set(labels)!={-1,1} or np.sum(labels==1)!=2:
        raise ValueError('Exactly two positive and two negative labels required')
    if not np.isfinite(sigma) or sigma<=0:
        raise ValueError('Positive finite observation noise required')
    negative=means[labels==-1]
    positive=means[labels==1]
    distances=np.linalg.norm(negative[:,None,:]-positive[None,:,:],axis=2)
    # Convexity of TV under either pairing of the two Gaussian mixtures.
    binary_accuracy=ndtr(distances/(2*sigma))
    upper=min((binary_accuracy[0,0]+binary_accuracy[1,1])/2,
              (binary_accuracy[0,1]+binary_accuracy[1,0])/2)
    # Error of nearest-four-template classification is bounded by the union
    # of opposite-label templates beating the true template. Average over 4.
    nearest_lower=1-.5*np.sum(ndtr(-distances/(2*sigma)))
    # A realizable context-specific affine readout and its exact Gaussian accuracy.
    w=positive.mean(axis=0)-negative.mean(axis=0)
    center=(positive.mean(axis=0)+negative.mean(axis=0))/2
    norm=np.linalg.norm(w)
    linear=.5 if norm==0 else np.mean(ndtr(labels*((means-center)@w)/(sigma*norm)))
    lower=max(.5,float(nearest_lower),float(linear))
    assert lower<=upper+2e-12,(lower,upper)
    return np.array([min(lower,upper),upper,linear,nearest_lower,distances.min()])


def batch(means,sigma):
    means=np.asarray(means)
    if means.ndim!=3 or means.shape[1]!=4:
        raise ValueError('Expected contexts by four sign cases by observations')
    return np.array([[bounds(m,sigma,LABELS[:,j]) for j in range(3)] for m in means])
