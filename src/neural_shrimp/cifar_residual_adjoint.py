"""Transpose of the two-step residual derivative for scalar-step full-batch GD.

For a smooth real scalar loss, H0 and H1 are symmetric individually, but
(I-eta H1)(I-eta H0) generally is not. Its transpose reverses the phase order.
This native helper makes two HVP calls through the supplied Problem, preserving
its operation accounting. It does not apply to momentum or rowwise step sizes.
"""
import math
import torch

def residual_adjoint(problem, theta, theta_next, vector, eta):
    if not isinstance(eta,(float,int)) or not math.isfinite(eta) or eta<=0:
        raise ValueError('finite positive scalar learning rate')
    if any(not isinstance(v,torch.Tensor) or v.dtype!=torch.float64 or v.ndim!=1 or v.shape!=theta.shape or not torch.isfinite(v).all() for v in (theta,theta_next,vector)):
        raise ValueError('finite native float64 states and cotangent')
    if not(theta.device==theta_next.device==vector.device):
        raise ValueError('same device')
    # A^T v = J0 J1 v - v. Each Jk uses the Hessian of the same scalar loss.
    right=vector-eta*problem.hvp(theta_next,vector)
    result=right-eta*problem.hvp(theta,right)-vector
    if not torch.isfinite(result).all():raise ValueError('finite adjoint return')
    return result
