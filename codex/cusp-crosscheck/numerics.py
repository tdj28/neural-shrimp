"""Task14 float64 directional AD and fold cubic algebra; no model construction.

Numerical diagnostics, not interval enclosures or an exact cusp certificate.
"""
import numpy as np
import torch


def directional(function, point, directions):
    """D^k function(point)[v1,...,vk], with fixed directions and k<=3."""
    if not 1 <= len(directions) <= 3:
        raise ValueError('one through three fixed directions required')
    if point.ndim != 1 or point.dtype != torch.float64 or point.device.type != 'cpu':
        raise ValueError('one full float64 CPU state required')
    if not bool(torch.isfinite(point).all()):
        raise ValueError('finite state required')
    for vector in directions:
        if vector.shape != point.shape or vector.dtype != point.dtype or vector.device != point.device or not bool(torch.isfinite(vector).all()):
            raise ValueError('matching finite float64 CPU directions required')
    result = function
    for vector in directions:
        previous = result
        result = lambda x, previous=previous, vector=vector: torch.func.jvp(previous, (x,), (vector,))[1]
    return result(point)


def fold_basis(matrix, *, critical_tolerance=1e-8, imaginary_tolerance=1e-10,
               unit_circle_gap=1e-3, isolation=1e-3, eigen_tolerance=1e-8):
    """Classify the complete dense spectrum BEFORE normal-form operations."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2 or not np.isfinite(matrix).all():
        raise ValueError('finite full square matrix required')
    values, vectors = np.linalg.eig(matrix)
    if not np.isfinite(values).all() or not np.isfinite(vectors).all():
        raise ValueError('nonfinite dense eigensystem')
    k = int(np.argmin(abs(values-1)))
    mu, other = values[k], np.delete(values, k)
    if abs(mu.imag) > imaginary_tolerance or abs(mu.real-1) > critical_tolerance:
        raise ValueError('not an admitted real +1 multiplier')
    if min(abs(abs(other)-1)) <= unit_circle_gap or min(abs(other-mu)) <= isolation:
        raise ValueError('unresolved critical isolation or hyperbolic complement')
    right = vectors[:, k].real.copy(); right /= np.linalg.norm(right)
    if not np.isfinite(right).all():raise ValueError('nonfinite normalized right mode')
    if right[np.argmax(abs(right))] < 0: right *= -1
    left_values, left_vectors = np.linalg.eig(matrix.T)
    if not np.isfinite(left_values).all() or not np.isfinite(left_vectors).all():raise ValueError('nonfinite left eigensystem')
    left = left_vectors[:, int(np.argmin(abs(left_values-mu)))].real.copy()
    pairing = float(left @ right)
    if not np.isfinite(pairing) or abs(pairing) < 1e-12: raise ValueError('unresolved left/right normalization')
    left /= pairing
    if not np.isfinite(left).all():raise ValueError('nonfinite normalized left mode')
    all_residuals = np.linalg.norm(matrix@vectors-vectors*values[None, :], axis=0)
    right_residual = float(np.linalg.norm(matrix@right-mu.real*right))
    left_residual = float(np.linalg.norm(matrix.T@left-mu.real*left)/np.linalg.norm(left))
    if not np.isfinite(all_residuals).all() or not np.isfinite([right_residual,left_residual]).all():raise ValueError('nonfinite eigenpair residual')
    if max(float(max(all_residuals)), right_residual, left_residual) >= eigen_tolerance:
        raise ValueError('dense eigenpair residual failure')
    return dict(multiplier=float(mu.real), multiplier_minus_one=float(mu.real-1),
                next_modulus=float(max(abs(other))), nearest_other_unit_circle=float(min(abs(abs(other)-1))),
                complement_attracting=bool(max(abs(other))<1), isolation=float(min(abs(other-mu))),
                right=right, left=left, eigenvalues=values, all_eigenpair_residuals=all_residuals,
                right_residual=right_residual, left_residual=left_residual,
                norm_right=float(np.linalg.norm(right)), pairing=float(left@right),
                orientation='largest-absolute right coordinate positive')


def homological(matrix, right, left, brr, *, multiplier=1.0):
    """Solve [(mu² I-M),r;l^T,0]*(h2,border)=(Brr-2br,0)."""
    matrix = np.asarray(matrix, dtype=np.float64)
    right, left, brr = [np.asarray(v, dtype=np.float64) for v in (right, left, brr)]
    n = len(right)
    if matrix.shape != (n,n) or left.shape != (n,) or brr.shape != (n,) or not np.isfinite(matrix).all() or not all(np.isfinite(v).all() for v in (right,left,brr)) or not np.isfinite(multiplier):
        raise ValueError('finite full-state homological inputs required')
    if abs(np.linalg.norm(right)-1) > 1e-10 or abs(left@right-1) > 1e-10:
        raise ValueError('unit right and l^T r=1 required')
    b = .5*float(left@brr)
    operator = multiplier**2*np.eye(n)-matrix
    bordered = np.block([[operator,right[:,None]],[left[None,:],np.zeros((1,1))]])
    rhs = np.concatenate([brr-2*b*right,[0.]])
    solution = np.linalg.solve(bordered,rhs); h2=solution[:n]
    denominator=max(1.,float(np.linalg.norm(rhs)))
    result=dict(b=b,h2=h2,border=float(solution[-1]),multiplier_used=float(multiplier),
                bordered_relative_residual=float(np.linalg.norm(bordered@solution-rhs)/denominator),
                unbordered_relative_residual=float(np.linalg.norm(operator@h2-rhs[:n])/denominator),
                orthogonality=float(left@h2),h2_norm=float(np.linalg.norm(h2)))
    if not np.isfinite(h2).all() or not all(np.isfinite(v) for k,v in result.items() if k!='h2'):raise ValueError('nonfinite homological output')
    return result


def cubic(left, crrr, brh2):
    left, crrr, brh2=[np.asarray(v,dtype=np.float64) for v in (left,crrr,brh2)]
    if left.ndim != 1 or crrr.shape != left.shape or brh2.shape != left.shape or not all(np.isfinite(v).all() for v in (left,crrr,brh2)):
        raise ValueError('finite full-state cubic inputs required')
    direct=float(left@crrr)/6; center=float(left@brh2)/2; total=direct+center
    result=dict(c=total,direct=direct,center_manifold=center,
                cancellation_ratio=(abs(direct)+abs(center))/max(abs(total),np.finfo(float).tiny))

    if not all(np.isfinite(v) for v in result.values()):raise ValueError('nonfinite cubic scalar output')
    return result
