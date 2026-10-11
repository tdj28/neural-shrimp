"""Opt-in float64 device derivatives and a measured CPU SciPy bridge.

This engineering candidate does not change the reference CPU implementation or
provide orbit, Krylov convergence, spectral-completeness or rigor guarantees.
"""
import time
import numpy as np
import torch
from scipy.sparse.linalg import LinearOperator
from .cifar import Problem


class DeviceProblem(Problem):
    def __init__(self, model, x, y, lam, device='cpu'):
        super().__init__(model, x, y, lam)
        self.device = torch.device(device)
        self.x = self.x.to(device=self.device, dtype=torch.float64)
        self.y = self.y.to(device=self.device, dtype=torch.float64)

    def tensor(self, value):
        return torch.as_tensor(value, device=self.device, dtype=torch.float64)

    def synchronize(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)


class HostTwoStepOperator(LinearOperator):
    """CPU NumPy/SciPy vectors, device J2VP, explicit synchronized phase costs.

    Data and both base states stay on the device. Complex Ritz residual calls
    split into two real derivatives. Subtracting identity supports Newton's
    linear system; SciPy's GMRES/Arnoldi and orthogonalization remain on CPU.
    """
    def __init__(self, problem, theta, eta, subtract_identity=False):
        self.problem, self.eta = problem, eta
        self.theta = problem.tensor(theta)
        self.theta_next = problem.update(self.theta, eta)
        self.subtract_identity = subtract_identity
        self.measurements = []
        problem.synchronize()
        super().__init__(dtype=np.dtype(np.float64), shape=(self.theta.numel(), self.theta.numel()))

    def _real(self, vector):
        p = self.problem
        p.synchronize()
        start = time.perf_counter()
        host = np.ascontiguousarray(vector, dtype=np.float64)
        tangent = p.tensor(host)
        p.synchronize()
        copied = time.perf_counter()
        result = p.jvp_two_step(self.theta, tangent, self.eta, theta_next=self.theta_next)
        p.synchronize()
        computed = time.perf_counter()
        output = result.detach().cpu().numpy().copy()
        p.synchronize()
        returned = time.perf_counter()
        self.measurements.append({'host_to_device_seconds': copied-start,
            'device_j2vp_seconds': computed-copied, 'device_to_host_seconds': returned-computed,
            'total_seconds': returned-start, 'input_bytes': host.nbytes, 'output_bytes': output.nbytes})
        return output

    def _matvec(self, vector):
        vector = np.asarray(vector).reshape(-1)
        output = (self._real(vector.real) + 1j*self._real(vector.imag)
                  if np.iscomplexobj(vector) else self._real(vector))
        return output-vector if self.subtract_identity else output


def parity_metrics(observed, expected, atol=1e-11, rtol=1e-9):
    """Coordinatewise mixed tolerance; report norm error too, never hide NaNs."""
    a, b = np.asarray(observed), np.asarray(expected)
    if a.shape != b.shape or atol <= 0 or rtol < 0:
        raise ValueError('invalid parity comparison')
    finite = bool(np.isfinite(a).all() and np.isfinite(b).all())
    if not finite:
        return {'passed': False, 'finite': False, 'atol': atol, 'rtol': rtol}
    error = np.abs(a-b)
    scaled = float(np.max(error/(atol+rtol*np.abs(b))))
    return {'passed': scaled <= 1., 'finite': True, 'max_abs_error': float(np.max(error)),
            'relative_l2_error': float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-300)),
            'max_scaled_error': scaled, 'atol': atol, 'rtol': rtol}
