"""Scheduled pulse sensitivities and finite trajectories; no file access.

All parameters, initial states, reference write directions and sampling times
are supplied by the caller. No target parameters or historical evidence here.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from two_write_interaction import _array, _finite, _sigma


METHODS = {
    'primary': dict(method='DOP853', rtol=2e-11, atol=2e-13, steps_per_period=512),
    'independent': dict(method='RK45', rtol=2e-12, atol=2e-14, steps_per_period=2048),
}


def tanh_flow(W):
    W = _array(W, 'W', ndim=2).copy()
    if W.shape[0] != W.shape[1]:
        raise ValueError('W must be square')
    eye = np.eye(W.shape[0])

    def rhs(t, x):
        return -x + W @ np.tanh(x)

    def jac(t, x):
        return -eye + W * (1 - np.tanh(x)**2)[None, :]

    def hessian(t, x, u, v):
        activation = np.tanh(x)
        return W @ (-2 * activation * (1 - activation**2) * u * v)

    return rhs, jac, hessian


def solve(rhs, start, stop, initial, period, method):
    if method not in METHODS:
        raise ValueError('Unknown frozen solver configuration')
    config = dict(METHODS[method])
    config['max_step'] = period / config.pop('steps_per_period')
    out = solve_ivp(rhs, (start, stop), initial, dense_output=True, **config)
    if not out.success or out.t[-1] != stop or not np.isfinite(out.y).all():
        raise RuntimeError('Incomplete or nonfinite integration: ' + out.message)
    return out


def sample_times(period):
    period = _sigma(period)
    return period * np.concatenate((np.linspace(1, 1.25, 32), np.linspace(1.75, 2, 32)))


def _inputs(initial, q1, q2, period, times):
    initial = _array(initial, 'initial', ndim=1)
    q1, q2 = _array(q1, 'q1', shape=initial.shape), _array(q2, 'q2', shape=initial.shape)
    period = _sigma(period)
    times = _array(times, 'times', ndim=1)
    if times[0] < period or np.any(np.diff(times) <= 0) or times[-1] <= period:
        raise ValueError('Times must strictly increase from at least the second write')
    return initial, q1, q2, period, times


def reference(rhs, jac, hessian, initial, q1, q2, period, times, method):
    """Return sampled [x,u1,u2,v11,v22,v12] including post-write time T."""
    initial, q1, q2, period, times = _inputs(initial, q1, q2, period, times)
    n = initial.size
    packed = np.zeros((6, n))
    packed[0], packed[1] = initial, q1

    def combined(t, flat):
        x, u1, u2, v11, v22, v12 = flat.reshape(6, n)
        J = jac(t, x)
        return np.concatenate((rhs(t, x), J @ u1, J @ u2,
            J @ v11 + hessian(t, x, u1, u1),
            J @ v22 + hessian(t, x, u2, u2),
            J @ v12 + hessian(t, x, u1, u2)))

    before = solve(combined, 0., period, packed.ravel(), period, method)
    event = before.y[:, -1].reshape(6, n).copy()
    event[2] += q2
    after = solve(combined, period, float(times[-1]), event.ravel(), period, method)
    samples = after.sol(times).T.reshape(len(times), 6, n)
    return dict(samples=_finite(samples, 'sensitivity samples'),
                nfev=before.nfev + after.nfev, method=method,
                event_before=before.y[:, -1].reshape(6, n).copy(), event_after=event)


def finite(rhs, initial, q1, q2, period, times, a1, a2, method):
    initial, q1, q2, period, times = _inputs(initial, q1, q2, period, times)
    if not np.isfinite([a1, a2]).all():
        raise ValueError('Pulse magnitudes must be finite')
    before = solve(rhs, 0., period, initial + a1*q1, period, method)
    event = before.y[:, -1] + a2*q2
    after = solve(rhs, period, float(times[-1]), event, period, method)
    return dict(samples=_finite(after.sol(times).T, 'finite samples'),
                nfev=before.nfev+after.nfev, method=method,
                event_before=before.y[:, -1].copy(), event_after=event)


def coefficients(samples, amplitude, observation='state', order='second'):
    """Time-major coefficient vectors with explicit observation cross terms.

    'ablated' removes v12 only, retaining self terms and any static-observation
    interaction. 'first' linearizes both the state and observation.
    """
    samples = _array(samples, 'samples', ndim=3)
    if samples.shape[1] != 6 or order not in ('first', 'second', 'ablated'):
        raise ValueError('Six sensitivity components and a known order required')
    amplitude = _sigma(amplitude)
    x, u1, u2, v11, v22, v12 = np.moveaxis(samples, 1, 0)
    if observation == 'state':
        baseline, gp, gpp = x, np.ones_like(x), np.zeros_like(x)
    elif observation == 'activation':
        baseline = np.tanh(x)
        gp = 1-baseline**2
        gpp = -2*baseline*gp
    else:
        raise ValueError('Unknown observation')
    A, B = amplitude*gp*u1, amplitude*gp*u2
    m = baseline.copy()
    state_cross = np.zeros_like(x)
    observation_cross = np.zeros_like(x)
    if order != 'first':
        m += amplitude**2 * (gp*(v11+v22) + gpp*(u1*u1+u2*u2))/2
        observation_cross = amplitude**2*gpp*u1*u2
        if order == 'second':
            state_cross = amplitude**2*gp*v12
    return {k: _finite(v.ravel(), k) for k, v in dict(m=m, A=A, B=B,
        C=state_cross+observation_cross, state_cross=state_cross,
        observation_cross=observation_cross).items()}


def phase_contexts(rhs, jac, section, terminal_adjoint, period, fractions, method):
    """Fresh dense phases, backward adjoint; no per-phase gradient repair."""
    section = _array(section, 'section', ndim=1)
    terminal_adjoint = _array(terminal_adjoint, 'terminal_adjoint', shape=section.shape)
    period = _sigma(period)
    fractions = _array(fractions, 'fractions', ndim=1)
    if np.any(fractions < 0) or np.any(fractions >= 1):
        raise ValueError('Fractions must be in [0,1)')
    base = solve(rhs, 0., period, section, period, method)
    adjoint = solve(lambda t,z: -jac(t,base.sol(t)).T@z, period, 0., terminal_adjoint, period, method)
    result=[]
    for fraction in fractions:
        t = float(fraction*period)
        x, z = base.sol(t), adjoint.sol(t)
        norm = np.linalg.norm(z)
        if norm == 0 or not np.isfinite(norm):
            raise ValueError('Undefined phase gradient')
        projector = np.eye(len(z)) - np.outer(z/norm, z/norm)
        coordinate = int(np.argmax(np.linalg.norm(projector, axis=0)))
        q = projector[:, coordinate]/np.linalg.norm(projector[:, coordinate])
        result.append(dict(initial=x, direction=q, phase_gradient=z,
            phase_fraction=float(fraction), coordinate=coordinate,
            normalization_residual=float(z@rhs(t,x)-1),
            direction_leakage=float(z@q),period=period))
    return dict(contexts=result, closure=float(np.max(np.abs(base.y[:,-1]-section))),
        adjoint_closure=float(np.max(np.abs(adjoint.y[:,-1]-terminal_adjoint))),
        nfev=base.nfev+adjoint.nfev, method=method)
