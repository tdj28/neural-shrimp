"""EXP017 reference-only acquisition: no IO, scheduling, or work at import.

Every adjoint inherits the exact old terminal vector. No normalization, new
neutral vector, orbit shooting, phase alignment, pulse or observer is performed.
The caller owns full census, numerical qualification, source/input seals and
immutable retention of AcquisitionFailure.payload.
"""
from __future__ import annotations

import copy
import time
import numpy as np
from scipy.integrate import solve_ivp

from hopfield_diagnostics import hopfield_rhs, hopfield_jacobian
from phase_memory_acquisition import AcquisitionFailure, reference_templates

METHODS = {
    'baseline-dop853': dict(kind='DOP853', rtol=1e-12, atol=1e-14, max_step=.025),
    'fine-dop853': dict(kind='DOP853', rtol=1e-12, atol=1e-14, max_step_period_divisor=8192),
    'fine-rk45': dict(kind='RK45', rtol=1e-12, atol=1e-14, max_step_period_divisor=8192),
}
EVENT_METHODS = ('fine-dop853', 'fine-rk45')
GRID_INTERVALS = (4096, 8192)


def _array(value, name, shape=None):
    result = np.asarray(value)
    if result.dtype != np.float64 or (shape is not None and result.shape != shape) or not np.isfinite(result).all():
        raise ValueError(f'{name} requires finite float64 shape {shape}')
    return result


def _scalar(value, name, *, positive=False):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value) or (positive and value <= 0)):
        raise ValueError(f'{name} requires a finite {"positive " if positive else ""}scalar')
    return float(value)


def method_spec(method, period):
    period = _scalar(period, 'period', positive=True)
    if method not in METHODS: raise ValueError('Unknown reference diagnostic method')
    spec = dict(METHODS[method], id=method)
    if 'max_step_period_divisor' in spec: spec['max_step'] = period/spec['max_step_period_divisor']
    _scalar(spec['max_step'], 'maximum step', positive=True)
    return spec


def build_grids(period):
    period = _scalar(period, 'period', positive=True)
    grids = {}
    for count in GRID_INTERVALS:
        times = np.arange(count+1, dtype=np.float64)*(period/count)
        times[-1] = period
        if not np.isfinite(times).all() or not np.all(np.diff(times) > 0):
            raise ValueError('Period must admit strictly increasing float64 grids')
        grids[str(count)] = times
    if not np.array_equal(grids['4096'], grids['8192'][::2]):
        raise ValueError('Nested grid identity failed')
    return grids


def _phases(rows, period, setting):
    if not isinstance(rows, list) or len(rows) != 24: raise ValueError('Exactly 24 ordered phase rows required')
    result = []
    for index, row in enumerate(rows):
        if type(row.get('phase_index')) is not int or row['phase_index'] != index:
            raise ValueError('Ordered unique phase indices 0..23 required')
        fraction = _scalar(row['phase_fraction'], 'phase fraction')
        if not 0 <= fraction < 1: raise ValueError('Phase fraction must be in [0,1)')
        split = 'development' if index < 8 else 'heldout'
        if row.get('split') != split: raise ValueError('Inherited phase split differs')
        result.append(dict(context_id=f'{setting}--p{index:02d}', phase_index=index,
            phase_fraction=fraction, split=split, phase_epoch=fraction*period))
    return result


def _solve(fun, span, initial, times, spec, *, solver, stage, offset=0.):
    """Dense solve with complete returned samples or explicitly partial evidence."""
    initial = _array(initial, 'solver initial').copy()
    times = _array(times, 'solver times')
    direction = np.sign(span[1]-span[0])
    if (times.ndim != 1 or len(times) < 2 or times[0] != span[0] or times[-1] != span[1]
            or not np.all(direction*np.diff(times) > 0)):
        raise ValueError('Solver grid must exactly span a strictly monotone interval')
    last_rhs = {}; started = time.monotonic()
    def tracked(local, packed):
        last_rhs.update(time=float(local), physical_time=float(offset+local),
            values=np.array(packed, copy=True), accepted_solver_state=False)
        return fun(offset+local, packed)
    common = dict(stage=stage, span=np.asarray(span, dtype=np.float64), offset=float(offset),
        requested_times=times.copy(), initial=initial, method_spec=dict(spec))
    try:
        solution = solver(tracked, span, initial, t_eval=times, dense_output=True,
            method=spec['kind'], rtol=spec['rtol'], atol=spec['atol'], max_step=spec['max_step'])
    except Exception as error:
        raise AcquisitionFailure('Reference diagnostic solver exception', dict(common,
            status='solver_exception', error=repr(error), last_rhs=last_rhs,
            wall_seconds=time.monotonic()-started)) from error
    snapshot = dict(common, times=np.asarray(solution.t).copy(), values=np.asarray(solution.y).T.copy(),
        solver=dict(success=bool(solution.success), message=str(solution.message),
            nfev=int(solution.nfev), njev=int(solution.njev), nlu=int(solution.nlu)),
        wall_seconds=time.monotonic()-started)
    try:
        _array(snapshot['times'], 'returned solver times', times.shape)
        _array(snapshot['values'], 'returned solver values', (len(times), len(initial)))
        if (not solution.success or not np.array_equal(snapshot['times'], times)
                or not np.array_equal(snapshot['values'][0], initial) or not callable(solution.sol)):
            raise ValueError('Incomplete solver, changed times/initial state, or missing dense output')
    except Exception as error:
        raise AcquisitionFailure('Invalid or incomplete diagnostic trajectory', dict(snapshot,
            status='invalid_or_incomplete', error=repr(error), last_rhs=last_rhs)) from error
    return snapshot, solution.sol


def _augmented(alpha, gamma, rhs, jacobian):
    def fun(t, packed):
        x, phi = packed[:3], packed[3:].reshape(3, 3)
        return np.r_[rhs(t, x, alpha, gamma), (jacobian(t, x, alpha, gamma)@phi).ravel()]
    return fun


def _trajectory(times, packed):
    packed = _array(packed, 'state/tangent samples', (len(times), 12))
    return dict(times=times.copy(), states=packed[:, :3].copy(), phi=packed[:, 3:].reshape(-1, 3, 3).copy())


def _derivatives(segment, alpha, gamma, rhs, jacobian, *, offset=0.):
    times, states, phi = segment['times'], segment['states'], segment['phi']
    segment['derivatives'] = np.asarray([rhs(offset+t, x, alpha, gamma) for t, x in zip(times, states, strict=True)])
    _array(segment['derivatives'], 'physical derivatives', states.shape)
    segment['phi_derivatives'] = np.asarray([jacobian(offset+t, x, alpha, gamma)@p
        for t, x, p in zip(times, states, phi, strict=True)])
    _array(segment['phi_derivatives'], 'tangent derivatives', phi.shape)
    if 'z' in segment:
        segment['z_derivatives'] = np.asarray([-jacobian(offset+t, x, alpha, gamma).T@z
            for t, x, z in zip(times, states, segment['z'], strict=True)])
        _array(segment['z_derivatives'], 'adjoint derivatives', states.shape)
        # Keep the same pointwise dot arithmetic used by scalar phase/event
        # diagnostics; changing contraction order can matter under cancellation.
        segment['normalization_residuals'] = np.asarray([float(z@flow-1.)
            for z,flow in zip(segment['z'],segment['derivatives'],strict=True)],dtype=np.float64)
        _array(segment['normalization_residuals'], 'normalization residuals', times.shape)


def _compact_solver(snapshot):
    return {k:copy.deepcopy(v) for k,v in snapshot.items() if k not in ('times', 'values', 'requested_times')}


def _fail(packet, stage, error):
    packet.update(status='failed', complete=False, failed_stage=stage, error=repr(error))
    if hasattr(error, 'payload'): packet['partial_failure'] = error.payload
    return AcquisitionFailure('Reference diagnostic acquisition failed', packet)


def acquire_base(oldreference, phase_rows, method, *, rhs=hopfield_rhs,
                 jacobian=hopfield_jacobian, solver=solve_ivp):
    """Direct 12D forward solve and backward adjoint with unchanged old boundary."""
    period = _scalar(oldreference['period'], 'period', positive=True)
    alpha = _scalar(oldreference['alpha'], 'alpha'); gamma = _scalar(oldreference['gamma'], 'gamma')
    setting = oldreference['setting']
    if not isinstance(setting, str) or not setting: raise ValueError('Nonempty setting identity required')
    spec = method_spec(method, period); grids = build_grids(period)
    phases = _phases(phase_rows, period, setting)
    initial = _array(oldreference['certificate']['section_state'], 'inherited section state', (3,)).copy()
    terminal = _array(oldreference['adjoint']['terminal_normalized_once'], 'inherited terminal adjoint', (3,)).copy()
    if (oldreference['certificate']['period'] != period
            or not np.array_equal(initial, oldreference['orbit']['states'][0])
            or not np.array_equal(terminal, oldreference['adjoint']['values'][-1])
            or type(oldreference['adjoint']['normalization_count']) is not int
            or oldreference['adjoint']['normalization_count'] != 1):
        raise ValueError('Inherited section state, period or fixed terminal-vector binding differs')
    oldstate, oldz = reference_templates(oldreference)
    packet = dict(schema='phase-reference-base-v1', id=f'{setting}--{method}', kind='base',
        setting=setting, alpha=alpha, gamma=gamma, period=period, method=method,
        method_spec=spec, status='started', complete=False,
        inherited=dict(section_state=initial.copy(), terminal_adjoint=terminal.copy(),
            normalization_count_from_old=1, new_normalizations=0),
        phase_rows=copy.deepcopy(phases), grids={}, phases=[],
        provenance=dict(trajectory='direct 12D forward state/full tangent; no periodic wrap',
            adjoint='backward on same solver dense state with original terminal vector',
            dense_phase='exact requested epoch evaluated by solver dense output; no saved-knot interpolation',
            qualifications='owned by evaluator; acquisition does not gate normalization or closure'))
    stage = 'forward'
    try:
        forward, dense = _solve(_augmented(alpha, gamma, rhs, jacobian), (0., period),
            np.r_[initial, np.eye(3).ravel()], grids['8192'], spec, solver=solver, stage=stage)
        packet['forward'] = forward
        fine = _trajectory(grids['8192'], forward['values']); packet['grids']['8192'] = fine
        packet['raw_endpoints'] = dict(initial_state=fine['states'][0].copy(), final_state=fine['states'][-1].copy(),
            initial_phi=fine['phi'][0].copy(), final_phi=fine['phi'][-1].copy())
        stage = 'backward'
        def adjoint(t, z):
            packed = _array(dense(t), 'dense forward state', (12,))
            return -jacobian(t, packed[:3], alpha, gamma).T@z
        backward, dense_z = _solve(adjoint, (period, 0.), terminal, grids['8192'][::-1].copy(),
            spec, solver=solver, stage=stage)
        packet['backward'] = backward
        fine['z'] = backward['values'][::-1].copy()
        packet['raw_endpoints'].update(initial_z=fine['z'][0].copy(), final_z=fine['z'][-1].copy())
        if not np.array_equal(fine['z'][-1], terminal): raise ValueError('Terminal adjoint changed')
        stage = 'grid_derivatives'
        _derivatives(fine, alpha, gamma, rhs, jacobian)
        packet['grids']['4096'] = {key:value[::2].copy() for key,value in fine.items()}
        stage = 'exact_phases'
        epochs = np.asarray([x['phase_epoch'] for x in phases], dtype=np.float64)
        phase_data = _trajectory(epochs, _array(dense(epochs), 'dense phase packed', (12, 24)).T)
        packet['exact_phase_arrays'] = phase_data
        phase_data['z'] = _array(dense_z(epochs), 'dense phase adjoint', (3, 24)).T.copy()
        _derivatives(phase_data, alpha, gamma, rhs, jacobian)
        for index,row in enumerate(phases):
            point = dict(row, state=phase_data['states'][index].copy(), phi=phase_data['phi'][index].copy(),
                flow=phase_data['derivatives'][index].copy(), phi_derivative=phase_data['phi_derivatives'][index].copy(),
                z=phase_data['z'][index].copy(), z_derivative=phase_data['z_derivatives'][index].copy(),
                normalization_residual=float(phase_data['normalization_residuals'][index]))
            packet['phases'].append(point)
            epoch = np.array(row['phase_epoch'], dtype=np.float64)
            ox = oldstate.evaluate(epoch); oz = oldz.evaluate(epoch)
            of = _array(rhs(float(epoch), ox, alpha, gamma), 'old-Hermite physical flow', (3,))
            point['old_hermite'] = dict(state=ox, flow=of.copy(), z=oz,
                state_derivative=oldstate.evaluate(epoch, derivative=True),
                z_derivative=oldz.evaluate(epoch, derivative=True), normalization_residual=float(oz@of-1.),
                provenance='unchanged inherited periodic cubic-Hermite templates; raw endpoints not substituted')
        stage = 'diagnostics'
        flow0, flowT = fine['derivatives'][0], fine['derivatives'][-1]
        phiT = fine['phi'][-1]; z0,zT=fine['z'][0],fine['z'][-1]
        flow_norm = float(np.linalg.norm(flow0)); z_norm = float(np.linalg.norm(terminal))
        if not np.isfinite([flow_norm,z_norm]).all() or flow_norm == 0 or z_norm == 0:
            raise ValueError('Nonzero finite physical flow and adjoint norms required')
        packet['diagnostics'] = dict(endpoint_difference=fine['states'][-1]-initial,
            relative_closure=float(np.linalg.norm(fine['states'][-1]-initial)/max(1., np.linalg.norm(initial))),
            neutral_flow_relative_residual=float(np.linalg.norm(phiT@flow0-flow0)/flow_norm),
            transported_flow_relative_residual=float(np.linalg.norm(phiT@flow0-flowT)/flow_norm),
            inherited_terminal_left_relative_residual=float(np.linalg.norm(phiT.T@terminal-terminal)/z_norm),
            inherited_terminal_normalization_start=float(terminal@flow0-1.),
            inherited_terminal_normalization_end=float(terminal@flowT-1.),
            adjoint_periodic_difference=z0-zT,
            adjoint_relative_periodicity=float(np.linalg.norm(z0-zT)/max(np.linalg.norm(z0),np.linalg.norm(zT))),
            maximum_grid_normalization_residual=float(np.max(np.abs(fine['normalization_residuals']))))
        for key,value in packet['diagnostics'].items():
            if isinstance(value,np.ndarray): _array(value,'diagnostic '+key)
            else: _scalar(value,'diagnostic '+key)
        packet['forward'] = _compact_solver(forward); packet['backward'] = _compact_solver(backward)
        packet.update(status='complete', complete=True)
        return packet
    except Exception as error:
        raise _fail(packet, stage, error) from error


def acquire_event(base, phase_index, *, rhs=hopfield_rhs, jacobian=hopfield_jacobian, solver=solve_ivp):
    """Own dense phase state, direct one-period continuation, nominal Z unchanged."""
    if base.get('complete') is not True or base.get('status') != 'complete':
        raise ValueError('Complete base required; numerical qualification is not an acquisition gate')
    if base.get('kind') != 'base' or base.get('schema') != 'phase-reference-base-v1': raise ValueError('Base schema differs')
    period = _scalar(base['period'], 'period', positive=True)
    alpha = _scalar(base['alpha'], 'alpha'); gamma = _scalar(base['gamma'], 'gamma')
    method = base['method']; setting = base['setting']; spec = method_spec(method, period)
    if method not in EVENT_METHODS: raise ValueError('Only two fixed fine methods have event jobs')
    if base['id'] != f'{setting}--{method}' or base['method_spec'] != spec: raise ValueError('Base method identity differs')
    if type(phase_index) is not int or phase_index not in range(24): raise ValueError('Phase index must be 0..23')
    phases = _phases(base['phase_rows'], period, setting)
    if len(base['phases']) != 24: raise ValueError('All 24 base phase records required')
    point = base['phases'][phase_index]; expected=phases[phase_index]
    if any(point[k] != v for k,v in expected.items()): raise ValueError('Base phase identity differs')
    initial = _array(point['state'], 'own dense initial state', (3,)).copy()
    gradient = _array(point['z'], 'own dense phase adjoint', (3,)).copy()
    for key,name in [('state','states'),('z','z'),('flow','derivatives'),('phi','phi')]:
        if not np.array_equal(point[key], base['exact_phase_arrays'][name][phase_index]):
            raise ValueError('Base phase sample binding differs')
    epoch = point['phase_epoch']; times=build_grids(period)['8192']
    packet = dict(schema='phase-reference-event-v1', kind='event', id=f'{point["context_id"]}--{method}',
        base_id=base['id'], context_id=point['context_id'], setting=setting, method=method,
        method_spec=spec, alpha=alpha, gamma=gamma, period=period, phase_index=phase_index,
        phase_fraction=point['phase_fraction'], split=point['split'], phase_epoch=epoch,
        event_time=period, physical_event_time=epoch+period, initial=initial.copy(),
        phase_gradient=gradient.copy(), old_hermite=copy.deepcopy(point['old_hermite']),
        status='started', complete=False,
        provenance=dict(initial='same method base exact dense phase state',
            adjoint='same method base exact dense nominal phase adjoint; unchanged at event',
            tangent='fresh identity at phase start; direct 12D one-period propagation',
            time='local 0..T with physical RHS time phase_epoch+local; no phase wrapping',
            new_normalizations=0))
    stage='forward'
    try:
        raw,_ = _solve(_augmented(alpha, gamma, rhs, jacobian), (0.,period),
            np.r_[initial,np.eye(3).ravel()], times, spec, solver=solver, stage=stage, offset=epoch)
        packet['forward']=raw
        trajectory=_trajectory(times,raw['values']);packet['trajectory']=trajectory
        trajectory['offset']=epoch
        stage='derivatives'
        _derivatives(trajectory,alpha,gamma,rhs,jacobian,offset=epoch)
        stage='event'
        state=trajectory['states'][-1].copy();flow=trajectory['derivatives'][-1].copy()
        initial_flow=trajectory['derivatives'][0].copy(); q=np.outer(flow,gradient);p=np.eye(3)-q
        old_gradient=_array(point['old_hermite']['z'],'old-Hermite phase adjoint',(3,))
        packet['event']=dict(state=state,flow=flow,phase_gradient=gradient.copy(),Q=q,P=p,
            normalization_residual=float(gradient@flow-1.),
            initial_normalization_residual=float(gradient@initial_flow-1.),
            event_minus_initial_normalization=float(gradient@(flow-initial_flow)),
            initial_flow=initial_flow, return_difference=state-initial,
            old_z_initial_normalization=float(old_gradient@initial_flow-1.),
            old_z_event_normalization=float(old_gradient@flow-1.),
            Q_idempotence_residual=q@q-q, P_idempotence_residual=p@p-p,
            QP_residual=q@p, phase_leakage=gradient@p,
            Q_operator_norm=float(np.linalg.norm(q,ord=2)),P_operator_norm=float(np.linalg.norm(p,ord=2)))
        for key,value in packet['event'].items():
            if isinstance(value,np.ndarray):_array(value,'event '+key)
            else:_scalar(value,'event '+key)
        packet['forward']=_compact_solver(raw)
        packet.update(status='complete',complete=True)
        return packet
    except Exception as error:
        raise _fail(packet,stage,error) from error
