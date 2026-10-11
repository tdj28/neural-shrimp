"""EXP-016 pure acquisition adapters: no IO, scheduling, or work at import.

Discontinuities have separate left/right records. All times inside a segment
are relative to its offset; no interpolant spans the intervention. Numerical
qualification belongs to the frozen evaluator, not these acquisition helpers.
"""
from __future__ import annotations

import copy
import time
import numpy as np
from scipy.integrate import solve_ivp

from hopfield_diagnostics import hopfield_rhs, hopfield_jacobian
from hopfield_drift_probe import integrate_trace, ProbeFailure
from phase_memory_acquisition import AcquisitionFailure, reference_templates

SAMPLES_PER_CYCLE = 4096
ARMS = ('identity', 'timing', 'transverse')
METHODS = {
    'rk4-coarse': dict(kind='RK4', steps_per_cycle=32768),
    'rk4-fine': dict(kind='RK4', steps_per_cycle=65536),
    'dop853-independent': dict(kind='DOP853', rtol=1e-13, atol=1e-15, max_step_period_divisor=4096),
}
TANGENT_METHODS = {
    'dop853': dict(kind='DOP853', rtol=1e-12, atol=1e-14, max_step_period_divisor=2048),
    'rk45': dict(kind='RK45', rtol=1e-12, atol=1e-14, max_step_period_divisor=4096),
}


def _array(value, name, shape=None):
    value = np.asarray(value)
    if value.dtype != np.float64 or (shape is not None and value.shape != shape) or not np.isfinite(value).all():
        raise ValueError(f'{name} requires finite float64 shape {shape}')
    return value


def _positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive finite scalar')
    return float(value)


def build_grid(period):
    period = _positive(period, 'period')
    spacing = period / SAMPLES_PER_CYCLE
    pre = np.arange(SAMPLES_PER_CYCLE+1, dtype=np.float64)*spacing
    post = np.arange(7*SAMPLES_PER_CYCLE+1, dtype=np.float64)*spacing
    pre[-1], post[-1] = period, 7*period
    if not np.isfinite(post).all() or not np.all(np.diff(pre) > 0) or not np.all(np.diff(post) > 0):
        raise ValueError('Period does not admit a finite strictly increasing float64 grid')
    return dict(pre=pre, post=post, post_offset=period, sample_dt=spacing,
                samples_per_cycle=SAMPLES_PER_CYCLE)


def _context(context):
    period = _positive(context['period'], 'period')
    for key in ('initial', 'phase_gradient', 'flow', 'direction', 'nominal_event_phase_gradient'):
        _array(context[key], key, (3,))
    if not np.isfinite([context['alpha'], context['gamma']]).all():
        raise ValueError('Finite model parameters required')
    if abs(float(np.linalg.norm(context['direction']))-1) > 1e-12:
        raise ValueError('Unit write direction required')
    return period


def prepare_context(reference, phase_fraction, *, context_id=None, phase_index=0,
                    rhs=hopfield_rhs, jacobian=hopfield_jacobian):
    """Sample fixed templates; preserve physical-vs-template and Z.f defects."""
    del jacobian  # Symmetric injection API; no tangent calculation in this stage.
    if isinstance(phase_fraction, (bool, np.bool_)) or not np.isfinite(phase_fraction) or not 0 <= phase_fraction < 1:
        raise ValueError('phase_fraction must lie in [0,1)')
    if type(phase_index) is not int or phase_index not in range(24):
        raise ValueError('phase_index must be an integer in 0..23')
    state, adjoint = reference_templates(reference)
    period = _positive(reference['period'], 'period'); epoch = float(phase_fraction)*period
    initial = _array(state.evaluate(np.array(epoch, dtype=np.float64)), 'initial', (3,)).copy()
    z = _array(adjoint.evaluate(np.array(epoch, dtype=np.float64)), 'adjoint', (3,)).copy()
    zt = _array(adjoint.evaluate(np.array(epoch+period, dtype=np.float64)), 'event adjoint', (3,)).copy()
    flow = _array(rhs(epoch, initial, reference['alpha'], reference['gamma']), 'physical flow', (3,)).copy()
    znorm = float(np.linalg.norm(z))
    if not np.isfinite(znorm) or znorm == 0: raise ValueError('Nonzero finite phase gradient required')
    unit = z/znorm; projection = np.eye(3)-np.outer(unit, unit)
    column = int(np.argmax(np.linalg.norm(projection, axis=0)))
    direction = projection[:, column]/np.linalg.norm(projection[:, column])
    result = dict(id=context_id or f"{reference['setting']}--p{phase_index:02d}",
        setting=reference['setting'], phase_index=phase_index,
        split='development' if phase_index < 8 else 'heldout',
        alpha=float(reference['alpha']), gamma=float(reference['gamma']), period=period,
        phase_fraction=float(phase_fraction), phase_epoch=epoch,
        initial=initial, phase_gradient=z, flow=flow, direction=direction,
        nominal_event_phase_gradient=zt, null_coordinate_index=column,
        normalization_residual=float(z@flow-1.), direction_phase_leakage=float(z@direction),
        template_velocity=state.evaluate(np.array(epoch, dtype=np.float64), derivative=True),
        provenance=dict(initial='saved periodic cubic-Hermite state template',
            adjoint='saved periodic cubic-Hermite adjoint, unrenormalized',
            event_adjoint='nominal phase_epoch+T, approximate periodic reference',
            projection_flow='physical RHS evaluated at independently integrated reference event state',
            reference_qualified=reference['qualified'], reference_status=reference['status']))
    multipliers = copy.deepcopy(reference.get('certificate', {}).get('transverse_multipliers', []))
    result['transverse_multipliers'] = multipliers
    dominant = max(multipliers, key=lambda x: x['modulus']) if multipliers else None
    result['rho'] = float(dominant['real']) if dominant is not None and dominant['imag'] == 0 else None
    result['rho_provenance'] = 'signed dominant real transverse multiplier of saved reference certificate; null if unavailable/nonreal'
    _context(result)
    return result


def _adaptive(rhs, initial, times, offset, spec, *, solver=solve_ivp):
    initial = _array(initial, 'adaptive initial').copy(); times = _array(times, 'adaptive times')
    if times.ndim != 1 or len(times) < 2 or times[0] != 0 or not np.all(np.diff(times) > 0):
        raise ValueError('Strictly increasing zero-origin adaptive times required')
    last_rhs = {}; start = time.monotonic()
    def tracked(local, value):
        last_rhs.update(time=float(local), global_time=float(offset+local),
                        values=np.array(value, copy=True), accepted_solver_state=False)
        return rhs(offset+local, value)
    try:
        solution = solver(tracked, (0., float(times[-1])), initial, t_eval=times,
            method=spec['kind'], rtol=spec['rtol'], atol=spec['atol'], max_step=spec['max_step'])
    except Exception as error:
        raise AcquisitionFailure('Adaptive solver exception', dict(status='failed', error=repr(error),
            offset=float(offset), requested_times=times.copy(), initial=initial, solver=dict(spec),
            last_rhs=last_rhs, wall_seconds=time.monotonic()-start)) from error
    result = dict(times=np.asarray(solution.t, dtype=np.float64).copy(),
        values=np.asarray(solution.y, dtype=np.float64).T.copy(), offset=float(offset),
        solver=dict(spec, success=bool(solution.success), message=str(solution.message),
            nfev=int(solution.nfev), njev=int(solution.njev), nlu=int(solution.nlu)),
        wall_seconds=time.monotonic()-start)
    if (not solution.success or not np.array_equal(result['times'], times)
        or result['values'].shape != (len(times), len(initial)) or not np.isfinite(result['values']).all()
        or not np.array_equal(result['values'][0], initial)):
        raise AcquisitionFailure('Incomplete or invalid adaptive trajectory',
            dict(result, status='failed', requested_times=times.copy(), initial=initial, last_rhs=last_rhs))
    return result


def _tangent_segment(context, initial, times, offset, spec, rhs, jacobian, solver):
    alpha, gamma = context['alpha'], context['gamma']
    def augmented(t, packed):
        x, phi = packed[:3], packed[3:].reshape(3, 3)
        return np.r_[rhs(t, x, alpha, gamma), (jacobian(t, x, alpha, gamma)@phi).ravel()]
    result = _adaptive(augmented, np.r_[initial, np.eye(3).ravel()], times, offset, spec, solver=solver)
    values = result.pop('values'); x = values[:, :3].copy(); phi = values[:, 3:].reshape(-1, 3, 3).copy()
    result.update(states=x, phi=phi)
    try:
        derivative = np.asarray([rhs(offset+t, v, alpha, gamma) for t, v in zip(times, x, strict=True)], dtype=np.float64)
        dphi = np.asarray([jacobian(offset+t, v, alpha, gamma)@p
            for t, v, p in zip(times, x, phi, strict=True)], dtype=np.float64)
        _array(derivative, 'state derivatives', x.shape); _array(dphi, 'tangent derivatives', phi.shape)
    except Exception as error:
        raise AcquisitionFailure('Tangent derivative reconstruction failed', dict(result, error=repr(error))) from error
    result.update(states=x, phi=phi, derivatives=derivative, phi_derivatives=dphi,
        initial=initial.copy(), initial_phi=np.eye(3), propagated_directly=True,
        full_tangent_reinitialized_at_segment_start=True)
    return result


def acquire_tangent(context, method='dop853', *, rhs=hopfield_rhs, jacobian=hopfield_jacobian, solver=solve_ivp):
    """Two direct 12D solves; post starts from pre endpoint with fresh I."""
    period = _context(context)
    if method not in TANGENT_METHODS: raise ValueError('Unknown tangent method')
    grid = build_grid(period); declared = TANGENT_METHODS[method]
    spec = dict(declared, id=method, max_step=period/declared['max_step_period_divisor'])
    packet = dict(context_id=context['id'], method=method, method_spec=spec,
        status='started', complete=False, period=period, event_time=period,
        provenance='Direct state and all three tangent columns; independent identity reset at T')
    stage = 'pre'
    try:
        packet['pre'] = _tangent_segment(context, context['initial'], grid['pre'], 0., spec, rhs, jacobian, solver)
        stage = 'post'
        packet['post'] = _tangent_segment(context, packet['pre']['states'][-1], grid['post'], period, spec, rhs, jacobian, solver)
        packet.update(status='complete', complete=True)
        return packet
    except Exception as error:
        packet.update(status='failed', complete=False, failed_stage=stage, error=repr(error))
        if hasattr(error, 'payload'): packet['partial_failure'] = error.payload
        raise AcquisitionFailure('Tangent acquisition failed', packet) from error


def seal_context(context, primary_tangent, *, rhs=hopfield_rhs):
    """Attach exact primary event reference and unmodified physical projectors."""
    period = _context(context)
    if (primary_tangent.get('complete') is not True or primary_tangent.get('status') != 'complete'
        or primary_tangent.get('context_id') != context['id'] or primary_tangent.get('method') != 'dop853'
        or primary_tangent.get('period') != period): raise ValueError('Matching completed primary tangent required')
    grid = build_grid(period)
    for key in ('pre', 'post'):
        segment = primary_tangent[key]
        if not np.array_equal(_array(segment['times'], 'reference times'), grid[key]): raise ValueError('Reference grid differs')
        _array(segment['states'], 'reference states', (len(grid[key]), 3))
        _array(segment['phi'], 'reference tangent', (len(grid[key]), 3, 3))
        if not np.array_equal(segment['phi'][0], np.eye(3)): raise ValueError('Fresh segment tangent identity required')
    if not np.array_equal(primary_tangent['pre']['states'][0], context['initial']): raise ValueError('Reference start differs')
    if not np.array_equal(primary_tangent['post']['states'][0], primary_tangent['pre']['states'][-1]): raise ValueError('Reference event continuity differs')
    result = copy.deepcopy(context); x = primary_tangent['pre']['states'][-1].copy()
    f = _array(rhs(period, x, context['alpha'], context['gamma']), 'event physical flow', (3,)).copy()
    z = context['nominal_event_phase_gradient'].copy(); q = np.outer(f, z); p = np.eye(3)-q
    _array(q, 'event Q', (3, 3)); _array(p, 'event P', (3, 3))
    result.update(reference_at_event=x, event_flow=f, event_phase_gradient=z, Q=q, P=p,
        event_normalization_residual=float(z@f-1.), event_projector_idempotence_residual=q@q-q,
        event_reference_method='dop853', event_reference_semantics='direct unforced state at T; nominal periodic adjoint at phase_epoch+T')
    return result


def _pulse(initial, direction, epsilon, sign):
    requested = float(sign)*epsilon*direction; final = initial+requested; realized = final-initial
    _array(final, 'pulse output', (3,))
    return dict(initial=final, requested_delta=requested, realized_delta=realized,
        realized_norm=float(np.linalg.norm(realized)), addition_rounding_residual=realized-requested)


def _trace_segment(context, initial, times, offset, method, native_fn, rhs, solver):
    period = context['period']; declared = METHODS[method]
    spec = dict(declared, id=method)
    if declared['kind'] == 'RK4':
        if native_fn is None: raise ValueError('Owned prebuilt native trace function required')
        dt = period/declared['steps_per_cycle']; spacing = period/SAMPLES_PER_CYCLE
        spec.update(dt=dt, sample_dt=spacing, steps=round(times[-1]/dt), samples=len(times))
        returned, states = integrate_trace(native_fn, context['alpha'], context['gamma'], initial,
            dt, duration=float(times[-1]), sample_dt=spacing)
        result = dict(times=returned, states=states, offset=float(offset), solver=spec)
    else:
        spec['max_step'] = period/declared['max_step_period_divisor']
        raw = _adaptive(lambda t, x: rhs(t, x, context['alpha'], context['gamma']), initial,
                        times, offset, spec, solver=solver)
        raw['states'] = raw.pop('values'); result = raw
    try:
        if (not np.array_equal(_array(result['times'], 'trace times'), times)
            or _array(result['states'], 'trace states', (len(times), 3)).shape != (len(times), 3)
            or not np.array_equal(result['states'][0], initial)): raise ValueError('Trace grid or initial state differs')
    except Exception as error:
        raise AcquisitionFailure('Trace validation failed', dict(result, error=repr(error))) from error
    return result


def acquire_trace(task, context, *, native_fn=None, rhs=hopfield_rhs, solver=solve_ivp):
    """Apply retain-then-write at exactly T; preserve every returned partial."""
    period = _context(context)
    for name in ('first_sign', 'second_sign'):
        if type(task.get(name)) is not int or task[name] not in (-1, 0, 1): raise ValueError('Signs must be integer -1,0,+1')
    if task.get('arm') not in ARMS or task.get('method') not in METHODS: raise ValueError('Unknown arm/method')
    epsilon = _positive(task['epsilon'], 'epsilon')
    if task.get('context_id') != context['id']: raise ValueError('Task/context identity differs')
    for name in ('reference_at_event', 'event_flow', 'event_phase_gradient'): _array(context[name], name, (3,))
    for name in ('Q', 'P'): _array(context[name], name, (3, 3))
    if not np.array_equal(context['Q'], np.outer(context['event_flow'], context['event_phase_gradient'])) or not np.array_equal(context['P'], np.eye(3)-context['Q']):
        raise ValueError('Sealed physical projectors differ')
    grid = build_grid(period); packet = dict(task=copy.deepcopy(task), context_id=context['id'],
        status='started', complete=False, period=period, event_time=period, duration=8*period,
        sample_dt=grid['sample_dt'], method_spec=dict(METHODS[task['method']]),
        alpha=context['alpha'], gamma=context['gamma']); stage = 'first_write'
    try:
        packet['first_pulse'] = _pulse(context['initial'], context['direction'], epsilon, task['first_sign'])
        stage = 'pre'
        packet['pre'] = _trace_segment(context, packet['first_pulse']['initial'], grid['pre'], 0., task['method'], native_fn, rhs, solver)
        stage = 'event'
        before = packet['pre']['states'][-1].copy(); base = context['reference_at_event']; displacement = before-base
        retain = {'identity':np.eye(3), 'timing':context['Q'], 'transverse':context['P']}[task['arm']]
        intended = retain@displacement
        # Identity must be bitwise identity, not a subtract/add round trip.
        after_retention = before.copy() if task['arm'] == 'identity' else base+intended
        pulse = _pulse(after_retention, context['direction'], epsilon, task['second_sign'])
        packet['event'] = dict(time=period, before=before, reference=base.copy(), displacement=displacement,
            retain_matrix=retain.copy(), intended_retained_displacement=intended,
            after_retention=after_retention, realized_retained_displacement=after_retention-base,
            retention_rounding_residual=after_retention-base-intended,
            replacement_delta=after_retention-before, replacement_norm=float(np.linalg.norm(after_retention-before)),
            second_pulse=pulse, after=pulse['initial'].copy(),
            phase_before=float(context['event_phase_gradient']@displacement),
            phase_after_retention=float(context['event_phase_gradient']@(after_retention-base)),
            Q_before=context['Q']@displacement, P_before=context['P']@displacement,
            Q_after_retention=context['Q']@(after_retention-base), P_after_retention=context['P']@(after_retention-base),
            operation_order='retention_then_second_write', event_clock='fixed elapsed T, no event detection or state feedback')
        stage = 'post'
        packet['post'] = _trace_segment(context, pulse['initial'], grid['post'], period, task['method'], native_fn, rhs, solver)
        packet.update(status='complete', complete=True)
        return packet
    except Exception as error:
        packet.update(status='failed', complete=False, failed_stage=stage, error=repr(error))
        if hasattr(error, 'payload'): packet['partial_failure'] = error.payload
        raise AcquisitionFailure('Two-write trace acquisition failed', packet) from error


def readout_samples(packet, delay_cycles):
    """Select exact 32 midpoint nodes without joining either side of the jump."""
    immediate = delay_cycles == 'immediate'
    if type(delay_cycles) is str and delay_cycles in ('k2', 'k4', 'k8'):
        delay_cycles = int(delay_cycles[1:])
    if not immediate and (type(delay_cycles) is not int or delay_cycles not in (1, 2, 4, 8)): raise ValueError('Known clock window required')
    if packet.get('complete') is not True or packet.get('status') != 'complete': raise ValueError('Complete packet required')
    key = 'pre' if immediate or delay_cycles == 1 else 'post'; segment = packet[key]
    start_cycle = 0 if key == 'pre' else delay_cycles-2
    index = ((2*np.arange(32)+1)*(SAMPLES_PER_CYCLE//512) if immediate else
             start_cycle*SAMPLES_PER_CYCLE + (2*np.arange(32)+1)*(SAMPLES_PER_CYCLE//64))
    state = _array(segment['states'], 'readout states')[index].copy()
    times = (packet['period']*(np.arange(32)+.5)/256 if immediate else
             packet['period']*(delay_cycles-1+(np.arange(32)+.5)/32))
    return dict(times=times,
                segment=key, indices=index, states=state, activations=np.tanh(state))
