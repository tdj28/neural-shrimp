"""EXP-016 fixed-observer evaluation, with no IO, fits, or integrations.

Only supplied saved packets are read. The primary observer is sealed from the
identity reference and is shared by every intervention and absent-write arm.
All probabilities are analytic Gaussian readout probabilities, not noisy trials
or claims about memory capacity. Failed prerequisites retain the full census.
"""
from __future__ import annotations

import copy
import math
from collections.abc import Mapping

import numpy as np

import two_write_acquisition as acquisition
import two_write_memory as memory

SETTINGS = ('core', 'failed-near-edge')
ARMS = ('identity', 'timing', 'transverse')
WINDOWS = ('immediate', 'k2', 'k4', 'k8')
METHODS = ('rk4-coarse', 'rk4-fine', 'dop853-independent')
EPSILON = SIGMA = 1e-4
STATE_LIMIT = 1e-6
ACTIVATION_LIMIT = 1e-7
MARGIN_LIMIT = .01
ACCURACY_LIMIT = .001
NORMALIZATION_LIMIT = 1e-7
PHASE_SEED = 2026091701
BOOTSTRAP_SEED = 20260917
BOOTSTRAP_RESAMPLES = 20_000


def _array(value, name, shape=None):
    if (type(value) is not np.ndarray or value.dtype != np.float64
            or (shape is not None and value.shape != shape)
            or not np.isfinite(value).all()):
        raise ValueError(f'{name} requires finite float64 array of shape {shape}')
    return value


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)) or not np.isfinite(value):
        raise ValueError(f'{name} requires finite real scalar')
    return float(value)


def _exact(left, right, name):
    if isinstance(right, np.ndarray):
        if not np.array_equal(_array(left, name, right.shape), right):
            raise ValueError(f'{name} differs')
    elif isinstance(right, Mapping):
        if not isinstance(left, Mapping) or set(left) != set(right):
            raise ValueError(f'{name} keys differ')
        for key in right:
            _exact(left[key], right[key], f'{name}.{key}')
    elif type(right) is int:
        if type(left) is not int or left != right:
            raise ValueError(f'{name} integer identity differs')
    elif isinstance(right, float):
        if _scalar(left, name) != right:
            raise ValueError(f'{name} scalar differs')
    elif type(left) is not type(right) or left != right:
        raise ValueError(f'{name} differs')


def phase_census():
    fractions = np.random.Generator(np.random.PCG64(PHASE_SEED)).random(24)
    return [dict(phase_index=i, split='development' if i < 8 else 'heldout', phase_fraction=float(f))
            for i, f in enumerate(fractions)]


def history_census(phases=None):
    """1,296 histories; common phase draws are declared, never outcome selected."""
    phases = phase_census() if phases is None else phases
    if not isinstance(phases, (list, tuple)) or len(phases) != 24:
        raise ValueError('Exactly 24 shared phase records required')
    for i, phase in enumerate(phases):
        if set(phase) != {'phase_index', 'split', 'phase_fraction'}:
            raise ValueError('Exact phase schema required')
        _exact(phase['phase_index'], i, 'phase_index')
        _exact(phase['split'], 'development' if i < 8 else 'heldout', 'split')
        if not 0 <= _scalar(phase['phase_fraction'], 'phase_fraction') < 1:
            raise ValueError('Phase fraction outside [0,1)')
    if len({float(p['phase_fraction']) for p in phases}) != 24:
        raise ValueError('Distinct phase fractions required')
    rows = []
    for setting in SETTINGS:
        for phase in phases:
            context_id = f"{setting}--p{phase['phase_index']:02d}"
            for arm in ARMS:
                for first in (-1, 0, 1):
                    for second in (-1, 0, 1):
                        rows.append(dict(id=f'{context_id}--{arm}--s1{first:+d}--s2{second:+d}',
                            context_id=context_id, setting=setting, **phase, arm=arm,
                            first_sign=first, second_sign=second, epsilon=EPSILON))
    return rows


def _history(history):
    if not isinstance(history, Mapping):
        raise ValueError('History mapping required')
    i = history.get('phase_index')
    if type(i) is not int or i not in range(24) or history.get('setting') not in SETTINGS:
        raise ValueError('Known setting and integer phase required')
    context_id = f"{history['setting']}--p{i:02d}"
    for key, value in dict(context_id=context_id, split='development' if i < 8 else 'heldout', epsilon=EPSILON).items():
        _exact(history.get(key), value, key)
    if not 0 <= _scalar(history.get('phase_fraction'), 'phase_fraction') < 1:
        raise ValueError('Phase fraction outside [0,1)')
    if history.get('arm') not in ARMS:
        raise ValueError('Known arm required')
    for key in ('first_sign', 'second_sign'):
        if type(history.get(key)) is not int or history[key] not in (-1, 0, 1):
            raise ValueError('Integer sign in -1,0,+1 required')
    wanted = f"{context_id}--{history['arm']}--s1{history['first_sign']:+d}--s2{history['second_sign']:+d}"
    _exact(history.get('id'), wanted, 'history id')
    if set(history) != {'id', 'context_id', 'setting', 'phase_index', 'split', 'phase_fraction', 'arm', 'first_sign', 'second_sign', 'epsilon'}:
        raise ValueError('Exact physical history schema required')
    return history


def methods_for_history(history):
    _history(history)
    return METHODS if history['phase_index'] in (0, 8) else METHODS[:2]


def task_census(histories=None):
    histories = history_census() if histories is None else histories
    return [dict(history, id=history['id']+'--'+method, history_id=history['id'], method=method)
            for history in histories for method in methods_for_history(history)]


def comparison_census(histories=None):
    histories = history_census() if histories is None else histories
    result = []
    for history in histories:
        pairs = [(METHODS[0], METHODS[1])]
        if len(methods_for_history(history)) == 3:
            pairs.append((METHODS[1], METHODS[2]))
        for left, right in pairs:
            result.append(dict(id=history['id']+'--'+left+'--vs--'+right,
                history_id=history['id'], context_id=history['context_id'],
                left_task_id=history['id']+'--'+left, right_task_id=history['id']+'--'+right,
                left_method=left, right_method=right))
    return result


def _task(task):
    if not isinstance(task, Mapping) or task.get('method') not in METHODS:
        raise ValueError('Known task/method required')
    history = {k: v for k, v in task.items() if k not in ('method', 'history_id')}
    history['id'] = task.get('history_id')
    _history(history)
    if task['method'] not in methods_for_history(history):
        raise ValueError('Unplanned independent method')
    _exact(task['id'], history['id']+'--'+task['method'], 'task id')
    return history


def _context(context):
    if not isinstance(context, Mapping):
        raise ValueError('Context mapping required')
    i = context.get('phase_index')
    if type(i) is not int or i not in range(24) or context.get('setting') not in SETTINGS:
        raise ValueError('Known context setting/phase required')
    _exact(context.get('id'), f"{context['setting']}--p{i:02d}", 'context id')
    _exact(context.get('split'), 'development' if i < 8 else 'heldout', 'context split')
    if not 0 <= _scalar(context.get('phase_fraction'), 'phase_fraction') < 1:
        raise ValueError('Context phase fraction outside [0,1)')
    period = _scalar(context['period'], 'period')
    if period <= 0:
        raise ValueError('Positive period required')
    for key in ('initial', 'flow', 'phase_gradient', 'direction', 'nominal_event_phase_gradient'):
        _array(context[key], key, (3,))
    if abs(float(np.linalg.norm(context['direction']))-1) > 1e-12:
        raise ValueError('Unit write direction required')
    _exact(context['phase_epoch'], context['phase_fraction']*period, 'phase_epoch')
    for key in ('alpha', 'gamma'):
        _scalar(context[key], key)
    return period


def _solver(segment, method, period, *, tangent=False):
    declared = acquisition.TANGENT_METHODS[method] if tangent else acquisition.METHODS[method]
    spec = segment['solver']
    for key, value in dict(declared, id=method).items():
        _exact(spec.get(key), value, f'solver.{key}')
    if declared['kind'] == 'RK4':
        dt = period/declared['steps_per_cycle']
        for key, value in dict(dt=dt, sample_dt=period/acquisition.SAMPLES_PER_CYCLE,
                steps=int(round(float(segment['times'][-1])/dt)), samples=len(segment['times'])).items():
            _exact(spec.get(key), value, f'solver.{key}')
    else:
        _exact(spec.get('max_step'), period/declared['max_step_period_divisor'], 'max_step')
        if spec.get('success') is not True or type(spec.get('nfev')) is not int or spec['nfev'] < 0:
            raise ValueError('Successful adaptive solver and integer nfev required')


def _segments(packet, period, method, *, tangent=False):
    if packet.get('complete') is not True or packet.get('status') != 'complete':
        raise ValueError('Complete saved trajectory required')
    _exact(packet.get('period'), period, 'period')
    _exact(packet.get('event_time'), period, 'event_time')
    grid = acquisition.build_grid(period)
    for key, offset in (('pre', 0.), ('post', period)):
        segment = packet[key]
        _exact(segment['times'], grid[key], f'{key}.times')
        _exact(segment['offset'], offset, f'{key}.offset')
        shape = (len(grid[key]), 3)
        _array(segment['states'], key+'.states', shape)
        if tangent:
            _array(segment['derivatives'], key+'.derivatives', shape)
        _solver(segment, method, period, tangent=tangent)
        if tangent:
            _array(segment['phi'], key+'.phi', (len(grid[key]), 3, 3))
            _array(segment['phi_derivatives'], key+'.phi_derivatives', (len(grid[key]), 3, 3))
            for field in ('propagated_directly', 'full_tangent_reinitialized_at_segment_start'):
                if segment.get(field) is not True:
                    raise ValueError('Direct nine-entry propagation required')
            _exact(segment['phi'][0], np.eye(3), key+'.phi initial')
            _exact(segment['initial_phi'], np.eye(3), key+'.initial_phi')
            _exact(segment['initial'], segment['states'][0], key+'.initial')
    return grid


def _tangent(packet, context, method):
    period = _context(context)
    _exact(packet.get('context_id'), context['id'], 'tangent context')
    _exact(packet.get('method'), method, 'tangent method')
    declared = dict(acquisition.TANGENT_METHODS[method], id=method, max_step=period/acquisition.TANGENT_METHODS[method]['max_step_period_divisor'])
    _exact(packet.get('method_spec'), declared, 'tangent method_spec')
    _segments(packet, period, method, tangent=True)
    _exact(packet['pre']['states'][0], context['initial'], 'reference initial')
    _exact(packet['post']['states'][0], packet['pre']['states'][-1], 'direct reference event')


def _difference(left, right):
    difference = _array(left, 'left') - _array(right, 'right', left.shape)
    if not np.isfinite(difference).all():
        raise ValueError('Nonfinite numerical difference')
    location = np.unravel_index(int(np.argmax(np.abs(difference))), difference.shape)
    return dict(max_abs=float(np.max(np.abs(difference))),
        rms=float(np.sqrt(np.mean(difference*difference))),
        location=[int(x) for x in location], signed_at_max=float(difference[location]))


def _normal_cdf(x):
    return .5*math.erfc(-float(x)/math.sqrt(2.))


def _project(h, vector, first_sign):
    h = _array(h, 'observer', (96,))
    vector = _array(vector, 'activation displacement', (96,))
    noise = SIGMA*float(np.linalg.norm(h))
    if noise <= 0 or not np.isfinite(noise):
        raise ValueError('Nonzero finite observer required')
    q = float(h@vector); z = q/noise
    if not np.isfinite([q, z]).all():
        raise ValueError('Nonfinite readout')
    return dict(projected_mean=q, noise_sd=noise, standardized_mean=z,
        signed_margin=None if first_sign == 0 else first_sign*z,
        probability_correct=.5 if first_sign == 0 else _normal_cdf(first_sign*z),
        balanced_pseudo_label=first_sign == 0)


def _predict(context, tangent, window):
    samples = acquisition.readout_samples(tangent, window)
    segment = tangent[samples['segment']]
    phi = segment['phi'][samples['indices']]
    direction = EPSILON*context['direction']
    derivative = 1.-samples['activations']**2
    result = {}
    for arm in ARMS:
        if window == 'immediate':
            first = phi@direction
            second = np.zeros_like(first)
            residual = np.zeros_like(first)
        else:
            H = {'identity': np.eye(3), 'timing': context['Q'], 'transverse': context['P']}[arm]
            composed = memory.compose_two_write(tangent['pre']['phi'][-1], phi, H, direction, direction)
            first, second = composed['first_response'], composed['second_response']
            residual = composed['composition_residual']
        result[arm] = dict(A=(derivative*first).ravel(), B=(derivative*second).ravel(),
            first_state=first, second_state=second,
            composition_residual=residual,
            composition_arithmetic_pass=bool(np.allclose(first+second, first+second-residual, rtol=1e-12, atol=1e-14)))
    return samples, result


def seal_reference(context, tangent_primary, tangent_alternate):
    """Build all observers from saved references before any pulse is scored.

    The alternate uses the primary's sealed Q/P, so it tests the same physical
    intervention. Direct saved midpoint nodes are used; no new solve or
    interpolation is performed. Exceptions become explicit failed prerequisites.
    """
    result = dict(context_id=context.get('id'), context=copy.deepcopy(context), complete=False,
        qualified=False, status='failed', gates={}, windows={}, reference_comparison=None,
        epsilon=EPSILON, sigma=SIGMA)
    try:
        period = _context(context)
        _tangent(tangent_primary, context, 'dop853')
        _tangent(tangent_alternate, context, 'rk45')
        ctx = copy.deepcopy(context)
        base = tangent_primary['pre']['states'][-1].copy()
        flow = tangent_primary['pre']['derivatives'][-1].copy()
        gradient = ctx['nominal_event_phase_gradient']
        Q = np.outer(flow, gradient)
        for key, expected in dict(reference_at_event=base,event_flow=flow,event_phase_gradient=gradient,Q=Q,P=np.eye(3)-Q).items():
            if key in context:
                _exact(context[key], expected, 'supplied sealed '+key)
        ctx.update(reference_at_event=base, event_flow=flow,
            event_phase_gradient=gradient.copy(), Q=Q, P=np.eye(3)-Q,
            event_normalization_residual=float(gradient@flow-1.),
            event_projector_idempotence_residual=Q@Q-Q,
            event_reference_method='dop853',
            event_reference_semantics='direct unforced state at T; nominal periodic adjoint at phase_epoch+T')
        result['context'] = ctx
        result['local_amplification'] = memory.oblique_components(flow, gradient,
            tangent_primary['pre']['phi'][-1]@(EPSILON*ctx['direction']))
        normalization_max = max(abs(float(ctx['phase_gradient']@ctx['flow']-1.)),
            abs(float(gradient@flow-1.)),
            abs(float(gradient@tangent_alternate['pre']['derivatives'][-1]-1.)))
        gates = dict(original_reference_qualified=ctx.get('provenance', {}).get('reference_qualified') is True,
            normalization=normalization_max <= NORMALIZATION_LIMIT,
            primary_observers=False, reference_states=True, reference_activation=True,
            scaled_full_tangent=True, scaled_activation_tangent=True,
            composed_state_predictions=True, composed_activation_predictions=True,
            prediction_margins=True, prediction_accuracies=True)
        comparison = dict(normalization_max=normalization_max, segments={}, windows={})
        result.update(gates=gates, reference_comparison=comparison)
        for key in ('pre', 'post'):
            p, a = tangent_primary[key], tangent_alternate[key]
            state = _difference(p['states'], a['states'])
            activation = _difference(np.tanh(p['states']), np.tanh(a['states']))
            tangent = _difference(EPSILON*p['phi'], EPSILON*a['phi'])
            p_activation_phi = (1.-np.tanh(p['states'])**2)[:, :, None]*p['phi']
            a_activation_phi = (1.-np.tanh(a['states'])**2)[:, :, None]*a['phi']
            activation_tangent = _difference(EPSILON*p_activation_phi, EPSILON*a_activation_phi)
            comparison['segments'][key] = dict(state=state, activation=activation,
                raw_phi_all_nine=_difference(p['phi'], a['phi']),
                pulse_scaled_phi_all_nine=tangent, pulse_scaled_activation_phi_all_nine=activation_tangent)
            gates['reference_states'] &= state['max_abs'] <= STATE_LIMIT
            gates['reference_activation'] &= activation['max_abs'] <= ACTIVATION_LIMIT
            gates['scaled_full_tangent'] &= tangent['max_abs'] <= STATE_LIMIT
            gates['scaled_activation_tangent'] &= activation_tangent['max_abs'] <= ACTIVATION_LIMIT
        for window in WINDOWS:
            samples, predictions = _predict(ctx, tangent_primary, window)
            _, alternate = _predict(ctx, tangent_alternate, window)
            observer = memory.linear_observer(predictions['identity']['A'], predictions['identity']['B'], SIGMA)
            h = observer['h']
            packet = dict(times=samples['times'], segment=samples['segment'], indices=samples['indices'],
                baseline=samples['activations'], baseline_states=samples['states'], h=h,
                observer=observer, linear_predictions=predictions, arm_observers={},
                one_mode_prediction=None)
            velocity = tangent_primary[samples['segment']]['derivatives'][samples['indices']]
            G = ((1.-samples['activations']**2)*velocity).ravel()
            G_norm = float(np.linalg.norm(G))
            packet['derivative_observer'] = (dict(available=True,h=G/G_norm,noise_sd=SIGMA,
                scope='direct-reference physical activation velocity; fixed positive-flow orientation; not the old periodic-Hermite clock')
                if np.isfinite(G_norm) and G_norm > 0 else dict(available=False,h=None,reason='Zero/unrepresentable activation velocity'))
            for arm in ARMS:
                try:
                    secondary = memory.linear_observer(predictions[arm]['A'], predictions[arm]['B'], SIGMA)
                    packet['arm_observers'][arm] = dict(available=True, observer=secondary)
                except ValueError as error:
                    packet['arm_observers'][arm] = dict(available=False, error=str(error), observer=None)
                predictions[arm]['primary_accuracy'] = memory.balanced_accuracy(h,
                    memory.linear_means(predictions[arm]['A'], predictions[arm]['B']), SIGMA)
            if window == 'immediate':
                one = dict(available=True, A=predictions['identity']['A'].copy(), B=np.zeros(96),
                    scope='matched immediate calibration; not a one-mode temporal prediction')
            elif ctx.get('rho') is None:
                one = dict(available=False, reason='Signed dominant real transverse multiplier unavailable')
            else:
                one = dict(memory.periodic_single_mode(predictions['identity']['B'], _scalar(ctx['rho'], 'rho')), available=True)
            if one['available']:
                one['primary_accuracy'] = memory.balanced_accuracy(h, memory.linear_means(one['A'], one['B']), SIGMA)
            packet['one_mode_prediction'] = one
            wc = {}
            for arm in ARMS:
                first = abs(float(h@(predictions[arm]['A']-alternate[arm]['A'])))/observer['noise_sd']
                second = abs(float(h@(predictions[arm]['B']-alternate[arm]['B'])))/observer['noise_sd']
                pmeans = memory.linear_means(predictions[arm]['A'], predictions[arm]['B'])
                ameans = memory.linear_means(alternate[arm]['A'], alternate[arm]['B'])
                pa = memory.balanced_accuracy(h, pmeans, SIGMA)
                aa = memory.balanced_accuracy(h, ameans, SIGMA)
                margin = _difference(pa['signed_standardized_means'], aa['signed_standardized_means'])
                accuracy = _difference(pa['probability_correct'], aa['probability_correct'])
                wc[arm] = dict(first_projection_noise_difference=first, second_projection_noise_difference=second,
                    signed_margins=margin, probability_correct=accuracy,
                    first_activation=_difference(predictions[arm]['A'], alternate[arm]['A']),
                    second_activation=_difference(predictions[arm]['B'], alternate[arm]['B']))
                wc[arm]['first_state'] = _difference(predictions[arm]['first_state'], alternate[arm]['first_state'])
                wc[arm]['second_state'] = _difference(predictions[arm]['second_state'], alternate[arm]['second_state'])
                gates['composed_state_predictions'] &= max(wc[arm]['first_state']['max_abs'], wc[arm]['second_state']['max_abs']) <= STATE_LIMIT
                gates['composed_activation_predictions'] &= max(wc[arm]['first_activation']['max_abs'], wc[arm]['second_activation']['max_abs']) <= ACTIVATION_LIMIT
                gates['prediction_margins'] &= max(first, second, margin['max_abs']) <= MARGIN_LIMIT
                gates['prediction_accuracies'] &= accuracy['max_abs'] <= ACCURACY_LIMIT
            comparison['windows'][window] = wc
            result['windows'][window] = packet
        gates['primary_observers'] = True
        result.update(complete=True, status='complete', gates={k: bool(v) for k,v in gates.items()},
            reference_comparison=comparison, qualified=bool(all(gates.values())),
            arithmetic_qualified=all(p['composition_arithmetic_pass'] for w in result['windows'].values()
                for p in w['linear_predictions'].values()))
    except (ValueError, KeyError, TypeError, IndexError, FloatingPointError, OverflowError) as error:
        result.update(error=f'{type(error).__name__}: {error}', qualified=False)
    return result


def contract():
    """Small prospective plan payload; no state or outcomes are accessed."""
    return dict(settings=list(SETTINGS), arms=list(ARMS), windows=list(WINDOWS), methods=list(METHODS),
        phases=24, development_phases=8, heldout_phases=16, phase_seed=PHASE_SEED,
        epsilon=EPSILON, sigma=SIGMA, observation_samples=32, observation_coordinates=3,
        thresholds=dict(state_max=STATE_LIMIT, activation_max=ACTIVATION_LIMIT,
            standardized_margin_max=MARGIN_LIMIT, probability_max=ACCURACY_LIMIT,
            zero_zero_standardized_drift_max=MARGIN_LIMIT, normalization_max=NORMALIZATION_LIMIT),
        expected=dict(contexts=48, reference_jobs=96, histories=1296, trace_tasks=2700,
            all_jobs=2796, comparisons=1404, coarse_fine_comparisons=1296, fine_independent_comparisons=108),
        bootstrap=dict(seed=BOOTSTRAP_SEED, resamples=BOOTSTRAP_RESAMPLES, paired_phase_blocks=16,
            interval='central 95%, descriptive conditional on two saved cycles'),
        primary='heldout identity k4, second present, near-edge minus core percentage points',
        arithmetic=dict(rtol=1e-12, atol=1e-14, reported_separately=True),
        scope='fixed phase-informed linear Gaussian observer, no memory-capacity or erasure claim')


def _bound_seal(task, sealed):
    context = sealed['context']
    _context(context)
    _exact(sealed.get('context_id'), task['context_id'], 'sealed context')
    for key in ('setting', 'phase_index', 'phase_fraction', 'split'):
        _exact(context[key], task[key], 'context.'+key)
    if sealed.get('complete') is not True or set(sealed.get('windows', {})) != set(WINDOWS):
        raise ValueError('Complete sealed reference windows required')
    _exact(sealed.get('epsilon'), EPSILON, 'sealed epsilon')
    _exact(sealed.get('sigma'), SIGMA, 'sealed sigma')
    for key in ('reference_at_event', 'event_flow', 'event_phase_gradient'):
        _array(context[key], key, (3,))
    _exact(context['Q'], np.outer(context['event_flow'], context['event_phase_gradient']), 'Q')
    _exact(context['P'], np.eye(3)-context['Q'], 'P')
    return context


def _pulse_binding(packet, original, direction, sign, name):
    requested = sign*EPSILON*direction
    initial = original+requested
    _exact(packet['initial'], initial, name+'.initial')
    _exact(packet['requested_delta'], requested, name+'.requested_delta')
    _exact(packet['realized_delta'], initial-original, name+'.realized_delta')
    _exact(packet['realized_norm'], float(np.linalg.norm(initial-original)), name+'.realized_norm')
    _exact(packet['addition_rounding_residual'], initial-original-requested, name+'.rounding')
    return initial


def _trace(task, packet, sealed):
    _task(task)
    context = _bound_seal(task, sealed)
    if not isinstance(packet, Mapping):
        raise ValueError('Saved trace packet unavailable')
    _exact(packet.get('task'), task, 'trace.task')
    _exact(packet.get('context_id'), context['id'], 'trace context')
    for key, value in dict(alpha=context['alpha'], gamma=context['gamma'], duration=8*context['period'],
            sample_dt=context['period']/acquisition.SAMPLES_PER_CYCLE).items():
        _exact(packet.get(key), value, 'trace.'+key)
    _segments(packet, context['period'], task['method'])
    _exact(packet.get('method_spec'), acquisition.METHODS[task['method']], 'trace method_spec')
    first = _pulse_binding(packet['first_pulse'], context['initial'], context['direction'], task['first_sign'], 'first_pulse')
    _exact(packet['pre']['states'][0], first, 'pre initial')
    event = packet['event']; before = packet['pre']['states'][-1]; base = context['reference_at_event']
    H = {'identity':np.eye(3), 'timing':context['Q'], 'transverse':context['P']}[task['arm']]
    retained = before.copy() if task['arm'] == 'identity' else base+H@(before-base)
    expected = dict(time=context['period'], before=before, reference=base,
        displacement=before-base, retain_matrix=H, intended_retained_displacement=H@(before-base),
        after_retention=retained, realized_retained_displacement=retained-base,
        retention_rounding_residual=retained-base-H@(before-base), replacement_delta=retained-before,
        replacement_norm=float(np.linalg.norm(retained-before)),
        phase_before=float(context['event_phase_gradient']@(before-base)),
        phase_after_retention=float(context['event_phase_gradient']@(retained-base)),
        Q_before=context['Q']@(before-base), P_before=context['P']@(before-base),
        Q_after_retention=context['Q']@(retained-base), P_after_retention=context['P']@(retained-base),
        operation_order='retention_then_second_write')
    for key, value in expected.items():
        _exact(event.get(key), value, 'event.'+key)
    second = _pulse_binding(event['second_pulse'], retained, context['direction'], task['second_sign'], 'second_pulse')
    _exact(event['after'], second, 'event.after')
    _exact(packet['post']['states'][0], second, 'post initial')
    return context


def score_trace(task, packet, sealed):
    """Score all four fixed clocks; absent-first labels are balanced at chance.

    The same reference baseline and identity h are used for every arm/sign.
    Only the zero/zero histories are drift controls. Secondary observers have
    separate availability and never rescue an undefined primary observer.
    """
    row = dict(task=copy.deepcopy(task), task_id=task.get('id'), history_id=task.get('history_id'),
        context_id=task.get('context_id'), complete=False, qualified=False, status='failed',
        windows={}, gates={})
    try:
        context = _trace(task, packet, sealed)
        drift_pass = True
        for window in WINDOWS:
            planned = sealed['windows'][window]
            samples = acquisition.readout_samples(packet, window)
            _exact(planned['times'], samples['times'], 'sealed clock times')
            _exact(planned['segment'], samples['segment'], 'sealed clock segment')
            if not np.array_equal(planned['indices'], samples['indices']):
                raise ValueError('Sealed midpoint indices differ')
            baseline = _array(planned['baseline'], 'shared baseline', (32,3))
            displacement = (samples['activations']-baseline).ravel()
            primary = _project(planned['h'], displacement, task['first_sign'])
            secondary_spec = planned['arm_observers'][task['arm']]
            secondary = dict(available=False, score=None)
            if secondary_spec.get('available') is True:
                secondary = dict(available=True, score=_project(secondary_spec['observer']['h'], displacement, task['first_sign']))
            zero_zero = task['first_sign'] == task['second_sign'] == 0
            drift = None if not zero_zero else abs(primary['standardized_mean']) <= MARGIN_LIMIT
            drift_pass &= drift is not False
            prediction = planned['linear_predictions'][task['arm']]
            expected = task['first_sign']*prediction['A']+task['second_sign']*prediction['B']
            linear = _project(planned['h'], expected, task['first_sign'])
            one = planned['one_mode_prediction'] if task['arm'] == 'identity' else None
            one_score = None if one is None or not one.get('available') else _project(planned['h'],
                task['first_sign']*one['A']+task['second_sign']*one['B'], task['first_sign'])
            derivative = dict(available=False,score=None,linear_prediction=None)
            if planned['derivative_observer']['available']:
                dh = planned['derivative_observer']['h']
                derivative = dict(available=True,score=_project(dh,displacement,task['first_sign']),
                    linear_prediction=_project(dh,expected,task['first_sign']))
            row['windows'][window] = dict(times=samples['times'], activation_displacement=displacement,
                derivative=derivative,
                primary=primary, secondary=secondary, linear_prediction=linear,
                one_mode_prediction=one_score, zero_zero_drift_pass=drift,
                primary_minus_linear_probability=primary['probability_correct']-linear['probability_correct'],
                primary_minus_linear_standardized_mean=primary['standardized_mean']-linear['standardized_mean'])
        row.update(complete=True, status='complete',
            gates=dict(reference_qualified=sealed.get('qualified') is True, zero_zero_drift=bool(drift_pass)),
            qualified=sealed.get('qualified') is True and bool(drift_pass),
            secondary_observer_available=all(w['secondary']['available'] for w in row['windows'].values()))
    except (ValueError, KeyError, TypeError, IndexError, FloatingPointError, OverflowError) as error:
        row['error'] = f'{type(error).__name__}: {error}'
    return row


def compare_traces(history, left_packet, right_packet, sealed, *, left_score=None, right_score=None, methods=None):
    """One required native/native or native/independent comparison, retained on failure.

    For absent-first controls the continuous standardized mean is compared
    directly; multiplying it by the absent label zero would hide numerical error.
    """
    _history(history)
    left_method = left_packet.get('task', {}).get('method') if isinstance(left_packet, Mapping) else None
    right_method = right_packet.get('task', {}).get('method') if isinstance(right_packet, Mapping) else None
    if left_method is None and isinstance(left_score, Mapping):
        left_method = left_score.get('task', {}).get('method')
    if right_method is None and isinstance(right_score, Mapping):
        right_method = right_score.get('task', {}).get('method')
    if methods is not None:
        if not isinstance(methods, (tuple, list)) or len(methods) != 2:
            raise ValueError('Exactly two ordered method names required')
        if left_method is not None and left_method != methods[0] or right_method is not None and right_method != methods[1]:
            raise ValueError('Explicit methods disagree with saved identities')
        left_method, right_method = methods
    # Missing packets retain the explicitly declared pair identity.
    specs = comparison_census([history])
    spec = next((x for x in specs if x['left_method'] == left_method and x['right_method'] == right_method), None)
    if spec is None:
        if left_method is None and right_method is None and len(specs) == 1:
            spec = specs[0]
        else:
            raise ValueError('Comparison methods unavailable or not a required ordered pair; supply failed score task identities')
    result = dict(spec, complete=False, qualified=False, status='failed', segments={}, windows={}, gates={},
        secondary=dict(available=False, qualified=False, windows={}),
        derivative=dict(available=False,qualified=False,windows={}))
    try:
        tasks = {t['method']:t for t in task_census([history])}
        _trace(tasks[spec['left_method']], left_packet, sealed)
        _trace(tasks[spec['right_method']], right_packet, sealed)
        left_score = score_trace(tasks[spec['left_method']], left_packet, sealed) if left_score is None else left_score
        right_score = score_trace(tasks[spec['right_method']], right_packet, sealed) if right_score is None else right_score
        for score, task in ((left_score, tasks[spec['left_method']]), (right_score, tasks[spec['right_method']])):
            _exact(score['task'], task, 'comparison score task')
            if score.get('complete') is not True or set(score.get('windows', {})) != set(WINDOWS):
                raise ValueError('Complete bound scores required')
        gates = dict(reference_qualified=sealed.get('qualified') is True, states=True, activation=True,
            margins=True, accuracy=True, zero_zero_drift=left_score['gates']['zero_zero_drift'] and right_score['gates']['zero_zero_drift'])
        for key in ('pre', 'post'):
            ls, rs = left_packet[key]['states'], right_packet[key]['states']
            state, activation = _difference(ls, rs), _difference(np.tanh(ls), np.tanh(rs))
            result['segments'][key] = dict(state=state, activation=activation)
            gates['states'] &= state['max_abs'] <= STATE_LIMIT
            gates['activation'] &= activation['max_abs'] <= ACTIVATION_LIMIT
        secondary_available, secondary_qualified = True, True
        derivative_available, derivative_qualified = True, True
        for window in WINDOWS:
            left, right = left_score['windows'][window], right_score['windows'][window]
            a, b = left['primary'], right['primary']
            margin = abs(_scalar(a['standardized_mean'], 'left margin')-_scalar(b['standardized_mean'], 'right margin'))
            accuracy = abs(_scalar(a['probability_correct'], 'left probability')-_scalar(b['probability_correct'], 'right probability'))
            result['windows'][window] = dict(standardized_margin_difference=margin, probability_difference=accuracy,
                left_standardized_mean=a['standardized_mean'], right_standardized_mean=b['standardized_mean'])
            gates['margins'] &= margin <= MARGIN_LIMIT
            gates['accuracy'] &= accuracy <= ACCURACY_LIMIT
            if left['secondary']['available'] and right['secondary']['available']:
                a, b = left['secondary']['score'], right['secondary']['score']
                sm = abs(_scalar(a['standardized_mean'], 'left secondary')-_scalar(b['standardized_mean'], 'right secondary'))
                sa = abs(_scalar(a['probability_correct'], 'left secondary probability')-_scalar(b['probability_correct'], 'right secondary probability'))
                result['secondary']['windows'][window] = dict(available=True, standardized_margin_difference=sm,
                    probability_difference=sa, qualified=sm <= MARGIN_LIMIT and sa <= ACCURACY_LIMIT)
                secondary_qualified &= sm <= MARGIN_LIMIT and sa <= ACCURACY_LIMIT
            else:
                secondary_available = False
                result['secondary']['windows'][window] = dict(available=False, qualified=False)
            if left['derivative']['available'] and right['derivative']['available']:
                a, b = left['derivative']['score'], right['derivative']['score']
                dm = abs(_scalar(a['standardized_mean'], 'left derivative')-_scalar(b['standardized_mean'], 'right derivative'))
                da = abs(_scalar(a['probability_correct'], 'left derivative probability')-_scalar(b['probability_correct'], 'right derivative probability'))
                result['derivative']['windows'][window] = dict(available=True,standardized_margin_difference=dm,
                    probability_difference=da,qualified=dm <= MARGIN_LIMIT and da <= ACCURACY_LIMIT)
                derivative_qualified &= dm <= MARGIN_LIMIT and da <= ACCURACY_LIMIT
            else:
                derivative_available = False
                result['derivative']['windows'][window] = dict(available=False,qualified=False)
        result.update(complete=True, status='complete', gates={k:bool(v) for k,v in gates.items()}, qualified=bool(all(gates.values())))
        result['derivative'].update(available=derivative_available,
            qualified=bool(derivative_available and derivative_qualified and all(gates[k] for k in ('reference_qualified','states','activation','zero_zero_drift'))))
        result['secondary'].update(available=secondary_available,
            qualified=bool(secondary_available and secondary_qualified and all(gates[k] for k in ('reference_qualified','states','activation','zero_zero_drift'))))
    except (ValueError, KeyError, TypeError, IndexError, FloatingPointError, OverflowError) as error:
        result['error'] = f'{type(error).__name__}: {error}'
    return result


def paired_phase_interval(core, edge):
    """Exactly 16 paired phases, fixed 20,000 PCG64 bootstrap draws; no missingness."""
    core, edge = _array(core, 'core', (16,)), _array(edge, 'edge', (16,))
    if np.any((core < 0)|(core > 1)) or np.any((edge < 0)|(edge > 1)):
        raise ValueError('Accuracy must lie in [0,1]')
    delta = 100.*(edge-core)
    draws = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(0,16,size=(BOOTSTRAP_RESAMPLES,16))
    values = np.mean(delta[draws], axis=1)
    return dict(estimate_percentage_points=float(np.mean(delta)),
        central_95_percentage_points=np.quantile(values,[.025,.975],method='linear').tolist(),
        phase_differences_percentage_points=delta, seed=BOOTSTRAP_SEED,
        resamples=BOOTSTRAP_RESAMPLES, paired_phases=16,
        scope='descriptive paired phase resampling conditional on two existing cycles')


def accuracy_failure_bounds(values):
    """A fixed 16-phase denominator; each missing/unqualified phase contributes [0,1]."""
    if len(values) != 16:
        raise ValueError('Exactly 16 planned heldout phases required')
    known = []
    for value in values:
        if value is None:
            continue
        value = _scalar(value, 'accuracy')
        if not 0 <= value <= 1:
            raise ValueError('Accuracy outside [0,1]')
        known.append(value)
    missing = 16-len(known)
    return dict(lower=math.fsum(known)/16, upper=(math.fsum(known)+missing)/16,
        known=len(known), unknown=missing, denominator=16)


def _complete_history_census(histories):
    if not isinstance(histories, (list, tuple)) or len(histories) != 1296:
        raise ValueError('Exactly 1296 planned histories required, including failures')
    for history in histories:
        _history(history)
    phases = []
    for i in range(24):
        members = [h for h in histories if h['phase_index'] == i]
        fractions = {h['phase_fraction'] for h in members}
        if len(fractions) != 1:
            raise ValueError('Phase pairing differs across settings')
        phases.append(dict(phase_index=i, split='development' if i<8 else 'heldout', phase_fraction=fractions.pop()))
    expected = history_census(phases)
    if len({h['id'] for h in histories}) != 1296:
        raise ValueError('Duplicate planned history')
    mapping = {h['id']:h for h in histories}
    for h in expected:
        _exact(mapping.get(h['id']), h, 'history census')
    return expected


def summarize(histories, scores_by_task_id, comparisons, prerequisites_by_context):
    """Streaming aggregation from compact evidence, with every planned row retained.

    Integration completeness, numerical qualification and descriptive accuracies
    remain separate. Missing methods, comparison rows and reference prerequisites
    never reduce the denominator or create a favorable replacement.
    """
    histories = _complete_history_census(histories)
    tasks = task_census(histories); specs = comparison_census(histories)
    expected_ids = {t['id'] for t in tasks}
    context_ids = sorted({h['context_id'] for h in histories})
    if not isinstance(scores_by_task_id, Mapping) or set(scores_by_task_id)-expected_ids:
        raise ValueError('Unexpected task score identity')
    if not isinstance(prerequisites_by_context, Mapping) or set(prerequisites_by_context)-set(context_ids):
        raise ValueError('Unexpected reference context identity')
    comparison_rows = list(comparisons.values()) if isinstance(comparisons, Mapping) else list(comparisons)
    if len({r.get('id') for r in comparison_rows}) != len(comparison_rows):
        raise ValueError('Duplicate comparison identity')
    actual_comparisons = {r['id']:r for r in comparison_rows}
    if set(actual_comparisons)-{r['id'] for r in specs}:
        raise ValueError('Unexpected comparison identity')
    retained_scores = []
    score_map = {}
    for task in tasks:
        score = scores_by_task_id.get(task['id'])
        if score is None:
            score = dict(task=copy.deepcopy(task),task_id=task['id'], history_id=task['history_id'],context_id=task['context_id'],
                complete=False, qualified=False,status='unavailable',windows={},gates={},error='Planned score unavailable')
        _exact(score.get('task'), task, 'score task')
        _exact(score.get('task_id'), task['id'], 'score task_id')
        _exact(score.get('history_id'), task['history_id'], 'score history_id')
        _exact(score.get('context_id'), task['context_id'], 'score context_id')
        for flag in ('complete', 'qualified'):
            if type(score.get(flag)) is not bool:
                raise ValueError('Score completeness/qualification must be Boolean')
        if score['qualified'] and not score['complete']:
            raise ValueError('Incomplete score cannot be qualified')
        if score.get('complete') is True:
            if set(score.get('windows', {})) != set(WINDOWS):
                raise ValueError('Complete score lacks required windows')
            for window in WINDOWS:
                primary = score['windows'][window]['primary']
                for key in ('projected_mean','noise_sd','standardized_mean','probability_correct'):
                    _scalar(primary[key], 'complete primary.'+key)
                if not 0 <= primary['probability_correct'] <= 1:
                    raise ValueError('Probability outside [0,1]')
        retained_scores.append(score); score_map[task['id']] = score
    retained_comparisons = []
    for spec in specs:
        row = actual_comparisons.get(spec['id'])
        if row is None:
            row = dict(spec, complete=False, qualified=False,status='unavailable',
                error='Planned comparison unavailable',gates={},secondary=dict(available=False,qualified=False))
        for key, value in spec.items():
            _exact(row.get(key), value, 'comparison.'+key)
        for flag in ('complete', 'qualified'):
            if type(row.get(flag)) is not bool:
                raise ValueError('Comparison completeness/qualification must be Boolean')
        if row['qualified'] and not row['complete']:
            raise ValueError('Incomplete comparison cannot be qualified')
        retained_comparisons.append(row)
    prerequisites = {}
    for context_id in context_ids:
        p = prerequisites_by_context.get(context_id)
        if p is None:
            p = dict(context_id=context_id,complete=False,qualified=False,error='Reference unavailable')
        _exact(p.get('context_id'), context_id, 'prerequisite context')
        for flag in ('complete', 'qualified'):
            if type(p.get(flag)) is not bool:
                raise ValueError('Prerequisite completeness/qualification must be Boolean')
        if p['qualified'] and not p['complete']:
            raise ValueError('Incomplete prerequisite cannot be qualified')
        prerequisites[context_id] = p
    context_status = {}
    for cid in context_ids:
        scores = [s for s in retained_scores if s['task']['context_id']==cid]
        rows = [r for r in retained_comparisons if r['context_id']==cid]
        p = prerequisites[cid]
        context_status[cid] = dict(context_id=cid,
            complete=p.get('complete') is True and all(s.get('complete') is True for s in scores) and all(r.get('complete') is True for r in rows),
            qualified=p.get('qualified') is True and all(s.get('qualified') is True for s in scores) and all(r.get('qualified') is True for r in rows),
            scores_expected=len(scores), scores_complete=sum(s.get('complete') is True for s in scores),
            comparisons_expected=len(rows), comparisons_complete=sum(r.get('complete') is True for r in rows),
            comparisons_qualified=sum(r.get('qualified') is True for r in rows))
    balances = []
    for cid in context_ids:
        group = [h for h in histories if h['context_id']==cid]
        base = group[0]
        for arm in ARMS:
            lookup = {(h['first_sign'],h['second_sign']):score_map[h['id']+'--rk4-fine'] for h in group if h['arm']==arm}
            for window in WINDOWS:
                row = dict(context_id=cid,setting=base['setting'],phase_index=base['phase_index'],split=base['split'],
                    phase_fraction=base['phase_fraction'],arm=arm,window=window,complete=False,
                    qualified=context_status[cid]['qualified'], with_second=None,without_second=None,interference_loss=None,
                    secondary_available=False,secondary_qualified=False)
                if all(s.get('complete') is True for s in lookup.values()):
                    probs = np.asarray([[lookup[f,s]['windows'][window]['primary']['probability_correct']
                        for s in (-1,0,1)] for f in (-1,1)], dtype=np.float64)
                    controls = [lookup[0,s]['windows'][window]['primary']['probability_correct'] for s in (-1,0,1)]
                    with_second = float(np.mean(probs[:,[0,2]])); without = float(np.mean(probs[:,1]))
                    row.update(complete=True,probabilities=probs,with_second=with_second,without_second=without,
                        interference_loss=without-with_second,no_first_balanced_probabilities=controls,
                        no_first_chance_pass=all(x==.5 for x in controls))
                    for prediction_key, output_key in (('linear_prediction','linear'),('one_mode_prediction','one_mode')):
                        pred = [lookup[f,s]['windows'][window].get(prediction_key) for f in (-1,1) for s in (-1,0,1)]
                        if all(x is not None for x in pred):
                            a = np.asarray([x['probability_correct'] for x in pred],dtype=np.float64).reshape(2,3)
                            row[output_key] = dict(with_second=float(np.mean(a[:,[0,2]])),without_second=float(np.mean(a[:,1])),
                                probabilities=a,finite_minus_prediction_with_second=with_second-float(np.mean(a[:,[0,2]])))
                        else:
                            row[output_key] = None
                    secondary = [lookup[f,s]['windows'][window]['secondary'] for f in (-1,1) for s in (-1,0,1)]
                    if all(x['available'] for x in secondary):
                        a = np.asarray([x['score']['probability_correct'] for x in secondary],dtype=np.float64).reshape(2,3)
                        relevant = [r for r in retained_comparisons if r['context_id']==cid
                            and score_map[r['left_task_id']]['task']['arm']==arm]
                        row.update(secondary_available=True,secondary=dict(with_second=float(np.mean(a[:,[0,2]])),
                            without_second=float(np.mean(a[:,1])),probabilities=a),
                            secondary_qualified=context_status[cid]['qualified'] and all(r.get('secondary',{}).get('qualified') is True for r in relevant))
                if row['complete']:
                    derivative = [lookup[f,s]['windows'][window].get('derivative',dict(available=False)) for f in (-1,1) for s in (-1,0,1)]
                    row['derivative_available'] = all(x['available'] for x in derivative)
                    row['derivative_qualified'] = False
                    if row['derivative_available']:
                        a = np.asarray([x['score']['probability_correct'] for x in derivative],dtype=np.float64).reshape(2,3)
                        relevant = [r for r in retained_comparisons if r['context_id']==cid
                            and score_map[r['left_task_id']]['task']['arm']==arm]
                        row['derivative'] = dict(with_second=float(np.mean(a[:,[0,2]])),without_second=float(np.mean(a[:,1])),probabilities=a)
                        row['derivative_qualified'] = context_status[cid]['qualified'] and all(r.get('derivative',{}).get('qualified') is True for r in relevant)
                balances.append(row)
    settings = {}
    for setting in SETTINGS:
        cids = [cid for cid in context_ids if prerequisites[cid].get('context',{}).get('setting',cid.split('--')[0])==setting]
        qualified = len(cids)==24 and all(context_status[cid]['qualified'] for cid in cids)
        means = None
        if qualified:
            means = []
            for split in ('development','heldout'):
                for arm in ARMS:
                    for window in WINDOWS:
                        rows = [r for r in balances if r['setting']==setting and r['split']==split and r['arm']==arm and r['window']==window]
                        means.append(dict(split=split,arm=arm,window=window,phases=len(rows),
                            with_second=float(np.mean([r['with_second'] for r in rows])),
                            without_second=float(np.mean([r['without_second'] for r in rows])),
                            interference_loss=float(np.mean([r['interference_loss'] for r in rows]))))
        settings[setting] = dict(qualified=qualified,contexts=24,
            qualified_contexts=sum(context_status[cid]['qualified'] for cid in cids),means=means)
    held = {}
    for setting in SETTINGS:
        by_phase = {r['phase_index']:r for r in balances if r['setting']==setting and r['split']=='heldout' and r['arm']=='identity' and r['window']=='k4'}
        held[setting] = [by_phase[i]['with_second'] if by_phase[i]['qualified'] and by_phase[i]['complete'] else None for i in range(8,24)]
    bounds = {s:accuracy_failure_bounds(held[s]) for s in SETTINGS}
    contrast = None
    if all(settings[s]['qualified'] for s in SETTINGS):
        contrast = paired_phase_interval(np.asarray(held['core'],dtype=np.float64),np.asarray(held['failed-near-edge'],dtype=np.float64))
    full_complete = all(x['complete'] for x in context_status.values())
    return dict(complete=full_complete,qualified=all(s['qualified'] for s in settings.values()),
        contract=contract(),counts=dict(contexts=48,reference_jobs=96,histories=1296,trace_tasks=2700,comparisons=1404,
            scores_complete=sum(s.get('complete') is True for s in retained_scores),
            comparisons_complete=sum(c.get('complete') is True for c in retained_comparisons),
            comparisons_qualified=sum(c.get('qualified') is True for c in retained_comparisons)),
        scores=retained_scores,comparisons=retained_comparisons,contexts=context_status,
        balanced_accuracies=balances,settings=settings,primary=contrast,
        primary_failure_bounds=dict(setting_accuracy=bounds,
            edge_minus_core_percentage_points=[100*(bounds['failed-near-edge']['lower']-bounds['core']['upper']),
                100*(bounds['failed-near-edge']['upper']-bounds['core']['lower'])]),
        scope='Analytic Gaussian fixed-observer accuracy, phase-known, conditional on two existing cycles; no capacity/erasure claim')
