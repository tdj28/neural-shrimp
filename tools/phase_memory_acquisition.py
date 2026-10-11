"""Draft phase-memory acquisition; imports perform no solve or file access.

The caller owns a prospectively frozen source/input plan, exclusive external
artifacts, resource limits, process ownership, stage seals and full task ledger.
These composable functions do not schedule jobs, build native code or retry a
failure. AcquisitionFailure.payload retains completed stages and partial solver
output. No outcome has been generated to develop this module.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import solve_ivp

import hopfield_diagnostics as diagnostics
from hopfield_drift_probe import ProbeFailure, integrate_trace
from phase_memory import PeriodicHermiteTemplate, neutral_left_phase, pulse_directions, realize_pulse

SETTINGS = {
    'core': dict(alpha=1.064, gamma=10.260722222222222,
        period=37.63637538160568,
        section_state=[-0.09541754833868374, -0.4837814189448931, 9.377779462204991]),
    'failed-near-edge': dict(alpha=1.0661333333333334, gamma=10.297166666666667,
        period=37.33149958853496,
        section_state=[-0.21395544204154604, 0.06262818434064273, -4.015837698327758]),
}
HISTORICAL = {
    'experiments/hopfield/EXP-005-plan.json': (3019, 'a74b737df2b98e390ce25fc42868c46de9f9b60a1d07150e0b396c12195e6463'),
    'evidence/EXP-005/attempt-001/manifest.json': (3019, 'a74b737df2b98e390ce25fc42868c46de9f9b60a1d07150e0b396c12195e6463'),
    'evidence/EXP-005/attempt-001/receipt.json': (4777, 'fb287eb68541e674c8dc095049c3e0a4e887e3b7297f8903dda39a9797ad3aee'),
    'evidence/EXP-005/attempt-001/points/core.json': (628323, 'c2fcd62b246835cc2a6e6a3e8849eb5a60981596d777200c8c7e0f7c5cef60dc'),
    'evidence/EXP-005/attempt-001/points/failed-near-edge.json': (628398, '8a300186196f84ffb58dd82c734f6a063e89cba95a5319cb42081eeef428fe05'),
}
FINE = dict(method='DOP853', rtol=1e-12, atol=1e-14, max_step=.025)
PHASE_NODES = 4096
DURATION = 320.
SAMPLE_DT = .025
TIMES = np.arange(12801, dtype=np.float64)*SAMPLE_DT
TIMES[-1] = DURATION
TIMES.setflags(write=False)
DOSES = (1e-4, 1e-3, 1e-2)
DIRECTIONS = ('phase_write', 'phase_null', 'neuron_one')
METHODS = ('rk4-0.0025', 'rk4-0.00125', 'dop853')
CERTIFICATE_GATES = ('closure', 'phase_condition', 'section_is_maximum', 'nonstationary',
                     'expected_peak_count', 'primitive', 'neutral_mode', 'transverse_stability')
REFERENCE_GATES = dict(relative_closure=1e-8, adjoint_relative_periodicity=1e-7,
    adjoint_normalization=1e-7, neutral_relative_residual=1e-7,
    unique_singular_rtol=1e-12, normalization_cosine_min=1e-12)


class AcquisitionFailure(RuntimeError):
    """The caller must persist payload even when acquisition cannot qualify."""
    def __init__(self, message, payload):
        super().__init__(message)
        self.payload = payload


def _array(value, name, shape):
    result = np.asarray(value)
    if result.dtype != np.float64 or result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{name} requires finite float64 shape {shape}')
    return result


def _finite_positive(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (float, int, np.floating, np.integer)) or not np.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')
    return float(value)


def _setting(value):
    if type(value) is not str or value not in SETTINGS:
        raise ValueError('Only the two frozen EXP005 settings are permitted')
    return SETTINGS[value]


def _validate_seed(seed):
    expected = _setting(seed['setting'])
    if any(seed[key] != expected[key] for key in ('alpha', 'gamma', 'period')):
        raise ValueError('Historical seed parameter/period identity differs')
    if not np.array_equal(_array(seed['section_state'], 'seed state', (3,)), expected['section_state']):
        raise ValueError('Historical seed state identity differs')


def _certificate_passes(certificate):
    return (certificate.get('verified') is True and certificate.get('attracting') is True
        and certificate.get('primitive') is True and certificate.get('expected_peak_count') == 9
        and certificate.get('peak_count') == 9 and set(certificate.get('gates', {})) == set(CERTIFICATE_GATES)
        and all(certificate['gates'][key] is True for key in CERTIFICATE_GATES))


def load_seed_records(repo_root):
    """Read only the five exact historical files; old orbit arrays are not reused."""
    root = Path(repo_root).resolve()
    decoded, bindings = {}, []
    for relative, (size, sha) in HISTORICAL.items():
        path = root/relative
        if not path.resolve().is_relative_to(root):
            raise ValueError('Historical input escapes repository')
        data = path.read_bytes()
        if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
            raise ValueError('Historical input bytes changed: '+relative)
        decoded[relative] = json.loads(data)
        bindings.append(dict(path=relative, bytes=size, sha256=sha))
    prefix = 'evidence/EXP-005/attempt-001/'
    receipt = decoded[prefix+'receipt.json']
    index = {entry['path']: entry for entry in receipt['artifacts']}
    if len(index) != len(receipt['artifacts']):
        raise ValueError('Duplicate historical receipt paths')
    seeds = []
    for setting, expected in SETTINGS.items():
        relative = prefix+f'points/{setting}.json'
        size, sha = HISTORICAL[relative]
        if index[f'points/{setting}.json'] != dict(path=f'points/{setting}.json', bytes=size, sha256=sha):
            raise ValueError('Historical receipt does not bind the exact seed')
        point = decoded[relative]; certificate = point['certificate']
        if (point['name'] != setting or any(point[key] != expected[key] or certificate[key] != expected[key]
                for key in ('alpha', 'gamma')) or not _certificate_passes(certificate)):
            raise ValueError('Historical certificate/setting identity differs')
        seed = dict(setting=setting, alpha=point['alpha'], gamma=point['gamma'],
            section_state=np.asarray(certificate['section_state'], dtype=np.float64),
            period=certificate['period'], historical_bindings=[dict(row) for row in bindings])
        _validate_seed(seed); seeds.append(seed)
    return seeds


def rerun_certificate(seed):
    """Exactly the unchanged independent nine-peak certification call."""
    _validate_seed(seed)
    return diagnostics.certify(seed['alpha'], seed['gamma'], seed['section_state'].copy(),
                               seed['period'], expected_peak_count=9)


def _solver_snapshot(solution):
    return dict(times=np.array(solution.t, dtype=np.float64, copy=True),
        values=np.array(solution.y, dtype=np.float64, copy=True),
        success=bool(solution.success), message=str(solution.message),
        nfev=int(solution.nfev), njev=int(solution.njev), nlu=int(solution.nlu))


def _solve(fun, span, initial, times, *, stage, dense=False):
    """Keep returned partial samples, and label last RHS stages as unaccepted."""
    last_rhs = {}
    def tracked(t, y):
        last_rhs.update(time=float(t), values=np.array(y, dtype=np.float64, copy=True),
                        accepted_solver_state=False)
        return fun(t, y)
    started = time.monotonic()
    try:
        solution = solve_ivp(tracked, span, initial, t_eval=times, dense_output=dense, **FINE)
    except Exception as error:
        raise AcquisitionFailure(f'{stage} solver raised: {error}', dict(stage=stage, status='solver_exception',
            error=repr(error), last_rhs=last_rhs, solver=dict(FINE), wall_seconds=time.monotonic()-started)) from error
    snapshot = _solver_snapshot(solution)
    snapshot.update(stage=stage, solver=dict(FINE), wall_seconds=time.monotonic()-started)
    if (not solution.success or snapshot['values'].shape != (len(initial), len(times))
            or not np.array_equal(snapshot['times'], times) or not np.isfinite(snapshot['values']).all()
            or not np.array_equal(snapshot['values'][:, 0], initial)):
        snapshot.update(status='incomplete_or_invalid_solver_output', last_rhs=last_rhs)
        raise AcquisitionFailure(f'{stage} incomplete, nonfinite, wrong grid or wrong initial state', snapshot)
    snapshot['status'] = 'complete'
    return snapshot, solution.sol


def _variational(alpha, gamma, initial, period, *, stage):
    times = np.linspace(0., period, PHASE_NODES+1, dtype=np.float64)
    initial = _array(initial, 'variational initial state', (3,)).copy()
    def rhs(t, packed):
        x, phi = packed[:3], packed[3:].reshape(3, 3)
        return np.concatenate((diagnostics.hopfield_rhs(t, x, alpha, gamma),
            (diagnostics.hopfield_jacobian(t, x, alpha, gamma)@phi).ravel()))
    result, dense = _solve(rhs, (0., period), np.r_[initial, np.eye(3).ravel()], times, stage=stage, dense=True)
    packed = result.pop('values').T
    states, phi = packed[:, :3].copy(), packed[:, 3:].reshape(-1, 3, 3).copy()
    result.update(states=states, phi=phi, initial=initial, monodromy=phi[-1].copy())
    try:
        derivatives = np.asarray([diagnostics.hopfield_rhs(t, x, alpha, gamma) for t, x in zip(times, states, strict=True)])
        result['derivatives'] = derivatives
        phi_derivatives = np.asarray([diagnostics.hopfield_jacobian(t, x, alpha, gamma)@p
            for t, x, p in zip(times, states, phi, strict=True)])
        result['phi_derivatives'] = phi_derivatives
    except Exception as error:
        result.update(status='derivative_reconstruction_failed', error=repr(error), qualified=False)
        raise AcquisitionFailure('Completed variational solve derivative reconstruction failed', result) from error
    flow0, flow_end = derivatives[0], derivatives[-1]
    flow_norm = float(np.linalg.norm(flow0))
    if not np.isfinite(flow_norm) or flow_norm == 0 or not np.isfinite(derivatives).all() or not np.isfinite(phi_derivatives).all():
        raise AcquisitionFailure('Invalid reference flow/derivative', dict(result, qualified=False))
    result.update(states=states, derivatives=derivatives, phi=phi, phi_derivatives=phi_derivatives,
        initial=initial, monodromy=phi[-1].copy(), endpoint_difference=states[-1]-initial,
        relative_closure=float(np.linalg.norm(states[-1]-initial)/max(1., np.linalg.norm(initial))),
        neutral_flow_relative_residual=float(np.linalg.norm(phi[-1]@flow0-flow0)/flow_norm),
        transported_flow_relative_residual=float(np.linalg.norm(phi[-1]@flow0-flow_end)/flow_norm))
    return result, dense


def _fail(packet, stage, error):
    packet.update(status='failed', qualified=False, failed_stage=stage, error=repr(error))
    if isinstance(error, AcquisitionFailure):
        packet['partial_failure'] = error.payload
    elif hasattr(error, 'diagnostics'):
        packet['partial_failure'] = error.diagnostics
    return AcquisitionFailure(f'{stage} failed: {error}', packet)


def acquire_reference(seed):
    """Certificate, raw fine orbit and backward adjoint; no pulse access."""
    _validate_seed(seed)
    alpha, gamma = seed['alpha'], seed['gamma']
    packet = dict(setting=seed['setting'], alpha=alpha, gamma=gamma, seed=seed,
                  status='started', qualified=False, gates={}, thresholds=dict(REFERENCE_GATES))
    stage = 'certificate'
    try:
        certificate = rerun_certificate(seed); packet['certificate'] = certificate
        packet['gates']['original_certificate'] = _certificate_passes(certificate)
        if not packet['gates']['original_certificate']:
            raise ValueError('Original numerical certificate did not pass unchanged gates')
        if certificate['alpha'] != alpha or certificate['gamma'] != gamma:
            raise ValueError('Certificate setting identity differs')
        period = _finite_positive(certificate['period'], 'reference period'); packet['period'] = period
        packet['gates']['fixed_duration_covers_eight_cycles'] = bool(8*period <= DURATION)
        if not packet['gates']['fixed_duration_covers_eight_cycles']:
            raise ValueError('Eight periods exceed frozen 320-unit horizon')
        initial = _array(np.asarray(certificate['section_state'], dtype=np.float64), 'certified initial state', (3,))
        stage = 'fine_orbit'
        orbit, dense = _variational(alpha, gamma, initial, period, stage=stage); packet['orbit'] = orbit
        packet['gates']['fine_orbit_closure'] = bool(orbit['relative_closure'] <= REFERENCE_GATES['relative_closure'])
        stage = 'neutral_left_phase'
        neutral = neutral_left_phase(orbit['monodromy'], orbit['derivatives'][0],
            residual_atol=REFERENCE_GATES['neutral_relative_residual'],
            unique_singular_rtol=REFERENCE_GATES['unique_singular_rtol'],
            normalization_cosine_min=REFERENCE_GATES['normalization_cosine_min'])
        packet['neutral'] = neutral
        stage = 'backward_adjoint'
        z_terminal = neutral['phase_gradient'].copy()
        def adjoint_rhs(t, z):
            x = dense(t)[:3]
            return -diagnostics.hopfield_jacobian(t, x, alpha, gamma).T@z
        descending = orbit['times'][::-1].copy()
        adjoint, _ = _solve(adjoint_rhs, (period, 0.), z_terminal, descending, stage=stage)
        packet['adjoint'] = adjoint
        values = adjoint.pop('values').T[::-1].copy()
        adjoint['values'] = values
        adjoint['times'] = adjoint['times'][::-1].copy()
        derivative = np.asarray([-diagnostics.hopfield_jacobian(t, x, alpha, gamma).T@z
            for t, x, z in zip(orbit['times'], orbit['states'], values, strict=True)])
        residuals = np.einsum('ij,ij->i', values, orbit['derivatives'])-1.
        periodic_vector = values[0]-values[-1]
        periodic_relative = float(np.linalg.norm(periodic_vector)/max(np.linalg.norm(values[0]), np.linalg.norm(values[-1])))
        adjoint.update(values=values, derivatives=derivative, terminal_normalized_once=z_terminal,
            periodic_residual_vector=periodic_vector, relative_periodic_residual=periodic_relative,
            normalization_residuals=residuals, maximum_normalization_residual=float(np.max(np.abs(residuals))),
            normalization_count=1, orbit_for_adjoint='raw_DOP853_dense_output_no_periodic_wrap')
        packet['adjoint'] = adjoint
        packet['gates'].update(adjoint_periodicity=bool(np.isfinite(periodic_relative) and periodic_relative <= REFERENCE_GATES['adjoint_relative_periodicity']),
            adjoint_normalization=bool(np.isfinite(residuals).all() and np.max(np.abs(residuals)) <= REFERENCE_GATES['adjoint_normalization']),
            finite_adjoint_derivatives=bool(np.isfinite(derivative).all()))
        if not all(packet['gates'].values()):
            raise ValueError('Reference closure or adjoint residual gates failed')
        stage = 'phase_directions'; packet['phases'] = []
        for j in range(8):
            index = j*(PHASE_NODES//8)
            direction = pulse_directions(values[index], orbit['derivatives'][index])
            packet['phases'].append(dict(phase_index=j, phase_epoch=float(orbit['times'][index]),
                initial=orbit['states'][index].copy(), flow=orbit['derivatives'][index].copy(),
                phase_gradient=values[index].copy(), **direction))
        packet.update(status='complete', qualified=True)
        return packet
    except Exception as error:
        raise _fail(packet, stage, error) from error


def reference_templates(reference):
    """Derived periodic interpolation; raw endpoints remain explicitly attached."""
    if reference.get('qualified') is not True or reference.get('status') != 'complete':
        raise ValueError('Qualified reference required')
    orbit, adjoint, period = reference['orbit'], reference['adjoint'], reference['period']
    state = PeriodicHermiteTemplate(orbit['times'][:-1], orbit['states'][:-1], orbit['derivatives'][:-1],
        period, raw_endpoint=orbit['states'][-1])
    phase = PeriodicHermiteTemplate(adjoint['times'][:-1], adjoint['values'][:-1], adjoint['derivatives'][:-1],
        period, raw_endpoint=adjoint['values'][-1])
    return state, phase


def _phase(reference, phase_index):
    if reference.get('qualified') is not True or reference.get('status') != 'complete':
        raise ValueError('Qualified reference required before dependent acquisition')
    expected = _setting(reference['setting'])
    if any(reference[key] != expected[key] for key in ('alpha', 'gamma')):
        raise ValueError('Reference setting differs')
    if type(phase_index) is not int or phase_index not in range(8):
        raise ValueError('Phase index must be an integer from zero through seven')
    phase = reference['phases'][phase_index]
    if phase['phase_index'] != phase_index or phase['phase_epoch'] != reference['orbit']['times'][phase_index*(PHASE_NODES//8)]:
        raise ValueError('Phase index/epoch binding differs')
    for key, raw_key in (('initial', 'states'), ('flow', 'derivatives')):
        if not np.array_equal(_array(phase[key], key, (3,)), reference['orbit'][raw_key][phase_index*(PHASE_NODES//8)]):
            raise ValueError('Phase state/flow binding differs')
    if not np.array_equal(_array(phase['phase_gradient'], 'phase gradient', (3,)), reference['adjoint']['values'][phase_index*(PHASE_NODES//8)]):
        raise ValueError('Phase adjoint binding differs')
    return phase


def acquire_phase_propagator(reference, phase_index):
    """Direct one-period propagation from identity; never invert a common Phi."""
    phase = _phase(reference, phase_index)
    packet = dict(setting=reference['setting'], phase_index=phase_index, phase_epoch=phase['phase_epoch'],
        alpha=reference['alpha'], gamma=reference['gamma'], period=reference['period'],
        initial=phase['initial'].copy(), status='started', qualified=False)
    try:
        orbit, _ = _variational(packet['alpha'], packet['gamma'], packet['initial'], packet['period'], stage='phase_propagator')
        packet.update(orbit)
        packet['gates'] = dict(relative_closure=bool(orbit['relative_closure'] <= REFERENCE_GATES['relative_closure']))
        packet['qualified'] = all(packet['gates'].values())
        if not packet['qualified']:
            raise ValueError('Phase propagator relative closure exceeds 1e-8')
        return packet
    except Exception as error:
        raise _fail(packet, 'phase_propagator', error) from error


def build_tasks():
    """All 912 named pulse/sham acquisitions, with no response-based filtering."""
    tasks = []
    for setting in SETTINGS:
        for phase in range(8):
            for dose_index, epsilon in enumerate(DOSES):
                for direction in DIRECTIONS:
                    for sign in (-1, 1):
                        for method in METHODS:
                            tasks.append(dict(id=f'{setting}--p{phase}--d{dose_index}--{direction}--s{sign:+d}--{method}',
                                setting=setting, phase_index=phase, kind='pulse', dose_index=dose_index,
                                epsilon=epsilon, direction=direction, sign=sign, method=method))
            for method in METHODS:
                tasks.append(dict(id=f'{setting}--p{phase}--sham--{method}', setting=setting, phase_index=phase,
                    kind='sham', dose_index=None, epsilon=None, direction=None, sign=None, method=method))
    return tasks


def _validate_task(task):
    if type(task) is not dict or set(task) != {'id','setting','phase_index','kind','dose_index','epsilon','direction','sign','method'}:
        raise ValueError('Exact task metadata required')
    if type(task['phase_index']) is not int or (task['kind'] == 'pulse' and
            (type(task['dose_index']) is not int or type(task['sign']) is not int or type(task['epsilon']) is not float)):
        raise ValueError('Strict task scalar identities required')
    expected = next((x for x in build_tasks() if x['id'] == task['id']), None)
    if expected is None or task != expected:
        raise ValueError('Task not in frozen 912-case census')


def case_context(task, reference):
    """Realize a reference-derived pulse before any response is acquired."""
    _validate_task(task)
    if task['setting'] != reference['setting']:
        raise ValueError('Task/reference setting differs')
    phase = _phase(reference, task['phase_index'])
    # Reconstruct the exact mathematical directions; reject stale/edited metadata.
    directions = pulse_directions(phase['phase_gradient'], phase['flow'])
    for name in DIRECTIONS:
        if not np.array_equal(phase['directions'][name], directions['directions'][name]):
            raise ValueError('Saved pulse direction differs from adjoint construction')
    base = phase['initial'].copy()
    if task['kind'] == 'pulse':
        realized = realize_pulse(base, directions['directions'][task['direction']], task['epsilon'], task['sign'])
    else:
        realized = dict(initial=base.copy(), requested_delta=np.zeros(3), realized_delta=np.zeros(3),
            realized_norm=0., addition_rounding_residual=np.zeros(3))
    return dict(alpha=reference['alpha'], gamma=reference['gamma'], phase_initial=base,
        phase_epoch=phase['phase_epoch'], period=reference['period'], **realized,
        initial_activation_displacement=np.tanh(realized['initial'])-np.tanh(base))


def acquire_case(task, reference, *, native_fn=None):
    """One exact 320-unit trace. Caller persists every returned/failed payload."""
    context = case_context(task, reference)
    packet = dict(task=dict(task), **context, status='started', complete=False,
                  duration=DURATION, sample_dt=SAMPLE_DT)
    started = time.monotonic()
    try:
        if task['method'].startswith('rk4-'):
            if native_fn is None:
                raise ValueError('Owned prebuilt native function required')
            dt = float(task['method'].removeprefix('rk4-'))
            packet['solver'] = dict(id=task['method'], kind='RK4', method='RK4', dt=dt, steps=round(DURATION/dt), samples=len(TIMES),
                                    implementation='unchanged hopfield_drift_probe.integrate_trace')
            times, states = integrate_trace(native_fn, context['alpha'], context['gamma'], context['initial'],
                                            dt, duration=DURATION, sample_dt=SAMPLE_DT)
            packet.update(times=times, states=states)
        else:
            packet['solver'] = dict(id=task['method'], kind='DOP853', **FINE)
            rhs = lambda t, x: diagnostics.hopfield_rhs(t, x, context['alpha'], context['gamma'])
            raw, _ = _solve(rhs, (0., DURATION), context['initial'], TIMES, stage='independent_trace')
            packet['solver'].update(success=True, nfev=raw['nfev'])
            packet.update(times=raw['times'], states=raw['values'].T.copy(),
                independent_diagnostics={key:value for key,value in raw.items() if key not in ('times','values')})
        if (not np.array_equal(_array(packet['times'], 'trace times', TIMES.shape), TIMES)
                or not np.array_equal(_array(packet['states'], 'trace states', (len(TIMES),3))[0], context['initial'])):
            raise ValueError('Trace grid/initial identity differs')
        packet.update(status='complete', complete=True, wall_seconds=time.monotonic()-started)
        return packet
    except Exception as error:
        if isinstance(error, ProbeFailure):
            packet.update(times=error.payload['times'], states=error.payload['states'], native_failure=error.payload)
        packet['wall_seconds'] = time.monotonic()-started
        raise _fail(packet, 'trace', error) from error


def trace_derivatives(packet):
    """RHS evaluation of saved states only; no integration or saved duplication."""
    if packet.get('complete') is not True or packet.get('status') != 'complete':
        raise ValueError('Complete saved trace required')
    _validate_task(packet['task'])
    expected = _setting(packet['task']['setting'])
    if any(packet[key] != expected[key] for key in ('alpha','gamma')):
        raise ValueError('Trace setting identity differs')
    times = _array(packet['times'], 'trace times', TIMES.shape)
    states = _array(packet['states'], 'trace states', (len(TIMES),3))
    if not np.array_equal(times, TIMES):
        raise ValueError('Trace grid differs')
    result = np.asarray([diagnostics.hopfield_rhs(t,x,packet['alpha'],packet['gamma'])
        for t,x in zip(times,states,strict=True)])
    return _array(result, 'trace derivatives', states.shape)


def replay_saved_variational(packet, alpha, gamma, period, initial):
    """Check saved derivatives, boundary identities and residuals without ODEs."""
    times=_array(packet['times'],'saved variational times',(PHASE_NODES+1,))
    if not np.array_equal(times,np.linspace(0.,period,PHASE_NODES+1)):
        raise ValueError('Saved variational grid differs')
    states=_array(packet['states'],'saved states',(PHASE_NODES+1,3))
    phi=_array(packet['phi'],'saved tangent',(PHASE_NODES+1,3,3))
    if not np.array_equal(states[0],initial) or not np.array_equal(phi[0],np.eye(3)) or not np.array_equal(phi[-1],packet['monodromy']):
        raise ValueError('Saved variational endpoint identities differ')
    velocity=np.asarray([diagnostics.hopfield_rhs(t,x,alpha,gamma) for t,x in zip(times,states,strict=True)])
    derivative=np.asarray([diagnostics.hopfield_jacobian(t,x,alpha,gamma)@p for t,x,p in zip(times,states,phi,strict=True)])
    for name,expected in [('derivatives',velocity),('phi_derivatives',derivative)]:
        if not np.allclose(_array(packet[name],name,expected.shape),expected,rtol=1e-12,atol=1e-14):
            raise ValueError('Saved derivative differs: '+name)
    flow_norm=np.linalg.norm(velocity[0])
    residuals=dict(relative_closure=float(np.linalg.norm(states[-1]-initial)/max(1.,np.linalg.norm(initial))),
        neutral_flow_relative_residual=float(np.linalg.norm(phi[-1]@velocity[0]-velocity[0])/flow_norm),
        transported_flow_relative_residual=float(np.linalg.norm(phi[-1]@velocity[0]-velocity[-1])/flow_norm))
    if not np.isfinite(list(residuals.values())).all() or any(not np.isclose(packet[k],v,rtol=1e-12,atol=1e-14) for k,v in residuals.items()):
        raise ValueError('Saved variational residual differs')
    return residuals


def replay_saved_reference(reference):
    """No recertification/integration: replay saved fine derivative/gate evidence."""
    _validate_seed(reference['seed'])
    if reference.get('qualified') is not True:return dict(qualified=False,scope='failed reference retained')
    if reference['thresholds']!=REFERENCE_GATES or not _certificate_passes(reference['certificate']):
        raise ValueError('Saved reference certificate/gate identity differs')
    period=reference['period'];alpha=reference['alpha'];gamma=reference['gamma']
    if period!=reference['certificate']['period'] or 8*period>DURATION:
        raise ValueError('Saved reference period differs')
    initial=np.asarray(reference['certificate']['section_state'],dtype=np.float64)
    replay_saved_variational(reference['orbit'],alpha,gamma,period,initial)
    orbit=reference['orbit'];adjoint=reference['adjoint'];times=orbit['times']
    values=_array(adjoint['values'],'saved adjoint',(PHASE_NODES+1,3))
    if not np.array_equal(adjoint['times'],times):raise ValueError('Saved adjoint grid differs')
    neutral=neutral_left_phase(orbit['monodromy'],orbit['derivatives'][0],
        residual_atol=REFERENCE_GATES['neutral_relative_residual'],unique_singular_rtol=REFERENCE_GATES['unique_singular_rtol'],
        normalization_cosine_min=REFERENCE_GATES['normalization_cosine_min'])
    if set(reference['neutral'])!=set(neutral):raise ValueError('Saved neutral diagnostic census differs')
    for key,value in neutral.items():
        saved=reference['neutral'][key]
        if isinstance(value,np.ndarray):
            if not np.allclose(_array(saved,key,value.shape),value,rtol=1e-12,atol=1e-14):raise ValueError('Saved neutral array differs')
        elif isinstance(value,float):
            if not np.isclose(saved,value,rtol=1e-12,atol=1e-14):raise ValueError('Saved neutral number differs')
        elif type(saved) is not type(value) or saved!=value:raise ValueError('Saved neutral category differs')
    if (type(adjoint['normalization_count']) is not int or adjoint['normalization_count']!=1 or not np.array_equal(values[-1],adjoint['terminal_normalized_once'])
            or not np.allclose(values[-1],neutral['phase_gradient'],rtol=1e-12,atol=1e-14)):
        raise ValueError('Saved single adjoint normalization differs')
    derivative=np.asarray([-diagnostics.hopfield_jacobian(t,x,alpha,gamma).T@z
        for t,x,z in zip(times,orbit['states'],values,strict=True)])
    residual=np.einsum('ij,ij->i',values,orbit['derivatives'])-1.
    relative=float(np.linalg.norm(values[0]-values[-1])/max(np.linalg.norm(values[0]),np.linalg.norm(values[-1])))
    for name,saved,expected in [('adjoint derivative',adjoint['derivatives'],derivative),
        ('normalization residuals',adjoint['normalization_residuals'],residual),
        ('periodic residual vector',adjoint['periodic_residual_vector'],values[0]-values[-1])]:
        if not np.allclose(_array(saved,name,expected.shape),expected,rtol=1e-12,atol=1e-14):raise ValueError('Saved '+name+' differs')
    if (not np.isclose(relative,adjoint['relative_periodic_residual'],rtol=1e-12,atol=1e-14)
            or not np.isclose(np.max(np.abs(residual)),adjoint['maximum_normalization_residual'],rtol=1e-12,atol=1e-14)
            or relative>REFERENCE_GATES['adjoint_relative_periodicity']
            or np.max(np.abs(residual))>REFERENCE_GATES['adjoint_normalization']
            or orbit['relative_closure']>REFERENCE_GATES['relative_closure']):
        raise ValueError('Saved fine-reference numerical gate failed')
    for j in range(8):
        phase=_phase(reference,j);expected=pulse_directions(phase['phase_gradient'],phase['flow'])
        if any(not np.array_equal(phase['directions'][key],expected['directions'][key]) for key in DIRECTIONS):
            raise ValueError('Saved reference direction differs')
    return dict(qualified=True,scope='Saved fine state/tangent/adjoint derivative and gate replay; no new certificate solve')
