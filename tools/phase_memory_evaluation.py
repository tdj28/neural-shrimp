"""Unnumbered draft phase-memory analysis; supplied arrays only, no I/O/ODE.

Fine RK4 is primary. Coarse RK4 and DOP853 are numerical checks, never
independent replicates. The caller binds reference/trace provenance, reconstructs
state derivatives from its unchanged RHS, and retains all raw failure packets.
The same source freeze must eventually bind this module and acquisition code.
"""
from __future__ import annotations

from dataclasses import asdict
from run_hopfield import safe
import numpy as np
from scipy.special import ndtr
import phase_memory as pm

SETTINGS = {'core': (1.064, 10.260722222222222),
            'failed-near-edge': (1.0661333333333334, 10.297166666666667)}
DOSES = (1e-4, 1e-3, 1e-2)
DIRECTIONS = ('phase_write', 'phase_null', 'neuron_one')
METHODS = ('rk4-0.0025', 'rk4-0.00125', 'dop853')
PRIMARY_METHOD = 'rk4-0.00125'
METHOD_PAIRS = ((METHODS[0], METHODS[1]), (METHODS[1], METHODS[2]))
NOISES = (1e-5, 1e-4, 1e-3)
PRIMARY_NOISE = 1e-4
TIMES = np.arange(12801, dtype=np.float64)*.025
TIMES[-1] = 320.
TIMES.setflags(write=False)


def task_census():
    """Independently reconstructed exact 912-row acquisition identity contract."""
    rows = []
    for setting in SETTINGS:
        for phase in range(8):
            for dose, epsilon in enumerate(DOSES):
                for direction in DIRECTIONS:
                    for sign in (-1, 1):
                        for method in METHODS:
                            rows.append(dict(id=f'{setting}--p{phase}--d{dose}--{direction}--s{sign:+d}--{method}',
                                setting=setting, phase_index=phase, kind='pulse', dose_index=dose,
                                direction=direction, sign=sign, epsilon=epsilon, method=method))
            for method in METHODS:
                rows.append(dict(id=f'{setting}--p{phase}--sham--{method}', setting=setting,
                    phase_index=phase, kind='sham', dose_index=None, direction=None, sign=None,
                    epsilon=None, method=method))
    return rows


_TASKS_BY_ID = {row['id']: row for row in task_census()}


def _task(task):
    if not isinstance(task, dict):
        raise ValueError('Exact task mapping required')
    if any(isinstance(value, np.ndarray) for value in task.values()):
        raise ValueError('Task metadata cannot contain arrays')
    task = safe(task)  # Normalize numpy scalar categories exactly as saved JSON.
    expected = _TASKS_BY_ID.get(task.get('id'))
    if expected != task or any(type(task[k]) is not type(expected[k]) for k in expected or {}):
        raise ValueError('Unplanned or type-substituted task identity')
    return expected


def _qualified(packet, name):
    if (packet.get('status') != 'complete' or packet.get('qualified') is not True
            or not isinstance(packet.get('gates'), dict) or not packet['gates']
            or any(value is not True for value in packet['gates'].values())):
        raise ValueError(f'{name} prerequisite is not fully qualified')


def _reference(task, reference, propagator):
    _qualified(reference, 'reference'); _qualified(propagator, 'phase propagator')
    period = pm._scalar(reference['period'], 'reference period', positive=True)
    if (reference['setting'] != task['setting'] or tuple(reference[k] for k in ('alpha', 'gamma')) != SETTINGS[task['setting']]
            or propagator['setting'] != task['setting'] or propagator['phase_index'] != task['phase_index']
            or propagator['period'] != period or 8*period > 320):
        raise ValueError('Reference/propagator identity or horizon differs')
    orbit, adjoint = reference['orbit'], reference['adjoint']
    expected_times = np.linspace(0., period, 4097)
    for packet in (orbit, adjoint, propagator):
        if not np.array_equal(pm._array(packet['times'], 'reference sample grid', (4097,)), expected_times):
            raise ValueError('Exact endpoint-included reference grid required')
    state = pm._array(orbit['states'], 'reference states', (4097, 3))
    derivative = pm._array(orbit['derivatives'], 'reference derivatives', (4097, 3))
    z = pm._array(adjoint['values'], 'adjoint values', (4097, 3))
    dz = pm._array(adjoint['derivatives'], 'adjoint derivatives', (4097, 3))
    template = pm.PeriodicHermiteTemplate(expected_times[:-1], state[:-1], derivative[:-1], period, raw_endpoint=state[-1])
    z_template = pm.PeriodicHermiteTemplate(expected_times[:-1], z[:-1], dz[:-1], period, raw_endpoint=z[-1])
    if len(reference['phases']) != 8 or [p['phase_index'] for p in reference['phases']] != list(range(8)):
        raise ValueError('Complete ordered eight-phase reference packet required')
    phase = reference['phases'][task['phase_index']]
    sample = 512*task['phase_index']
    if (phase['phase_epoch'] != task['phase_index']*period/8
            or not np.array_equal(phase['initial'], state[sample])
            or not np.array_equal(phase['flow'], derivative[sample])
            or not np.array_equal(phase['phase_gradient'], z[sample])
            or not np.array_equal(propagator['initial'], phase['initial'])):
        raise ValueError('Realized phase state/gradient/epoch binding differs')
    directions = pm.pulse_directions(z[sample], derivative[sample])
    if (set(phase['directions']) != set(DIRECTIONS)
            or any(not np.array_equal(phase['directions'][key], directions['directions'][key]) for key in DIRECTIONS)
            or phase['classifier_orientations'] != directions['classifier_orientations']):
        raise ValueError('Saved directions/classifier orientations differ')
    phi = pm._array(propagator['phi'], 'phase propagator', (4097, 3, 3))
    dphi = pm._array(propagator['phi_derivatives'], 'phase propagator derivative', (4097, 3, 3))
    segment = pm.HermiteSegment(expected_times, phi, dphi)
    matrix = pm._array(propagator['monodromy'], 'monodromy', (3, 3))
    if not np.array_equal(phi[0], np.eye(3)) or not np.array_equal(phi[-1], matrix):
        raise ValueError('Phase propagator endpoint identities differ')
    return template, z_template, phase, segment, matrix


def _trace(task, trace, reference, phase, state_derivatives):
    if _task(trace.get('task')) != task or trace.get('status') != 'complete' or trace.get('complete') is not True:
        raise ValueError('Trace incomplete or task identity differs')
    if tuple(trace[k] for k in ('alpha', 'gamma')) != SETTINGS[task['setting']]:
        raise ValueError('Trace weights differ from the fixed setting')
    if not np.array_equal(pm._array(trace['times'], 'trace grid', TIMES.shape), TIMES):
        raise ValueError('Complete 320-unit/12801-sample trace grid required')
    state = pm._array(trace['states'], 'trace states', (12801, 3))
    derivatives = pm._array(state_derivatives, 'RHS-derived trace derivatives', state.shape)
    initial = pm._array(phase['initial'], 'phase initial', (3,))
    pulse = (dict(initial=initial, requested_delta=np.zeros(3), realized_delta=np.zeros(3),
                  realized_norm=0., addition_rounding_residual=np.zeros(3)) if task['kind'] == 'sham'
             else pm.realize_pulse(initial, phase['directions'][task['direction']], task['epsilon'], task['sign']))
    for name, value in pulse.items():
        if not np.array_equal(trace[name], value):
            raise ValueError(f'Realized pulse field differs: {name}')
    if not np.array_equal(trace['phase_initial'], initial) or not np.array_equal(state[0], pulse['initial']):
        raise ValueError('Saved initial state differs from exact realized pulse')
    if not np.array_equal(trace['initial_activation_displacement'], np.tanh(pulse['initial'])-np.tanh(initial)):
        raise ValueError('Initial activation-displacement diagnostic differs')
    solver = trace['solver']
    if task['method'].startswith('rk4-'):
        dt = float(task['method'].split('-')[1])
        required = dict(id=task['method'], kind='RK4', dt=dt, steps=round(320/dt))
    else:
        required = dict(id='dop853', kind='DOP853', rtol=1e-12, atol=1e-14, max_step=.025, success=True)
        if type(solver.get('nfev')) is not int or solver['nfev'] < 0:
            raise ValueError('Independent solver work/status invalid')
    if any(solver.get(key) != value or type(solver.get(key)) is not type(value) for key, value in required.items()):
        raise ValueError('Trace solver specification differs')
    return state, derivatives, pulse


def _difference(array):
    values = pm._array(array, 'diagnostic array')
    if not values.size:
        raise ValueError('Empty diagnostic array')
    location = tuple(int(i) for i in np.unravel_index(np.abs(values).argmax(), values.shape))
    maximum = float(np.abs(values).max())
    # Scaling keeps finite large diagnostics finite rather than squaring to inf.
    scaled = values/maximum if maximum else values
    return dict(rms=float(maximum*np.sqrt(np.mean(scaled**2))), maximum_absolute=maximum,
        maximum_location=list(location), value_at_maximum=float(values[location]),
        component_rms=(maximum*np.sqrt(np.mean(scaled**2, axis=0))).tolist())


def response_arrays(template, adjoint_template, segment, monodromy, phase_epoch, realized_delta, times, states):
    """Full saved-time diagnostics only on t<=8T, not the remaining 320-unit tail.

    Linear phase/transverse coordinates are not exact finite-amplitude
    isochrons/isostables. Full linear response uses the actual rounded pulse.
    The caller may retain these arrays externally; scoring keeps summaries.
    """
    times = pm._array(times, 'diagnostic times')
    states = pm._array(states, 'diagnostic states', (len(times), 3))
    mask = times <= 8*template.period
    selected = times[mask]
    baseline = template.evaluate(selected+phase_epoch)
    velocity = template.evaluate(selected+phase_epoch, derivative=True)
    adjoint = adjoint_template.evaluate(selected+phase_epoch)
    difference = states[mask]-baseline
    phase = np.einsum('ij,ij->i', adjoint, difference)
    transverse = difference-velocity*phase[:, None]
    linear = pm.full_linear_response(segment, monodromy, realized_delta, selected)['state_delta']
    return dict(times=selected, state_delta=difference, full_linear_state_delta=linear,
        state_prediction_error=difference-linear, time_phase_linear=phase, transverse_linear=transverse,
        activation_delta=np.tanh(states[mask])-np.tanh(baseline),
        activation_linear_prediction=pm.linear_activation_delta(baseline, linear))


def _context_id(task):
    return f"{task['setting']}--p{task['phase_index']}"


def _case_id(task):
    return task['id'].rsplit('--', 1)[0]


def _pulse(task, phase):
    initial = pm._array(phase['initial'], 'phase initial', (3,))
    if task['kind'] == 'sham':
        return dict(initial=initial.copy(), requested_delta=np.zeros(3), realized_delta=np.zeros(3),
                    realized_norm=0., addition_rounding_residual=np.zeros(3))
    return pm.realize_pulse(initial, phase['directions'][task['direction']], task['epsilon'], task['sign'])


def _prepared_context(task, reference, propagator):
    template, z_template, phase, segment, matrix = _reference(task, reference, propagator)
    epoch = phase['phase_epoch']
    times = TIMES[TIMES <= 8*template.period].copy()
    context = dict(id=_context_id(task), setting=task['setting'], phase_index=task['phase_index'],
        status='complete', period=template.period, phase_epoch=epoch,
        initial=np.asarray(phase['initial']).copy(), flow=np.asarray(phase['flow']).copy(),
        phase_gradient=np.asarray(phase['phase_gradient']).copy(),
        directions={k:np.asarray(v).copy() for k,v in phase['directions'].items()},
        classifier_orientations=safe(phase['classifier_orientations']),
        diagnostic_times=times, reference_states=template.evaluate(times+epoch),
        reference_velocity=template.evaluate(times+epoch, derivative=True),
        reference_adjoint=z_template.evaluate(times+epoch),
        readouts={str(delay):asdict(pm.clock_readout(template, epoch, delay)) for delay in pm.DELAYS},
        derivative_scope='Clock G uses the periodic cubic-Hermite template derivative; not a new RHS evaluation.',
        diagnostic_scope='Saved times t<=8T only; all declared readout times are strictly below8T.')
    return context, (template, phase, segment, matrix)


def _prepared_case(task, context, mathematics):
    template, phase, segment, matrix = mathematics
    pulse = _pulse(task, phase)
    z, epoch = np.asarray(phase['phase_gradient']), phase['phase_epoch']
    orientation = 1 if task['kind'] == 'sham' else phase['classifier_orientations'][task['direction']]
    result = dict(id=_case_id(task), context_id=_context_id(task), status='complete',
        case={k:v for k,v in task.items() if k not in ('id','method')}, pulse=pulse,
        orientation=orientation, requested_time_displacement=float(z@pulse['requested_delta']),
        realized_time_displacement=float(z@pulse['realized_delta']),
        time_displacement_rounding_difference=float(z@(pulse['realized_delta']-pulse['requested_delta'])),
        pulse_used_for_phase_only='requested_delta', pulse_used_for_full_linear='realized_delta', readouts={})
    for delay in pm.DELAYS:
        readout = pm.ClockReadout(**context['readouts'][str(delay)])
        null = pm.phase_only_prediction(template, pulse['requested_delta'], z, readout.times, phase_epoch=epoch)
        linear = pm.full_linear_response(segment, matrix, pulse['realized_delta'], readout.times)['state_delta']
        baseline = template.evaluate(readout.times+epoch)
        activation_linear = pm.linear_activation_delta(baseline, linear)
        linear_q = pm._scalar(float(np.sum(readout.weights*activation_linear)), 'linear readout score')
        result['readouts'][str(delay)] = dict(phase_only_states=null['states'], phase_only_activations=null['activations'],
            full_linear_state_delta=linear, full_linear_activation_delta=activation_linear,
            noise_gain=float(np.linalg.norm(readout.weights)), phase_only_q=readout.score(null['activations']),
            full_linear_q=linear_q, zero_response_q=0.)
    times = context['diagnostic_times']
    result['full_linear_state_delta'] = pm.full_linear_response(segment,matrix,pulse['realized_delta'],times)['state_delta']
    result['full_linear_activation_delta'] = pm.linear_activation_delta(context['reference_states'],result['full_linear_state_delta'])
    result['phase_only_activations'] = pm.phase_only_prediction(template,pulse['requested_delta'],z,times,phase_epoch=epoch)['activations']
    return result


def prepare_reference_outputs(references, propagators, tasks=None):
    """Reference-only complete ledger to persist/hash/seal BEFORE any trace.

    references: setting -> reference packet; propagators: (setting,phase_index)
    -> phase packet. tasks, when supplied, must be the exact ordered912 census.
    Produces16 shared contexts/readout coefficients and304 method-independent
    pulse/sham null/linear packets. No argument accepts pulse outcomes. Missing
    prerequisites retain failed contexts/cases. Parent owns actual disk seals,
    qualification of reference acquisition, and chronology enforcement.
    """
    expected = task_census()
    if tasks is not None and (len(tasks)!=len(expected) or [_task(t) for t in tasks]!=expected):
        raise ValueError('Exact complete ordered912-task plan required')
    reference_keys=set(SETTINGS); propagator_keys={(setting,p) for setting in SETTINGS for p in range(8)}
    if set(references)-reference_keys or set(propagators)-propagator_keys:
        raise ValueError('Unknown prerequisite identity')
    result=dict(schema_version='draft-phase-memory-prepared-v1', tasks=expected, contexts={},cases={},
        required_counts=dict(contexts=16,cases=304,traces=912), failures=[])
    mathematics={}
    for task in expected:
        cid, caseid=_context_id(task), _case_id(task)
        if cid not in result['contexts']:
            try:
                context, math=_prepared_context(task,references[task['setting']],propagators[(task['setting'],task['phase_index'])])
                result['contexts'][cid]=context; mathematics[cid]=math
            except (ValueError,KeyError,TypeError,FloatingPointError,np.linalg.LinAlgError) as error:
                result['contexts'][cid]=dict(id=cid,setting=task['setting'],phase_index=task['phase_index'],status='failed',error=repr(error))
                result['failures'].append(dict(id=cid,error=repr(error)))
        if caseid in result['cases']:continue
        try:
            if cid not in mathematics:raise ValueError('Unqualified or missing reference/phase prerequisite')
            result['cases'][caseid]=_prepared_case(task,result['contexts'][cid],mathematics[cid])
        except (ValueError,KeyError,TypeError,FloatingPointError,np.linalg.LinAlgError) as error:
            result['cases'][caseid]=dict(id=caseid,context_id=cid,status='failed',error=repr(error))
            result['failures'].append(dict(id=caseid,error=repr(error)))
    # One method-independent paired null table is sealed before any trace.
    result['paired_nulls']=[]
    for task in expected:
        if task['kind']!='pulse' or task['sign']!=-1 or task['method']!=PRIMARY_METHOD:continue
        minus=result['cases'][_case_id(task)]
        plus=result['cases'][_case_id(task).replace('--s-1','--s+1')]
        for delay in pm.DELAYS:
            for sigma in NOISES:
                cell={key:task[key] for key in ('setting','phase_index','dose_index','direction')}
                cell.update(delay_cycles=delay,sigma=sigma,status='unavailable',
                    phase_only_balanced_accuracy=None,full_linear_balanced_accuracy=None,zero_balanced_accuracy=.5)
                if minus['status']==plus['status']=='complete':
                    a,b=plus['readouts'][str(delay)],minus['readouts'][str(delay)]
                    cell.update(status='complete',phase_only_balanced_accuracy=_accuracy(a['phase_only_q'],b['phase_only_q'],a['noise_gain'],plus['orientation'],sigma),
                        full_linear_balanced_accuracy=_accuracy(a['full_linear_q'],b['full_linear_q'],a['noise_gain'],plus['orientation'],sigma))
                result['paired_nulls'].append(cell)
    result['complete']=not result['failures']
    result['qualified']=result['complete']
    return result


def packet_differences(left, right, path='packet'):
    """Exact saved replay: strict raw arrays, safe()-normalized JSON scalars.

    Nonfinite numeric leaves are always errors, even matching NaN/inf. This is
    value/schema replay; the controller separately verifies bytes/hashes and
    rejects extra/missing NPZ members before reconstruction.
    """
    failures=[]
    def walk(a,b,p):
        if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
            if (not isinstance(a,np.ndarray) or not isinstance(b,np.ndarray)
                or a.dtype!=np.float64 or b.dtype!=np.float64 or a.shape!=b.shape
                or not np.isfinite(a).all() or not np.isfinite(b).all()
                or a.tobytes()!=b.tobytes()):failures.append(p+': array dtype/shape/finite/bitwise mismatch')
        elif isinstance(a,dict) or isinstance(b,dict):
            if not isinstance(a,dict) or not isinstance(b,dict) or set(a)!=set(b):
                failures.append(p+': dictionary identity mismatch');return
            for key in a:walk(a[key],b[key],p+'.'+str(key))
        elif isinstance(a,(list,tuple)) or isinstance(b,(list,tuple)):
            if not isinstance(a,(list,tuple)) or not isinstance(b,(list,tuple)) or len(a)!=len(b):
                failures.append(p+': sequence identity mismatch');return
            for index,(x,y) in enumerate(zip(a,b)):walk(x,y,f'{p}[{index}]')
        else:
            if any(isinstance(v,(float,np.floating)) and not np.isfinite(v) for v in (a,b)):
                failures.append(p+': nonfinite numeric leaf');return
            a,b=safe(a),safe(b)
            if type(a) is not type(b) or a!=b:failures.append(p+': normalized metadata mismatch')
    walk(left,right,path)
    return failures


def replay_prepared_reference_outputs(prepared, references, propagators, tasks=None):
    """Reference-only exact saved packet replay; returns all field mismatches."""
    rebuilt=prepare_reference_outputs(references,propagators,tasks)
    differences=packet_differences(prepared,rebuilt)
    return dict(match=not differences, qualified=not differences and rebuilt['qualified'],
                differences=differences,required_counts=rebuilt['required_counts'])


def score_trace(task, trace, reference, phase_propagator, *, state_derivatives, prepared):
    """Consume sealed reference-only predictions, then score one supplied trace.

    The corresponding context/case is independently rebuilt and compared before
    trace access. The full prepared packet is replayed once by the controller;
    this per-trace check protects local bindings. No outcome enters preparation.
    Primary null uses requested pulses; linear diagnostics use realized pulses.
    """
    task = _task(task)
    result = dict(task=task.copy(), status='failed', readouts={}, error=None)
    try:
        context, math = _prepared_context(task,reference,phase_propagator)
        case = _prepared_case(task,context,math)
        if (prepared['schema_version']!='draft-phase-memory-prepared-v1'
                or prepared['required_counts']!=dict(contexts=16,cases=304,traces=912)):
            raise ValueError('Prepared packet schema/census differs')
        differences=packet_differences(prepared['contexts'][_context_id(task)],context)
        differences+=packet_differences(prepared['cases'][_case_id(task)],case)
        if differences:raise ValueError('Prepared reference replay differs: '+ '; '.join(differences))
        context,case=prepared['contexts'][_context_id(task)],prepared['cases'][_case_id(task)]
        template, phase, segment, matrix = math
        state, derivative, pulse = _trace(task, trace, reference, phase, state_derivatives)
        sample_trace = pm.HermiteSegment(TIMES, state, derivative)
        for key in ('orientation','requested_time_displacement','realized_time_displacement','time_displacement_rounding_difference'):
            result[key]=safe(case[key])
        for delay in pm.DELAYS:
            readout = pm.ClockReadout(**context['readouts'][str(delay)])
            activations=np.tanh(sample_trace.evaluate(readout.times)); q=readout.score(activations)
            null=case['readouts'][str(delay)]; gain=null['noise_gain']
            result['readouts'][str(delay)] = dict(q=q, noise_gain=gain, phase_only_q=null['phase_only_q'],
                full_linear_q=null['full_linear_q'], zero_response_q=0., sample_count=32, activation_components=3,
                standardized_sham_bias=q/(PRIMARY_NOISE*gain) if task['kind']=='sham' else None)
        size=len(context['diagnostic_times'])
        actual=state[:size]-context['reference_states']
        coordinate=np.einsum('ij,ij->i',context['reference_adjoint'],actual)
        arrays=dict(state_delta=actual,full_linear_state_delta=case['full_linear_state_delta'],
            state_prediction_error=actual-case['full_linear_state_delta'],
            transverse_linear=actual-context['reference_velocity']*coordinate[:,None],
            activation_delta=np.tanh(state[:size])-np.tanh(context['reference_states']),
            activation_linear_prediction=case['full_linear_activation_delta'],
            phase_only_activation_prediction_error=np.tanh(state[:size])-case['phase_only_activations'])
        result['dynamics']=dict(samples=size,last_saved_time=float(context['diagnostic_times'][-1]),
            maximum_declared_time=8*template.period,phase_linear_initial=float(coordinate[0]),
            phase_linear_final=float(coordinate[-1]),**{name:_difference(value) for name,value in arrays.items()},
            coordinate_scope='Linear phase/transverse coordinates, not exact finite-amplitude phase or isostables.',
            pulse_used_for_full_linear='realized_delta',pulse_used_for_phase_only='requested_delta')
        result['status']='complete'
    except (ValueError,KeyError,TypeError,FloatingPointError,np.linalg.LinAlgError) as error:
        result['error']=repr(error)
        if isinstance(error,pm.PhaseQualificationError):result['qualification_diagnostics']=error.diagnostics
    return result


def compare_trace_pair(task_a, trace_a, score_a, task_b, trace_b, score_b):
    """All 320 units for state/waveform gates; fixed clock readout mean gate."""
    a, b = _task(task_a), _task(task_b)
    same = lambda r: {k:v for k,v in r.items() if k not in ('id', 'method')}
    if same(a) != same(b) or (a['method'], b['method']) not in METHOD_PAIRS:
        raise ValueError('Exact same-case coarse/fine or fine/independent comparison required')
    result = dict(ids=[a['id'], b['id']], setting=a['setting'], complete=False, qualified=False, error=None)
    try:
        if (_task(trace_a['task']) != a or _task(trace_b['task']) != b or _task(score_a['task']) != a or _task(score_b['task']) != b
                or trace_a.get('complete') is not True or trace_b.get('complete') is not True
                or any(r['status'] != 'complete' for r in (trace_a, trace_b, score_a, score_b))
                or any(not np.array_equal(t['times'], TIMES) for t in (trace_a, trace_b))):
            raise ValueError('Incomplete comparison inputs or substituted identity/grid')
        left = pm._array(trace_a['states'], 'left trace', (12801, 3))
        right = pm._array(trace_b['states'], 'right trace', (12801, 3))
        state, activation = _difference(left-right), _difference(np.tanh(left)-np.tanh(right))
        readouts = {}
        for delay in pm.DELAYS:
            sa, sb = score_a['readouts'][str(delay)], score_b['readouts'][str(delay)]
            gain = pm._scalar(sa['noise_gain'], 'noise gain', positive=True)
            if sb['noise_gain'] != gain:
                raise ValueError('Readout weights differ across numerical methods')
            difference = abs(pm._scalar(sa['q'], 'left score')-pm._scalar(sb['q'], 'right score'))/(PRIMARY_NOISE*gain)
            readouts[str(delay)] = dict(standardized_mean_difference=difference, passed=bool(difference <= .01))
        result.update(complete=True, state_difference=state, activation_difference=activation, readouts=readouts,
            state_pass=state['maximum_absolute'] <= 1e-6, activation_pass=activation['maximum_absolute'] <= 1e-7)
        result['qualified'] = result['state_pass'] and result['activation_pass'] and all(r['passed'] for r in readouts.values())
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        result['error'] = repr(error)
    return result


def paired_response_diagnostics(task_plus, trace_plus, score_plus, task_minus, trace_minus, score_minus, *, prepared):
    """Signed odd/even response around the unchanged template, t<=8T only.

    q_odd=(q_plus-q_minus)/2 and q_even=(q_plus+q_minus)/2; physical
    state/activation responses subtract the reference before the even average.
    No sham recentering and no numerical-method pooling. Parent retains all432
    planned same-method pair records even when either sign is unavailable.
    """
    plus,minus=_task(task_plus),_task(task_minus)
    fields=('setting','phase_index','kind','dose_index','direction','epsilon','method')
    if (plus['kind']!='pulse' or plus['sign']!=1 or minus['sign']!=-1
            or any(plus[key]!=minus[key] for key in fields)):
        raise ValueError('Exact same-method positive/negative pulse pair required')
    result=dict(ids=[plus['id'],minus['id']],status='failed',setting=plus['setting'],
                phase_index=plus['phase_index'],dose_index=plus['dose_index'],direction=plus['direction'],method=plus['method'])
    try:
        for task,trace,score in ((plus,trace_plus,score_plus),(minus,trace_minus,score_minus)):
            if (_task(trace['task'])!=task or _task(score['task'])!=task
                    or trace.get('complete') is not True or trace['status']!='complete' or score['status']!='complete'
                    or not np.array_equal(pm._array(trace['times'],'signed trace times',TIMES.shape),TIMES)):
                raise ValueError('Unavailable or misbound signed pair')
        context=prepared['contexts'][_context_id(plus)]
        if context['status']!='complete':raise ValueError('Unavailable prepared signed-pair context')
        size=len(context['diagnostic_times']);baseline=pm._array(context['reference_states'],'signed-pair reference',(size,3))
        xp=pm._array(trace_plus['states'],'positive states',(12801,3))[:size]
        xm=pm._array(trace_minus['states'],'negative states',(12801,3))[:size]
        dp,dm=xp-baseline,xm-baseline
        ap,am=np.tanh(xp)-np.tanh(baseline),np.tanh(xm)-np.tanh(baseline)
        result.update(status='complete',samples=size,last_saved_time=float(context['diagnostic_times'][-1]),
            state_odd=_difference((dp-dm)/2),state_even=_difference((dp+dm)/2),
            activation_odd=_difference((ap-am)/2),activation_even=_difference((ap+am)/2),readouts={})
        for delay in pm.DELAYS:
            a=pm._scalar(score_plus['readouts'][str(delay)]['q'],'positive q')
            b=pm._scalar(score_minus['readouts'][str(delay)]['q'],'negative q')
            result['readouts'][str(delay)]=dict(q_odd=(a-b)/2,q_even=(a+b)/2)
    except (ValueError,KeyError,TypeError,AttributeError) as error:
        result.update(status='failed',error=repr(error))
    return result


def _accuracy(q_plus, q_minus, gain, orientation, sigma):
    q_plus, q_minus = pm._scalar(q_plus, 'positive score'), pm._scalar(q_minus, 'negative score')
    gain = pm._scalar(gain, 'readout noise gain', positive=True)
    if type(orientation) is not int or orientation not in (-1, 1):
        raise ValueError('Fixed integer classifier orientation required')
    means = np.array([orientation*q_plus, -orientation*q_minus])/(sigma*gain)
    if not np.isfinite(means).all():
        raise ValueError('Nonfinite standardized class score')
    return float(np.mean(ndtr(means)))


def _indexed(records, expected, key):
    indexed, failures = {}, []
    for position, row in enumerate(records):
        identity = None
        try:
            identity = key(row)
            if identity not in expected or identity in indexed:
                raise ValueError('Unknown or duplicate record identity')
            indexed[identity] = row
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            try:
                planned = expected.get(identity)
                setting = planned.get('setting') if isinstance(planned,dict) else planned
            except (TypeError,AttributeError):setting=None
            failures.append(dict(position=position, setting=setting, error=str(error)))
    return indexed, failures


def summarize(scored_records, trace_comparisons, *, prerequisites):
    """Complete-denominator paired readouts and per-setting descriptive means.

    prerequisites maps each setting to exactly {reference: bool,
    propagators: [bool]*8}; false means unqualified, never a dropped origin.
    912 trace rows,608 trace comparisons,5,184 paired readouts and3,456 BA
    comparisons are retained even with missing inputs. No bootstrap or pooling
    numerical methods as replicates. Saved diagnostic packets are caller-owned.
    """
    tasks = task_census(); expected = {t['id']:t for t in tasks}
    if (set(prerequisites) != set(SETTINGS) or any(set(p) != {'reference', 'propagators'}
            or type(p['reference']) is not bool or len(p['propagators']) != 8
            or any(type(x) is not bool for x in p['propagators']) for p in prerequisites.values())):
        raise ValueError('Explicit full two-setting reference/propagator prerequisite gates required')
    scores, invalid_scores = _indexed(scored_records, expected, lambda row:row['task']['id'])
    terminal = []
    for task in tasks:
        row = scores.get(task['id'])
        error = None
        if row is None: error = 'missing_scored_trace'
        elif packet_differences(row.get('task'),task): error = 'substituted_task_identity'
        elif row.get('status') != 'complete': error = 'failed_scored_trace'
        elif set(row.get('readouts', {})) != {str(k) for k in pm.DELAYS}: error = 'incomplete_readout_delays'
        if error:
            terminal.append(dict(id=task['id'], setting=task['setting'], status=error)); continue
        try:
            for readout in row['readouts'].values():
                for name in ('q', 'phase_only_q', 'full_linear_q'):
                    pm._scalar(readout[name], name)
                pm._scalar(readout['noise_gain'], 'noise gain', positive=True)
                if readout['zero_response_q'] != 0. or readout['sample_count'] != 32 or readout['activation_components'] != 3:
                    raise ValueError('Fixed readout dimensions or zero null differ')
            if type(row['orientation']) is not int or row['orientation'] not in (-1, 1):
                raise ValueError('Invalid classifier orientation')
            sham_pass = task['kind'] != 'sham' or all(abs(r['q']/(PRIMARY_NOISE*r['noise_gain'])) <= .01 for r in row['readouts'].values())
            terminal.append(dict(id=task['id'], setting=task['setting'], status='complete', sham_pass=sham_pass))
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            terminal.append(dict(id=task['id'], setting=task['setting'], status='invalid_scored_trace', error=str(error)))
    valid_ids = {r['id'] for r in terminal if r['status'] == 'complete'}
    expected_comparisons = {}
    for task in tasks:
        for left, right in METHOD_PAIRS:
            if task['method'] != left:continue
            peer = dict(task, method=right, id=task['id'].rsplit('--', 1)[0]+'--'+right)
            expected_comparisons[(task['id'], peer['id'])] = task['setting']
    comparisons, invalid_comparisons = _indexed(trace_comparisons, expected_comparisons, lambda row:tuple(row['ids']))
    comparison_rows = []
    for pair, setting in expected_comparisons.items():
        row = comparisons.get(pair)
        passed = False
        try:
            if row is not None and row.get('complete') is True and all(ident in valid_ids for ident in pair):
                readouts = row['readouts']
                if row.get('setting') != setting:raise ValueError('Comparison setting differs')
                for delay in pm.DELAYS:
                    a,b=scores[pair[0]]['readouts'][str(delay)],scores[pair[1]]['readouts'][str(delay)]
                    if a['noise_gain']!=b['noise_gain']:raise ValueError('Comparison readout weight differs')
                    expected_difference=abs(a['q']-b['q'])/(PRIMARY_NOISE*a['noise_gain'])
                    if readouts[str(delay)]['standardized_mean_difference']!=expected_difference:
                        raise ValueError('Saved comparison mean differs from scored rows')
                # Reconstruct gates, never trust a supplied qualified=True alone.
                passed = (pm._scalar(row['state_difference']['maximum_absolute'], 'state difference') <= 1e-6
                    and pm._scalar(row['activation_difference']['maximum_absolute'], 'activation difference') <= 1e-7
                    and set(readouts) == {str(k) for k in pm.DELAYS}
                    and all(0 <= pm._scalar(r['standardized_mean_difference'], 'readout difference') <= .01 for r in readouts.values())
                    and row['state_difference']['maximum_absolute'] >= 0 and row['activation_difference']['maximum_absolute'] >= 0)
        except (ValueError, KeyError, TypeError):pass
        comparison_rows.append(dict(ids=list(pair), setting=setting, qualified=bool(passed), missing=row is None))
    paired = []
    for task in tasks:
        if task['kind'] != 'pulse' or task['sign'] != -1:continue
        plus_id = task['id'].replace('--s-1--', '--s+1--')
        minus, plus = scores.get(task['id']), scores.get(plus_id)
        for delay in pm.DELAYS:
            for sigma in NOISES:
                cell = {k:task[k] for k in ('setting', 'phase_index', 'dose_index', 'direction', 'method')}
                cell.update(delay_cycles=delay, sigma=sigma, status='unavailable', balanced_accuracy=None,
                    phase_only_balanced_accuracy=None, full_linear_balanced_accuracy=None, zero_balanced_accuracy=.5)
                try:
                    if task['id'] not in valid_ids or plus_id not in valid_ids:raise ValueError('Missing signed response')
                    left, right = plus['readouts'][str(delay)], minus['readouts'][str(delay)]
                    if plus['orientation'] != minus['orientation'] or left['noise_gain'] != right['noise_gain']:
                        raise ValueError('Paired readout/noise identity differs')
                    for output, name in [('balanced_accuracy','q'), ('phase_only_balanced_accuracy','phase_only_q'), ('full_linear_balanced_accuracy','full_linear_q')]:
                        cell[output] = _accuracy(left[name],right[name],left['noise_gain'],plus['orientation'],sigma)
                    cell['status'] = 'complete'
                except (ValueError, KeyError, TypeError, AttributeError) as error:cell['error']=str(error)
                paired.append(cell)
    signed_readouts=[]
    for task in tasks:
        if task['kind']!='pulse' or task['sign']!=-1:continue
        plus_id=task['id'].replace('--s-1--','--s+1--')
        for delay in pm.DELAYS:
            row={key:task[key] for key in ('setting','phase_index','dose_index','direction','method')}
            row.update(delay_cycles=delay,status='unavailable',q_odd=None,q_even=None,
                phase_only_q_odd=None,phase_only_q_even=None,full_linear_q_odd=None,full_linear_q_even=None)
            if task['id'] in valid_ids and plus_id in valid_ids:
                a,b=scores[plus_id]['readouts'][str(delay)],scores[task['id']]['readouts'][str(delay)]
                for name in ('q','phase_only_q','full_linear_q'):
                    row[name+'_odd']=(a[name]-b[name])/2
                    row[name+'_even']=(a[name]+b[name])/2
                row['status']='complete'
            signed_readouts.append(row)
    by_key = {(c['setting'],c['phase_index'],c['dose_index'],c['direction'],c['method'],c['delay_cycles'],c['sigma']):c for c in paired}
    ba_comparisons=[]
    for cell in paired:
        for left,right in METHOD_PAIRS:
            if cell['method']!=left:continue
            key=(cell['setting'],cell['phase_index'],cell['dose_index'],cell['direction'],right,cell['delay_cycles'],cell['sigma'])
            other=by_key[key]
            difference=abs(cell['balanced_accuracy']-other['balanced_accuracy']) if cell['status']==other['status']=='complete' else None
            ba_comparisons.append(dict(setting=cell['setting'],phase_index=cell['phase_index'],dose_index=cell['dose_index'],
                direction=cell['direction'],methods=[left,right],delay_cycles=cell['delay_cycles'],sigma=cell['sigma'],
                absolute_ba_difference=difference,qualified=difference is not None and difference<=.001))
    settings={}
    for setting in SETTINGS:
        needed=[r for r in terminal if r['setting']==setting]
        complete=all(r['status']=='complete' for r in needed)
        qualified=(complete and not any(r['setting'] in (None,setting) for r in invalid_scores+invalid_comparisons) and prerequisites[setting]['reference']
            and all(prerequisites[setting]['propagators']) and all(r['sham_pass'] for r in needed)
            and all(r['qualified'] for r in comparison_rows+ba_comparisons if r['setting']==setting)
            and all(r['status']=='complete' for r in paired if r['setting']==setting))
        aggregates=None
        if qualified:
            aggregates=[]
            for method in METHODS:
                for delay in pm.DELAYS:
                    for sigma in NOISES:
                        means={}
                        for direction in DIRECTIONS:
                            cells=[c for c in paired if (c['setting'],c['method'],c['delay_cycles'],c['sigma'],c['direction'])==(setting,method,delay,sigma,direction)]
                            if len(cells)!=24:raise ValueError('Incomplete fixed eight-phase/three-dose denominator')
                            means[direction]={name:float(np.mean([c[name] for c in cells])) for name in
                                ('balanced_accuracy','phase_only_balanced_accuracy','full_linear_balanced_accuracy','zero_balanced_accuracy')}
                        aggregates.append(dict(method=method,delay_cycles=delay,sigma=sigma,cells_per_direction=24,means=means,
                            phase_write_minus_phase_null=means['phase_write']['balanced_accuracy']-means['phase_null']['balanced_accuracy'],
                            primary=method==PRIMARY_METHOD and delay==4 and sigma==PRIMARY_NOISE))
        settings[setting]=dict(complete=complete,qualified=bool(qualified),required_traces=456,aggregates=aggregates)
    return dict(schema_version='draft-phase-memory-analysis-v1',primary_method=PRIMARY_METHOD,
        complete=all(s['complete'] for s in settings.values()),qualified=all(s['qualified'] for s in settings.values()),
        required_counts=dict(trace_rows=912,trace_readout_scores=3648,pulse_readout_scores=3456,sham_readout_scores=192,trace_comparisons=608,trace_comparisons_per_setting=304,paired_readout_cells=5184,method_independent_null_cells=1728,signed_response_pairs=432,signed_readout_cells=1728,balanced_accuracy_comparisons=3456),
        invalid_input_records=dict(scored=invalid_scores,comparisons=invalid_comparisons),
        terminal_rows=terminal,trace_comparisons=comparison_rows,paired_readouts=paired,signed_response_readouts=signed_readouts,balanced_accuracy_comparisons=ba_comparisons,
        prerequisites=prerequisites,settings=settings,
        scope='Two fixed settings; fine RK4 primary, other methods numerical diagnostics. No bootstrap, solver pooling, population guarantee or geometry-utility claim.')
