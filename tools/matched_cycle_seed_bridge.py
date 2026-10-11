"""Bounded changed-parameter seed correction, without reference qualification.

Two predeclared interpolated gamma values are inputs, never optimized here.
The tight EXP018 state/tangent primitive and bordered maximum-section equations
are reused; the larger trust region belongs only to this provisional bridge.
No IO, scheduling, global monkeypatch or integrations occur at import.
"""
from __future__ import annotations

import copy
import re
from collections.abc import Mapping

import numpy as np
from scipy.integrate import solve_ivp

from hopfield_diagnostics import hopfield_rhs, hopfield_jacobian
from phase_memory_acquisition import AcquisitionFailure
import periodic_reference_refinement as refinement
import phase_reference_diagnostic as prior

METHOD = 'tight-dop853'
CYCLES = ('core-rho025', 'core-rho050')
MAX_UPDATES = 6
FACTORS = (1., .5, .25, .125)
TRIALS = 25
BINDING_FIELDS = ('experiment_id', 'source_commit', 'plan_sha256',
                  'packet_sha256', 'packet_semantic_sha256')


def contract():
    return dict(schema='matched-cycle-seed-bridge-v1', cycle_ids=list(CYCLES),
        bridge_jobs=2, trial_slots_per_bridge=TRIALS, bridge_ivp_slots=50,
        total_successor_jobs=116, total_successor_ivp_slots=578,
        method=METHOD, integration=copy.deepcopy(refinement.METHODS[METHOD]),
        maximum_updates=MAX_UPDATES, line_factors=list(FACTORS), armijo=1e-4,
        relative_closure=1e-12, section_absolute=1e-12, condition_max=1e10,
        state_radius='2*max(abs(x_right-x_left)/max(1,abs(x_seed)))+1e-6',
        period_radius='2*abs(T_right-T_left)/T_seed+1e-6',
        fixed_gamma=True, alternate_seed_or_parameter_retry=False,
        bridge_is_reference_qualification=False, adjoint_normalizations=0,
        subsequent_independent_tight_refinement_required=True)


def _same(actual, expected, name):
    """Exact input identity, including scalar types, arrays and nested maps."""
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise ValueError(name+': mapping fields differ')
        for key in expected: _same(actual[key], expected[key], name+'.'+str(key))
    elif isinstance(expected, np.ndarray):
        if (type(actual) is not np.ndarray or actual.dtype != expected.dtype or
                actual.shape != expected.shape or not np.array_equal(actual, expected)):
            raise ValueError(name+': exact array differs')
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(actual) != len(expected):
            raise ValueError(name+': sequence differs')
        for i, (a, b) in enumerate(zip(actual, expected, strict=True)): _same(a, b, name+f'[{i}]')
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(name+': exact value/type differs')


def _endpoint(row, name):
    if not isinstance(row, Mapping): raise ValueError(name+' endpoint required')
    x = prior._array(np.asarray(row['first_maximum_state']), name+' maximum', (3,)).copy()
    T = prior._scalar(row['period'], name+' period', positive=True)
    a, g, rho = (prior._scalar(row[k], name+' '+k) for k in ('alpha', 'gamma', 'dominant_rho'))
    if type(row['index']) is not int or row['index'] < 0: raise ValueError('Integer endpoint index required')
    if type(row['node_id']) is not str or not row['node_id']: raise ValueError('Endpoint node identity required')
    digest = row['node_metadata_sha256']
    if type(digest) is not str or re.fullmatch('[0-9a-f]{64}', digest) is None:
        raise ValueError('Bound endpoint metadata required')
    return x, T, a, g, rho


def build_bridge_spec(slot, bracket):
    """Bind the preselected gamma and nearest original endpoint without solves.

    The caller binds the original branch and closure-report bytes. ``bracket``
    is their retained endpoint record, not newly measured reference evidence.
    """
    if not isinstance(slot, Mapping) or not isinstance(bracket, Mapping):
        raise ValueError('Frozen slot and original bracket required')
    cycle = slot['id']
    if cycle not in CYCLES or slot.get('cycle_id', cycle) != cycle: raise ValueError('One of two fixed core bridge slots required')
    target = .25 if cycle == CYCLES[0] else .5
    _same(slot['target_rho'], target, 'slot target')
    _same(bracket['target'], target, 'bracket target')
    if slot.get('role') != 'candidate' or slot.get('source_setting') != 'core': raise ValueError('Core candidate bridge required')
    if (slot.get('available') is not True or slot.get('status') != 'bridge_required' or
            slot.get('seed_kind') != 'fixed_gamma_proposal'):
        raise ValueError('Available fixed-gamma proposal requiring a bridge is required')
    left, right = bracket['left'], bracket['right']
    xl, Tl, al, gl, rl = _endpoint(left, 'left')
    xr, Tr, ar, gr, rr = _endpoint(right, 'right')
    if al != ar or not (gl < gr and rl < target < rr) or right['index'] != left['index']+1:
        raise ValueError('Consecutive increasing original core bracket required')
    fraction = float((target-rl)/(rr-rl))
    gamma = float(gl+fraction*(gr-gl))
    _same(bracket['fraction'], fraction, 'retained interpolation fraction')
    _same(bracket['proposed_interpolated_gamma'], gamma, 'retained fixed gamma')
    if not gl < gamma < gr: raise ValueError('Strictly internal fixed parameter required')
    nearest = left if fraction <= .5 else right
    _same(bracket['proposed_nearest_endpoint_index'], nearest['index'], 'nearest original endpoint')
    x, T, a, seed_gamma, _ = _endpoint(nearest, 'nearest')
    for key, value in [('alpha', a), ('gamma', gamma), ('period', T),
                       ('node_index', nearest['index']), ('node_id', nearest['node_id'])]:
        _same(slot[key], value, 'slot '+key)
    if slot.get('branch_id') != 'core-direction-+1': raise ValueError('Original core branch required')
    _same(np.asarray(slot['first_maximum_state']), x, 'slot original maximum')
    if not isinstance(slot.get('seed_provenance'), Mapping) or not slot['seed_provenance']:
        raise ValueError('Original seed provenance required')
    scales = np.maximum(1., np.abs(x))
    state_separation = float(np.max(np.abs(xr-xl)/scales))
    period_separation = float(abs(Tr-Tl)/T)
    for key, value in [('endpoint_scaled_state_separation', state_separation),
                       ('endpoint_relative_period_separation', period_separation)]:
        _same(bracket[key], value, 'retained '+key)
    return dict(schema='matched-cycle-bridge-spec-v1', id=cycle+'--bridge', kind='bridge',
        cycle_id=cycle, setting=cycle, source_setting='core', role='candidate',
        alpha=a, gamma=gamma, original_gamma=seed_gamma, original_state=x, original_period=T,
        state_scales=scales, state_radius=2.*state_separation+1e-6,
        period_radius=2.*period_separation+1e-6, method=METHOD,
        slot=copy.deepcopy(dict(slot)), bracket=copy.deepcopy(dict(bracket)),
        seed_provenance=copy.deepcopy(dict(slot['seed_provenance'])),
        trust_scope='Fixed bracket-sized seed safeguard; not a proof of branch uniqueness')


def _validate_spec(spec):
    if not isinstance(spec, Mapping): raise ValueError('Bridge specification required')
    rebuilt = build_bridge_spec(spec['slot'], spec['bracket'])
    _same(spec, rebuilt, 'exact frozen bridge specification')
    return rebuilt


def acquire_bridge(spec, *, rhs=hopfield_rhs, jacobian=hopfield_jacobian, solver=solve_ivp):
    """Correct only the fixed gamma, retaining every planned/tried IVP slot.

    A complete unconverged packet is a scientific failure, not an exception.
    Solver/arithmetic exceptions carry the full prefix as AcquisitionFailure.
    Even a converged bridge is provisional and ``qualified`` stays False.
    """
    spec = _validate_spec(spec)
    seed, T0, S = spec['original_state'], spec['original_period'], spec['state_scales']
    alpha, gamma, ident = spec['alpha'], spec['gamma'], spec['id']
    J0 = prior._array(jacobian(0., seed, alpha, gamma), 'target-parameter seed Jacobian', (3, 3))
    section_scale = max(1., float(np.linalg.norm(J0[2]*S)))
    packet = dict(schema='matched-cycle-seed-bridge-v1', kind='bridge', id=ident,
        cycle_id=spec['cycle_id'], setting=spec['setting'], source_setting='core', role='candidate',
        alpha=alpha, gamma=gamma, method=METHOD, spec=copy.deepcopy(spec),
        original_state=seed.copy(), original_period=T0, original_gamma=spec['original_gamma'],
        state_scales=S.copy(), section_scale=section_scale, seed_provenance=copy.deepcopy(spec['seed_provenance']),
        complete=False, converged=False, qualified=False, new_reference_qualified=False, status='started',
        limits=dict(maximum_updates=6, line_factors=list(FACTORS), trial_slots=25,
            armijo=1e-4, state_radius=spec['state_radius'], period_radius=spec['period_radius'],
            condition_max=1e10, relative_closure=1e-12, section_residual=1e-12),
        trials=[dict(index=i, id=f'{ident}--trial-{i:02d}', status='not_needed', attempted=False,
            ivp_attempted=False, ivp_started=False, ivp_returned=False, ivp_completed=False,
            reason='not_reached') for i in range(TRIALS)], iterations=[], ivp_evaluations=0)

    def evaluate(index, state, period, factor=None):
        row = packet['trials'][index]
        row.update(attempted=True, status='started', state=state.copy(), period=float(period), factor=factor)
        if (not np.isfinite(state).all() or not np.isfinite(period) or period <= 0 or
                np.max(np.abs(state-seed)/S) > spec['state_radius'] or
                abs(period-T0)/T0 > spec['period_radius']):
            row.update(status='outside_neighborhood', complete=True, accepted=False)
            return None
        try:
            row['ivp_attempted'] = True
            packet['ivp_evaluations'] += 1
            def tracked_solver(*args, **kwargs):
                row['ivp_started'] = True
                out = solver(*args, **kwargs)
                row['ivp_returned'] = True
                return out
            forward, _ = refinement._forward(state, period, alpha, gamma, METHOD,
                rhs=rhs, jacobian=jacobian, solver=tracked_solver)
            row['ivp_completed'] = forward['ivp_completed']
            row['forward'] = forward
            row['metrics'] = refinement.shooting_metrics(state, period, forward['trajectory'],
                seed, T0, S, section_scale, alpha, gamma, rhs=rhs, jacobian=jacobian)
            row.update(status='complete', complete=True, accepted=False)
            return row
        except Exception as error:
            row.update(status='failed', complete=False, error=repr(error))
            if hasattr(error, 'payload'):
                row['partial_failure'] = error.payload
                row['ivp_completed'] = bool(error.payload.get('ivp_completed', False))
            raise

    stage = 'initial_trial'
    try:
        current = evaluate(0, seed, T0)
        current['accepted'] = True
        reason = 'maximum_updates'
        for iteration in range(MAX_UPDATES+1):
            metrics = current['metrics']
            if refinement._converged(metrics):
                packet['converged'] = True
                reason = 'residual_targets'
                break
            if iteration == MAX_UPDATES: break
            if metrics['scaled_condition'] is None or metrics['scaled_condition'] > 1e10:
                reason = 'singular_or_ill_conditioned_bordered_system'
                break
            stage = f'newton_update_{iteration}'
            delta = np.linalg.solve(metrics['scaled_bordered'], -metrics['scaled_residual'])
            prior._array(delta, 'scaled bridge Newton correction', (4,))
            record = dict(iteration=iteration, from_trial=current['index'], scaled_correction=delta.copy(), trial_indices=[])
            packet['iterations'].append(record)
            moved = False
            for half, factor in enumerate(FACTORS):
                index = 1+iteration*len(FACTORS)+half
                record['trial_indices'].append(index)
                state = current['state']+factor*delta[:3]*S
                period = float(current['period']+factor*delta[3]*T0)
                candidate = evaluate(index, state, period, factor)
                if candidate is not None and candidate['metrics']['merit'] <= (1.-1e-4*factor)*metrics['merit']:
                    candidate['accepted'] = True
                    current, moved = candidate, True
                    for skipped in range(index+1, 1+(iteration+1)*len(FACTORS)):
                        packet['trials'][skipped]['reason'] = 'line_search_accepted'
                    break
            if not moved:
                reason = 'line_search_no_decrease'
                break
        packet['candidate'] = dict(state=current['state'].copy(), period=current['period'],
            source_trial_index=current['index'], metrics=copy.deepcopy(current['metrics']))
        for row in packet['trials']:
            if not row['attempted'] and row['reason'] == 'not_reached': row['reason'] = 'unreached_after_'+reason
        packet.update(status='complete', complete=True, stop_reason=reason,
            attempted_trials=sum(r['attempted'] for r in packet['trials']))
        return packet
    except Exception as error:
        packet.update(status='failed', complete=False, converged=False,
            failed_stage=stage, error=repr(error))
        if hasattr(error, 'payload'): packet['partial_failure'] = error.payload
        raise AcquisitionFailure('Fixed-parameter seed bridge failed', packet) from error


def prepare_bridged_seed(slot, bridge, *, bridge_binding):
    """Admit a sealed bridge as provisional input to both independent shoots.

    Hash verification/readback is the controller's responsibility. The public
    seed and all later metadata explicitly distinguish this new correction from
    historical qualification. No orbit/adjoint certificate is invented.
    """
    import matched_cycle_reference_acquisition as acquisition
    if not isinstance(bridge, Mapping): raise ValueError('Retained bridge packet required')
    spec = _validate_spec(bridge['spec'])
    _same(slot, spec['slot'], 'original fixed selected slot')
    for field, value in [('id', spec['id']), ('kind', 'bridge'), ('cycle_id', spec['cycle_id']),
                         ('alpha', spec['alpha']), ('gamma', spec['gamma']), ('method', METHOD),
                         ('complete', True), ('converged', True), ('status', 'complete'),
                         ('qualified', False), ('new_reference_qualified', False)]:
        _same(bridge[field], value, 'admitted bridge '+field)
    if not isinstance(bridge_binding, Mapping) or set(bridge_binding) != set(BINDING_FIELDS):
        raise ValueError('Exact bridge artifact binding required')
    _same(bridge_binding['experiment_id'], 'EXP-023', 'bridge experiment')
    for key in BINDING_FIELDS[1:]:
        length = 40 if key == 'source_commit' else 64
        if type(bridge_binding[key]) is not str or re.fullmatch('[0-9a-f]{'+str(length)+'}', bridge_binding[key]) is None:
            raise ValueError('Valid bridge '+key+' required')
    candidate = bridge['candidate'];state = prior._array(candidate['state'], 'corrected seed', (3,)).copy()
    period = prior._scalar(candidate['period'], 'corrected period', positive=True)
    index = candidate['source_trial_index']
    if type(index) is not int or index not in range(25): raise ValueError('Exact bridge source trial required')
    if len(bridge['trials']) != 25: raise ValueError('All bridge trial slots required')
    trial = bridge['trials'][index]
    _same(trial['state'], state, 'candidate source state')
    _same(trial['period'], period, 'candidate source period')
    _same(trial['metrics'], candidate['metrics'], 'candidate source metrics')
    for key in ('accepted', 'complete', 'ivp_started', 'ivp_returned', 'ivp_completed'):
        _same(trial[key], True, 'admitted trial '+key)
    if (not refinement._converged(candidate['metrics']) or
            np.max(np.abs(state-spec['original_state'])/spec['state_scales']) > spec['state_radius'] or
            abs(period-spec['original_period'])/spec['original_period'] > spec['period_radius']):
        raise ValueError('Converged bridge must satisfy its frozen trust/residual limits')
    resolved = copy.deepcopy(dict(slot))
    resolved.update(status='selected', first_maximum_state=state, period=period, seed_kind='provisional_parameter_bridge',
        seed_provenance=dict(kind='provisional_parameter_bridge', original_selection=copy.deepcopy(dict(slot)),
            original_seed_provenance=copy.deepcopy(spec['seed_provenance']), bridge=copy.deepcopy(dict(bridge_binding)),
            bridge_id=spec['id'], original_gamma=spec['original_gamma'], target_gamma=spec['gamma'],
            source_trial_index=index, reference_qualified=False))
    return acquisition.prepare_seed(resolved)
