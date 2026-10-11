"""EXP-013 metadata census and supplied-input, pre-preview features.

This module has no loaders, fits or integration entry points. The caller binds
the historical grid/manifests and qualifies/persists every fresh unperturbed
reference and sensitivity before supplying them here. No preview/target label
is accepted or read. Validation checks representation and input identity; it
does not replace the caller's numerical certificates or saved-evidence audit.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

import preview_shortlist_features as old
from hopfield_diagnostics import hopfield_rhs
from hopfield_information_design import build_atlas_geometry, select_information_anchors

SCHEMA_VERSION = 'EXP-013-fresh-prepreview-inputs-v1'
FEATURE_SCHEMA_VERSION = old.SCHEMA_VERSION
SHARED_FEATURE_NAMES = old.SHARED_FEATURE_NAMES
GEOMETRY_FEATURE_NAMES = old.GEOMETRY_FEATURE_NAMES
SOURCE_PATHS = list(dict.fromkeys(['tools/fresh_preview_features.py',
    'tools/hopfield_diagnostics.py', *old.SOURCE_PATHS]))
CORE_INDEX = (53, 60)
ANCHOR_COUNT = 24
PHASES = (675, 1350)
RADII = (.00025, .001, .004, .016)
CANDIDATE_COUNT = 18
EXCLUSION_RADIUS = .04
SENSITIVITY_TIMES = np.linspace(0., 300., 3001)
SENSITIVITY_TIMES.setflags(write=False)
CERTIFICATE_GATES = ('closure', 'phase_condition', 'section_is_maximum',
    'nonstationary', 'expected_peak_count', 'primitive', 'neutral_mode',
    'transverse_stability')


def _coordinate(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError('Finite real physical coordinates required')
    value = float(value)
    if not np.isfinite(value):
        raise ValueError('Finite real physical coordinates required')
    return value


def _prior_union(prior_manifests):
    """Exact physical deduplication, with every historical alias retained."""
    expected = {'EXP-006': 12, 'EXP-007': 12, 'EXP-008': 24}
    if not isinstance(prior_manifests, Mapping) or set(prior_manifests) != set(expected):
        raise ValueError('Exactly the EXP-006/007/008 prior manifests are required')
    exclusions, groups_by_experiment = {}, {}
    for experiment, count in expected.items():
        manifest = prior_manifests[experiment]
        anchors = manifest['anchors']
        if manifest['experiment_id'] != experiment or len(anchors) != count:
            raise ValueError(f'Complete {experiment} origin census required')
        seen_ids, groups = set(), set()
        for anchor in sorted(anchors, key=lambda a: a['id']):
            identity = anchor['id']
            if not isinstance(identity, str) or not identity or identity in seen_ids:
                raise ValueError('Unique nonempty prior aliases required within each manifest')
            seen_ids.add(identity)
            alpha, gamma = _coordinate(anchor['alpha']), _coordinate(anchor['gamma'])
            group = old.physical_group(alpha, gamma)
            if group in groups:
                raise ValueError('A prior manifest contains duplicate physical origins')
            groups.add(group)
            item = exclusions.setdefault(group, dict(alpha=alpha, gamma=gamma, group=group, aliases=[]))
            item['aliases'].append(dict(experiment_id=experiment, id=identity))
        groups_by_experiment[experiment] = groups
    if (len(exclusions) != 36 or not groups_by_experiment['EXP-007'] <= groups_by_experiment['EXP-008']
            or groups_by_experiment['EXP-006'] & groups_by_experiment['EXP-008']):
        raise ValueError('The exact 36-origin union and EXP-007/008 alias relationship are required')
    return list(exclusions.values())


def _atlas_contract(atlas):
    if (atlas.period.shape != (181, 181) or tuple(atlas.core_index) != CORE_INDEX
            or not np.array_equal(atlas.alpha[[0, -1]], [1.056, 1.08])
            or not np.array_equal(atlas.gamma[[0, -1]], [10.14, 10.55])):
        raise ValueError('Complete EXP-004 focus rectangle and core [53,60] required')


def select_origins(grid, prior_manifests):
    """Select 24 fixed representatives using only supplied historical metadata.

    ``prior_manifests`` maps EXP-006/007/008 to their decoded manifests. Only
    experiment identity and anchor id/alpha/gamma are inspected. The caller
    verifies their historical hashes. Exclusion order is experiment then
    sorted alias ID; this affects only the retained nearest-exclusion index.
    All eligible pixels, block memberships and physical alias identities are
    retained. No baseline success, new feature, preview or target is an input.
    """
    exclusions = _prior_union(prior_manifests)
    atlas = build_atlas_geometry(grid, CORE_INDEX)
    _atlas_contract(atlas)
    packet = select_information_anchors(grid, CORE_INDEX, exclusions,
        block_count=ANCHOR_COUNT, exclusion_radius=EXCLUSION_RADIUS)
    for index, (anchor, block) in enumerate(zip(packet['anchors'], packet['blocks'], strict=True)):
        identity, block_id = f'fresh-{index:02d}', f'fresh-block-{index:02d}'
        anchor.update(id=identity, anchor=identity, role='fresh', block_id=block_id,
            group=old.physical_group(anchor['alpha'], anchor['gamma']))
        block.update(id=block_id, anchor_id=identity)
    eligible = sorted(cell for block in packet['blocks'] for cell in block['members'])
    if len(eligible) != len({tuple(cell) for cell in eligible}) or len(packet['anchors']) != ANCHOR_COUNT:
        raise ValueError('Overlapping or incomplete spatial selection')
    packet.update(schema_version=SCHEMA_VERSION, experiment_id='EXP-013',
        source_experiment='EXP-004', exclusions=exclusions, eligible_pixels=eligible,
        exclusion_order='EXP-006, EXP-007, EXP-008; each manifest sorted by alias ID; first physical occurrence',
        alias_count=sum(len(e['aliases']) for e in exclusions))
    return packet


def _anchors(anchors):
    if len(anchors) != ANCHOR_COUNT:
        raise ValueError('All 24 planned fresh origins are required; no replacement or subset')
    result, groups, pixels = [], set(), set()
    for index, anchor in enumerate(anchors):
        identity = f'fresh-{index:02d}'
        if anchor['id'] != identity:
            raise ValueError('Fresh IDs must retain their fixed spatial leaf order')
        alpha, gamma = _coordinate(anchor['alpha']), _coordinate(anchor['gamma'])
        group = old.physical_group(alpha, gamma)
        if anchor['group'] != group or group in groups:
            raise ValueError('Fresh physical-origin alias or mismatched group')
        row, column = anchor['row'], anchor['column']
        if (any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in (row, column))
                or not (0 <= row < 181 and 0 <= column < 181) or (int(row), int(column)) in pixels):
            raise ValueError('Unique valid fresh atlas pixels required')
        groups.add(group); pixels.add((int(row), int(column)))
        result.append(dict(id=identity, alpha=alpha, gamma=gamma, group=group,
            row=int(row), column=int(column)))
    return result


def build_census(anchors):
    """Deterministic IDs/recipes before fresh directions or outcomes exist."""
    anchors = _anchors(anchors)
    cases, contexts, sensitivity_contexts, shams, sentinels = [], [], [], [], []
    for index, anchor in enumerate(anchors):
        for phase_index, phase in enumerate(PHASES):
            context_id = f'{anchor["id"]}-p{phase}'
            common = dict(role='fresh', anchor=anchor['id'], group=anchor['group'], phase=phase)
            sensitivity_contexts.append(dict(id=context_id, **common))
            shams.append(dict(id=f'{context_id}-sham', radius=0., **common))
            for radius_index, radius in enumerate(RADII):
                decision_id = f'{context_id}-r{radius:g}'
                candidate_ids = []
                for candidate in range(CANDIDATE_COUNT):
                    identity = f'{decision_id}-c{candidate:02d}'
                    candidate_ids.append(identity)
                    recipe = (dict(kind='uniform_16', index=candidate) if candidate < 16
                        else dict(kind='saved_tail_direction', sign=1 if candidate == 16 else -1))
                    cases.append(dict(id=identity, **common, radius=radius,
                        candidate=candidate, sensitivity_context=context_id, direction_recipe=recipe))
                    if radius_index == (index + 2*phase_index) % 4 and candidate == (index + 9*phase_index) % 18:
                        sentinels.append(identity)
                contexts.append(dict(id=decision_id, **common, radius=radius, candidate_ids=candidate_ids))
    counts = dict(origins=24, decision_contexts=192, sensitivity_contexts=48,
        candidate_cases=3456, sham_cases=48, independent_sentinels=48,
        native_candidate_traces=10368, native_sham_traces=144,
        independent_candidate_traces=96, total_short_traces=10608,
        policy_context_resolution_readouts=6144)
    if (len(cases) != counts['candidate_cases'] or len({c['id'] for c in cases}) != len(cases)
            or len(sentinels) != 48 or len(set(sentinels)) != 48):
        raise ValueError('Fresh metadata census changed')
    return dict(schema_version=SCHEMA_VERSION, cases=cases, decision_contexts=contexts,
        sensitivity_contexts=sensitivity_contexts, shams=shams, sentinel_ids=sentinels, counts=counts)


def _array(value, shape, name):
    result = np.asarray(value)
    if result.dtype != np.float64 or result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'Complete finite float64 {name} array {shape} required')
    return result


def _certificate(anchor, supplied):
    # Explicit field access: do not copy or inspect other fields of the supplied
    # certificate/mapping. They cannot introduce preview/target inputs.
    if (supplied['verified'] is not True or supplied['attracting'] is not True
            or supplied['primitive'] is not True or supplied['peak_count'] != 9
            or supplied['expected_peak_count'] != 9):
        raise ValueError('Verified attracting primitive nine-peak reference required')
    gates = supplied['gates']
    if set(gates) != set(CERTIFICATE_GATES) or any(gates[k] is not True for k in CERTIFICATE_GATES):
        raise ValueError('All unchanged reference certificate gates must pass')
    if (_coordinate(supplied['alpha']) != anchor['alpha']
            or _coordinate(supplied['gamma']) != anchor['gamma']):
        raise ValueError('Reference certificate belongs to a different physical origin')
    period, rho = _coordinate(supplied['period']), _coordinate(supplied['transverse_spectral_radius'])
    if period <= 0 or not 0 < rho < 1:
        raise ValueError('Positive period and attracting spectral radius required')
    times = _array(supplied['orbit_times'], (2701,), 'reference times')
    states = _array(supplied['orbit_states'], (2701, 3), 'reference states')
    activation = _array(supplied['orbit_activations'], (2701, 3), 'reference activations')
    if not np.array_equal(times, np.linspace(0., period, 2701)):
        raise ValueError('Reference must include exactly the original 2701 uniform endpoint samples')
    if not np.array_equal(activation, np.tanh(states)):
        raise ValueError('Reference activations differ from tanh of their bound states')
    return dict(period=period, transverse_spectral_radius=rho, orbit_activations=activation), states


def _sensitivity(anchor, phase, orbit_states, supplied):
    """Validate the actual local_parameter_sensitivity result plus tail_direction."""
    times = _array(supplied['times'], (3001,), 'unperturbed times')
    states = _array(supplied['states'], (3001, 3), 'unperturbed states')
    state_sensitivity = _array(supplied['state_sensitivity'], (3001, 3, 2), 'state sensitivity')
    sensitivity = _array(supplied['activation_sensitivity'], (3001, 3, 2), 'activation sensitivity')
    velocity = _array(supplied['activation_velocity'], (3001, 3), 'activation velocity')
    direction = _array(supplied['tail_direction'], (2,), 'realized tail direction')
    if not np.array_equal(times, SENSITIVITY_TIMES):
        raise ValueError('Unperturbed sensitivity must cover 0..300 at exact original times')
    if not np.array_equal(states[0], orbit_states[phase]):
        raise ValueError('Unperturbed initial state differs from the exact reference phase')
    if not np.array_equal(state_sensitivity[0], np.zeros((3, 2))):
        raise ValueError('Parameter sensitivity initial condition must be zero')
    derivative = 1. - np.tanh(states)**2
    if not np.array_equal(sensitivity, derivative[:, :, None] * state_sensitivity):
        raise ValueError('Activation sensitivity differs from its supplied state derivative')
    expected_velocity = derivative * np.asarray([hopfield_rhs(t, x, anchor['alpha'], anchor['gamma'])
                                                for t, x in zip(times, states, strict=True)])
    if not np.array_equal(velocity, expected_velocity):
        raise ValueError('Activation velocity differs from the unperturbed physical origin')
    if not np.isclose(np.linalg.norm(direction), 1., rtol=0, atol=1e-12):
        raise ValueError('Saved local direction must be finite and unit length')
    solver = supplied['solver']
    expected_solver = dict(method='DOP853', rtol=1e-10, atol=1e-12, max_step=.1)
    if set(solver) != set(expected_solver) or any(solver[k] != v for k, v in expected_solver.items()):
        raise ValueError('Unchanged unperturbed sensitivity solver specification required')
    mask = times > 75.
    return sensitivity[mask], velocity[mask], direction


def build_fresh_features(anchors, atlas, certificates, sensitivities):
    """Build all 37 shared + six geometry fields from supplied inputs only.

    ``certificates`` maps all 24 anchor IDs to complete qualified certificate
    dictionaries. ``sensitivities`` maps all 48 ``fresh-XX-pPHASE`` IDs to
    original unperturbed sensitivity results augmented with their realized
    ``tail_direction``. No state or direction is integrated/recomputed here;
    algebraic validation checks the saved arrays' origin/time conventions.
    The original shared feature function receives the exact times>75 tail.
    Both the 2700-sample reference convention and all 18 candidate IDs survive
    unchanged, including duplicate physical directions. Returned matrices own
    their memory and are read-only. Caller-owned bindings/timings stay outside
    this pure function and must accompany its sealed output.
    """
    anchors = _anchors(anchors)
    census = build_census(anchors)
    _atlas_contract(atlas)
    if (not isinstance(certificates, Mapping) or set(certificates) != {a['id'] for a in anchors}
            or not isinstance(sensitivities, Mapping)
            or set(sensitivities) != {c['id'] for c in census['sensitivity_contexts']}):
        raise ValueError('All and only 24 reference certificates and 48 sensitivity contexts required')
    rows, shared, geometry = [], [], []
    for anchor in anchors:
        # Validate geometry identity before consuming this origin's arrays.
        old.geometry_features(atlas, anchor, [1., 0.], RADII[0])
        certificate, orbit_states = _certificate(anchor, certificates[anchor['id']])
        for phase in PHASES:
            context_id = f'{anchor["id"]}-p{phase}'
            sensitivity, velocity, local = _sensitivity(anchor, phase, orbit_states, sensitivities[context_id])
            directions = old.direction_candidates('uniform_16') + old.direction_candidates('local_signed', local_direction=local)
            for radius in RADII:
                for candidate, action in enumerate(directions):
                    direction = action['direction']
                    shared.append(old.shared_features(anchor, phase, direction, radius, certificate, sensitivity, velocity))
                    geometry.append(old.geometry_features(atlas, anchor, direction, radius)['values'])
                    rows.append(dict(id=f'{context_id}-r{radius:g}-c{candidate:02d}',
                        role='fresh', anchor=anchor['id'], group=anchor['group'], phase=phase,
                        radius=radius, candidate=candidate, direction=list(direction)))
    shared, geometry = np.asarray(shared, dtype=np.float64), np.asarray(geometry, dtype=np.float64)
    if ([r['id'] for r in rows] != [r['id'] for r in census['cases']]
            or shared.shape != (3456, 37) or geometry.shape != (3456, 6)
            or not np.isfinite(shared).all() or not np.isfinite(geometry).all()):
        raise ValueError('Full fresh pre-preview feature census/schema differs')
    shared.setflags(write=False); geometry.setflags(write=False)
    return dict(schema_version=FEATURE_SCHEMA_VERSION, adapter_schema_version=SCHEMA_VERSION,
        shared_names=list(SHARED_FEATURE_NAMES), geometry_names=list(GEOMETRY_FEATURE_NAMES),
        rows=rows, shared=shared, geometry=geometry)
