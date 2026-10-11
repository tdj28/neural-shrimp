"""Pure EXP-019 bridge from globally refined references to frozen two-write code.

The caller verifies the complete EXP-018 receipt and packet byte/semantic hashes
before passing a bundle. This adapter checks the selected numerical identities;
it performs no IO, integration, interpolation, fitting or phase normalization.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
import re

import numpy as np

from hopfield_diagnostics import hopfield_rhs
import two_write_acquisition as legacy

CANONICAL_METHOD = 'tight-dop853'
CANONICAL_VERIFICATION = 'tight-dop853--tight-dop853'
SETTINGS = ('core', 'failed-near-edge')
SOURCE_FIELDS = ('experiment_id', 'source_commit', 'plan_sha256',
    'receipt_sha256', 'reference_id', 'packet_sha256', 'packet_semantic_sha256')


def _same(value, expected, name):
    if isinstance(expected, np.ndarray):
        actual = legacy._array(value, name, expected.shape)
        if not np.array_equal(actual, expected):
            raise ValueError(name+' differs')
    elif type(value) is not type(expected) or value != expected:
        raise ValueError(name+' differs')


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isfinite(value):
        raise ValueError(name+' must be a finite scalar')
    return float(value)


def _source(source_binding, setting):
    if not isinstance(source_binding, Mapping) or set(source_binding) != set(SOURCE_FIELDS):
        raise ValueError('Exact EXP-018 source binding fields required')
    _same(source_binding['experiment_id'], 'EXP-018', 'source experiment')
    _same(source_binding['reference_id'], setting+'--reference-pair', 'source reference')
    for key, length in [('source_commit', 40), ('plan_sha256', 64),
            ('receipt_sha256', 64), ('packet_sha256', 64), ('packet_semantic_sha256', 64)]:
        value = source_binding[key]
        if type(value) is not str or re.fullmatch('[0-9a-f]{'+str(length)+'}', value) is None:
            raise ValueError('Valid source '+key+' required')
    return copy.deepcopy(dict(source_binding))


def _multipliers(certificate):
    values = copy.deepcopy(certificate.get('transverse_multipliers', []))
    if not isinstance(values, list) or len(values) not in (0, 2):
        raise ValueError('Zero or two transverse multiplier records required')
    for value in values:
        if not isinstance(value, Mapping) or set(value) != {'real', 'imag', 'modulus'}:
            raise ValueError('Complex transverse multiplier record required')
        real, imag, modulus = (_number(value[k], 'multiplier '+k) for k in ('real', 'imag', 'modulus'))
        if modulus < 0 or not np.isclose(modulus, np.hypot(real, imag), rtol=1e-12, atol=1e-14):
            raise ValueError('Transverse multiplier modulus differs')
    dominant = max(values, key=lambda value: value['modulus']) if values else None
    rho = float(dominant['real']) if dominant is not None and dominant['imag'] == 0 else None
    return values, rho


def prepare_context(reference_pair, phase_row, *, source_binding, rhs=hopfield_rhs):
    """Use one canonical exact dense state/Z for both future 8T algorithms.

    ``source_binding`` contains the seven fields in SOURCE_FIELDS. Its hashes
    identify caller-verified immutable artifacts; they are not recomputed here.
    ``phase_row`` contains phase_index, phase_fraction and the inherited split.
    The same24 fractions are regression phases, not a newly held-out cohort.
    """
    pair = reference_pair
    if not isinstance(pair, Mapping) or pair.get('complete') is not True or pair.get('qualified') is not True:
        raise ValueError('Complete qualified global reference pair required')
    _same(pair.get('status'), 'complete', 'reference status')
    _same(pair.get('schema'), 'periodic-reference-pair-v1', 'reference schema')
    _same(pair.get('kind'), 'reference', 'reference kind')
    setting = pair.get('setting')
    if setting not in SETTINGS:
        raise ValueError('Known original setting required')
    _same(pair.get('id'), setting+'--reference-pair', 'reference identity')
    _same(pair.get('canonical_method'), CANONICAL_METHOD, 'canonical method')
    provenance_binding = _source(source_binding, setting)
    if not isinstance(pair.get('gates'), Mapping) or not pair['gates'] or any(v is not True for v in pair['gates'].values()):
        raise ValueError('Every reference-pair prerequisite must qualify')
    canonical = pair['canonical']
    _same(canonical.get('source_verification'), CANONICAL_VERIFICATION, 'canonical verification')
    _same(canonical.get('normalization_count'), 1, 'one original global normalization')
    period = legacy._positive(canonical['period'], 'canonical period')
    section = legacy._array(canonical['state'], 'canonical section', (3,))
    terminal = legacy._array(canonical['terminal_adjoint'], 'canonical terminal gradient', (3,))
    _same(canonical['neutral']['phase_gradient'], terminal, 'canonical gradient')
    if set(pair['bases']) != {'tight-dop853', 'tight-rk45'}:
        raise ValueError('Both canonical reference reconstructions required')
    for method, candidate_base in pair['bases'].items():
        if candidate_base.get('complete') is not True or candidate_base.get('qualified') is not True:
            raise ValueError('Both canonical reference reconstructions must qualify')
        _same(candidate_base['method'], method, 'base method')
        _same(candidate_base['id'], setting+'--'+method, 'base identity')
        _same(candidate_base['period'], period, 'base canonical period')
        _same(candidate_base['canonical_state'], section, 'base canonical section')
        _same(candidate_base['global_terminal_adjoint'], terminal, 'shared terminal gradient')
    base = pair['bases'][CANONICAL_METHOD]
    _same(base.get('schema'), 'refined-phase-base-v1', 'canonical base schema')
    _same(base.get('status'), 'complete', 'canonical base status')
    _same(base.get('new_normalizations'), 0, 'no base gradient repair')
    _same(base['raw_endpoints']['final_z'], terminal, 'unchanged backward terminal')
    alpha, gamma = (_number(pair[name], name) for name in ('alpha', 'gamma'))
    for name, value in [('alpha', alpha), ('gamma', gamma)]:
        _same(base[name], value, 'base '+name)
    if not isinstance(phase_row, Mapping) or set(phase_row) != {'phase_index', 'phase_fraction', 'split'}:
        raise ValueError('Exact declared phase row required')
    index = phase_row['phase_index']
    if type(index) is not int or index not in range(24):
        raise ValueError('Original phase index0..23 required')
    if len(base['phases']) != 24 or len(base['phase_rows']) != 24:
        raise ValueError('All24 original phases required')
    exact = base['exact_phase_arrays']
    for key, shape in [('times', (24,)), ('states', (24, 3)), ('z', (24, 3)), ('derivatives', (24, 3))]:
        legacy._array(exact[key], 'exact dense '+key, shape)
    for i, phase in enumerate(base['phases']):
        fraction = _number(phase['phase_fraction'], 'phase fraction')
        if not 0 <= fraction < 1:
            raise ValueError('Original phase fraction must be in [0,1)')
        metadata = dict(context_id=f'{setting}--p{i:02d}', phase_index=i,
            phase_fraction=fraction, split='development' if i < 8 else 'heldout', phase_epoch=fraction*period)
        if set(base['phase_rows'][i]) != set(metadata):
            raise ValueError('Original phase metadata fields differ')
        for key, value in metadata.items():
            _same(phase[key], value, 'phase '+key)
            _same(base['phase_rows'][i][key], value, 'phase row '+key)
        for pk, ak in [('state', 'states'), ('z', 'z'), ('flow', 'derivatives'), ('phase_epoch', 'times')]:
            value = exact[ak][i]
            if pk == 'phase_epoch':
                value = float(value)
            _same(phase[pk], value, 'exact dense '+pk)
    phase = base['phases'][index]
    for key, value in phase_row.items():
        _same(phase[key], value, 'declared '+key)
    state = phase['state'].copy()
    z = phase['z'].copy()
    flow = legacy._array(rhs(phase['phase_epoch'], state, alpha, gamma), 'physical phase flow', (3,)).copy()
    if not np.allclose(flow, phase['flow'], rtol=1e-12, atol=1e-14):
        raise ValueError('Saved dense phase flow differs from physical RHS')
    znorm = float(np.linalg.norm(z))
    if not np.isfinite(znorm) or znorm == 0:
        raise ValueError('Nonzero finite phase gradient required')
    unit = z/znorm
    projector = np.eye(3)-np.outer(unit, unit)
    coordinate = int(np.argmax(np.linalg.norm(projector, axis=0)))
    direction = projector[:, coordinate]/np.linalg.norm(projector[:, coordinate])
    verification = pair['verifications'][CANONICAL_VERIFICATION]
    if verification.get('complete') is not True or verification.get('qualified') is not True or verification['certificate'].get('qualified') is not True:
        raise ValueError('Qualified canonical verification certificate required')
    _same(verification['candidate_method'], CANONICAL_METHOD, 'certificate candidate')
    _same(verification['method'], CANONICAL_METHOD, 'certificate verifier')
    multipliers, rho = _multipliers(verification['certificate'])
    context = dict(id=phase['context_id'], setting=setting, phase_index=index,
        split=phase['split'], alpha=alpha, gamma=gamma, period=period,
        phase_fraction=phase['phase_fraction'], phase_epoch=phase['phase_epoch'],
        initial=state, phase_gradient=z, flow=flow, direction=direction,
        nominal_event_phase_gradient=z.copy(), null_coordinate_index=coordinate,
        normalization_residual=float(z@flow-1.), direction_phase_leakage=float(z@direction),
        transverse_multipliers=multipliers, rho=rho,
        rho_provenance='signed dominant real transverse multiplier of the new canonical DOP853 candidate/DOP853 verification certificate; null if unavailable/nonreal',
        context_schema='refined-two-write-context-v1', new_normalizations=0,
        phase_cohort='reused EXP-016/018 regression fractions; inherited split labels, no fresh holdout',
        provenance=dict(initial='new canonical DOP853 exact solver-dense phase state',
            adjoint='new canonical DOP853 exact dense adjoint, unchanged after its one global construction normalization',
            event_adjoint='same unchanged phase adjoint at nominal phase_epoch+T; approximate periodic reference',
            projection_flow='physical RHS of independently integrated fresh8T primary reference event state',
            reference_qualified=True, reference_status='complete', source_binding=provenance_binding,
            canonical_source_verification=CANONICAL_VERIFICATION, canonical_base_id=base['id'],
            source_global_normalization_count=1, new_normalizations=0,
            both_fresh_reference_methods_use_identical_initial_state=True,
            eight_period_reference_qualified=False))
    legacy._context(context)
    return context
