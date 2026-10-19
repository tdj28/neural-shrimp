"""Source preparation: independent exact saved-scalar/coverage consistency checks.

This is NOT an executable evidence auditor: no files, Git, processes, model or
backend are accessed. The caller supplies decoded JSON records and pin records.
Their authenticity is NOT established here. Even success never admits a target
operator bound. A reviewed future authority wrapper must supply the exact joins
listed in tube-saved-audit-preparation.md before interpreting L_candidate.
"""
from fractions import Fraction
import hashlib
import json
import re

DIM = 896
RAW = '/data2/neural-shrimp-data/codex/coexistence/tube-target-001'
FLAGS = {'trajectory_enclosure': False, 'certificate_passed': False,
         'reachability_proved': False, 'conditional_operator_bound_only': True}
NAMES = ('admission.json', 'selection.json', 'prefix-binding.json',
         'tube-inputs.json', 'actual-receiver.json', 'tube-operator.json',
         'coverage.json', 'status.json')
INPUT_FIELDS = ('centres', 'nominal_radius', 'live_primal', 'live_directions',
                'dimension', 'direction_count')
STEP_CHECKS = {'exact_input_chain', 'full_image_endpoints', 'exact_image_midpoints',
               'signed_defect_contains_image_minus_next',
               'saved_upper_contains_all_defect_endpoints'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def integer(value, expected, label):
    require(type(value) is int and value == expected, label)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def qr(value):
    return {'numerator': value.numerator, 'denominator': value.denominator}


def rational(value):
    require(type(value) is dict and set(value) == {'numerator', 'denominator'},
            'exact rational schema')
    n, d = value['numerator'], value['denominator']
    require(type(n) is int and type(d) is int and d > 0, 'exact rational integers')
    result = Fraction(n, d)
    require(qr(result) == value, 'canonical rational representation')
    return result


def dyadic(value):
    require(type(value) is list and len(value) == 2 and
            all(type(v) is int for v in value), 'finite native integer dyadic pair')
    return Fraction(value[0]) * Fraction(2) ** value[1]


def ball(value):
    require(type(value) is dict and set(value) == {'mid', 'rad'}, 'native ball schema')
    m, r = dyadic(value['mid']), dyadic(value['rad'])
    require(r >= 0, 'nonnegative native radius')
    return m, r, m-r, m+r


def vector(value, exact=False):
    require(type(value) is list and len(value) == DIM, 'all 896 native components')
    rows = [ball(x) for x in value]
    if exact:
        require(all(x[1] == 0 for x in rows), 'exact saved centre coordinates')
    return rows


def column(value):
    require(type(value) is list and len(value) == DIM and
            all(type(x) is list and len(x) == 1 for x in value), 'native 896 by 1 column')
    return [ball(x[0]) for x in value]


def flags(value):
    require(type(value) is dict and all(value.get(k) is v for k, v in FLAGS.items()),
            'conditional-only saved scope flags')


def thin_flags(value):
    require(value.get('trajectory_enclosure') is False and
            value.get('certificate_passed') is False, 'thin scope flags')


def pin(value):
    require(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}, 'pin schema')
    require(type(value['path']) is str and value['path'].startswith('/') and
            type(value['bytes']) is int and value['bytes'] >= 0 and
            type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']),
            'full explicit pin fields')
    return value


def first_step(initial, step, report, selected, upstream_pins):
    """Recombine first-step arithmetic; report and upstream pins are untrusted I/O inputs.

    The future wrapper must authenticate successful audit002 and its BF16 row198
    authority before using the e1 interpretation; this function does not read the
    checkpoint and does not turn a passing JSON claim into source authority.
    """
    flags(selected); thin_flags(initial); thin_flags(step); thin_flags(report)
    require(report.get('reachability_proved') is False and report['status'] == 'complete_success'
            and report['original_partial_evidence_preserved'] is True,
            'complete successful original thin audit required')
    integer(report['steps_checked'], 20, 'all original 20 point steps audited')
    require(type(report['steps']) is list and len(report['steps']) == 20, 'full audit rows')
    integer(initial['step'], 0, 'initial index'); integer(step['step'], 1, 'first step index')
    c0, c1 = vector(initial['centre'], True), vector(step['next_centre'], True)
    require(vector(step['input_centre'], True) == c0, 'exact first input chain')
    images, defects = vector(step['full_image']), vector(step['defects'])
    for image, centre, defect in zip(images, c1, defects):
        require(centre[0] == image[0], 'original exact image midpoint selection')
        require(defect[2] <= image[2]-centre[0] and defect[3] >= image[3]-centre[0],
                'full signed defect must contain image minus next centre')
    maximum = max(max(abs(x[2]), abs(x[3])) for x in defects)
    _, _, lower, delta = ball(step['defect_inf_upper'])
    require(lower >= 0 and delta >= maximum, 'saved bound covers every signed defect')
    radius = 2*delta if delta > 0 else Fraction(1, 2**2048)
    require(radius > 0 and radius.denominator & (radius.denominator-1) == 0,
            'positive exact dyadic radius')
    require(selected['schema'] == 'codex-task33-first-tube-selection-v1' and
            selected['centre'] == step['next_centre'] and
            selected['centre_canonical_sha256'] == canonical(step['next_centre']) and
            rational(selected['delta0']) == delta and rational(selected['radius']) == radius and
            selected['radius_rule'] == '2*delta0 if positive; otherwise exact 2^-2048' and
            selected['incoming_error_le_radius'] is True and delta <= radius,
            'exact selected centre/radius/error rule')
    for field in ('initial_file', 'step1_file', 'audit_result'):
        require(pin(selected[field]) == pin(upstream_pins[field]), 'selected upstream pin differs')
    ia, entry, checked = report['initial_state'], report['entry_candidate'], report['steps'][0]
    integer(ia['component_count'], DIM, 'audited full initial dimension')
    integer(checked['component_count'], DIM, 'audited full first step dimension')
    integer(checked['step'], 1, 'audited first row index'); integer(entry['step'], 1, 'entry index')
    require(ia['file'] == selected['initial_file'] and ia['exact_checkpoint_row_match'] is True and
            entry['file'] == checked['file'] == selected['step1_file'] and
            set(checked['checks']) == STEP_CHECKS and
            all(v is True for v in checked['checks'].values()), 'audited first-step pin/check joins')
    require(rational(checked['computed_defect_endpoint_max']) == maximum and
            rational(checked['saved_defect_inf_upper_exact']) == rational(entry['delta0']) ==
            rational(entry['e1_upper']) == delta and
            checked['next_centre_canonical_sha256'] == entry['next_centre_canonical_sha256'] ==
            selected['centre_canonical_sha256'], 'audited exact first-step scalars')
    embedding = ia['embedding']
    require(selected['embedding'] == embedding and embedding['tensor'] == 'model.embed_tokens.weight'
            and embedding['stored_dtype'] == 'BF16' and embedding['shape'] == [151936, DIM],
            'audited checkpoint row description')
    integer(embedding['row'], 198, 'exact row198'); integer(embedding['row_bytes'], 2*DIM, 'row bytes')
    require(embedding['exact_rational_values_sha256'] == canonical([qr(v[0]) for v in c0]),
            'audited exact embedding row value hash')
    return {'delta0': qr(delta), 'radius': qr(radius), 'defect_endpoint_max': qr(maximum),
            'centre_canonical_sha256': selected['centre_canonical_sha256']}


def inputs(value, selected):
    flags(value)
    integer(value['dimension'], DIM, 'full tube dimension')
    integer(value['direction_count'], 1, 'single full cube column')
    integer(value['precision'], 2048, '2048 bit receiver precision')
    require(value['centres'] == selected['centre'] and value['nominal_radius'] == selected['radius'],
            'selected exact tube equals actual tube')
    radius = rational(value['nominal_radius'])
    require(radius > 0 and radius.denominator & (radius.denominator-1) == 0, 'positive dyadic tube')
    centres, primal, directions = vector(value['centres'], True), vector(value['live_primal']), column(value['live_directions'])
    for c, p, v in zip(centres, primal, directions):
        require(p[2] <= c[0]-radius and p[3] >= c[0]+radius, 'nominal closed cube containment')
        require(v[2] <= -1 and v[3] >= 1, 'independent unit direction cube containment')


def coverage(value):
    flags(value)
    for key in ('started_layers', 'ended_layers'):
        require(type(value[key]) is list and all(type(v) is int for v in value[key]) and
                value[key] == list(range(24)), 'complete ordered 24 layer coverage')
    expected = [[k, start, min(start+1024, 151936)] for k in ('head_logits', 'head_derivatives')
                for start in range(0, 151936, 1024)]
    rows = value['head_passes']
    require(type(rows) is list and all(type(r) is list and len(r) == 3 and
            type(r[0]) is str and type(r[1]) is int and type(r[2]) is int for r in rows)
            and rows == expected, 'both complete ordered 149 chunk head passes')
    require(value['actual_receiver_equals_original'] is True and
            value['all_direction_cube_retained'] is True, 'successful full receiver coverage')


def successful_records(records, inventory, receipt, expected_binding):
    """Structural success/inventory gate, deliberately NOT file authentication.

    Receipt and expected_binding must later come from independent original-source
    authority. No caller-provided boolean can override a failed original outcome.
    """
    require(type(records) is dict and set(records) == set(NAMES), 'exact eight successful output records')
    require(type(inventory) is dict and set(inventory) == set(NAMES), 'exact full pinned successful inventory')
    for name in NAMES:
        flags(records[name])
        require(pin(inventory[name])['path'] == RAW+'/'+name, 'exact owned output pin allocation')
    require(type(receipt['returncode']) is int and receipt['returncode'] == 0 and
            receipt['failure'] is None and all(receipt[k] is True for k in
            ('worker_reaped', 'worker_identity_absent', 'worker_group_session_absent')),
            'successful complete original supervision closure required')
    require(type(receipt['output_files']) is list and len(receipt['output_files']) == len(NAMES),
            'complete original output inventory count')
    require({p['path'] for p in receipt['output_files']} == {p['path'] for p in inventory.values()} and
            all(pin(p) == inventory[p['path'].removeprefix(RAW+'/')] for p in receipt['output_files']),
            'original receipt exact output pin join')
    require(records['admission.json'] == expected_binding, 'exact expected original admission binding')
    s = records['status.json']
    require(s['schema'] == 'codex-task33-one-tube-status-v1' and
            s['status'] == 'operator_measurement_completed' and s['full_domain_gates_succeeded'] is True,
            'overall complete domain/coverage success required')
    integer(s['target_calls'], 1, 'one original target call')
    require(s['binding'] == expected_binding and s['selection'] == inventory['selection.json'],
            'status original admission/selection join')
    require(type(s['artifacts']) is list and len(s['artifacts']) == len(NAMES)-1 and
            {p['path'] for p in s['artifacts']} == {inventory[k]['path'] for k in NAMES if k != 'status.json'} and
            all(pin(p) == inventory[p['path'].removeprefix(RAW+'/')] for p in s['artifacts']),
            'status full preceding artifact pin join')


def check_bundle(records, inventory, receipt, expected_binding, initial, step, report, upstream_pins):
    """Return only unauthenticated scalar consistency; never an admitted bound."""
    successful_records(records, inventory, receipt, expected_binding)
    selection = records['selection.json']
    first = first_step(initial, step, report, selection, upstream_pins)
    for name in ('tube-inputs.json', 'actual-receiver.json', 'tube-operator.json'):
        inputs(records[name], selection)
    original = records['tube-inputs.json']
    for name in ('actual-receiver.json', 'tube-operator.json'):
        require(all(records[name][k] == original[k] for k in INPUT_FIELDS),
                'all original native input/actual receiver/operator fields identical')
    operator = records['tube-operator.json']
    require(operator['schema'] == 'conditional-whole-tube-infinity-operator-v1', 'operator schema')
    for value in (original, operator): integer(value['threads'], 1, 'one arithmetic thread')
    vector(operator['full_output'])
    derivatives = column(operator['signed_derivative'])
    require(type(operator['row_abs_endpoint_upper']) is list and len(operator['row_abs_endpoint_upper']) == DIM,
            'all 896 signed row bounds')
    recomputed = [max(abs(v[2]), abs(v[3])) for v in derivatives]
    require([rational(v) for v in operator['row_abs_endpoint_upper']] == recomputed,
            'every original signed row endpoint maximum exact')
    maximum = max(recomputed)
    require(rational(operator['L']) == rational(records['status.json']['L']) == maximum,
            'exact maximum L and successful status join')
    coverage(records['coverage.json'])
    return {'schema': 'codex-task33-tube-scalar-helper-v1',
            'status': 'scalar_and_coverage_consistent_only', 'components_checked': DIM,
            'layer_pairs_checked': 24, 'head_chunks_checked': [149, 149],
            'first_step': first, 'L_candidate': qr(maximum),
            'original_file_authentication_performed': False,
            'original_source_authentication_performed': False, 'operator_bound_admitted': False,
            **FLAGS, 'scope': 'Pure decoded-record checks only. The authority wrapper is not implemented; no actual target evidence was read. No trajectory, reachability or certificate.'}
