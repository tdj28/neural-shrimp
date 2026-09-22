"""EXP-022's pure, historical-metadata-only six-slot selection.

No files, integrations, eigensolvers or finite-response data are accessed here.
The caller verifies original receipt inventories before supplying parsed inputs.
Malformed metadata raises ValueError; valid absence is retained in its slot.
"""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from numbers import Real
import re


SETTINGS = ('core', 'failed-near-edge')
BRANCHES = {'core': 'core-direction-+1',
            'failed-near-edge': 'failed-near-edge-direction--1'}
TARGETS = (0.25, 0.50)
CALIPER = Fraction(1, 40)
OLD_GATES = ('closure', 'phase', 'neutral_flow', 'primitive_nine_peaks')
CROSSINGS = ('plus_one_bracket', 'minus_one_bracket', 'unit_circle_bracket')


def contract():
    return dict(schema='cycle-reference-selection-v1', slots=6, anchors=2,
        candidates=4, branches=deepcopy(BRANCHES), targets=list(TARGETS),
        caliper_exact='1/40', distances='exact rational values of saved binary64 real parts',
        ties='smallest original node index', target_order=list(TARGETS),
        prefix='stop before first non-attracting, unstable or recorded crossing node',
        recorded_gates='all recorded booleans true, including four required old gates',
        dominant='unique largest recorded modulus; imaginary part exactly zero; real part positive',
        seed='saved first_maximum_state; EXP-018 anchors use canonical tight-dop853 maximum',
        missing_seed='rank first; chosen absent maximum is missing_seed_evidence, no fallback',
        no_reuse=True, replacement_search=False, checkpoint_gate=False,
        duplicate_scope='exact historical node identity only; physical loop duplicates require later qualification',
        scientific_qualification=False)


def slot_census():
    rows = [dict(id='anchor-'+s, cycle_id='anchor-'+s, role='anchor',
                 source_setting=s, target_rho=None, branch_id=None) for s in SETTINGS]
    for s in SETTINGS:
        for target, suffix in zip(TARGETS, ('025', '050'), strict=True):
            name = s+'-rho'+suffix
            rows.append(dict(id=name, cycle_id=name, role='candidate', source_setting=s,
                             target_rho=target, branch_id=BRANCHES[s]))
    return rows


def _number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(name+' must be a real number, not a boolean')
    value = float(value)
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(name+' must be finite'+(' and positive' if positive else ''))
    return value


def _bool(value, name):
    if type(value) is not bool:
        raise ValueError(name+' must be a JSON boolean')
    return value


def _vector(value, name):
    # Loaded canonical packets use float64 ndarray; historical JSON uses lists.
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(name+' must contain exactly three finite scalars')
    return [_number(v, name) for v in value]


def _gates(value, name, required=()):
    if not isinstance(value, dict) or not value or not set(required) <= value.keys():
        raise ValueError(name+' has missing gates')
    return {str(k): _bool(v, name+'.'+str(k)) for k, v in value.items()}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _binding(value, experiment):
    if not isinstance(value, dict) or value.get('experiment_id') != experiment:
        raise ValueError('source binding experiment mismatch')
    for key, length in (('source_commit', 40), ('receipt_sha256', 64)):
        if not isinstance(value.get(key), str) or not re.fullmatch('[0-9a-f]{'+str(length)+'}', value[key]):
            raise ValueError('invalid source binding '+key)
    artifacts = value.get('artifacts')
    if not isinstance(artifacts, list) or not artifacts:
        raise ValueError('source binding requires original artifact hashes')
    paths = []
    for row in artifacts:
        if not isinstance(row, dict) or set(row) != {'path', 'sha256', 'bytes'}:
            raise ValueError('invalid artifact binding fields')
        if not isinstance(row['path'], str) or not row['path'] or '..' in row['path'].split('/'):
            raise ValueError('invalid artifact path')
        if not isinstance(row['sha256'], str) or not re.fullmatch('[0-9a-f]{64}', row['sha256']):
            raise ValueError('invalid artifact hash')
        if type(row['bytes']) is not int or row['bytes'] <= 0:
            raise ValueError('invalid artifact byte count')
        paths.append(row['path'])
    if len(set(paths)) != len(paths):
        raise ValueError('duplicate source artifact path')
    return deepcopy(value)


def _empty(spec, status, reasons):
    return dict(deepcopy(spec), available=False, status=status, reasons=list(reasons),
                node_id=None, node_index=None, alpha=None, gamma=None, period=None,
                first_maximum_state=None, dominant_rho=None, seed_provenance=None)


def _anchor(spec, packet, binding):
    s = spec['source_setting']
    if not isinstance(packet, dict) or packet.get('setting') != s or packet.get('id') != s+'--reference-pair':
        raise ValueError('anchor identity mismatch')
    if packet.get('canonical_method') != 'tight-dop853':
        raise ValueError('anchor canonical method mismatch')
    complete = _bool(packet.get('complete'), 'anchor.complete')
    qualified = _bool(packet.get('qualified'), 'anchor.qualified')
    gates = _gates(packet.get('gates'), 'anchor.gates')
    if qualified and (not complete or not all(gates.values()) or packet.get('status') != 'complete'):
        raise ValueError('anchor qualification contradicts recorded state')
    if not complete or not qualified:
        row = _empty(spec, 'invalid_anchor', ['anchor_not_complete_and_qualified'])
        row['seed_provenance'] = dict(source_binding=binding)
        return row
    canonical = packet.get('canonical', {})
    if canonical.get('source_verification') != 'tight-dop853--tight-dop853':
        raise ValueError('anchor canonical verification mismatch')
    row = dict(deepcopy(spec), available=True, status='selected', reasons=[], node_id=None,
        node_index=None, alpha=_number(packet.get('alpha'), 'anchor.alpha'),
        gamma=_number(packet.get('gamma'), 'anchor.gamma'),
        period=_number(canonical.get('period'), 'anchor.period', positive=True),
        first_maximum_state=_vector(canonical.get('state'), 'anchor.canonical.state'), dominant_rho=None)
    row['seed_provenance'] = dict(source_binding=binding, original_reference_id=packet['id'],
        state_field='canonical.state', period_field='canonical.period',
        canonical_method='tight-dop853', original_qualified=True,
        seed_metadata_sha256=_digest({k: row[k] for k in ('alpha','gamma','period','first_maximum_state')}))
    return row


def _branch(branch_id, branch):
    setting = next(s for s, b in BRANCHES.items() if b == branch_id)
    if not isinstance(branch, dict) or type(branch.get('direction')) is not int or branch['direction'] != (1 if setting == 'core' else -1):
        raise ValueError('branch direction mismatch')
    alpha = _number(branch.get('alpha_seed'), 'branch.alpha_seed')
    gamma = _number(branch.get('gamma_seed'), 'branch.gamma_seed')
    if branch.get('parameterization') != 'alpha=alpha_seed; gamma=gamma_seed+0.41*s':
        raise ValueError('branch parameterization mismatch')
    nodes = branch.get('nodes')
    if not isinstance(nodes, list) or not nodes:
        raise ValueError('branch nodes must be a nonempty original ordered list')
    audits, stopped = [], None
    for index, node in enumerate(nodes):
        if not isinstance(node, dict) or type(node.get('index')) is not int or node['index'] != index:
            raise ValueError('original node indices must be unique, ordered and contiguous from zero')
        node_alpha = _number(node.get('alpha'), 'node.alpha')
        node_gamma = _number(node.get('gamma'), 'node.gamma')
        parameter = _number(node.get('parameter'), 'node.parameter')
        if node_alpha != alpha or node_gamma != gamma + 0.41*parameter:
            raise ValueError('node parameters disagree with frozen branch parameterization')
        period = _number(node.get('period'), 'node.period', positive=True)
        _vector(node.get('state'), 'node.state')
        maximum = None if node.get('first_maximum_state') is None else _vector(node['first_maximum_state'], 'node.first_maximum_state')
        gates = _gates(node.get('gates'), 'node.gates', OLD_GATES)
        if _bool(node.get('verified'), 'node.verified') != all(gates.values()):
            raise ValueError('node verified flag contradicts gates')
        attracting = _bool(node.get('attracting'), 'node.attracting')
        unstable = _bool(node.get('unstable'), 'node.unstable')
        radius = _number(node.get('transverse_spectral_radius'), 'node.radius')
        modes = node.get('transverse_multipliers')
        if not isinstance(modes, list) or len(modes) != 2:
            raise ValueError('exactly two recorded transverse modes required')
        values = []
        for mode in modes:
            if not isinstance(mode, dict) or not {'real','imag','modulus'} <= mode.keys():
                raise ValueError('malformed transverse multiplier')
            real, imag, modulus = (_number(mode[k], 'mode.'+k) for k in ('real','imag','modulus'))
            # This is schema consistency, not a newly estimated uncertainty gate.
            if modulus < 0 or not math.isclose(modulus, math.hypot(real, imag), rel_tol=2e-15, abs_tol=0):
                raise ValueError('recorded multiplier modulus inconsistent with components')
            values.append((real, imag, modulus))
        if radius != max(m[2] for m in values) or attracting != (radius < 1-1e-5) or unstable != (radius > 1+1e-5):
            raise ValueError('recorded stability flags inconsistent with old thresholds')
        flags = node.get('bracket_flags')
        if not isinstance(flags, list) or any(f not in CROSSINGS for f in flags) or len(set(flags)) != len(flags):
            raise ValueError('malformed recorded crossing flags')
        stops = ([] if attracting else ['not_attracting']) + (['unstable'] if unstable else []) + list(flags)
        if stopped is None and stops:
            stopped = dict(index=index, reasons=stops)
        reasons = []
        if stopped is not None:
            reasons.append('outside_initial_attracting_prefix')
        reasons += ['old_gate_failed:'+k for k, v in gates.items() if not v]
        seed_reasons = ['absent_first_maximum_state'] if maximum is None else []
        largest = [m for m in values if m[2] == radius]
        dominant = largest[0] if len(largest) == 1 else None
        if dominant is None:
            reasons.append('nonunique_dominant_modulus')
        elif dominant[1] != 0 or dominant[0] <= 0:
            reasons.append('dominant_not_exact_real_positive')
        audits.append(dict(node_id=branch_id+'--node-'+str(index), index=index,
            in_initial_attracting_prefix=stopped is None, eligible=not reasons, reasons=reasons,
            seed_available=not seed_reasons, seed_reasons=seed_reasons,
            alpha=node_alpha, gamma=node_gamma, period=period, first_maximum_state=maximum,
            dominant_rho=None if dominant is None else dominant[0],
            recorded_modes=deepcopy(modes), recorded_gates=gates, bracket_flags=list(flags),
            node_metadata_sha256=_digest(node)))
    return dict(branch_id=branch_id, prefix_stop=stopped, nodes=audits)


def select_cycles(branches, anchors, *, bindings):
    """Return six slots and complete node/target decisions from supplied metadata.

    branches maps the two literal branch IDs to full EXP-006 branch JSON.
    anchors maps the two source settings to EXP-018 reference-pair packets (or
    their identity/qualification/alpha/gamma/canonical projection). bindings has
    four keys: the two branch IDs and two anchor slot IDs. Its entries contain
    experiment_id, source_commit, receipt_sha256 and original artifact triples.
    Hash/receipt membership verification is the caller's explicit responsibility.
    """
    specs = slot_census()
    expected_bindings = set(BRANCHES.values()) | {'anchor-'+s for s in SETTINGS}
    if not isinstance(branches, dict) or set(branches) != set(BRANCHES.values()):
        raise ValueError('exact two historical branches required')
    if not isinstance(anchors, dict) or set(anchors) != set(SETTINGS):
        raise ValueError('exact two canonical anchors required')
    if not isinstance(bindings, dict) or set(bindings) != expected_bindings:
        raise ValueError('exact four source bindings required')
    bound = {k: _binding(v, 'EXP-018' if k.startswith('anchor-') else 'EXP-006') for k,v in bindings.items()}
    audits = {b: _branch(b, branches[b]) for b in BRANCHES.values()}
    rows, decisions, used = [], [], set()
    for spec in specs:
        if spec['role'] == 'anchor':
            rows.append(_anchor(spec, anchors[spec['source_setting']], bound[spec['id']]))
            continue
        target = Fraction.from_float(spec['target_rho'])
        candidates, choice_rows = [], []
        for node in audits[spec['branch_id']]['nodes']:
            rho = node['dominant_rho']
            distance = None if rho is None else abs(Fraction.from_float(rho)-target)
            reasons = list(node['reasons'])
            if distance is not None and distance > CALIPER:
                reasons.append('outside_fixed_multiplier_bin')
            if node['node_id'] in used:
                reasons.append('already_selected_node')
            choice_rows.append(dict(node_id=node['node_id'], eligible=not reasons, reasons=reasons,
                seed_available=node['seed_available'], seed_reasons=list(node['seed_reasons']),
                distance=None if distance is None else float(distance),
                exact_distance=None if distance is None else str(distance), selected=False))
            if not reasons:
                candidates.append((distance, node['index'], node))
        if not candidates:
            row = _empty(spec, 'no_historical_match', ['no_eligible_unused_node_within_fixed_bin'])
            row['seed_provenance'] = dict(source_binding=bound[spec['branch_id']])
        else:
            _, _, node = min(candidates, key=lambda item: item[:2])
            used.add(node['node_id'])
            choice_rows[node['index']]['selected'] = True
            row = dict(deepcopy(spec), available=node['seed_available'],
                status='selected' if node['seed_available'] else 'missing_seed_evidence',
                reasons=list(node['seed_reasons']), node_index=node['index'])
            for key in ('node_id','alpha','gamma','period','first_maximum_state','dominant_rho'):
                row[key] = deepcopy(node[key])
            row['seed_provenance'] = dict(source_binding=bound[spec['branch_id']],
                state_field='first_maximum_state', period_field='period', original_verified=True,
                node_metadata_sha256=node['node_metadata_sha256'],
                recorded_transverse_multipliers=deepcopy(node['recorded_modes']))
        decisions.append(dict(slot_id=spec['id'], nodes=choice_rows))
        rows.append(row)
    return dict(schema=contract()['schema'], contract=contract(), slots=rows,
        branch_audits=list(audits.values()), candidate_decisions=decisions,
        counts=dict(planned=6, available=sum(r['available'] for r in rows),
                    unavailable=sum(not r['available'] for r in rows)),
        all_slots_available=all(r['available'] for r in rows),
        scientific_qualification=False, new_neural_computations=0)
