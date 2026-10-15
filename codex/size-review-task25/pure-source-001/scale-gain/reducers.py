"""Task25 own pure saved-data reducers. No file loading or producer imports.

The eventual admitted driver must authenticate and normalize the selected
original inputs. This source alone neither chooses attempts nor admits a run.
"""
import math
import numpy as np

P = 'llm-soft-loop/pod64/'
# Literal current source roster/order; historical invocation binding is pending.
MODEL_SPECS = [
    ('Qwen2.5', '0.5B', .49, ['scan64/q05-alpha0'], []),
    ('Qwen2.5', '1.5B', 1.54, ['big/qwen1.5b-alpha0-wide'], ['big/qwen1.5b-alpha0-upper']),
    ('Qwen2.5', '3B', 3.09, ['big/qwen3b-alpha0-wide'], ['big/qwen3b-alpha0-upper']),
    ('Qwen2.5', '7B', 7.62, ['controls/q7-alpha0'], ['big/qwen7b-alpha0-wide', 'big/qwen7b-alpha0-upper']),
    ('Qwen2.5', '14B', 14.7, ['big/qwen14b-alpha0-wide'], ['big/qwen14b-alpha0-upper']),
    ('Qwen2.5', '32B', 32.76, ['big/qwen32b-alpha0-wide'], ['big/qwen32b-alpha0-upper']),
    ('Qwen2.5', '72B', 72.7, ['big/qwen72b-alpha0', 'big/qwen72b-alpha0-highT'], []),
    ('Llama-3.1', '8B', 8.03, ['llama8b-hires/alpha0'], ['big/llama3.1-8b-alpha0-wide', 'big/llama3.1-8b-alpha0-upper']),
    ('Llama-3.1', '70B', 70.6, ['big/llama3.1-70b-alpha0'], []),
    ('Llama-3.3', '70B', 70.6, ['big/llama3.3-70b-alpha0'], []),
    ('OLMo-2', '1B', 1.48, ['big/olmo2-1b-alpha0-wide'], ['big/olmo2-1b-alpha0-upper']),
    ('OLMo-2', '7B', 7.30, ['controls/o-alpha0c'], ['big/olmo2-7b-alpha0-wide', 'big/olmo2-7b-alpha0-upper']),
    ('OLMo-2', '13B', 13.7, ['big/olmo2-13b-alpha0-wide'], ['big/olmo2-13b-alpha0-upper']),
    ('OLMo-2', '32B', 32.2, ['big/olmo2-32b-alpha0-wide'], ['big/olmo2-32b-alpha0-upper']),
    ('Qwen3.5 MoE', '35B-A3B', 35., ['controls/m1t-alpha0/scan'], []),
]
MODEL_SPECS = [dict(family=f, model=m, params_b=p,
                    primary_sources=[P+x for x in a], supplemental_sources=[P+x for x in b])
               for f, m, p, a, b in MODEL_SPECS]
BINS = ((0, 1), (1, 3), (3, 6), (6, 10), (10, 99))
FAMILIES = (('Qwen2.5', ('7B', '14B', '32B', '72B')),
            ('OLMo-2', ('7B', '13B', '32B')))
P3_ASSUMPTION = (
    "Frozen P3 says 'the 14B and 32B models'. The same frozen roster and P1/P2 "
    "name OLMo-2 13B. Explicit family-specific interpretation: Qwen2.5 14B/32B "
    "and OLMo-2 13B/32B versus their own 7B. This is not a literal OLMo-2 14B test."
)


def number(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise ValueError('numeric scalar required; no Boolean or string coercion')
    if np.ndim(value) or np.iscomplexobj(value):
        raise ValueError('real scalar required')
    return float(value)


def json_ready(value):
    """Lossless named nonfinite dispositions, not null or zero substitution."""
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite': repr(value)}
    return value


def _source(sources, key):
    if key not in sources:
        raise ValueError('missing inventory descriptor: ' + key)
    source = sources[key]
    if source['id'] != key or type(source['available']) is not bool:
        raise ValueError('source identity/availability must be explicit')
    # Availability is independent of historical process success.
    for field in ('run_disposition', 'expected_rows', 'rows'):
        if field not in source:
            raise ValueError('missing source metadata: ' + field)
    if type(source['expected_rows']) is not int or source['expected_rows'] < 0 or not isinstance(source['rows'], list):
        raise ValueError('integer row budget and explicit row list required')
    if source['available'] and len(source['rows']) != source['expected_rows']:
        raise ValueError('source row coverage differs from admitted metadata')
    if not source['available'] and source['rows']:
        raise ValueError('unavailable source cannot silently supply rows')
    return source


def _period_valid(p, maximum):
    return math.isfinite(p) and p.is_integer() and 0 <= p <= maximum


def _scale_join(keys, sources):
    rows, status, missing = [], [], []
    for occurrence, key in enumerate(keys):
        source = _source(sources, key)
        status.append({'source_id': key, 'available': source['available'],
                       'run_disposition': source['run_disposition'], 'expected_rows': source['expected_rows']})
        if not source['available']:
            missing.append(key)
            continue
        maximum = source['max_period']
        if type(maximum) is not int or maximum < 1:
            raise ValueError('original maximum period required')
        alpha = number(source['alpha'])
        for i, raw in enumerate(source['rows']):
            if type(raw['completed']) is not bool:
                raise ValueError('explicit original cell completion required')
            values = {k: number(raw[k]) for k in ('log10_temperature', 'lyapunov', 'period', 'entropy')}
            t, le, per, ent = (values[k] for k in ('log10_temperature', 'lyapunov', 'period', 'entropy'))
            chaotic = le > .001 and per <= 0
            invalid = [k + '_nonfinite' for k, v in values.items() if not math.isfinite(v)]
            if not _period_valid(per, maximum): invalid.append('period_outside_integer_writer_range')
            if not raw['completed']: invalid.append('cell_not_completed')
            if not math.isfinite(alpha) or alpha != 0: invalid.append('alpha_not_exact_zero')
            rows.append({'id': f'{occurrence}:{key}:{i}', 'source_id': key, 'source_row': i,
                         'source_occurrence': occurrence, 'input_ordinal': len(rows), **values,
                         'alpha': alpha, 'completed': raw['completed'], 'max_period': maximum,
                         'literal_chaotic': bool(chaotic), 'literal_unresolved': bool(per <= 0 and not chaotic),
                         'literal_periodic': bool(per > 0), 'finite_qualified': not invalid,
                         'qualification_failures': invalid})
    order = np.argsort(np.asarray([r['log10_temperature'] for r in rows]), kind='stable')
    sorted_rows = [rows[int(i)] for i in order]
    kept, duplicates, conflicts = [], [], []
    previous, anchor = None, None
    for rank, row in enumerate(sorted_rows):
        # Literal source compares immediate sorted neighbors, not each chain's anchor.
        keep = previous is None or abs(row['log10_temperature'] - previous['log10_temperature']) > 1e-9
        row.update({'sorted_ordinal': rank, 'selected': bool(keep)})
        if keep:
            kept.append(row)
            anchor = row
        else:
            delta = {k: row[k] - previous[k] for k in ('log10_temperature', 'lyapunov', 'period', 'entropy')}
            anchor_delta = {k: row[k] - anchor[k] for k in delta}
            disagree = row['literal_chaotic'] != previous['literal_chaotic']
            duplicates.append({'discarded_id': row['id'], 'previous_id': previous['id'], 'retained_chain_id': anchor['id'],
                               'exact_coordinate_equal': row['log10_temperature'] == previous['log10_temperature'],
                               'signed_previous_differences': delta, 'signed_anchor_differences': anchor_delta,
                               'adjacent_chaos_label_disagrees': disagree})
            if disagree: conflicts.append(row['id'])
        previous = row
    nonfinite_le = [r['id'] for r in kept if not math.isfinite(r['lyapunov'])]
    refusals = ([] if not conflicts else ['adjacent_duplicate_chaos_disagreement'])
    if nonfinite_le: refusals.append('retained_nonfinite_exponent')
    return {'rows': sorted_rows, 'selected': kept, 'duplicates': duplicates, 'source_status': status,
            'missing': missing, 'source_refusals': refusals if not missing else [],
            'present_row_diagnostic_refusals': refusals, 'conflict_ids': conflicts,
            'retained_nonfinite_exponent_ids': nonfinite_le}


def _extrema(rows, key):
    finite = [r for r in rows if math.isfinite(r[key])]
    if not finite:
        return {'min': None, 'max': None, 'min_ids': [], 'max_ids': [], 'finite_count': 0}
    low, high = min(r[key] for r in finite), max(r[key] for r in finite)
    return {'min': low, 'max': high, 'min_ids': [r['id'] for r in finite if r[key] == low],
            'max_ids': [r['id'] for r in finite if r[key] == high], 'finite_count': len(finite)}


def _one_scale(spec, sources):
    primary = _scale_join(spec['primary_sources'], sources)
    extras = [key for key in spec['supplemental_sources'] if _source(sources, key)['available']]
    full = _scale_join(spec['primary_sources'] + extras, sources)
    missing_extras = [key for key in spec['supplemental_sources'] if key not in extras]
    base = dict(family=spec['family'], model=spec['model'], params_b=number(spec['params_b']))
    missing = bool(primary['missing'] or full['missing'])
    literal = {**base, 'missing': True} if missing else None
    refusals = sorted(set(primary['source_refusals'] + full['source_refusals']))
    core_rows = [r for r in primary['selected'] if r['log10_temperature'] <= .3 + 1e-9]
    for row in primary['selected']:
        row['source_core'] = row['log10_temperature'] <= .3 + 1e-9
        row['declared_exact_core'] = -1 <= row['log10_temperature'] <= .3
    all_rows = full['selected']
    if len(all_rows) < 2 and not missing:
        refusals.append('source_gradient_requires_at_least_two_coordinates')
    if not missing and not refusals:
        lt = np.asarray([r['log10_temperature'] for r in all_rows])
        le = np.asarray([r['lyapunov'] for r in all_rows])
        chaos = np.asarray([r['literal_chaotic'] for r in all_rows])
        mid = np.asarray([1 < r['entropy'] < 10 for r in all_rows])
        weights = np.gradient(lt)
        for r, w, c, h in zip(all_rows, weights, chaos, mid):
            r.update(spacing_weight=float(w), chaos_weight=float(w*c), intermediate_entropy_weight=float(w*h))
        literal = {**base, 'missing': False, 'temperatures_core': len(core_rows),
                   'chaotic_core': sum(r['literal_chaotic'] for r in core_rows),
                   'unresolved_core': sum(r['literal_unresolved'] for r in core_rows),
                   'share_core': float(np.mean([r['literal_chaotic'] for r in core_rows])) if core_rows else float('nan'),
                   'temperatures_all': len(all_rows), 'chaotic_all': int(chaos.sum()),
                   'log10T_range': [float(lt.min()), float(lt.max())], 'max_exponent': float(le.max()),
                   'chaos_decades': float((weights*chaos).sum()), 'intermediate_entropy_decades': float((weights*mid).sum()),
                   'sweeps': spec['primary_sources']+extras,
                   'chaotic_log10T': [round(r['log10_temperature'], 4) for r in all_rows if r['literal_chaotic']]}
    return {'identity': base, 'literal_summary': literal, 'source_refusals': refusals,
            'primary_join': primary, 'full_join': full, 'absent_optional_sources': missing_extras,
            'lower_axis_check': {'below_minus_one_ids': [r['id'] for r in primary['selected'] if r['log10_temperature'] < -1],
                                 'core_membership_difference_ids': [r['id'] for r in primary['selected']
                                                                   if r['source_core'] != r['declared_exact_core']]},
            'qualification': {'primary_selected': len(primary['selected']), 'full_selected': len(all_rows),
                              'primary_invalid_ids': [r['id'] for r in primary['selected'] if not r['finite_qualified']],
                              'full_invalid_ids': [r['id'] for r in all_rows if not r['finite_qualified']],
                              'core_qualified_count': sum(r['finite_qualified'] for r in core_rows),
                              'core_qualified_chaotic_count': sum(r['finite_qualified'] and r['literal_chaotic'] for r in core_rows)},
            'full_finite_extrema': {k: _extrema(all_rows, k) for k in ('log10_temperature', 'lyapunov', 'entropy')},
            'source_core_rule': 'log10T <= 0.3 + 1e-9 only; exact lower endpoint is separately checked'}


def _ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    rank = np.empty(len(values), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]: j += 1
        rank[order[i:j]] = (i+1+j)/2
        i = j
    return rank


def _spearman(x, y):
    if len(x) < 2 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return {'rho': float('nan'), 'reason': 'insufficient_or_nonfinite', 'x': x, 'y': y}
    a, b = _ranks(x), _ranks(y)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return {'rho': float('nan'), 'reason': 'constant_ranks', 'x': x, 'y': y, 'x_ranks': a, 'y_ranks': b}
    return {'rho': float(np.corrcoef(a, b)[0, 1]), 'reason': None, 'x': x, 'y': y, 'x_ranks': a, 'y_ranks': b}


def scale_summary(sources, specs=MODEL_SPECS):
    """Pure reduction; explicit missing descriptors and every duplicate retained."""
    keys = [(s['family'], s['model']) for s in specs]
    if len(set(keys)) != len(keys): raise ValueError('ambiguous model roster')
    models = [_one_scale(s, sources) for s in specs]
    by = {tuple(m['identity'][k] for k in ('family', 'model')): m for m in models}
    def share(family, model):
        m = by.get((family, model))
        r = None if m is None else m['literal_summary']
        return None if r is None or r['missing'] else r['share_core']
    p1, p2, ledger = {}, {}, []
    for family, order in FAMILIES:
        reference = share(family, order[0])
        for model in order[1:3]:
            value = share(family, model)
            name = f'{family} {model} < {order[0]}'
            p1[name] = None if value is None or reference is None else bool(value < reference)
            ledger.append({'prediction': 'P1', 'name': name, 'left': value, 'right': reference,
                           'source_operator_result': p1[name], 'finite_defined': value is not None and reference is not None
                           and math.isfinite(value) and math.isfinite(reference)})
        values = [share(family, model) for model in order]
        name = family + ' ' + ' >= '.join(order)
        p2[name] = None if any(v is None for v in values) else all(values[i] >= values[i+1] for i in range(len(values)-1))
        ledger.append({'prediction': 'P2', 'name': name, 'values': values, 'source_operator_result': p2[name],
                       'finite_defined': all(v is not None and math.isfinite(v) for v in values)})
    good = [m['literal_summary'] for m in models if m['literal_summary'] is not None and not m['literal_summary']['missing']]
    correlations = {'breadth': _spearman([r['intermediate_entropy_decades'] for r in good], [r['chaos_decades'] for r in good]),
                    'log_params': _spearman([float(np.log(r['params_b'])) for r in good], [r['chaos_decades'] for r in good]),
                    'model_ids': [[r['family'], r['model']] for r in good]}
    projection = {'rows': [m['literal_summary'] for m in models], 'P1': p1, 'P2': p2,
                  'P1_holds': None if any(v is None for v in p1.values()) else all(p1.values()),
                  'P2_holds': None if any(v is None for v in p2.values()) else all(p2.values()),
                  'spearman': {'breadth': correlations['breadth']['rho'], 'log_params': correlations['log_params']['rho'], 'models': len(good)}}
    return json_ready({'component': 'scale', 'models': models, 'producer_projection': projection,
                       'prediction_ledger': ledger, 'descriptive_correlations': correlations,
                       'literal_summary_numeric_stage_has_no_refusal': not any(m['source_refusals'] for m in models),
                       'scope': 'Conditional per-model results retained even if a source refusal would abort the producer program; qualified counts never replace literal counts.'})


def gain_summary(specs, sources):
    """All bins/all cells; specs bind source IDs to original labels and families."""
    models, projection, duplicates = [], {}, []
    for ordinal, spec in enumerate(specs):
        source = _source(sources, spec['source_id'])
        rows = []
        for i, raw in enumerate(source['rows']):
            if type(raw['completed']) is not bool: raise ValueError('explicit cell completion required')
            values = {k: number(raw[k]) for k in ('log10_temperature', 'entropy_mean', 'log_g_net', 'log_g_read', 'lyapunov', 'period', 'boundary_term')}
            invalid = [k+'_nonfinite' for k, v in values.items() if not math.isfinite(v)]
            if not _period_valid(values['period'], 64): invalid.append('period_outside_integer_writer_range')
            if not raw['completed']: invalid.append('cell_not_completed')
            rows.append({'id': f'{ordinal}:{source["id"]}:{i}', 'source_row': i, **values,
                         'completed': raw['completed'], 'literal_chaotic': values['lyapunov'] > .001 and values['period'] == 0,
                         'finite_qualified': not invalid, 'qualification_failures': invalid})
        bins, literal = [], {}
        for low, high in BINS:
            name = f'{low}-{high}'
            members = [r for r in rows if low <= r['entropy_mean'] < high]
            row = {'bin': name, 'lower_inclusive': low, 'upper_exclusive': high, 'cells': len(members),
                   'member_ids': [r['id'] for r in members], 'chaotic': sum(r['literal_chaotic'] for r in members),
                   'finite_qualified_cells': sum(r['finite_qualified'] for r in members),
                   'status': 'missing_source' if not source['available'] else ('populated' if members else 'empty')}
            for factor in ('net', 'read'):
                values = [r['log_g_'+factor] for r in members]
                row['median_log_g_'+factor] = float(np.median(values)) if values else None
                row[f'{factor}_finite_count'] = sum(math.isfinite(v) for v in values)
                row[f'{factor}_nonfinite_ids'] = [r['id'] for r in members if not math.isfinite(r['log_g_'+factor])]
            bins.append(row)
            if members:
                literal[name] = {k: row[k] for k in ('cells', 'median_log_g_net', 'median_log_g_read', 'chaotic')}
        if spec['label'] in projection:
            duplicates.append({'label': spec['label'], 'overwriting_spec_ordinal': ordinal,
                               'earlier_literal_bins': projection[spec['label']]})
        projection[spec['label']] = literal
        models.append({'spec': spec, 'available': source['available'], 'run_disposition': source['run_disposition'],
                       'rows': rows, 'bins': bins, 'outside_all_bin_ids': [r['id'] for r in rows
                           if not any(low <= r['entropy_mean'] < high for low, high in BINS)]})
    comparisons = []
    for family, order in FAMILIES:
        for model in order[1:3]:
            for low, high in BINS[1:4]:
                for factor in ('net', 'read'):
                    sides = []
                    for name in (model, '7B'):
                        matches = [r for r in models if r['spec']['family'] == family and r['spec']['model'] == name]
                        if len(matches) != 1 or not matches[0]['available']:
                            sides.append({'model': name, 'status': 'missing_or_ambiguous_model', 'cells': None, 'median': None})
                        else:
                            b = next(x for x in matches[0]['bins'] if x['bin'] == f'{low}-{high}')
                            sides.append({'model': name, 'status': b['status'], 'cells': b['cells'],
                                          'finite_qualified_cells': b['finite_qualified_cells'],
                                          'finite_factor_count': b[f'{factor}_finite_count'],
                                          'median': b['median_log_g_'+factor]})
                    a, b = [s['median'] for s in sides]
                    defined = a is not None and b is not None and math.isfinite(a) and math.isfinite(b)
                    comparisons.append({'family': family, 'model': model, 'reference': '7B', 'bin': f'{low}-{high}',
                                        'factor': factor, 'sides': sides, 'source_operator_result': None if a is None or b is None else bool(a <= b),
                                        'finite_defined': defined, 'P3_verdict': bool(a <= b) if defined else None,
                                        'finite_defined_scope': 'Both literal all-cell medians finite; not a claim every contributing cell is qualified.',
                                        'signed_median_difference': None if a is None or b is None else a-b,
                                        'roster_interpretation': P3_ASSUMPTION})
    return json_ready({'component': 'gain', 'models': models, 'producer_projection': projection,
                       'duplicate_label_ledger': duplicates, 'P3': comparisons,
                       'P3_holds': None if any(r['P3_verdict'] is None for r in comparisons) else all(r['P3_verdict'] for r in comparisons),
                       'P3_roster_interpretation': P3_ASSUMPTION,
                       'literal_summary_inputs_available': all(m['available'] for m in models),
                       'scope': 'All-cell bin medians, not chaos-only medians; fixed coarse entropy bins are observational matching, not a causal size effect.'})


_MISSING = object()


def compare_leaves(own, producer, path=''):
    """Retain every producer leaf, including tiny/nonfinite differences; no tolerance."""
    rows = []
    if isinstance(producer, dict):
        if not producer:
            rows.append({'path': path, 'disposition': 'exact_empty_mapping' if own == {} else 'empty_mapping_difference',
                         'own': None if own is _MISSING else json_ready(own), 'producer': {}})
        if isinstance(own, dict):
            for k in own.keys() - producer.keys():
                rows.append({'path': path+'/'+str(k).replace('~', '~0').replace('/', '~1'),
                             'disposition': 'extra_own_key', 'own': json_ready(own[k])})
        for k, v in producer.items():
            token = str(k).replace('~', '~0').replace('/', '~1')
            value = own.get(k, _MISSING) if isinstance(own, dict) else _MISSING
            rows.extend(compare_leaves(value, v, path+'/'+token))
    elif isinstance(producer, list):
        if not isinstance(own, list) or len(own) != len(producer):
            rows.append({'path': path, 'disposition': 'sequence_shape_mismatch',
                         'own_length': len(own) if isinstance(own, list) else None, 'producer_length': len(producer)})
        for i, v in enumerate(producer):
            value = own[i] if isinstance(own, list) and i < len(own) else _MISSING
            rows.extend(compare_leaves(value, v, path+'/'+str(i)))
    else:
        if own is _MISSING:
            return [{'path': path, 'disposition': 'missing_own_leaf', 'producer': json_ready(producer)}]
        if isinstance(own, dict) and set(own) == {'nonfinite'}:
            own = float(own['nonfinite'])
        equal = own == producer and (isinstance(own, bool) == isinstance(producer, bool))
        numeric = type(own) in (int, float) and type(producer) in (int, float)
        rows.append({'path': path, 'disposition': 'exact' if equal else 'different', 'own': json_ready(own),
                     'producer': json_ready(producer), 'signed_difference': json_ready(own-producer) if numeric else None,
                     'numeric_pair_finite': math.isfinite(own) and math.isfinite(producer) if numeric else None})
    return rows
