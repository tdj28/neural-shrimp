"""Independent PM2F saved-scalar arithmetic. No model/producer imports.

Fitted rates and distance scalars are inherited observations; this module never
reconstructs a trajectory from them or equates a positive slope with endpoint growth.
"""
import math
from collections import Counter
import numpy as np

ROW_KEYS = {'index', 'gold', 'answer', 'correct', 'switch_step', 'twins'}
TWIN_KEYS = {'answer', 'flip', 'switch_step', 'same_switch', 'growth_rate',
             'log10_max_think', 'log10_end_think'}


def finite(x):
    return type(x) in (int, float) and math.isfinite(x)


def number_kind(x):
    if x is None:
        return 'null'
    if type(x) not in (int, float):
        return 'wrong_type'
    if math.isnan(x):
        return 'nan'
    if math.isinf(x):
        return 'positive_infinity' if x > 0 else 'negative_infinity'
    return 'finite'


def scalar_summary(values):
    """Finite-qualified descriptive summary, with every exclusion counted."""
    kinds = Counter(number_kind(x) for x in values)
    a = np.asarray([x for x in values if finite(x)], dtype=np.float64)
    return {'total': len(values), 'kinds': dict(sorted(kinds.items())),
            'finite_count': len(a),
            'minimum': float(a.min()) if len(a) else None,
            'maximum': float(a.max()) if len(a) else None,
            'mean': float(a.mean()) if len(a) else None,
            'median': float(np.median(a)) if len(a) else None}


def relation(value, threshold):
    if not finite(value):
        return 'undefined'
    return 'above' if value > threshold else 'below' if value < threshold else 'equal'


def switch_valid(value, budget):
    return value is None or (type(value) is int and 0 <= value <= budget)


def fraction(n, d):
    return n / d if d else None


def reduce_rows(rows, args):
    """Retain source order and duplicate witnesses; no last-write selection.

    Structural omissions refuse the component. Scalar nonfinite observations and
    stored-vs-reconstructed Boolean discrepancies are science diagnostics, not
    discarded rows. All denominator variants are named.
    """
    for k in ('model', 'twins', 'eps0', 'think_budget', 'answer_budget', 'group', 'limit'):
        if k not in args:
            raise ValueError('missing run argument: ' + k)
    K, eps, budget = args['twins'], args['eps0'], args['think_budget']
    if type(K) is not int or K < 1 or not finite(eps) or not 0 < eps < 1:
        raise ValueError('invalid twins or eps0')
    if type(budget) is not int or budget < 1:
        raise ValueError('invalid thinking budget')
    logeps = math.log10(eps)
    observed, ledger, twins = {}, [], []
    for line, r in enumerate(rows, 1):
        if not isinstance(r, dict) or not ROW_KEYS <= r.keys():
            raise ValueError(f'row {line}: missing row fields')
        if type(r['index']) is not int or r['index'] < 0:
            raise ValueError(f'row {line}: invalid index')
        if not isinstance(r['gold'], str) or not (r['answer'] is None or isinstance(r['answer'], str)):
            raise ValueError(f'row {line}: invalid answer/gold types')
        if type(r['correct']) is not bool or not switch_valid(r['switch_step'], budget):
            raise ValueError(f'row {line}: invalid correct/switch type or range')
        if not isinstance(r['twins'], list) or len(r['twins']) != K:
            raise ValueError(f'row {line}: twin count mismatch')
        previous = observed.get(r['index'], [])
        observed.setdefault(r['index'], []).append(line)
        base_correct = r['answer'] == r['gold']
        row = {'line': line, 'index': r['index'], 'duplicate_of_lines': list(previous),
               'original': r, 'base_correct_reconstructed': base_correct,
               'base_correct_matches': r['correct'] == base_correct,
               'base_window_stop': r['switch_step'] if r['switch_step'] is not None else budget,
               'twins': []}
        for k, t in enumerate(r['twins']):
            if not isinstance(t, dict) or not TWIN_KEYS <= t.keys():
                raise ValueError(f'row {line} twin {k}: missing twin fields')
            if not (t['answer'] is None or isinstance(t['answer'], str)):
                raise ValueError(f'row {line} twin {k}: invalid answer type')
            if any(type(t[x]) is not bool for x in ('flip', 'same_switch')) or not switch_valid(t['switch_step'], budget):
                raise ValueError(f'row {line} twin {k}: invalid flag/switch type or range')
            g, mx, en = (t[x] for x in ('growth_rate', 'log10_max_think', 'log10_end_think'))
            flip = t['answer'] != r['answer']
            same = t['switch_step'] == r['switch_step']
            reasons = []
            for key, v in [('growth_rate', g), ('log10_max_think', mx), ('log10_end_think', en)]:
                if v is not None and not finite(v): reasons.append(key + ':' + number_kind(v))
            if finite(mx) and finite(en) and en > mx: reasons.append('endpoint_exceeds_saved_maximum')
            if finite(mx) and mx < -300: reasons.append('maximum_below_writer_log_floor')
            if finite(en) and en < -300: reasons.append('endpoint_below_writer_log_floor')
            if (mx is None) != (en is None): reasons.append('maximum_endpoint_null_mismatch')
            if row['base_window_stop'] == 0 and any(v is not None for v in (g, mx, en)):
                reasons.append('scalar_present_for_empty_base_window')
            if g is not None and (mx is None or en is None): reasons.append('fitted_rate_without_distance_scalars')
            if row['base_window_stop'] < 2 and g is not None: reasons.append('fitted_rate_with_fewer_than_two_possible_steps')
            z = {'line': line, 'index': r['index'], 'twin': k,
                 'flip_reconstructed': flip, 'flip_matches': t['flip'] == flip,
                 'same_switch_reconstructed': same, 'same_switch_matches': t['same_switch'] == same,
                 'twin_correct_reconstructed': t['answer'] == r['gold'],
                 'endpoint_vs_nominal_eps0': relation(en, logeps),
                 'maximum_vs_nominal_eps0': relation(mx, logeps),
                 'fitted_rate_vs_zero': relation(g, 0),
                 'endpoint_at_writer_floor': finite(en) and en == -300,
                 'maximum_at_writer_floor': finite(mx) and mx == -300,
                 'twin_switches_before_base': (t['switch_step'] is not None and
                                               t['switch_step'] < row['base_window_stop']),
                 'diagnostics': reasons}
            row['twins'].append(z)
            twins.append((r, t, z))
        ledger.append(row)
    n, nt = len(rows), len(twins)
    counts = lambda key: dict(sorted(Counter(z[key] for _, _, z in twins).items()))
    anyflip = sum(any(t['flip_reconstructed'] for t in r['twins']) for r in ledger)
    reconstructed_flips = sum(z['flip_reconstructed'] for _, _, z in twins)
    stored_flips = sum(t['flip'] for _, t, _ in twins)
    return {'schema': 'pm2f-independent-scalars-v1', 'model': args['model'], 'args': args,
            'scope': {'endpoint_reference': 'nominal eps0, not a saved realized initial distance',
                      'distance_window': 'indices before base switch; NaNs removed by original writer',
                      'fit_window': 'inclusive 10*eps0 <= distance <= 1e-3; at least two selected samples',
                      'slope_and_distances': 'inherited; native series not saved',
                      'growth_is_not_endpoint_growth': True,
                      'twins_are_not_independent_problem_units': True},
            'coverage': {'records': n, 'unique_problem_indices': len(observed), 'twins': nt,
                         'duplicate_indices': {str(k): v for k, v in observed.items() if len(v) > 1},
                         'all_rows_retained_in_source_order': True},
            'counts': {'base_correct_stored': sum(r['correct'] for r in rows),
                       'base_correct_reconstructed': sum(r['base_correct_reconstructed'] for r in ledger),
                       'base_correct_mismatches': sum(not r['base_correct_matches'] for r in ledger),
                       'base_accuracy_reconstructed': fraction(sum(r['base_correct_reconstructed'] for r in ledger), n),
                       'answer_flips_stored': stored_flips, 'answer_flips_reconstructed': reconstructed_flips,
                       'answer_flip_fraction_all_twins': fraction(reconstructed_flips, nt),
                       'flip_flag_mismatches': sum(not z['flip_matches'] for _, _, z in twins),
                       'problem_any_flip': anyflip, 'problem_any_flip_fraction': fraction(anyflip, n),
                       'same_switch_stored': sum(t['same_switch'] for _, t, _ in twins),
                       'same_switch_reconstructed': sum(z['same_switch_reconstructed'] for _, _, z in twins),
                       'same_switch_flag_mismatches': sum(not z['same_switch_matches'] for _, _, z in twins),
                       'endpoint_vs_nominal_eps0': counts('endpoint_vs_nominal_eps0'),
                       'maximum_vs_nominal_eps0': counts('maximum_vs_nominal_eps0'),
                       'fitted_rate_vs_zero': counts('fitted_rate_vs_zero'),
                       'positive_fit_but_endpoint_below_eps0': sum(z['fitted_rate_vs_zero'] == 'above' and z['endpoint_vs_nominal_eps0'] == 'below' for _, _, z in twins),
                       'endpoint_writer_floor': sum(z['endpoint_at_writer_floor'] for _, _, z in twins),
                       'maximum_writer_floor': sum(z['maximum_at_writer_floor'] for _, _, z in twins),
                       'earlier_twin_switch': sum(z['twin_switches_before_base'] for _, _, z in twins),
                       'diagnostic_twin_rows': sum(bool(z['diagnostics']) for _, _, z in twins)},
            'finite_scalar_summaries': {key: scalar_summary([t[key] for _, t, _ in twins])
                                        for key in ('growth_rate', 'log10_max_think', 'log10_end_think')},
            'per_twin_index': {str(k): {
                'twins': sum(z['twin']==k for _,_,z in twins),
                'answer_flips_reconstructed': sum(z['flip_reconstructed'] for _,_,z in twins if z['twin']==k),
                'same_switch_reconstructed': sum(z['same_switch_reconstructed'] for _,_,z in twins if z['twin']==k),
                'endpoint_vs_nominal_eps0': dict(sorted(Counter(z['endpoint_vs_nominal_eps0'] for _,_,z in twins if z['twin']==k).items())),
                'fitted_rate_vs_zero': dict(sorted(Counter(z['fitted_rate_vs_zero'] for _,_,z in twins if z['twin']==k).items())),
                'maximum_vs_nominal_eps0': dict(sorted(Counter(z['maximum_vs_nominal_eps0'] for _,_,z in twins if z['twin']==k).items()))}
                for k in range(K)},
            'base_switch_summary': scalar_summary([r['switch_step'] for r in rows]),
            'twin_switch_summary': scalar_summary([t['switch_step'] for _, t, _ in twins]),
            'rows': ledger}


def compare_leaves(own, published, path='$'):
    """No tolerance and no dropped published or own field; retain tiny deltas."""
    out = []
    if isinstance(own, dict) and isinstance(published, dict):
        for k in sorted(set(own) | set(published)):
            if k not in own or k not in published:
                out.append({'path': path + '.' + k, 'status': 'missing_own' if k not in own else 'missing_published',
                            'own': own.get(k), 'published': published.get(k)})
            else: out.extend(compare_leaves(own[k], published[k], path + '.' + k))
    elif isinstance(own, list) and isinstance(published, list):
        for k in range(max(len(own), len(published))):
            if k >= len(own) or k >= len(published):
                out.append({'path': f'{path}[{k}]', 'status': 'missing_own' if k >= len(own) else 'missing_published',
                            'own': own[k] if k < len(own) else None,
                            'published': published[k] if k < len(published) else None})
            else: out.extend(compare_leaves(own[k], published[k], f'{path}[{k}]'))
    else:
        numeric = type(own) in (int, float) and type(published) in (int, float)
        equal = (own == published and (numeric or type(own) is type(published)))
        out.append({'path': path, 'status': 'exact' if equal else 'different', 'own': own, 'published': published,
                    'signed_difference': own - published if finite(own) and finite(published) else None})
    return out


def json_safe(value):
    if isinstance(value, dict): return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    if type(value) is float and not math.isfinite(value): return {'nonfinite': number_kind(value)}
    return value


def declared_publication_projection(rows, args, growth_definition='endpoint'):
    """Prospectively named interpretation, NOT attested missing inline source.

    Endpoint and fitted-slope versions are both retained. No selection by which
    matches publication. Quartiles use the disclosed NumPy linear convention on
    non-null saved slopes; nonfinite values propagate, with separate diagnostics.
    """
    if growth_definition not in ('endpoint', 'fitted_slope'):
        raise ValueError('unknown growth definition')
    ts = [t for r in rows for t in r['twins']]
    rates = [t['growth_rate'] for t in ts if t['growth_rate'] is not None]
    ends = [t['log10_end_think'] for t in ts if t['log10_end_think'] is not None]
    logeps = math.log10(args['eps0'])
    values = ends if growth_definition == 'endpoint' else rates
    threshold = logeps if growth_definition == 'endpoint' else 0.
    numeric_rates = all(type(v) in (int, float) for v in rates)
    numeric_ends = all(type(v) in (int, float) for v in ends)
    numeric_values = all(type(v) in (int, float) for v in values)
    return {'problems': len(rows), 'twins': len(ts),
            'accuracy_base': fraction(sum(r['answer'] == r['gold'] for r in rows), len(rows)),
            'twins_grew': sum(v > threshold for v in values) if numeric_values else None,
            'twins_shrank': sum(v < threshold for v in values) if numeric_values else None,
            'twin_flips': sum(t['answer'] != r['answer'] for r in rows for t in r['twins']),
            'twin_flip_rate': fraction(sum(t['answer'] != r['answer'] for r in rows for t in r['twins']), len(ts)),
            'problems_any_flip': fraction(sum(any(t['answer'] != r['answer'] for t in r['twins']) for r in rows), len(rows)),
            'different_switch': sum(t['switch_step'] != r['switch_step'] for r in rows for t in r['twins']),
            'growth_rate_quartiles': [float(x) for x in np.percentile(np.asarray(rates, dtype=float), [25,50,75], method='linear')] if rates and numeric_rates else None,
            'growth_rate_n': len(rates),
            'log10_end_think_median': float(np.median(np.asarray(ends, dtype=float))) if ends and numeric_ends else None}
