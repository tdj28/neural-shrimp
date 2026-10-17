"""Independent pure saved-record arithmetic. No files, producer imports or models.

Inputs are already hash-admitted mappings. Each ordered run has {id,args,records}.
The caller must supply frozen expected indices and the exact original analysis order.
Absent rows are retained as coverage failures; duplicate row keys refuse rather than
silently reproduce the producer's overwrite. Statistics remain descriptive.
"""
import math
import re
import hashlib
import json
from collections import defaultdict

import numpy as np

SEED = 20261016
BOOT = 10000


def selection(document, expected_groups=32, per_group=16):
    """Structural selection audit only; no tokenizer/dataset reconstruction."""
    groups = document['groups']
    indices = [i for g in groups for i in g['indices']]
    ordered = sorted(indices)
    lengths = [g['length'] for g in groups]
    checks = {'group_count': len(groups) == expected_groups,
              'per_group': all(len(g['indices']) == per_group for g in groups),
              'unique_indices': len(set(indices)) == len(indices),
              'indices_sorted_exact': document['indices'] == ordered,
              'lengths_sorted_unique': lengths == sorted(set(lengths)),
              'indices_at_or_above_start': all(i >= document['start'] for i in indices),
              'index_digest': hashlib.sha256(json.dumps(ordered).encode()).hexdigest() == document['sha256_indices']}
    return {'checks': checks, 'all_structural_checks': all(checks.values()), 'groups': groups,
            'split': document['split'], 'start': document['start'], 'end': document.get('end'),
            'eligible_lengths': document['eligible_lengths'],
            'unavailable': 'full tokenizer/group eligibility and first-in-dataset selection require original full selection inputs'}


def mean(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()) if a.size and np.isfinite(a).all() else None


def extract(text):
    marker = text.lower().rfind('answer is')
    found = re.findall(r'-?\d[\d,]*\.?\d*', text[marker + 9:] if marker >= 0 else text)
    if not found:
        return None
    token = found[0] if marker >= 0 else found[-1]
    try:
        x = float(token.replace(',', '').rstrip('.'))
        return str(int(x)) if x == int(x) else str(x)
    except (ValueError, OverflowError):
        return None


def plurality(answers):
    counts, winner, maximum = {}, None, 0
    for answer in answers:
        if answer is not None:
            counts[answer] = counts.get(answer, 0) + 1
            if counts[answer] > maximum:
                winner, maximum = answer, counts[answer]
    return winner


def bootstrap(grouped, rng, draws=BOOT):
    """Source's weighted problem mean after cluster resampling; keep every draw."""
    keys = list(grouped)
    groups = [np.asarray(grouped[k], dtype=float) for k in keys]
    if not groups or any(not len(g) or not np.isfinite(g).all() for g in groups):
        return {'interval': None, 'status': 'undefined: empty/nonfinite groups',
                'cluster_order': keys, 'draws': []}
    totals = np.array([g.sum() for g in groups])
    sizes = np.array([len(g) for g in groups])
    pick = rng.integers(len(groups), size=(draws, len(groups)))
    values = totals[pick].sum(axis=1) / sizes[pick].sum(axis=1)
    return {'interval': np.quantile(values, [.025, .975]).tolist(), 'status': 'defined',
            'cluster_order': keys, 'cluster_counts': sizes.tolist(), 'draws': values.tolist(),
            'undefined_draws': int((~np.isfinite(values)).sum())}


def wilson(success, total):
    if not total:
        return None
    z = 1.959964
    p, z2 = success / total, z * z
    denominator = 1 + z2 / total
    middle = (p + z2 / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / denominator
    return [middle - radius, middle + radius]


def mcnemar(base_only, arm_only):
    n = base_only + arm_only
    return min(1., 2 * sum(math.comb(n, k) for k in range(min(base_only, arm_only) + 1)) / 2**n)


def admit_rows(run, expected, layouts):
    expected = list(expected)
    if len(set(expected)) != len(expected) or len(set(layouts)) != len(layouts):
        raise ValueError('duplicate declared index/layout')
    rows, ledger, ignored = {}, [], []
    for record in run['records']:
        i, layout = record['index'], record['layout']
        if type(i) is not int or type(layout) is not int:
            raise ValueError('noninteger row identity')
        if i < 0:
            ignored.append(record)
            continue
        if i not in expected or layout not in layouts:
            raise ValueError('undeclared scored row')
        key = (i, layout)
        if key in rows:
            raise ValueError('duplicate scored row; producer would overwrite')
        answer, gold = record['answer'], record['gold']
        if (answer is not None and not isinstance(answer, str)) or not isinstance(gold, str):
            raise ValueError('answer/gold schema')
        if type(record['correct']) is not bool:
            raise ValueError('correct flag must be Boolean')
        if type(record['batch']) is not int or type(record['row']) is not int:
            raise ValueError('batch/position schema')
        correct = answer is not None and answer == gold
        stored_text = record.get('text')
        rows[key] = dict(record, correct=correct)
        ledger.append({'index': i, 'layout': layout, 'batch': record['batch'], 'row': record['row'],
                       'prompt_len': record['prompt_len'], 'answer': answer, 'gold': gold,
                       'correct': correct, 'saved_correct': record['correct'],
                       'correct_match': correct == record['correct'],
                       'stored_text_length': len(stored_text) if stored_text is not None else None,
                       'stored_text_parse': extract(stored_text) if stored_text is not None else None,
                       'text_parse_scope': 'suffix of at most 400 chars; diagnostic, not full-answer re-extraction'})
    missing = [[i, layout] for i in expected for layout in layouts if (i, layout) not in rows]
    return rows, {'run': run['id'], 'missing': missing, 'rows': ledger, 'ignored_negative_rows': ignored,
                  'full_coverage': not missing,
                  'correct_flags_match': all(r['correct_match'] for r in ledger)}


def paired_difference(common, baseline, arm, key, rng, draws):
    groups = defaultdict(list)
    rows = []
    for i in common:
        delta = float(arm[i][key]) - float(baseline[i][key])
        groups[baseline[i]['cluster']].append(delta)
        rows.append({'index': i, 'cluster': baseline[i]['cluster'], 'delta': delta})
    boot = bootstrap(groups, rng, draws)
    return {'difference': mean([r['delta'] for r in rows]), 'interval': boot['interval'],
            'rows': rows, 'bootstrap': boot}


def e2(ordered_arms, base, expected, draws=BOOT):
    """One original analyzer invocation; alternate baseline is a separate invocation/RNG reset."""
    rng = np.random.default_rng(SEED)
    if len({r['id'] for r in ordered_arms}) != len(ordered_arms):
        raise ValueError('duplicate arm identity')
    out = {'arms': {}, 'contrasts': {}, 'coverage': [], 'row_statistics': {}, 'bootstrap': {}}
    paired = {}
    for run in ordered_arms:
        cfg, name = run['args'], run['id']
        layouts = sorted(cfg['layouts'])
        if layouts != [8, 16]:
            raise ValueError('E2 requires frozen layouts 8 and 16')
        recs, coverage = admit_rows(run, expected, layouts)
        out['coverage'].append(coverage)
        rr, groups, gains, temps, met = {}, defaultdict(list), [], [], []
        capped = unmet = steps = 0
        cap = cfg.get('gain_cap') or 0.
        for i in sorted(expected):
            if any((i, layout) not in recs for layout in layouts):
                continue
            a, b = [recs[i, layout] for layout in layouts]
            if a['gold'] != b['gold'] or a['prompt_len'] != b['prompt_len']:
                raise ValueError('paired gold/prompt mismatch')
            rr[i] = {'cluster': b['batch'], 'flip': a['answer'] != b['answer'],
                     'accuracy': .5 * (a['correct'] + b['correct']),
                     'one': a['correct'] != b['correct'], 'correct_a': a['correct'],
                     'correct_b': b['correct'], 'missing_a': a['answer'] is None,
                     'missing_b': b['answer'] is None}
            groups[b['batch']].append(float(rr[i]['flip']))
            for record in [a, b]:
                g = list(record['gain'])
                if len(g) != cfg['soft_steps']:
                    raise ValueError('gain step count differs from run contract')
                gains.extend(g); steps += len(g)
                if cap:
                    unmet += sum(x > cap * 1.001 for x in g)
                    met.extend(x for x in g if x <= cap * 1.001)
                if 'temp' in record:
                    if len(record['temp']) != len(g):
                        raise ValueError('temperature/gain length mismatch')
                    temps.extend(record['temp'])
                    capped += sum(t < cfg['temperature'] for t in record['temp'])
        n = len(rr)
        k = sum(r['flip'] for r in rr.values())
        boot = bootstrap(groups, rng, draws)
        accuracy = {str(layout): mean([r['correct_' + suffix] for r in rr.values()])
                    for layout, suffix in zip(layouts, 'ab')}
        finite = all(math.isfinite(float(v)) for v in gains + temps)
        out['arms'][name] = {
            'run': cfg.get('out'), 'gain_cap': cfg.get('gain_cap'), 'top_n': cfg.get('top_n'),
            'temperature': cfg['temperature'], 'soft_steps': cfg.get('soft_steps'), 'layouts': layouts,
            'problems': n, 'clusters': len(groups), 'answer_flips': k, 'flip_rate': k / n if n else None,
            'flip_wilson95': wilson(k, n), 'flip_cluster95': boot['interval'], 'accuracy': accuracy,
            'accuracy_mean': mean(list(accuracy.values())) if n else None,
            'exactly_one_correct': mean([r['one'] for r in rr.values()]),
            'missing_answers': {str(layout): sum(r['missing_' + s] for r in rr.values())
                                for layout, s in zip(layouts, 'ab')},
            'steps_capped_share': capped / steps if temps and steps else 0.,
            'mean_step_temperature': mean(temps) if temps else cfg['temperature'],
            'mean_gain': mean(gains), 'max_gain': max(gains) if gains and finite else None,
            'steps_cap_unmet_share': unmet / steps if cap and steps else None,
            'median_gain_where_met': float(np.median(met)) if met and finite else None}
        coverage.update({'finite_step_values': finite, 'nonfinite_gain': sum(not math.isfinite(float(v)) for v in gains),
                         'nonfinite_temperature': sum(not math.isfinite(float(v)) for v in temps),
                         'step_count': steps, 'rounding': 'gain and temperature saved to 5 decimals',
                         'gain_boundary_ambiguous_count': sum(abs(float(g) - cap * 1.001) <= .000005 for g in gains) if cap else 0,
                         'temperature_boundary_ambiguous_count': sum(abs(float(t) - cfg['temperature']) <= .000005 for t in temps)})
        out['bootstrap'][name] = boot
        out['row_statistics'][name] = rr
        paired[name] = rr
    if base not in paired:
        raise ValueError('missing declared baseline')
    for name in paired:
        if name == base:
            continue
        x, y = paired[base], paired[name]
        common = sorted(x.keys() & y.keys())
        diffs = {key: paired_difference(common, x, y, key, rng, draws) for key in ['flip', 'accuracy', 'one']}
        b = sum(x[i]['flip'] and not y[i]['flip'] for i in common)
        c = sum(y[i]['flip'] and not x[i]['flip'] for i in common)
        out['contrasts'][name + ' - ' + base] = {
            'problems': len(common), 'flip_rate_difference': diffs['flip']['difference'],
            'flip_cluster95': diffs['flip']['interval'], 'flip_discordant_base_only_arm_only': [b, c],
            'flip_mcnemar_p': mcnemar(b, c), 'accuracy_difference': diffs['accuracy']['difference'],
            'accuracy_cluster95': diffs['accuracy']['interval'], 'exactly_one_correct_difference': diffs['one']['difference'],
            'exactly_one_cluster95': diffs['one']['interval']}
        out.setdefault('contrast_evidence', {})[name + ' - ' + base] = {
            'statistics': diffs, 'missing_base': sorted(y.keys() - x.keys()),
            'missing_arm': sorted(x.keys() - y.keys()),
            'cluster_mismatches': [i for i in common if x[i]['cluster'] != y[i]['cluster']],
            'reduces_flips_descriptive': diffs['flip']['interval'][1] < 0 if diffs['flip']['interval'] else None,
            'without_costing_accuracy_descriptive': diffs['accuracy']['interval'][0] > -.03 if diffs['accuracy']['interval'] else None}
    return out


def e3(soft0, soft, greedy, sampled, expected, draws=BOOT):
    rng = np.random.default_rng(SEED)
    runs = [soft0] + list(soft) + [greedy] + list(sampled)
    if len(soft) != 8 or len(sampled) != 8 or len({r['id'] for r in runs}) != len(runs):
        raise ValueError('E3 requires eight distinct soft and eight sampled replicates')
    normalized, coverage = [], []
    for run in runs:
        if run['args']['layouts'] != [16]:
            raise ValueError('E3 requires layout 16')
        r, c = admit_rows(run, expected, [16])
        normalized.append({i: row for (i, _), row in r.items()}); coverage.append(c)
    base, ss, gg, hh = normalized[0], normalized[1:9], normalized[9], normalized[10:]
    common = sorted(set(expected).intersection(*(set(r) for r in normalized)))
    per, realized = {}, []
    for i in common:
        gold = base[i]['gold']
        if any(r[i]['gold'] != gold or r[i]['prompt_len'] != base[i]['prompt_len'] for r in normalized):
            raise ValueError('E3 paired gold/prompt mismatch')
        sa, ha = [r[i]['answer'] for r in ss], [r[i]['answer'] for r in hh]
        sv, hv = plurality(sa), plurality(ha)
        pert = [r[i].get('perturb_realized') for r in ss]
        realized.extend(v for v in pert if v is not None)
        per[i] = {'cluster': base[i]['batch'], 'soft_vote': sv is not None and sv == gold,
                  'soft_single': mean([r[i]['correct'] for r in ss]), 'soft0': base[i]['correct'],
                  'greedy': gg[i]['correct'], 'sample_vote': hv is not None and hv == gold,
                  'sample_single': mean([r[i]['correct'] for r in hh]),
                  'soft_pass': any(r[i]['correct'] for r in ss), 'sample_pass': any(r[i]['correct'] for r in hh),
                  'soft_distinct': len(set(sa)), 'sample_distinct': len(set(ha)),
                  'soft_changed_vs0': mean([a != base[i]['answer'] for a in sa]),
                  'perturb_realized': pert, 'soft_answers': sa, 'sample_answers': ha,
                  'soft_winner': sv, 'sample_winner': hv}
    keys = ['soft_vote', 'soft_single', 'soft0', 'greedy', 'sample_vote', 'sample_single', 'soft_pass', 'sample_pass']
    acc = {key: mean([r[key] for r in per.values()]) for key in keys}
    pairs = [('soft_vote','soft_single'),('soft_vote','soft0'),('soft_vote','greedy'),
             ('soft_vote','sample_vote'),('sample_vote','sample_single'),('sample_vote','greedy'),('soft_single','soft0')]
    contrasts, evidence = {}, {}
    for a, b in pairs:
        groups = defaultdict(list)
        for row in per.values():
            groups[row['cluster']].append(float(row[a]) - float(row[b]))
        boot = bootstrap(groups, rng, draws)
        key = a + ' - ' + b
        contrasts[key] = {'difference': acc[a] - acc[b] if common else None, 'cluster95': boot['interval']}
        evidence[key] = boot
    sensitivity = {}
    for count in sorted({r['soft_distinct'] for r in per.values()}):
        rows = [r for r in per.values() if r['soft_distinct'] == count]
        sensitivity[str(count)] = {'problems': len(rows), **{k: mean([r[k] for r in rows]) for k in keys[:5]}}
    finite = all(math.isfinite(float(v)) for v in realized)
    projection = {'problems': len(common), 'clusters': len({r['cluster'] for r in per.values()}),
                  'K_soft': len(soft), 'K_sample': len(sampled), 'accuracy': acc, 'contrasts': contrasts,
                  'soft_answer_changed_vs_unperturbed': mean([r['soft_changed_vs0'] for r in per.values()]),
                  'soft_distinct_answers_mean': mean([r['soft_distinct'] for r in per.values()]),
                  'sample_distinct_answers_mean': mean([r['sample_distinct'] for r in per.values()]),
                  'by_soft_sensitivity_distinct_answers': sensitivity,
                  'perturb_realized_relative': {'min': min(realized), 'median': float(np.median(realized)), 'max': max(realized)}
                  if realized and finite else None,
                  'runs': {'soft0': soft0['id'], 'soft': [r['id'] for r in soft], 'greedy': greedy['id'],
                           'sample': [r['id'] for r in sampled]}}
    return {'projection': projection, 'coverage': coverage, 'rows': per, 'bootstrap': evidence,
            'common_indices': common, 'excluded_from_common': sorted(set(expected) - set(common)),
            'realized_nonfinite': sum(not math.isfinite(float(v)) for v in realized),
            'single_run_accuracies': {'soft': [mean([r[i]['correct'] for i in common]) for r in ss],
                                      'sample': [mean([r[i]['correct'] for i in common]) for r in hh]}}
