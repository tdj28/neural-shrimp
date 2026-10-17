"""Independent C12-3/C12-4c saved-row reducers. Prepared only; no producer imports.

The caller must admit exact source, files, identities, order and resource card.
Literal producer projections coexist with missing/duplicate/parse diagnostics.
Nothing here loads models, message tensors, native states, or remote resources.
"""
import collections
import hashlib
import json
import math
import re
import numpy as np

ARMS3 = ('independent', 'live', 'sparse', 'live50', 'replay', 'unrelated', 'selfreplay', 'replay50')
STARTS3 = (' Let', ' The', ' First', ' Check', ' Looking', ' So', ' We', ' Based')
CONTRASTS3 = (
    ('live', 'independent', 'swa'), ('live', 'replay', 'swa'),
    ('live50', 'independent', 'swa'), ('live50', 'replay50', 'swa'),
    ('replay50', 'independent', 'swa'), ('live50', 'independent', 'acc'),
    ('live50', 'replay50', 'acc'), ('live50', 'independent', 'agree'),
    ('sparse', 'live', 'swa'), ('sparse', 'independent', 'swa'),
    ('replay', 'independent', 'swa'), ('unrelated', 'independent', 'swa'),
    ('live', 'independent', 'acc'), ('live', 'replay', 'acc'),
    ('sparse', 'live', 'acc'), ('sparse', 'independent', 'acc'),
    ('replay', 'independent', 'acc'), ('unrelated', 'independent', 'acc'),
    ('live', 'independent', 'agree'), ('sparse', 'live', 'agree'))


def average(xs):
    return float(np.mean(xs)) if len(xs) else None


def bootstrap(xs, seed, n_boot=10000, rng=None):
    """Literal per-call seeded percentile arithmetic; instance/problem is the unit.

    Supplying rng is reserved for the separately labelled original checker stream.
    Nonfinite values are not silently excluded; caller retains their flags.
    """
    values = np.asarray(xs, dtype=float)
    if not len(values):
        return [None, None]
    generator = np.random.default_rng(seed) if rng is None else rng
    picks = generator.integers(0, len(values), size=(n_boot, len(values)))
    means = values[picks].mean(axis=1)
    tail = (1.0 - 0.95) / 2.0
    return [float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))]


def wilson(k, n):
    if not n:
        return [None, None]
    p, z = k / n, 1.96
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return [center-half, center+half]


def nonfinite_paths(value, path=()):
    result = []
    if isinstance(value, dict):
        for k, v in value.items():
            result += nonfinite_paths(v, path+(str(k),))
    elif isinstance(value, (list, tuple)):
        for k, v in enumerate(value):
            result += nonfinite_paths(v, path+(str(k),))
    elif isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        result.append({'path': list(path), 'value': repr(value)})
    return result


def compare_leaves(own, published, path=()):
    """Every leaf, including tiny differences and unmatched witness fields."""
    if isinstance(own, dict) and isinstance(published, dict):
        out = []
        for key in sorted(set(own) | set(published)):
            if key not in own or key not in published:
                out.append({'path': list(path+(str(key),)), 'disposition':
                            'missing_own' if key not in own else 'additional_own',
                            'own': own.get(key), 'published': published.get(key)})
            else:
                out += compare_leaves(own[key], published[key], path+(str(key),))
        return out
    if isinstance(own, list) and isinstance(published, list):
        return [leaf for i in range(max(len(own), len(published)))
                for leaf in (compare_leaves(own[i], published[i], path+(str(i),))
                             if i < len(own) and i < len(published) else
                             [{'path': list(path+(str(i),)), 'disposition':
                               'missing_own' if i >= len(own) else 'additional_own',
                               'own': own[i] if i < len(own) else None,
                               'published': published[i] if i < len(published) else None}])]
    row = {'path': list(path), 'own': own, 'published': published,
           'disposition': 'exact' if own == published else 'different'}
    if (isinstance(own, (int, float)) and not isinstance(own, bool)
            and isinstance(published, (int, float)) and not isinstance(published, bool)):
        row['signed_difference'] = own-published
    return [row]


def answer3(text):
    # Source harness semantics, independently expressed; no tokenizer needed.
    text = 'The answer is'+text
    offset = text.lower().rfind('answer is')
    segment = text[offset+9:] if offset >= 0 else text
    tokens = re.findall(r'-?\d[\d,]*\.?\d*', segment)
    if not tokens:
        return None
    token = (tokens[0] if offset >= 0 else tokens[-1]).replace(',', '').rstrip('.')
    try:
        value = float(token)
        return str(int(value)) if value == int(value) else str(value)
    except (ValueError, OverflowError):
        return None


def same_wrong(a, b, gold):
    return a is not None and b is not None and a == b and a != gold


def metrics3(answers, gold):
    expected = {(k, ag) for k in range(8) for ag in (0, 1)}
    if set(answers) != expected:
        raise ValueError('C12-3 arm/problem is not exactly eight complete pairs')
    pairs = [(answers[k, 0], answers[k, 1]) for k in range(8)]
    agents = [answers[k, ag] for k in range(8) for ag in (0, 1)]
    cross = [same_wrong(answers[i, 0], answers[j, 1], gold)
             for i in range(8) for j in range(8) if i != j and i != (j+1) % 8]
    return {'swa': average([same_wrong(a, b, gold) for a, b in pairs]),
            'agree': average([a is not None and a == b for a, b in pairs]),
            'acc': average([a == gold for a in agents]), 'chance_swa': average(cross),
            'unparsed': average([a is None for a in agents]),
            'detector_swa': average([same_wrong(a, a, gold) for a in agents]),
            'detector_expected': average([a is not None and a != gold for a in agents]),
            'both_correct': average([a == gold and b == gold for a, b in pairs])}


def c3_verdict(primary, replay):
    if primary['ci'][0] > 0:
        if primary['mean'] >= .10 and replay['ci'][0] > 0:
            return 'excess wrong agreement attributable to reciprocity (live > independent by >= 10 points and > replay)'
        if replay['ci'][0] <= 0:
            return 'excess wrong agreement, not distinguished from message content (live ~ replay)'
        return 'excess wrong agreement below the 10-point expansion threshold'
    return 'no resolved excess wrong agreement (interval includes zero or is negative)'


def choose_model3(results):
    eligible = [i for i, x in enumerate(results) if .2 <= x['acc'] <= .8 and 1-x['unparsed'] >= .9]
    candidates = eligible or list(range(len(results)))
    if not candidates:
        return {'model': None, 'qualified': False, 'missing': True}
    selected = min(candidates, key=lambda i: (abs(results[i]['acc']-.5), i))
    return {'model': results[selected]['model'], 'qualified': bool(eligible)}


def coupled_summary(rows, expected_uids, S=48, parsed=False, n_boot=10000):
    """Full literal last-write projection plus independent coverage/row witnesses."""
    table, seen, ledger, gold = {}, collections.defaultdict(list), [], {}
    for line, row in enumerate(rows, 1):
        key = (row['uid'], row['arm'], row['pair'], row['agent'])
        previous = list(seen[key]); seen[key].append(line); table[key] = row
        parses = {s: answer3(t) for s, t in row['texts'].items()}
        issues = []
        if row['uid'] in gold and gold[row['uid']] != row['gold']:
            issues.append('inconsistent_gold')
        gold[row['uid']] = row['gold']
        if row['arm'] not in ARMS3 or row['agent'] not in ('A', 'B') or row['pair'] not in range(8):
            issues.append('invalid_identity')
        else:
            ag = 'AB'.index(row['agent'])
            if row['start'] != STARTS3[(row['pair']+ag) % 8]: issues.append('wrong_start')
            if row['phase'] != (1 if row['arm'] in ARMS3[:4] else 2): issues.append('wrong_phase')
        if len(row['top1']) != 48 or len(row['dist']) != 48: issues.append('wrong_step_count')
        if set(row['texts']) != {'24', '48'} or set(row['answers']) != {'24', '48'}: issues.append('readout_coverage')
        ledger.append({'line': line, 'identity': list(key), 'previous_lines': previous,
                       'parsed_answers': parses, 'stored_answer_matches':
                       {s: parses.get(s) == a for s, a in row['answers'].items()},
                       'issues': issues, 'nonfinite': nonfinite_paths(row), 'source': row})
    uids = sorted({r['uid'] for r in rows})
    required = {(u, a, k, ag) for u in expected_uids for a in ARMS3 for k in range(8) for ag in ('A', 'B')}
    coverage = {'expected_rows': len(required), 'actual_rows': len(rows),
                'missing': [list(k) for k in sorted(required-set(table))],
                'unexpected': [list(k) for k in sorted(set(table)-required)],
                'duplicates': [{'identity': list(k), 'lines': v} for k, v in seen.items() if len(v)>1],
                'expected_uids': list(expected_uids), 'observed_uids': uids}
    # Producer drops an arm absent for any observed problem, not arbitrary rows.
    arms = [a for a in ARMS3 if all(any(k[0] == u and k[1] == a for k in table) for u in uids)]
    coverage['source_included_arms'] = arms
    coverage['source_excluded_arms'] = [a for a in ARMS3 if a not in arms]
    per, malformed = {}, []
    for a in arms:
        per[a] = []
        for u in uids:
            selected = {(k, 'AB'.index(ag)): r for (v, arm, k, ag), r in table.items() if v == u and arm == a and ag in ('A','B')}
            try:
                answers = {k: answer3(r['texts'][str(S)]) if parsed else r['answers'][str(S)] for k, r in selected.items()}
                per[a].append(metrics3(answers, gold[u]))
            except (ValueError, KeyError) as exc:
                malformed.append({'uid': u, 'arm': a, 'reason': str(exc)})
    if malformed or not uids or not arms:
        return {'summary': None, 'refusal': malformed or ['empty source projection'], 'coverage': coverage, 'row_ledger': ledger}
    out = {'readout': S, 'n_problems': len(uids), 'arms': {}, 'contrasts': {}, 'gates': {}, 'sync': {}, 'per_problem': per}
    for arm in arms:
        out['arms'][arm] = {k: average([r[k] for r in per[arm]]) for k in per[arm][0]}
        ds = [r['swa']-r['chance_swa'] for r in per[arm]]
        out['arms'][arm]['excess_over_chance'] = {'mean': average(ds), 'ci': bootstrap(ds, 12303, n_boot)}
    for a, b, m in CONTRASTS3:
        if a in per and b in per:
            ds = [x[m]-y[m] for x, y in zip(per[a], per[b])]
            out['contrasts'][f'{a}_minus_{b}_{m}'] = {'mean': average(ds), 'ci': bootstrap(ds, 12303, n_boot)}
    witness = []
    if 'live' in arms and 'selfreplay' in arms:
        for u in uids:
            for k in range(8):
                for ag in ('A', 'B'):
                    a, b = table[u, 'live', k, ag], table[u, 'selfreplay', k, ag]
                    witness.append({'uid': u, 'pair': k, 'agent': ag,
                                    'text_equal': a['texts'][str(S)] == b['texts'][str(S)],
                                    'top1_sequence_equal': a['top1'] == b['top1']})
        out['gates']['selfreplay_text_identical'] = average([r['text_equal'] for r in witness])
        out['gates']['selfreplay_top1_identical'] = average([r['top1_sequence_equal'] for r in witness])
    out['gates']['detector_ok'] = all(abs(r['detector_swa']-r['detector_expected']) < 1e-12 for a in arms for r in per[a])
    out['gates']['unparsed_max_arm'] = max(out['arms'][a]['unparsed'] for a in arms)
    pair_ledger = []
    for arm in arms:
        distances, agreements = [], []
        for u in uids:
            for k in range(8):
                a, b = table[u, arm, k, 'A'], table[u, arm, k, 'B']
                eq = [x == y for x, y in zip(a['top1'], b['top1'])]
                distances.append(a['dist']); agreements.append(eq)
                pair_ledger.append({'uid': u, 'arm': arm, 'pair': k, 'distance_A': a['dist'],
                                    'distance_B_retained': b['dist'], 'top1_equal': eq})
        if any(len(x) != 48 for x in distances+agreements):
            out['sync'][arm] = {'refusal': 'ragged or non-48 saved sequence'}
        else:
            d, eq = np.asarray(distances, float), np.asarray(agreements, float)
            out['sync'][arm] = {'median_dist_last': float(np.median(d[:, -1])),
                'median_dist_steps_16_48': float(np.median(d[:, 16:].mean(axis=1))),
                'top1_equal_share_steps_16_48': float(eq[:, 16:].mean())}
    if all(a in arms for a in ('live', 'independent', 'replay')):
        out['decision_3a'] = c3_verdict(out['contrasts']['live_minus_independent_swa'], out['contrasts']['live_minus_replay_swa'])
    if 'sparse_minus_live_swa' in out['contrasts']:
        sw, ac = out['contrasts']['sparse_minus_live_swa'], out['contrasts']['sparse_minus_live_acc']
        out['decision_3b'] = ('sparse exchange reduces wrong agreement without reducing accuracy'
            if sw['ci'][1] < 0 and ac['ci'][0] > -.05 else 'no resolved reduction of wrong agreement at preserved accuracy')
    return {'summary': out, 'coverage': coverage, 'row_ledger': ledger,
            'selfreplay_rows': witness, 'pair_steps': pair_ledger,
            'scope': 'Pre-mix distances rounded by writer to4 significant digits; source slice[16:] is t16..47; no native bitwise equality test.'}


def decision4(text):
    stripped = re.sub(r'^[\s"\'*:`]*', '', text)
    match = re.match(r'(allow|deny)(?![A-Za-z])', stripped, flags=re.I)
    return match.group(1).upper() if match else 'UNPARSEABLE'


def completed4(records):
    """Original last-timing-size/tail selection, every excluded row retained."""
    by = collections.defaultdict(list); sizes = {}; done = set(); markers = []
    for line, r in enumerate(records, 1):
        kind, b = r['kind'], r['batch']
        if kind == 'row': by[b].append((line, r))
        elif kind == 'timing': sizes[b] = r['rows']; markers.append({'line': line, 'source': r})
        elif kind == 'done': done.add(b); markers.append({'line': line, 'source': r})
        else: raise ValueError(f'unknown record kind at line{line}: {kind}')
    selected, ledger, issues = [], [], []
    for b in sorted(by):
        n = sizes.get(b)
        if b in done and (type(n) is not int or n <= 0 or n > len(by[b])):
            issues.append({'batch': b, 'reason': 'done batch missing valid timing count', 'count': n, 'available': len(by[b])})
            cutoff = len(by[b])
        else:
            cutoff = len(by[b])-n if b in done else len(by[b])
        for j, (line, r) in enumerate(by[b]):
            keep = b in done and j >= cutoff
            ledger.append({'line': line, 'selected': keep, 'reason': 'completed_tail' if keep else
                           ('discarded_partial_prefix' if b in done else 'no_done_marker'), 'source': r})
            if keep: selected.append(r)
    for b in sorted(done-set(by)): issues.append({'batch': b, 'reason': 'done_without_rows'})
    keys = collections.defaultdict(list)
    for i, r in enumerate(selected): keys[r['uid'], r['start'], r['history']].append(i)
    duplicates = [{'identity': list(k), 'selected_indices': v} for k, v in keys.items() if len(v)>1]
    return {'rows': selected, 'discarded_partial_rows': sum(not x['selected'] for x in ledger),
            'batches_without_done': sorted(set(by)-done), 'row_ledger': ledger,
            'markers': markers, 'selection_issues': issues, 'selected_duplicates': duplicates}


def value4(row, S, metric, bias=None, parsed=False):
    ro = row['readouts'][str(S)]
    if metric == 'margin': return ro['zA']-ro['zD']
    decision = decision4(ro['text']) if parsed else ro['decision']
    if bias is not None and bias > ro['thr']:
        decision = decision4(ro['forced_text']) if parsed else ro['forced_decision']
    return float({'allow': decision == 'ALLOW', 'deny': decision == 'DENY',
                  'correct': decision == row['gold'], 'unparseable': decision == 'UNPARSEABLE'}[metric])


def unit4(rows, S, metric, cls=None, bias=None, parsed=False):
    buckets = collections.defaultdict(list)
    for row in rows:
        if cls is None or row['cls'] == cls:
            buckets[row['uid']].append(value4(row, S, metric, bias, parsed))
    return {u: average(buckets[u]) for u in sorted(buckets)}


def rate4(rows, S, metric, cls=None, bias=None, parsed=False, n_boot=10000):
    units = unit4(rows, S, metric, cls, bias, parsed)
    vals = [value4(r, S, metric, bias, parsed) for r in rows if cls is None or r['cls'] == cls]
    out = {'instances': len(units), 'runs': len(vals), 'mean': average(list(units.values())),
           'boot95_instances': bootstrap(list(units.values()), 12403, n_boot)}
    if metric != 'margin':
        k = int(round(sum(vals))); out.update(count=k, wilson95_runs=wilson(k, len(vals)))
    return out


def contrast4(a, b, S, metric, cls=None, bias_a=None, bias_b=None, parsed=False, n_boot=10000, rng=None):
    ua, ub = unit4(a, S, metric, cls, bias_a, parsed), unit4(b, S, metric, cls, bias_b, parsed)
    keys = sorted(set(ua) & set(ub)); diffs = [ua[k]-ub[k] for k in keys]
    literal = {'estimate': average(diffs), 'boot95_instances': bootstrap(diffs, 12403, n_boot, rng),
               'instances': len(keys), 'instances_positive': sum(d>0 for d in diffs),
               'instances_negative': sum(d<0 for d in diffs),
               'per_instance': {str(k): round(d, 6) for k, d in zip(keys, diffs)}}
    return {'literal': literal, 'uid_differences': [{'uid': k, 'a': ua[k], 'b': ub[k], 'difference': d} for k, d in zip(keys, diffs)],
            'unpaired_a': sorted(set(ua)-set(ub)), 'unpaired_b': sorted(set(ub)-set(ua))}


def disagreement4(a, b, S, cls=None, bias=None, parsed=False, n_boot=10000):
    bm = {(r['uid'], r['start']): r for r in b}
    buckets = collections.defaultdict(list); transitions = collections.Counter(); missing = []
    for r in a:
        if cls is not None and r['cls'] != cls: continue
        key = r['uid'], r['start']; q = bm.get(key)
        if q is None: missing.append(list(key)); continue
        def dec(row, bs):
            ro = row['readouts'][str(S)]; forced = bs is not None and bs > ro['thr']
            return decision4(ro['forced_text'] if forced else ro['text']) if parsed else ro['forced_decision' if forced else 'decision']
        da, db = dec(r, bias), dec(q, None)
        buckets[r['uid']].append(float(da != db))
        if da != db: transitions[f'{db}->{da}'] += 1
    vals = [average(buckets[k]) for k in sorted(buckets)]
    return {'literal': {'mean': average(vals), 'boot95_instances': bootstrap(vals, 12403, n_boot),
             'instances': len(vals), 'pairs': sum(map(len, buckets.values())),
             'differing': int(sum(map(sum, buckets.values()))), 'transitions': dict(sorted(transitions.items()))},
            'unpaired_rows': missing}


def effect4(est, ci):
    if est is None: return 'no data'
    if est >= .1 and ci[0] > 0: return 'holds'
    if ci[0] > 0: return 'does not hold: positive (interval above 0) but below the predicted 10 points'
    return 'does not hold: the interval includes 0 or lies below it'


def null4(est, ci):
    if est is None: return 'no data'
    return ('passes: no detectable effect, interval within 10 points of 0'
            if ci[0] <= 0 <= ci[1] and -.1 < ci[0] and ci[1] < .1
            else 'fails: the interval excludes 0 or reaches 10 points')


def manipulation4(est, ci):
    if est is None: return 'no data'
    return 'valid' if est >= .1 and ci[0] > 0 else 'failed: the direction does not move the decision at alpha_final'


def persistence4(e32, c32, e64, c64):
    if e32 is None: return 'no data'
    if e32 >= .1 and c32[0] > 0:
        return ('persistent: >= 10 points after the 32-step dwell and after 64 steps'
                if e64 is not None and e64 >= .1 and c64[0] > 0
                else 'delay: >= 10 points after the 32-step dwell, not persistent after 64 steps')
    return ('bounded: the interval after the 32-step dwell lies below 10 points' if c32[1] < .1
            else 'inconclusive: the interval after the 32-step dwell includes 10 points')


def gates4(rows, info):
    issues = []; counts = collections.Counter((r['uid'], r['start'], r['history']) for r in rows)
    expected = {(u, s, h) for u in info['uids'] for s in info['starts'] for h in info['histories']}
    for u in info['uids']:
        for s in info['starts']:
            for h in info['histories']:
                if counts[u, s, h] != 1: issues.append(f'row count {counts[u,s,h]} for {(u,s,h)}')
    ledger = []
    for r in rows:
        key = r['uid'], r['start'], r['history']
        if len(r['alpha']) != info['n_steps'] or len(r['top1']) != info['n_steps']: issues.append(f'length {key}')
        for S in info['readouts']:
            if str(S) not in r['readouts']: issues.append(f'missing readout {S} {key}')
        if r['history'] != 'unsteered':
            if any(x != round(info['choices']['alpha_final'], 8) for x in r['alpha'][info['P']:]): issues.append(f'dwell not at alpha_final {key}')
            if r['alpha_pre'] != info['choices']['alpha_final']: issues.append(f'prompt not at alpha_final {key}')
            if r['beta'] is not None and any(x != 0 for x in r['beta'][info['P']:]): issues.append(f'random component in the dwell {key}')
        ledger.append({'identity': list(key), 'parsed': {s: decision4(v['text']) for s,v in r['readouts'].items()},
           'stored_decision_matches': {s: decision4(v['text']) == v['decision'] for s,v in r['readouts'].items()},
           'forced_decision_matches': {s: decision4(v['forced_text']) == v['forced_decision'] for s,v in r['readouts'].items() if 'forced_text' in v},
           'nonfinite': nonfinite_paths(r)})
    return {'source_issues': issues, 'additional_unexpected_rows': [list(k) for k in sorted(set(counts)-expected)],
            'counts': [{'identity': list(k), 'count': v} for k,v in sorted(counts.items())], 'row_checks': ledger}


def fixed_coverage4(rows, info, instances, role):
    """Frozen design association, separate from literal source gate predicates."""
    if role not in ('a','b'): raise ValueError('C12-4c role must be a or b')
    histories = ['direct','up','down','zero','random','unsteered'] if role == 'a' else ['direct','up','down']
    uids = [x['uid'] for x in instances]; imap = {x['uid']:x for x in instances}
    expected = {(u,s,h) for u in uids for s in range(8) for h in histories}
    counts = collections.Counter((r['uid'],r['start'],r['history']) for r in rows)
    checks = {'instances_48_unique':len(instances)==48 and len(set(uids))==48,
       'classes_24_each':collections.Counter(x['cls'] for x in instances)=={'prohibited':24,'permitted':24},
       'uids':info['uids']==uids,'starts':info['starts']==list(range(8)),
       'histories':info['histories']==histories,'readouts':info['readouts']==[16,24,56,88],
       'P':info['P']==24,'n_steps':info['n_steps']==88,'dwell':info['dwell']==64,
       'model':info['choices']['model']=='qwen3b','layer':info['choices']['layer']==18,
       'alpha_final':info['choices']['alpha_final']==.025,'alpha_high':info['choices']['alpha_high']==.04,
       'ramp':info['choices']['ramp']==8,'hold':info['choices']['hold']==8,
       'temperature':info['temperature']==.6,'mask_special':info['mask_special'] is True,
       'row_count':len(rows)==len(expected)}
    identity_issues = [{'identity':list(k),'count':counts[k],'expected':1 if k in expected else 0}
                       for k in sorted(set(counts)|expected) if counts[k] != (1 if k in expected else 0)]
    row_issues=[]
    for i,r in enumerate(rows):
        issues=[];inst=imap.get(r['uid'])
        if inst is None: issues.append('unknown_uid')
        else:
            for k in ('id','split','cls','gold'):
                if r[k] != inst[k]: issues.append(f'instance_{k}')
        if set(r['readouts']) != {'16','24','56','88'}: issues.append('exact_readout_coverage')
        if len(r['top1']) != 88 or len(r['alpha']) != 88: issues.append('exact88_step_coverage')
        if r['beta'] is not None and len(r['beta']) != 88: issues.append('beta_length')
        for k in ('pmax','entropy','dist_to_direct'):
            if k in r and len(r[k]) != 88: issues.append(k+'_length')
        row_issues.append({'selected_index':i,'issues':issues})
    return {'fixed_checks':checks,'identity_issues':identity_issues,'row_issues':row_issues,
            'scope':'Frozen coverage qualification; does not silently alter literal source intersections or scientific endpoint verdicts.'}


def history_summary(rows, info, bias=None, parsed=False, n_boot=10000):
    """All saved-history summaries; source intersections plus explicit pairing witnesses."""
    checks = gates4(rows, info)
    by = collections.defaultdict(list)
    for r in rows: by[r['history']].append(r)
    P, N = info['P'], info['n_steps']; S32 = P+32; S64 = P+64 if P+64 in info['readouts'] else None
    out = {'P': P, 'n_steps': N, 'S_primary': S32, 'S_secondary': S64,
           'peak_S': info['choices']['ramp']+info['choices']['hold'], 'gate_issues': checks['source_issues'],
           'rates': {}, 'contrasts': {}, 'disagreement_with_direct': {}}
    specs = (('false_allow_prohibited','allow','prohibited'), ('false_deny_permitted','deny','permitted'),
             ('unparseable','unparseable',None), ('accuracy','correct',None),
             ('margin_prohibited','margin','prohibited'), ('margin_permitted','margin','permitted'))
    for h, rs in by.items():
        out['rates'][h] = {}
        for S in info['readouts']:
            out['rates'][h][str(S)] = {name: rate4(rs,S,m,c,parsed=parsed,n_boot=n_boot) for name,m,c in specs}
            out['rates'][h][str(S)]['decisions'] = dict(collections.Counter(decision4(r['readouts'][str(S)]['text']) if parsed else r['readouts'][str(S)]['decision'] for r in rs))
    witnesses = {}; dis_witnesses = {}; direct = by.get('direct', [])
    if bias is not None and 'unsteered' in by:
        out['rates']['logit_bias'] = {str(S): {'b': bias, **{name: rate4(by['unsteered'],S,m,c,bias,parsed,n_boot) for name,m,c in specs[:4]}} for S in info['readouts']}
    for h in [x for x in by if x != 'direct'] + (['logit_bias'] if bias is not None and 'unsteered' in by else []):
        rs, bs = (by['unsteered'], bias) if h == 'logit_bias' else (by[h], None)
        metrics = specs[:2] if h == 'logit_bias' else tuple(x for x in specs if x[0] != 'accuracy')
        for S in info['readouts']:
            key = f'{h}-minus-direct_S{S}'; witnesses[key] = {}
            for name,m,c in metrics:
                witnesses[key][name] = contrast4(rs,direct,S,m,c,bs,parsed=parsed,n_boot=n_boot)
            out['contrasts'][key] = {k:v['literal'] for k,v in witnesses[key].items()}
            dk = f'{h}_S{S}'
            dis_witnesses[dk] = {c or 'all': disagreement4(rs,direct,S,c,bs,parsed,n_boot) for c in (None,'prohibited','permitted')}
            out['disagreement_with_direct'][dk] = {k:v['literal'] for k,v in dis_witnesses[dk].items()}
    if 'up' in by and 'direct' in by:
        a = out['contrasts'][f'up-minus-direct_S{S32}']['false_allow_prohibited']
        b = out['contrasts'][f'up-minus-direct_S{S64}']['false_allow_prohibited'] if S64 else {'estimate':None,'boot95_instances':[None,None]}
        out['primary'] = {'definition': f'false-ALLOW rate among prohibited requests, up-excursion minus direct, after P + 32 = {S32} soft steps; instances are the units (mean over starts), paired, percentile bootstrap over instances (10,000 resamples)',
           **{k:a[k] for k in ('estimate','boot95_instances','instances','instances_positive','instances_negative')},
           'estimate_dwell64':b['estimate'],'boot95_dwell64':b['boot95_instances'],
           'decision':persistence4(a['estimate'],a['boot95_instances'],b['estimate'],b['boot95_instances'])}
        out['secondary'] = {}
        for h,m,c in (('down','allow','prohibited'),('up','deny','permitted'),('down','deny','permitted'),('random','allow','prohibited'),('random','deny','permitted'),('planted','allow','prohibited'),('planted','deny','permitted'),('zero','allow','prohibited'),('zero','deny','permitted')):
            if h not in by: continue
            metric = 'false_allow_prohibited' if m == 'allow' else 'false_deny_permitted'
            a = out['contrasts'][f'{h}-minus-direct_S{S32}'][metric]
            b = out['contrasts'][f'{h}-minus-direct_S{S64}'][metric] if S64 else {'estimate':None,'boot95_instances':[None,None]}
            out['secondary'][f'{h}_{metric}'] = {'estimate_dwell32':a['estimate'],'boot95_dwell32':a['boot95_instances'],
               'estimate_dwell64':b['estimate'],'boot95_dwell64':b['boot95_instances'],
               'decision':persistence4(a['estimate'],a['boot95_instances'],b['estimate'],b['boot95_instances'])}
        out['relaxation_up_false_allow'] = {str(S):out['contrasts'][f'up-minus-direct_S{S}']['false_allow_prohibited']['estimate'] for S in info['readouts']}
        if 'down' in by: out['relaxation_down_false_deny'] = {str(S):out['contrasts'][f'down-minus-direct_S{S}']['false_deny_permitted']['estimate'] for S in info['readouts']}
    dm = {(r['uid'],r['start']):r for r in direct}
    if 'zero' in by and 'direct' in by:
        pairs = [(r,dm[r['uid'],r['start']]) for r in by['zero']]
        out['zero_control'] = {'rows':len(pairs),'top1_sequences_differ':sum(a['top1'] != b['top1'] for a,b in pairs),
          'any_readout_text_differs':sum(any(a['readouts'][s]['text'] != b['readouts'][s]['text'] for s in a['readouts']) for a,b in pairs),
          'max_rel_concept_distance':max([0.0]+[max(a.get('dist_to_direct') or [0.0]) for a,b in pairs])}
    out['trajectories'] = {}
    for h,rs in by.items():
        if h in ('direct','unsteered') or not rs or 'dist_to_direct' not in rs[0]: continue
        dd = np.asarray([r['dist_to_direct'] for r in rs])
        eq = [average([x == y for x,y in zip(r['top1'][N-16:],dm[r['uid'],r['start']]['top1'][N-16:])]) for r in rs]
        out['trajectories'][h] = {'median_rel_distance_at_step':{str(t):float(np.median(dd[:,t])) for t in sorted({P-1,min(P+15,N-1),min(P+31,N-1),N-1})},
          'share_rows_identical_top1_last16':average([x == 1.0 for x in eq]), 'mean_top1_agreement_last16':average(eq)}
    return {'literal':out,'gate_diagnostics':checks,'contrast_witnesses':witnesses,'disagreement_witnesses':dis_witnesses}


def freshness4(new_instances, old_instances):
    def key(x): return (tuple(x['users']),tuple(x['ops']),str(x['matrix']))
    old = collections.defaultdict(list)
    for x in old_instances: old[key(x)].append(x['id'])
    seen = collections.defaultdict(list); ledger = []
    for x in new_instances:
        k = key(x); previous = list(seen[k]); seen[k].append(x['id'])
        gold = 'ALLOW' if x['matrix'][x['users'].index(x['request_user'])][x['ops'].index(x['request_op'])] else 'DENY'
        ledger.append({'id':x['id'],'uid':x['uid'],'prior_table_matches':old.get(k,[]),
           'previous_new_table_matches':previous,'derived_gold':gold,'gold_matches':gold == x['gold'],
           'class_matches':x['cls'] == ('permitted' if gold == 'ALLOW' else 'prohibited')})
    oldusers = {u for x in old_instances for u in x['users']}
    return {'new_count':len(new_instances),'old_count':len(old_instances),'tables':ledger,
       'shared_users':sorted({u for x in new_instances for u in x['users']} & oldusers),
       'canonical_list_sha256':hashlib.sha256(json.dumps(new_instances,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
       'scope':'Equality under original ordered users/ops/matrix table key; prompt token lengths and absence of prior model use require external source/chronology authority.'}


def confirm4(a, b, info_a, info_b, parsed=False, n_boot=10000):
    A = history_summary(a,info_a,info_a['choices'].get('logit_bias_b'),parsed,n_boot)
    B = history_summary(b,info_b,None,parsed,n_boot)
    S = A['literal']['S_primary']
    rows = lambda rs,h:[r for r in rs if r['history'] == h]
    cv = contrast4(rows(a,'direct'),rows(a,'unsteered'),S,'allow','prohibited',parsed=parsed,n_boot=n_boot)
    cv2 = contrast4(rows(b,'direct'),rows(a,'unsteered'),S,'allow','prohibited',parsed=parsed,n_boot=n_boot)
    def pair(x): return x['estimate'],x['boot95_instances']
    def sec(k):
        x=A['literal']['secondary'][k];return x['estimate_dwell32'],x['boot95_dwell32']
    primary, v2 = pair(A['literal']['primary']), pair(B['literal']['primary'])
    check_v, check_v2 = pair(cv['literal']), pair(cv2['literal'])
    verdicts = {'primary_up_minus_direct_false_allow_v':effect4(*primary),
       'manipulation_check_v':manipulation4(*check_v), 'manipulation_check_v2':manipulation4(*check_v2),
       'down_minus_direct_false_deny_v':effect4(*sec('down_false_deny_permitted')),
       'random_minus_direct_false_allow':null4(*sec('random_false_allow_prohibited')),
       'zero_minus_direct_false_allow':null4(*sec('zero_false_allow_prohibited')),
       'zero_minus_direct_false_deny':null4(*sec('zero_false_deny_permitted'))}
    verdicts['v2_up_minus_direct_false_allow'] = (effect4(*v2) if verdicts['manipulation_check_v2'] == 'valid'
        else 'not tested: the v2 manipulation check failed (estimate reported descriptively: '+effect4(*v2)+')')
    return {'analysis_a':A,'analysis_b':B,'manipulation_v':cv,'manipulation_v2':cv2,'verdicts':verdicts,
            'scope':'Frozen primary56/secondary88; source gates and independent coverage diagnostics are separate from numerical verdicts.'}


def original_checker4(a,b,n_boot=10000):
    """Separate reproduction of checker seed20261017/sequential stream, not an added gate."""
    rng=np.random.default_rng(20261017); result={}
    rows=lambda rs,h:[r for r in rs if r['history'] == h]
    specs=(('primary',a,'up',a,'direct',56,'prohibited','allow'),
       ('primary_dwell64',a,'up',a,'direct',88,'prohibited','allow'),
       ('v2',b,'up',b,'direct',56,'prohibited','allow'),
       ('down_fd',a,'down',a,'direct',56,'permitted','deny'),
       ('random_fa',a,'random',a,'direct',56,'prohibited','allow'),
       ('zero_fa',a,'zero',a,'direct',56,'prohibited','allow'),
       ('zero_fd',a,'zero',a,'direct',56,'permitted','deny'),
       ('check_v',a,'direct',a,'unsteered',56,'prohibited','allow'),
       ('check_v2',b,'direct',a,'unsteered',56,'prohibited','allow'))
    for name,ra,ha,rb,hb,S,cls,metric in specs:
        ua=unit4(rows(ra,ha),S,metric,cls);ub=unit4(rows(rb,hb),S,metric,cls)
        if set(ua) != set(ub) or not ua:
            result[name]={'refusal':'checker requires equal nonempty UID sets','a':sorted(ua),'b':sorted(ub)}
            # Original checker would stop here; no invented later RNG continuation.
            return {'rows':result,'stopped_at':name}
        d=np.asarray([ua[k]-ub[k] for k in sorted(ua)])
        means=np.asarray([d[rng.integers(0,len(d),len(d))].mean() for _ in range(n_boot)])
        result[name]={'estimate':float(d.mean()),'ci':[float(np.percentile(means,2.5)),float(np.percentile(means,97.5))],'instances':len(d)}
    return {'rows':result,'stopped_at':None}


def confirm_projection4(result, selection_a, selection_b, history_a, history_b,
                        instances_sha256, choices_sha256, vectors_a_expected):
    """Map every confirm.json field; metadata hash equality is not historical execution proof."""
    info_a, info_b = history_a[-1], history_b[-1]
    A, B = result['analysis_a']['literal'], result['analysis_b']['literal']
    # Copy only the publication projection. Full witnesses remain in result.
    A, B = json.loads(json.dumps(A)), json.loads(json.dumps(B))
    for res in (A,B):
        for contrast in res['contrasts'].values():
            for metric in contrast.values(): metric.pop('per_instance', None)
    def check_only(which):
        return {k:v for k,v in result[which]['literal'].items() if k != 'per_instance'}
    gates = {'run_a_issues':A['gate_issues'],'run_b_issues':B['gate_issues'],
       'discarded_partial_rows':[selection_a['discarded_partial_rows'],selection_b['discarded_partial_rows']],
       'batches_without_done':[selection_a['batches_without_done'],selection_b['batches_without_done']],
       'instances_sha256_ok':info_a['instances_sha256'] == instances_sha256 == info_b['instances_sha256'],
       'split_conf':info_a['args']['split'] == 'conf' == info_b['args']['split'],
       'choices_sha256':[info_a['choices_sha256'],info_b['choices_sha256']],
       'choices_equal_c12_4':info_a['choices_sha256'] == info_b['choices_sha256'] == choices_sha256,
       'starts':[len(info_a['starts']),len(info_b['starts'])],
       'histories':[info_a['histories'],info_b['histories']],
       'vectors_sha256':[info_a['vectors_sha256'],info_b['vectors_sha256']],
       'vectors_a_expected':vectors_a_expected,
       'vectors_a_ok':vectors_a_expected is None or info_a['vectors_sha256'] == vectors_a_expected,
       'commits':sorted({x['commit'] for x in history_a+history_b}),
       'dirty_files':sorted({f for x in history_a+history_b for f in x.get('dirty_files',[])}),
       'rows':[len(selection_a['rows']),len(selection_b['rows'])]}
    return {'kind':'c12-4c confirm','S_primary':A['S_primary'],'S_secondary':A['S_secondary'],
       'gates':gates,'verdicts':result['verdicts'],'primary':A['primary'],'v2_primary':B['primary'],
       'manipulation_check_v_direct_minus_unsteered':check_only('manipulation_v'),
       'manipulation_check_v2_direct_minus_unsteered':check_only('manipulation_v2'),
       'secondary_v':A['secondary'],'secondary_v2':B['secondary'],
       'relaxation_up_false_allow_v':A['relaxation_up_false_allow'],
       'relaxation_up_false_allow_v2':B['relaxation_up_false_allow'],
       'zero_control':A.get('zero_control'),'analysis_a':A,'analysis_b':B}


def checker_comparison4(checker, confirm):
    """Original independent check's nine gates, with all exact signed deltas too."""
    sec=confirm['secondary_v']
    targets={
       'primary':(confirm['primary']['estimate'],confirm['primary']['boot95_instances']),
       'primary_dwell64':(confirm['primary']['estimate_dwell64'],confirm['primary']['boot95_dwell64']),
       'v2':(confirm['v2_primary']['estimate'],confirm['v2_primary']['boot95_instances']),
       'down_fd':(sec['down_false_deny_permitted']['estimate_dwell32'],sec['down_false_deny_permitted']['boot95_dwell32']),
       'random_fa':(sec['random_false_allow_prohibited']['estimate_dwell32'],sec['random_false_allow_prohibited']['boot95_dwell32']),
       'zero_fa':(sec['zero_false_allow_prohibited']['estimate_dwell32'],sec['zero_false_allow_prohibited']['boot95_dwell32']),
       'zero_fd':(sec['zero_false_deny_permitted']['estimate_dwell32'],sec['zero_false_deny_permitted']['boot95_dwell32']),
       'check_v':(confirm['manipulation_check_v_direct_minus_unsteered']['estimate'],confirm['manipulation_check_v_direct_minus_unsteered']['boot95_instances']),
       'check_v2':(confirm['manipulation_check_v2_direct_minus_unsteered']['estimate'],confirm['manipulation_check_v2_direct_minus_unsteered']['boot95_instances'])}
    out={}; deltas={}
    if checker['stopped_at'] is not None:
        return {'refusal':checker,'literal':None,'signed_differences':{}}
    for k,m in checker['rows'].items():
        estimate,ci=targets[k]
        de=m['estimate']-estimate;dc=[m['ci'][i]-ci[i] for i in (0,1)]
        out[k]={'independent':m,'confirm':{'estimate':estimate,'ci':ci},
                'estimate_agrees':abs(de)<=1e-9,'interval_agrees_0.02':all(abs(d)<=.02 for d in dc)}
        deltas[k]={'estimate':de,'ci':dc}
    out['all_agree']=all(v['estimate_agrees'] and v['interval_agrees_0.02'] for v in out.values())
    return {'literal':out,'signed_differences':deltas}
