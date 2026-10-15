"""Own BZ/BP saved-arithmetic components, source-only prepared; no target I/O.

Original source association and actual row allocation remain adapter prerequisites.
Caller supplies one RNG per original analysis invocation and preserves its call order.
"""
import math
import re
import string
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def normalize_answer(text):
    s = ''.join(c for c in text.lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', s).split())


def arithmetic_score(text, answer):
    first = re.search(r'-?\d+', text.replace(',', ''))
    return bool(first and int(first.group()) == answer)


def nq_score(text, answers):
    response = normalize_answer(text)
    return any(normalize_answer(a) and normalize_answer(a) in response for a in answers)


def text_score_record(text, saved_correct, task, item, cap):
    require(isinstance(text, str) and type(saved_correct) is bool, 'invalid reply/flag')
    require(len(text) <= cap, 'saved reply exceeds original cap')
    if task == 'mbpp':
        return {'task': task, 'saved_correct': saved_correct, 'own_score': None,
                'comparison': 'unavailable: generated code is not executed'}
    require(task in ('arithmetic', 'nq'), 'unknown task')
    own = arithmetic_score(text, item['answer']) if task == 'arithmetic' else nq_score(text, item['answers'])
    # At the cap the original may be longer; do not promote a prefix score to a full-response proof.
    return {'task': task, 'saved_correct': saved_correct, 'own_score': bool(own),
            'equal': bool(own == saved_correct), 'saved_characters': len(text),
            'full_text_available_from_length': len(text) < cap,
            'comparison': 'full saved response' if len(text) < cap else 'possibly truncated prefix only'}


def sign_test(values):
    """Literal source nonzero/binomial arithmetic; invalid values remain flagged."""
    x = np.asarray(values, dtype=np.float64)
    require(x.ndim == 1, 'expected scalar vector')
    nz = x[x != 0]
    positive = int((nz > 0).sum())
    n = len(nz)
    numerator = 2 * sum(math.comb(n, k) for k in range(min(positive, n-positive)+1)) if n else 1
    denominator = 2 ** n if n else 1
    return {'positive': positive, 'nonzero': n,
            'sign_test_p': min(1.0, numerator / denominator),
            'finite_qualified': bool(np.isfinite(x).all()),
            'nonfinite_values': int((~np.isfinite(x)).sum())}


def paired_bootstrap(values, rng, draws=10000):
    """One pair-difference resample per draw; original linear percentile convention."""
    x = np.asarray(values, dtype=np.float64)
    require(x.ndim == 1 and draws > 0, 'invalid pair vector/draw count')
    if not len(x):
        return {'defined': False, 'reason': 'no complete pairs', 'pairs': 0, 'draws_consumed': 0}
    samples = np.empty(draws, dtype=np.float64)
    for b in range(draws):
        samples[b] = np.mean(rng.choice(x, len(x)))
    with np.errstate(invalid='ignore', over='ignore'):
        interval = np.percentile(samples, [2.5, 97.5], method='linear')
    return {'defined': bool(np.isfinite(x).all() and np.isfinite(samples).all()),
            'pairs': len(x), 'mean': float(np.mean(x)),
            'bootstrap_95': interval.tolist(), 'draws_consumed': draws,
            'finite_draws': int(np.isfinite(samples).sum()),
            'nonfinite_draws': int((~np.isfinite(samples)).sum()), **sign_test(x)}


def prompt_permutation(chaotic, periodic, rng, draws=10000):
    """Prompt accuracies, not response rows; preserve concatenated-group/source order."""
    a, b = np.asarray(chaotic, dtype=np.float64), np.asarray(periodic, dtype=np.float64)
    require(a.ndim == b.ndim == 1 and draws > 0, 'invalid prompt vector/draw count')
    if not len(a) or not len(b):
        return {'defined': False, 'reason': 'missing label group', 'chaotic': len(a),
                'periodic': len(b), 'draws_consumed': 0}
    require(np.isfinite(a).all() and np.isfinite(b).all(), 'nonfinite prompt accuracy')
    require(((0 <= a) & (a <= 1)).all() and ((0 <= b) & (b <= 1)).all(), 'accuracy outside [0,1]')
    observed = float(np.mean(a) - np.mean(b))
    pool, n = np.concatenate((a, b)), len(a)
    exceed = 0
    for _ in range(draws):
        shuffled = rng.permutation(pool)
        statistic = float(np.mean(shuffled[:n]) - np.mean(shuffled[n:]))
        exceed += abs(statistic) >= abs(observed)
    return {'defined': True, 'chaotic': len(a), 'periodic': len(b),
            'acc_chaotic': float(np.mean(a)), 'acc_periodic': float(np.mean(b)),
            'difference': observed, 'permutation_p': exceed / draws,
            'exceedances': int(exceed), 'draws_consumed': draws,
            'plus_one_p_descriptive': (exceed + 1) / (draws + 1),
            'calibration': 'Inclusive absolute comparison, raw Monte Carlo fraction; not a randomized-treatment test.'}


def task_composition(rows, tasks=('arithmetic', 'nq')):
    """Descriptive symmetric decomposition; no extrapolation across absent support.

On all-stratum common support: raw C-P = sum mean(group shares)*within-task
contrast + sum share contrast*mean(group accuracies). This is an algebraic
composition description, not adjustment for causal confounding.
"""
    require(len(tasks) == len(set(tasks)), 'duplicate task names')
    require(all(r['task'] in tasks and r['label'] in ('chaotic', 'periodic') for r in rows), 'unexpected task/label')
    require(all(np.isfinite(r['acc']) and 0 <= r['acc'] <= 1 for r in rows), 'invalid prompt accuracy')
    values = {t: {g: [float(r['acc']) for r in rows if r['task'] == t and r['label'] == g]
                  for g in ('chaotic', 'periodic')} for t in tasks}
    totals = {g: sum(len(values[t][g]) for t in tasks) for g in ('chaotic', 'periodic')}
    table = []
    for t in tasks:
        item = {'task': t}
        for g in ('chaotic', 'periodic'):
            x = values[t][g]
            item[g] = {'prompts': len(x), 'accuracy': float(np.mean(x)) if x else None,
                       'within_label_task_share': len(x)/totals[g] if totals[g] else None}
        item['common_support'] = bool(values[t]['chaotic'] and values[t]['periodic'])
        item['difference'] = item['chaotic']['accuracy']-item['periodic']['accuracy'] if item['common_support'] else None
        table.append(item)
    eligible = [x for x in table if x['common_support']]
    out = {'tasks': table, 'totals': totals,
           'common_support_tasks': [x['task'] for x in eligible],
           'complete_support': len(eligible) == len(tasks),
           'interpretation': 'Observed task-composition algebra; no causal effect or historical regression reconstruction.'}
    if all(totals.values()):
        means = {g: float(np.mean([r['acc'] for r in rows if r['label'] == g])) for g in totals}
        out['raw_difference'] = means['chaotic']-means['periodic']
    if len(eligible) != len(tasks):
        out['decomposition'] = {'defined': False, 'reason': 'at least one task lacks both labels; no imputed task mean'}
        return out
    within, mix = 0.0, 0.0
    for item in table:
        c, p = item['chaotic'], item['periodic']
        item['symmetric_within_contribution'] = .5*(c['within_label_task_share']+p['within_label_task_share'])*item['difference']
        item['symmetric_composition_contribution'] = .5*(c['accuracy']+p['accuracy'])*(c['within_label_task_share']-p['within_label_task_share'])
        within += item['symmetric_within_contribution']; mix += item['symmetric_composition_contribution']
    out['decomposition'] = {'defined': True, 'within_task': within, 'composition': mix,
                            'sum': within+mix, 'closure_difference': within+mix-out['raw_difference']}
    return out
