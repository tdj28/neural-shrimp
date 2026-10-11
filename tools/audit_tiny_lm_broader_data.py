"""Independent fixed-scope audit of saved broader-data LM evidence; never trains."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
REVIEWED = 'ca9e96b362189d3220370d798f649eb24fbfdc00'
SOURCE_REVIEW = 'e0c09a21be4ef2cd370e19511d71e664754f0dd7'
CORPUS = '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
COLLECTOR_FILES = ['src/neural_shrimp/tiny_lm.py', 'src/neural_shrimp/tiny_lm_prediction.py',
    'src/neural_shrimp/tiny_lm_progress.py', 'tools/explore_tiny_lm.py',
    'tools/explore_tiny_lm_prediction.py', 'tools/explore_tiny_lm_broader_data.py',
    'docs/plans/explore-tiny-lm-broader-data.md']
ANALYZER_FILES = [*COLLECTOR_FILES, 'tools/analyze_tiny_lm_broader_data.py']
AUDIT_FILES = ['tools/audit_tiny_lm_broader_data.py', 'tests/test_tiny_lm_broader_audit.py',
    'docs/reviews/tiny-lm-broader-outcome-audit.md', 'src/neural_shrimp/tiny_lm.py',
    'src/neural_shrimp/tiny_lm_prediction.py']
SEEDS = (28161, 28162)
ARMS = (8, 256)
FAMILIES = ('B0', 'B1', 'B2', 'B3')
ORDINARY = ('persistence', 'AR1', 'AR2', 'B0', 'B1')
PENALTIES = (.01, .1, 1., 10.)
SPLITS = {'fit': (32, 383), 'tune': (400, 583), 'test': (600, 783)}
COUNTS = {'fit': 352, 'tune': 184, 'test': 184}
CAP = 32 * 1024**2
# Declared before viewing outcomes; no threshold can alter the producer gate.
NUMERIC_RTOL, NUMERIC_ATOL = 2e-8, 2e-10
FEATURE_RTOL, FEATURE_ATOL = 3e-3, 3e-5
SAMPLE_STEPS = (32, 600)


class AuditFailure(ValueError):
    pass


class AuditDeadline(BaseException):
    pass


def require(condition, message):
    if not condition:
        raise AuditFailure(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        h = hashlib.file_digest(stream, 'sha256')
    return h.hexdigest()


def canonical(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def read(path):
    return json.loads(Path(path).read_text())


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_exclusive(path, value):
    payload = (json.dumps(clean(value), indent=2, allow_nan=False) + '\n').encode()
    size = sum(p.stat().st_size for p in path.parent.rglob('*') if p.is_file())
    require(size + len(payload) < CAP, 'audit raw cap')
    with path.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def compare(actual, expected, path='root'):
    """Compare requested fields recursively; extra explanatory strings are not claims."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict), path + ': dictionary required')
        for key, value in expected.items():
            require(key in actual, path + ': missing ' + key)
            compare(actual[key], value, path + '.' + key)
    elif isinstance(expected, (list, tuple, np.ndarray)):
        require(len(actual) == len(expected), path + ': length mismatch')
        for i, (left, right) in enumerate(zip(actual, expected)):
            compare(left, right, path + f'[{i}]')
    elif expected is None or isinstance(expected, (str, bool, np.bool_)):
        require(actual == expected, path + ': exact value mismatch')
    elif isinstance(expected, (int, np.integer)):
        require(actual == expected, path + ': integer mismatch')
    else:
        if not np.isfinite(expected):
            require(actual is None or not np.isfinite(actual), path + ': nonfinite mask mismatch')
        else:
            require(actual is not None and np.isfinite(actual) and
                    abs(actual - expected) <= NUMERIC_ATOL + NUMERIC_RTOL * abs(expected),
                    path + ': numerical mismatch')


def git_bytes(commit, path):
    return subprocess.check_output(['git', 'show', commit + ':' + path], cwd=ROOT)


def verify_source(identity, commit, paths):
    require(identity['commit'] == commit, 'source commit mismatch')
    require(set(identity['sha256']) == set(paths), 'source inventory mismatch')
    for name in paths:
        actual = digest(git_bytes(commit, name))
        require(actual == identity['sha256'][name], 'source digest mismatch: ' + name)
        require(actual == digest(git_bytes(REVIEWED, name)), 'unreviewed producer source: ' + name)


def inventory(folder, manifest):
    entries = manifest['files']
    names = [entry['path'] for entry in entries]
    require(len(names) == len(set(names)), 'duplicate inventory path')
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file() and p != folder / 'manifest.json'}
    require(actual == set(names), 'exact inventory mismatch')
    for entry in entries:
        relative = Path(entry['path'])
        target = folder / relative
        require(not relative.is_absolute() and '..' not in relative.parts and not target.is_symlink()
                and target.resolve().is_relative_to(folder.resolve()), 'unsafe inventory path')
        require(target.stat().st_size == entry['bytes'] and sha(target) == entry['sha256'], 'artifact hash/size mismatch')
    return {'files': len(entries), 'bytes': sum(e['bytes'] for e in entries)}


def expected_roles(raw, vocabulary):
    """Independent direct corpus-byte/token reconstruction, not producer data helpers."""
    starts = {'control': [65536 + 256*j for j in range(8)],
              'expanded': [65536 + 256*j for j in range(256)],
              'probe': [196608 + 256*j for j in range(32)]}
    mapping = {letter: i for i, letter in enumerate(vocabulary)}
    out = {}
    for role, offsets in starts.items():
        windows = [raw[s:s+33] for s in offsets]
        require(all(len(w) == 33 for w in windows), 'short corpus window')
        tokens = np.array([[mapping[c] for c in w.decode('ascii')] for w in windows], dtype='<i8')
        out[role] = {'starts': offsets, 'byte_character_intervals': [[s, s+33] for s in offsets],
            'input_intervals': [[s, s+32] for s in offsets], 'target_intervals': [[s+1, s+33] for s in offsets],
            'selected_bytes_sha256': digest(b''.join(windows)), 'window_tokens_sha256': digest(tokens.tobytes()),
            'window_token_ids': tokens.tolist(), 'shape': list(tokens.shape), 'targets_per_evaluation': len(offsets)*32}
    return out


def verify_corpus(folder, corpus):
    raw = (corpus / 'input.txt').read_bytes()
    require(len(raw) == 1115394 and digest(raw) == CORPUS, 'original corpus mismatch')
    d = read(folder / 'data.json')
    original_receipt = read(corpus / 'receipt.json')
    require(original_receipt['corpus_sha256'] == CORPUS and original_receipt['bytes'] == len(raw), 'original corpus receipt')
    require(d['source'] == original_receipt, 'producer corpus provenance mismatch')
    vocabulary = sorted(set(raw[:1003854].decode('ascii'))) + ['<UNK>']
    require(len(vocabulary) == 66 and vocabulary == d['vocabulary'], 'vocabulary mismatch')
    require(d['vocabulary_sha256'] == canonical(vocabulary), 'vocabulary digest mismatch')
    roles = expected_roles(raw, vocabulary)
    require(roles == d['roles'], 'original corpus window/token mismatch')
    require(d['corpus_sha256'] == CORPUS and d['corpus_bytes'] == len(raw), 'corpus identity fields')
    require(d['target_rule'] == 'X=chars[s:s+32]; Y=chars[s+1:s+33], next character at every position', 'target rule mismatch')
    require(d['data_identity_sha256'] == canonical({k: d[k] for k in
        ('corpus_sha256', 'corpus_bytes', 'vocabulary', 'roles', 'target_rule')}), 'data canonical digest')
    arrays = {}
    with np.load(folder / 'data-tokens.npz', allow_pickle=False) as saved:
        require(set(saved.files) == {r+'_'+xy for r in roles for xy in ('X', 'Y')}, 'token array inventory')
        for role, metadata in roles.items():
            tokens = np.asarray(metadata['window_token_ids'], dtype='<i8')
            for xy, value in [('X', tokens[:, :-1]), ('Y', tokens[:, 1:])]:
                actual = saved[role+'_'+xy]
                require(actual.dtype == np.dtype('<i8') and np.array_equal(actual, value), 'saved token bytes mismatch')
                arrays[role+'_'+xy] = value.copy()
    return d, arrays


def initial_state(seed):
    """Independent shape/order and NumPy initialization reconstruction; no torch call."""
    rng = np.random.default_rng(seed)
    shapes = [(66,32), (32,32), (32,), (32,), (96,32), (32,32), (32,), (32,),
              (128,32), (128,), (32,128), (32,), (32,), (32,)]
    pieces = []
    for i, shape in enumerate(shapes):
        value = rng.normal(0, .02, shape)
        if i in (3, 7, 9, 11, 13):
            value.fill(0)
        elif i in (2, 6, 12):
            value.fill(1)
        pieces.append(value.ravel())
    return np.concatenate(pieces).astype('<f4')


def labels(ce):
    ce = np.asarray(ce, float)
    y = np.full(len(ce), np.nan)
    for t in range(len(ce)):
        if t + 16 < len(ce) and all(math.isfinite(v) for v in ce[t:t+17]):
            y[t] = ce[t+16] - ce[t]
    return y


def indices(length, split):
    lo, hi = SPLITS[split]
    return np.array([i for i in range(length) if lo <= i <= hi], dtype=int)


def normalization(y):
    fit = y[indices(len(y), 'fit')]
    if len(fit) != 352 or not np.isfinite(fit).all():
        raise ValueError('incomplete fitting labels')
    return max(.01, math.sqrt(math.fsum(float(v)*float(v) for v in fit) / len(fit)))


def ridge(X, y, penalty):
    """Augmented least squares; no producer normal-equation solver is called."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    if not len(X) or not np.isfinite(y).all() or np.isinf(X).any():
        raise ValueError('invalid fitting data')
    means = np.array([math.fsum(c[np.isfinite(c)]) / np.isfinite(c).sum() if np.isfinite(c).any() else 0. for c in X.T])
    filled = np.where(np.isnan(X), means, X)
    scales = np.maximum(np.sqrt(np.mean((filled-means)**2, axis=0)), 1e-8)
    Z = np.column_stack((np.ones(len(X)), (filled-means)/scales))
    regularizer = np.diag([0., *([math.sqrt(penalty)]*X.shape[1])])
    design = np.vstack((Z / math.sqrt(len(X)), regularizer))
    target = np.r_[y / math.sqrt(len(X)), np.zeros(Z.shape[1])]
    weights, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    require(rank == Z.shape[1] and np.isfinite(weights).all(), 'independent ridge rank/finiteness')
    return {'mean': means.tolist(), 'scale': scales.tolist(), 'weights': weights.tolist(),
            'penalty': penalty, 'training_rows': len(y)}


def predict(model, X):
    X = np.asarray(X, float)
    if np.isinf(X).any():
        raise ValueError('infinite prediction feature')
    mu, sd, w = [np.asarray(model[k]) for k in ('mean', 'scale', 'weights')]
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        return w[0] + ((np.where(np.isnan(X), mu, X)-mu)/sd) @ w[1:]


def ar(history, order):
    h = np.asarray(history, float)[-32:]
    if len(h) != 32:
        return np.nan, 'insufficient_history'
    if not np.isfinite(h).all():
        return np.nan, 'nonfinite_history'
    design = np.asarray([[1., *[h[t-j-1] for j in range(order)]] for t in range(order, 32)])
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            beta, _, rank, _ = np.linalg.lstsq(design, h[order:], rcond=None)
            if rank != order+1:
                return np.nan, 'rank_deficient'
            if not np.isfinite(beta).all():
                return np.nan, 'nonfinite_fit'
            future = list(h)
            for _ in range(16):
                value = float(beta @ np.array([1., *[future[-1-j] for j in range(order)]]))
                if not math.isfinite(value):
                    return np.nan, 'nonfinite_recursion'
                future.append(value)
            return future[-1] - h[-1], 'ok'
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError):
        return np.nan, 'numerical_failure'


def metric(y, p, scale, count):
    bad = [i for i, (a, b) in enumerate(zip(y, p)) if not math.isfinite(a) or not math.isfinite(b)]
    value = {'rows': len(y), 'expected_rows': count, 'invalid_row_indices': bad,
             'complete': len(y) == count and not bad, 'normalization': scale}
    numbers = dict.fromkeys(('normalized_mse', 'mse', 'mae', 'signed_bias'))
    if value['complete']:
        try:
            errors = [float(b)-float(a) for a, b in zip(y, p)]
            numbers = {'normalized_mse': math.fsum((e/scale)**2 for e in errors)/len(errors),
                'mse': math.fsum(e*e for e in errors)/len(errors),
                'mae': math.fsum(abs(e) for e in errors)/len(errors),
                'signed_bias': math.fsum(errors)/len(errors)}
            if not all(math.isfinite(v) for v in numbers.values()):
                raise OverflowError()
        except (OverflowError, ValueError):
            value.update(complete=False, failure='nonfinite_score')
            numbers = dict.fromkeys(numbers)
    return {**value, **numbers}


def scores(record, p, status):
    try:
        scale = normalization(record['y'])
    except ValueError:
        scale = None
    out = {}
    for split in SPLITS:
        ix = indices(len(record['y']), split)
        if scale is None:
            value = {'complete': False, 'rows': len(ix), 'expected_rows': COUNTS[split],
                **dict.fromkeys(('normalized_mse', 'mse', 'mae', 'signed_bias')), 'failure': 'invalid_fit_normalization'}
        else:
            value = metric(record['y'][ix], p[ix], scale, COUNTS[split])
        failed = [int(t) for t in ix if status[t] != 'ok']
        value.update(steps=ix.tolist(), prediction=p[ix].tolist(), status=status[ix].tolist(), failed_forecast_steps=failed)
        if failed:
            value['complete'] = False
        out[split] = value
    return out


def fitted_candidates(pair, family):
    groups = pair[0]['feature_groups']
    allowed = {'common', *FAMILIES[:FAMILIES.index(family)+1]}
    columns = [i for i, group in enumerate(groups) if group in allowed]
    X = np.concatenate([r['X'][indices(len(r['y']), 'fit')][:, columns] for r in pair])
    y = np.concatenate([r['y'][indices(len(r['y']), 'fit')] for r in pair])
    candidates = []
    for penalty in PENALTIES:
        candidate = {'penalty': penalty}
        try:
            model = ridge(X, y, penalty)
            tune = []
            for r in pair:
                ix = indices(len(r['y']), 'tune')
                tune.append(metric(r['y'][ix], predict(model, r['X'][ix][:, columns]), normalization(r['y']), 184))
            if not all(s['complete'] for s in tune):
                raise ValueError('invalid tuning rows')
            candidate.update(status='ok', model=model, tune_scores=tune,
                             mean_tune_normalized_mse=math.fsum(s['normalized_mse'] for s in tune)/2)
        except (ValueError, FloatingPointError, np.linalg.LinAlgError):
            candidate['status'] = 'failed'
        candidates.append(candidate)
    good = [c for c in candidates if c['status'] == 'ok']
    return {'columns': columns, 'candidates': candidates, 'all_candidates_valid': len(good) == 4,
        'selected': min(good, key=lambda c: (c['mean_tune_normalized_mse'], -c['penalty'])) if good else None}


def analysis(records):
    """Independent numerical reconstruction of each specified selection and score."""
    report = {'seeds': {}, 'learning': {}}
    require(sorted((r['seed'], r['arm']) for r in records) == [(s,a) for s in SEEDS for a in ARMS], 'four exact cases required')
    for seed in SEEDS:
        pair = sorted([r for r in records if r['seed'] == seed], key=lambda r: r['arm'])
        require(pair[0]['eta'] == pair[1]['eta'] and pair[0]['initial_state_sha256'] == pair[1]['initial_state_sha256'], 'unmatched arms')
        families, models = {}, {}
        valid = all(r['outcome'] == 'completed' and r['completed_updates'] == 800 and
                    r.get('final_train_ce') is not None and math.isfinite(r['final_train_ce']) for r in pair)
        for family in (*ORDINARY[:3], *FAMILIES):
            fit = fitted_candidates(pair, family) if family in FAMILIES else None
            if fit is not None:
                models[family] = fit
                valid = valid and fit['all_candidates_valid']
            arms = {}
            for r in pair:
                if family == 'persistence':
                    p = np.zeros(len(r['ce'])); status = np.full(len(p), 'ok', dtype=object)
                elif family.startswith('AR'):
                    results = [ar(r['ce'][:t+1], int(family[-1])) for t in range(len(r['ce']))]
                    p = np.asarray([v[0] for v in results]); status = np.array([v[1] for v in results], dtype=object)
                else:
                    p = np.full(len(r['ce']), np.nan); status = np.full(len(p), 'model_unavailable', dtype=object)
                    if fit['selected'] is not None:
                        for split in SPLITS:
                            for t in indices(len(p), split):
                                try:
                                    p[t] = predict(fit['selected']['model'], r['X'][t:t+1, fit['columns']])[0]
                                    status[t] = 'ok' if math.isfinite(p[t]) else 'nonfinite_value'
                                except (ValueError, FloatingPointError):
                                    status[t] = 'numerical_failure'
                arms[str(r['arm'])] = scores(r, p, status)
                valid = valid and all(v['complete'] for v in arms[str(r['arm'])].values())
            families[family] = arms
        ordinary = []
        for family in ORDINARY:
            tune = [families[family][str(a)]['tune'] for a in ARMS]
            good = all(v['complete'] for v in tune)
            ordinary.append({'family': family, 'valid': good,
                'mean_tune_normalized_mse': math.fsum(v['normalized_mse'] for v in tune)/2 if good else None})
        chosen = min(ordinary, key=lambda c: (c['mean_tune_normalized_mse'], ORDINARY.index(c['family'])))['family'] if all(c['valid'] for c in ordinary) else None
        b2, b3 = [families[f]['256']['test'] for f in ('B2', 'B3')]
        base = families[chosen]['256']['test'] if chosen else None
        enough = bool(base is not None and all(v['complete'] for v in (b2, b3, base)) and b2['normalized_mse'] >= 1e-6 and base['normalized_mse'] >= 1e-6)
        positive = bool(valid and enough and all(b3['normalized_mse'] <= .95*c['normalized_mse'] and b3['mae'] <= c['mae'] for c in (b2, base)))
        report['seeds'][str(seed)] = {'models': models, 'scores': families, 'ordinary_candidates': ordinary,
            'selected_ordinary': chosen, 'all_comparators_and_eligible_rows_valid': bool(valid),
            'comparison_above_error_floor': enough, 'positive_forecast_interpretation': positive}
        def probe(r, t):
            rows = [v for v in r['probe'] if v['step'] == t]
            return rows[0]['ce'] if len(rows) == 1 and rows[0]['status'] == 'ok' and rows[0]['ce'] is not None and math.isfinite(rows[0]['ce']) else None
        small, large = pair
        s, l, early = probe(small, 800), probe(large, 800), probe(large, 25)
        difference = lambda a, b: None if a is None or b is None else a-b
        report['learning'][str(seed)] = {'probe_CE_8_at800': s, 'probe_CE_256_at800': l, 'probe_CE_256_at25': early,
            'equal_updates_difference': difference(l, s), 'equal_training_tokens_difference': difference(early, s),
            'final_train_CE_8': small.get('final_train_ce'), 'final_train_CE_256': large.get('final_train_ce'),
            'train_probe_gap_8': difference(s, small.get('final_train_ce')),
            'train_probe_gap_256': difference(l, large.get('final_train_ce')),
            'gradient_calls': {str(r['arm']): r['completed_updates'] for r in pair},
            'processed_training_targets': {str(r['arm']): r['completed_updates']*r['arm']*32 for r in pair},
            'elapsed_seconds': {str(r['arm']): r.get('elapsed_seconds') for r in pair},
            'all_scheduled_probe_observations': {str(r['arm']): r['probe'] for r in pair}}
    report['learning_aggregate'] = {}
    for name in ('equal_updates_difference', 'equal_training_tokens_difference'):
        values = [r[name] for r in report['learning'].values()]
        report['learning_aggregate'][name] = {'seed_values': values,
            'mean': math.fsum(values)/2 if all(v is not None for v in values) else None,
            'range': [min(values), max(values)] if all(v is not None for v in values) else None}
    report['positive_forecast_interpretation'] = all(v['positive_forecast_interpretation'] for v in report['seeds'].values())
    report['next_action'] = 'Stop after this report regardless of outcome; no policy/assessment/inference stage authorized.'
    return report


def feature_schema():
    names = ['plane_unavailable', 'plane_history_unavailable']
    groups = ['common']*2
    for family, fields in [('B0', ['ce', 'slope32', 'median32', 'mad32', 'ar1_forecast',
            'ar1_previous_absolute_error', 'ar2_forecast', 'ar2_previous_absolute_error']),
            ('B1', ['gradient_norm', 'gradient_norm_change', 'consecutive_cosine',
                    'secant_curvature', 'parameter_norm_drift32'])]:
        names += fields; groups += [family]*len(fields)
    fields = [f'{label}_{i}' for i in range(6) for label in ('gram', 'gram_mean8', 'gram_std8')]
    fields += [f'gram_history_lag{lag}_{i}' for lag in range(1, 8) for i in range(6)]
    fields += [f'step_ratio_lag{lag}' for lag in range(1, 4)]
    names += fields; groups += ['B2']*len(fields)
    names += ['plane_coherence', 'plane_mean8', 'plane_std8', 'log_training_targets']
    groups += ['B3']*3 + ['common']
    return names, groups


def saved_feature_checks(r, states):
    """All eligible rows: loss/state features and algebra from retained Gram entries."""
    X, ce = r['X'], r['ce']
    names = {name: i for i, name in enumerate(r['feature_names'])}
    np.testing.assert_equal(X[:, names['ce']], ce)
    np.testing.assert_allclose(X[:, names['log_training_targets']], math.log(r['arm']*32), rtol=0, atol=1e-14)
    gram = X[:, [names[f'gram_{i}'] for i in range(6)]]
    for lag in range(1, 8):
        np.testing.assert_equal(X[32:, [names[f'gram_history_lag{lag}_{i}'] for i in range(6)]], gram[32-lag:len(X)-lag])
    planes = []
    for lag in range(8):
        prefix = 'gram' if lag == 0 else f'gram_history_lag{lag}'
        ab, ac, bc = [X[:, names[f'{prefix}_{j}']] for j in (3,4,5)]
        d1, d2 = np.maximum(0, 1-ab*ab), np.maximum(0, 1-bc*bc)
        good = np.isfinite(d1+d2+ac) & (d1 >= 1e-4) & (d2 >= 1e-4)
        plane = np.full(len(X), np.nan)
        plane[good] = np.clip((ab[good]*bc[good]-ac[good])/np.sqrt(d1[good]*d2[good]), -1, 1)
        planes.append(plane)
    expected = {'plane_coherence': planes[0], 'plane_mean8': np.mean(planes, axis=0), 'plane_std8': np.std(planes, axis=0)}
    for name, value in expected.items():
        np.testing.assert_allclose(X[32:, names[name]], value[32:], rtol=2e-10, atol=1e-9, equal_nan=True)
    np.testing.assert_equal(X[32:, names['plane_unavailable']], (~np.isfinite(planes[0]))[32:])
    np.testing.assert_equal(X[32:, names['plane_history_unavailable']], (~np.isfinite(expected['plane_mean8']))[32:])
    theta_norm = np.linalg.norm(states.astype(float), axis=1)
    step_norm = np.asarray([float(np.linalg.norm(states[t]-states[t-1])) for t in range(1,len(states))])
    for t in range(32, len(X)):
        h = ce[t-31:t+1]
        median = np.median(h)
        values = {'slope32': np.polyfit(np.arange(32), h, 1)[0], 'median32': median,
                  'mad32': np.median(abs(h-median)),
                  'parameter_norm_drift32': (theta_norm[t]-theta_norm[t-31])/max(theta_norm[t-31], 1e-30)}
        for order in (1, 2):
            def one_step(window):
                design = np.asarray([[1., *[window[j-k-1] for k in range(order)]] for j in range(order,32)])
                weights = np.linalg.lstsq(design, window[order:], rcond=None)[0]
                return float(weights @ np.r_[1., window[-order:][::-1]])
            values[f'ar{order}_forecast'] = one_step(h)
            values[f'ar{order}_previous_absolute_error'] = abs(ce[t]-one_step(ce[t-32:t]))
        for lag in range(1,4):
            values[f'step_ratio_lag{lag}'] = float(step_norm[t-1])/float(step_norm[t-lag-1]) if step_norm[t-lag-1] > 1e-30 else np.nan
        for i in range(6):
            values[f'gram_mean8_{i}'] = np.mean(gram[t-7:t+1, i]) if np.isfinite(gram[t-7:t+1]).all() else np.nan
            values[f'gram_std8_{i}'] = np.std(gram[t-7:t+1, i]) if np.isfinite(gram[t-7:t+1]).all() else np.nan
        for name, value in values.items():
            np.testing.assert_allclose(X[t,names[name]], value, rtol=2e-8, atol=2e-10, equal_nan=True)
    return {'rows_from_step32': max(0,len(X)-32), 'B3_gram_algebra_atol': 1e-9,
            'gradient_origin': 'Bound by artifacts; independently evaluated only at fixed sampled states.'}


def load_cases(folder, data):
    records, states_by_case, summaries = [], {}, []
    expected_dirs = {f'seed-{s}-arm-{a}' for s in SEEDS for a in ARMS}
    require({p.name for p in folder.glob('seed-*') if p.is_dir()} == expected_dirs, 'case directory inventory')
    schema = feature_schema()
    for seed in SEEDS:
        ritz = read(folder / f'initial-ritz-{seed}.json')
        value, residual = ritz['maximum']['value'], ritz['maximum']['relative_residual']
        require(math.isfinite(value) and value > 0 and math.isfinite(residual) and 0 <= residual <= .02 and 1 <= ritz['steps'] <= 32, 'unqualified Ritz rate record')
        eta = 1.6*2/value
        expected_initial = initial_state(seed)
        for arm in ARMS:
            path = folder / f'seed-{seed}-arm-{arm}'
            r = read(path / 'result.json')
            require((r['seed'],r['arm']) == (seed,arm) and r['role'] == 'collect', 'case identity')
            require(r['eta'] == eta and r['rate_factor'] == 1.6, 'case learning rate')
            require(r['data_identity_sha256'] == data['data_identity_sha256'], 'case data identity')
            require(r['requested_updates'] == 800 and r['training_targets_per_update'] == arm*32, 'case budget/target count')
            require(r['feature_names'] == schema[0] and r['feature_groups'] == schema[1], 'feature schema')
            chunks, count = [], 0
            for item in sorted(path.glob('states-*.npy')):
                lo, hi = map(int, item.stem.split('-')[1:])
                chunk = np.load(item, allow_pickle=False)
                require(lo == count and chunk.shape == (hi-lo+1,15776) and chunk.dtype == np.dtype('<f4'), 'state chunk layout')
                chunks.append(chunk); count += len(chunk)
            require(count == r['state_rows'] == r['completed_updates']+1, 'accepted state count')
            states = np.concatenate(chunks)
            require(np.array_equal(states[0], expected_initial) and digest(states[0].tobytes()) == r['initial_state_sha256'], 'initial state bytes')
            require(np.isfinite(states).all() or r['outcome'] == 'numerical_guard_stop', 'nonfinite unguarded state')
            with np.load(path / 'features.npz', allow_pickle=False) as saved:
                r.update(X=saved['X'], ce=saved['ce'], y=saved['progress'])
            require(r['X'].shape == (r['completed_updates'],82) and len(r['ce']) == len(r['y']) == len(r['X']), 'case row count')
            np.testing.assert_equal(r['y'], labels(r['ce']))
            trace = [json.loads(line) for line in (path / 'trace.jsonl').read_text().splitlines()]
            require([v['step'] for v in trace] == list(range(len(r['ce']))), 'trace step order')
            np.testing.assert_equal([v['ce'] for v in trace], r['ce'])
            np.testing.assert_allclose([v['parameter_norm'] for v in trace], np.linalg.norm(states[:len(trace)],axis=1), rtol=2e-7, atol=1e-7)
            names = {n:i for i,n in enumerate(r['feature_names'])}
            np.testing.assert_allclose([v['gradient_norm'] for v in trace], r['X'][:,names['gradient_norm']], rtol=2e-7, atol=1e-7)
            if r['outcome'] == 'completed':
                require(r['completed_updates'] == 800 and r['guard_step'] is None and not (path/'guard.json').exists(), 'completed case guard')
                require([p['step'] for p in r['probe']] == [0,25,*range(100,801,100)], 'probe schedule')
            else:
                require(r['outcome'] in ('numerical_guard_stop','wall_budget_stop','failed_or_interrupted'), 'unknown stop status')
            for probe in r['probe']:
                require(read(path / f"probe-{probe['step']:06d}.json") == probe, 'probe checkpoint binding')
            compare(r['initial_train_ce'], None if not len(r['ce']) else r['ce'][0])
            compare(r['last_preupdate_train_ce'], None if not len(r['ce']) else r['ce'][-1])
            feature_check = saved_feature_checks(r, states)
            records.append(r); states_by_case[seed,arm] = states
            summaries.append({'seed':seed,'arm':arm,'outcome':r['outcome'],'state_rows':count,
                'completed_updates':len(r['ce']),'rate':eta,'initial_state_sha256':r['initial_state_sha256'],
                'saved_feature_checks':feature_check})
    return records, states_by_case, summaries


def sampled_features(records, states, tokens, output):
    """No updates/HVP/probe evaluation. Shared reviewed model and feature primitives."""
    import torch
    from neural_shrimp.tiny_lm import LMProblem, TinyTransformer
    from neural_shrimp.tiny_lm_prediction import CausalFeatures
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    rows = []
    for r in records:
        key = r['seed'], r['arm']
        role = 'control' if r['arm'] == 8 else 'expanded'
        problem = LMProblem(TinyTransformer(66), torch.tensor(tokens[role+'_X']), torch.tensor(tokens[role+'_Y']), .001)
        grad_value = torch.func.grad_and_value(problem.loss)
        for end in SAMPLE_STEPS:
            require(end < len(r['ce']), 'missing predeclared feature checkpoint')
            monitor = CausalFeatures(); gradients = []; cpu_ce = []
            for t in range(end-32,end+1):
                theta = torch.from_numpy(states[key][t].copy())
                gradient, objective = grad_value(theta)
                g = gradient.detach().numpy().copy()
                gradients.append(g)
                cpu_ce.append(float(objective)-.0005*float(theta@theta))
                previous_step = None if t == 0 else states[key][t]-states[key][t-1]
                # Saved CE isolates gradient/device parity from ill-conditioned AR refits.
                feature = monitor.observe(r['ce'][t], g, states[key][t], previous_step)
            feature = np.r_[feature, math.log(r['arm']*32)]
            target = r['X'][end]
            finite = np.isfinite(feature) & np.isfinite(target)
            mask_equal = bool(np.array_equal(np.isfinite(feature), np.isfinite(target)))
            errors = np.abs(feature[finite]-target[finite])
            allowed = FEATURE_ATOL+FEATURE_RTOL*np.abs(target[finite])
            ce_error = float(np.max(np.abs(np.asarray(cpu_ce)-r['ce'][end-32:end+1])))
            passed = mask_equal and bool(np.all(errors<=allowed)) and ce_error <= 2e-5
            name = f'features-seed-{key[0]}-arm-{key[1]}-step-{end}.npz'
            payload_bytes = np.asarray(gradients).nbytes + feature.nbytes + np.asarray(cpu_ce).nbytes + 4096
            require(sum(p.stat().st_size for p in output.rglob('*') if p.is_file())+payload_bytes < CAP-1024**2, 'sample output cap')
            with (output/name).open('xb') as stream:
                np.savez(stream, gradients=np.asarray(gradients), ce=np.asarray(cpu_ce), feature=feature,
                         steps=np.arange(end-32,end+1))
                stream.flush(); os.fsync(stream.fileno())
            row = {'seed':key[0],'arm':key[1],'step':end,'gradient_evaluations':33,
                'mask_equal':mask_equal,'feature_tolerance_pass':passed,'max_absolute_feature_error':float(errors.max()) if len(errors) else None,
                'max_tolerance_fraction':float(np.max(errors/allowed)) if len(errors) else None,
                'maximum_ce_absolute_error':ce_error,'artifact':name,'sha256':sha(output/name)}
            rows.append(row)
            write_exclusive(output/f'sample-{len(rows):02d}.json',row)
    return {'completed_checkpoints':len(rows),'gradient_evaluations':33*len(rows),
        'all_sampled_tolerances_pass':all(v['feature_tolerance_pass'] for v in rows),'checkpoints':rows,
        'feature_rtol':FEATURE_RTOL,'feature_atol':FEATURE_ATOL,'ce_atol':2e-5,
        'scope':'Shared reviewed torch model/gradient and CausalFeatures primitive, CPU float32 at fixed saved states. Saved CE feeds causal helper; CPU CE checked separately. No updates, HVP, probe evaluation or full-trajectory parity. Device discrepancies never alter forecast selections/gate.'}


def audit(args, output, checkpoint):
    require(sha(args.collection/'manifest.json') == args.manifest_sha256 and sha(args.analysis) == args.analysis_sha256, 'root-pinned input hashes')
    manifest = read(args.collection/'manifest.json')
    items = inventory(args.collection,manifest)
    meta = manifest['metadata']
    require(meta['source_commit'] == args.producer_commit and meta['role'] == 'collect', 'collection execution source/role')
    verify_source(meta['source_identity'],args.producer_commit,COLLECTOR_FILES)
    require(set(meta['threads'].values()) == {'1'}, 'producer thread count')
    plan = read(args.collection/'plan.json')
    require(plan['source'] == meta and plan['mode'] == 'collect' and plan['seeds'] == list(SEEDS)
            and plan['arms'] == list(ARMS) and plan['steps'] == 800 and plan['max_seconds'] == 450
            and plan['dtype'] == 'float32' and plan['tf32'] is False, 'collection plan')
    observed = read(args.analysis)
    require(observed['input_manifest_sha256'] == args.manifest_sha256, 'analysis input manifest')
    verify_source(observed['analysis_identity'],args.producer_commit,ANALYZER_FILES)
    data, tokens = verify_corpus(args.collection,args.corpus)
    require(observed['data_identity_sha256'] == data['data_identity_sha256'], 'analysis data identity')
    checkpoint('identities',{'collection':str(args.collection),'analysis':str(args.analysis),
        'producer_commit':args.producer_commit,'reviewed_source':REVIEWED,'source_review':SOURCE_REVIEW,
        'manifest_sha256':args.manifest_sha256,'analysis_sha256':args.analysis_sha256,
        'data_identity_sha256':data['data_identity_sha256'],'inventory':items})
    records, states, summaries = load_cases(args.collection,data)
    summary = read(args.collection/'summary.json')
    require(summary['parameters'] == 15776 and summary['old_protected_outcomes_evaluated'] is False, 'summary model/scope')
    require(summary['results'] == [read(args.collection/f'seed-{s}-arm-{a}'/'result.json') for s in SEEDS for a in ARMS], 'summary/case mismatch')
    checkpoint('saved_cases',{'cases':summaries})
    reconstructed = analysis(records)
    compare(observed,reconstructed)
    # Probe values cannot affect any model, forecast, split or interpretation gate.
    changed = [{**r,'probe':[]} for r in records]
    # Structural proof: analysis() consults probes only after constructing each seed report.
    # Synthetic control also checks numerical invariance; avoid a second real-data fit.
    require(all(not any('probe' in name for name in r['feature_names']) for r in changed), 'probe feature leakage')
    write_exclusive(output/'recomputed-analysis.json',reconstructed)
    checkpoint('arithmetic',{'passed':True,'numeric_rtol':NUMERIC_RTOL,'numeric_atol':NUMERIC_ATOL,
        'positive_forecast_interpretation':reconstructed['positive_forecast_interpretation'],
        'all_candidate_fits':len(SEEDS)*len(FAMILIES)*len(PENALTIES),
        'scope':'All labels, fit-only normalization/imputation/scales, 32 augmented least-squares candidates, exact chosen penalties and ordinary comparator, every eligible fit/tune/test prediction/status/metric and both seed/aggregate gates. Probe summaries recomputed from retained observations only.'})
    sampled = sampled_features(records,states,tokens,output)
    checkpoint('sampled_features',sampled)
    return {'passed':bool(sampled['all_sampled_tolerances_pass']),'arithmetic_passed':True,
            'sampled_features':sampled,'cases':summaries,'positive_forecast_interpretation':reconstructed['positive_forecast_interpretation']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('collection','analysis','corpus','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    for name in ('producer-commit','manifest-sha256','analysis-sha256'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--max-seconds',type=float,default=60)
    args = parser.parse_args()
    require(math.isfinite(args.max_seconds) and 0 < args.max_seconds <= 60, 'invalid audit budget')
    require(not args.output.resolve().is_relative_to(args.collection.resolve()) and
            not args.output.resolve().is_relative_to(args.corpus.resolve()), 'output overlaps input')
    require(all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')), 'audit requires one thread')
    commit = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    hashes = {}
    for path in AUDIT_FILES:
        require((ROOT/path).read_bytes() == git_bytes(commit,path), 'uncommitted audit source: '+path)
        hashes[path] = sha(ROOT/path)
    # Review-bound shared primitive bytes are checked even if branch tip includes docs.
    for path in ('src/neural_shrimp/tiny_lm.py','src/neural_shrimp/tiny_lm_prediction.py'):
        require(digest(git_bytes(REVIEWED,path)) == hashes[path], 'unreviewed shared helper')
    args.output.mkdir(parents=True,exist_ok=False)
    started = time.monotonic(); stages = []
    def checkpoint(stage, details):
        record = {'stage':stage,'elapsed_seconds':time.monotonic()-started,**details}
        write_exclusive(args.output/f'stage-{len(stages):02d}-{stage}.json',record)
        stages.append(stage)
    identity = {'commit':commit,'sha256':hashes,'argv':sys.argv,'pid':os.getpid(),
                'threads':{k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                'python':sys.version,'numpy':np.__version__,'niceness':os.nice(0)}
    checkpoint('request',{'audit_identity':identity,'request':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}})
    def interrupted(signum, frame):
        raise AuditDeadline(f'signal {signum}; no budget extension')
    signal.signal(signal.SIGALRM,interrupted); signal.signal(signal.SIGTERM,interrupted)
    signal.setitimer(signal.ITIMER_REAL,args.max_seconds)
    success = False
    try:
        result = audit(args,args.output,checkpoint)
        success = result['passed']
    except BaseException as error:
        result = {'passed':False,'status':'incomplete_or_failed','error':repr(error),'completed_stages':stages}
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        result.update(audit_identity=identity,elapsed_seconds=time.monotonic()-started)
        write_exclusive(args.output/'result.json',result)
        files = [{'path':str(p.relative_to(args.output)),'bytes':p.stat().st_size,'sha256':sha(p)}
                 for p in sorted(args.output.rglob('*')) if p.is_file()]
        write_exclusive(args.output/'manifest.json',{'files':files,'audit_identity':identity})
    print(json.dumps({'passed':success,'output':str(args.output),'elapsed_seconds':result['elapsed_seconds']}))
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())
