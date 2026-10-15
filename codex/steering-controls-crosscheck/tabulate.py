"""Independent Task16 saved-scan arithmetic; no producer imports or model calls."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np

THRESHOLD = 1e-3
EDGES = (0., .01, .03, .06, .09, .12)
MAX_ARCHIVE_BYTES = 128 * 1024**2
REQUIRED = ('log10_temperature', 'alpha', 'lyapunov', 'period')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def integer(value, label, minimum=0):
    require(type(value) is int and value >= minimum, label + ': integer required')
    return value


def bound_path(root, relative):
    rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts, 'nonrelative input path')
    root = Path(root).resolve()
    p = root / rel
    require(not p.is_symlink() and p.resolve().is_relative_to(root), 'input path escapes authority')
    return p


def authenticate(root, spec):
    p = bound_path(root, spec['path'])
    require(p.is_file(), 'missing input: ' + spec['path'])
    require(p.stat().st_size == spec['bytes'], 'input size: ' + spec['path'])
    require(digest(p) == spec['sha256'], 'input hash: ' + spec['path'])
    return p


def numeric(a, name):
    require(a.dtype.kind in 'fiu' and a.dtype.itemsize <= 8, name + ': real numeric dtype required')
    return a


def axis(a, n, label):
    numeric(a, label)
    require(a.ndim == 1 and a.size == n, label + ': axis shape')
    a = a.astype(np.float64)
    require(np.isfinite(a).all() and (n == 1 or (np.diff(a) > 0).all()), label + ': finite increasing unique axis')
    return a


def load_segment(root, spec):
    """Only four numeric arrays, plus partial done; never deserialize object members."""
    p = authenticate(root, spec)
    shape = tuple(spec['shape'])
    require(len(shape) == 2 and all(type(n) is int and n > 0 for n in shape), 'segment shape contract')
    require(spec['layout'] in ('full', 'partial'), 'unknown layout')
    with zipfile.ZipFile(p) as z:
        names = [i.filename for i in z.infolist()]
        require(len(names) == len(set(names)), 'duplicate archive member')
        require(sum(i.file_size for i in z.infolist()) <= MAX_ARCHIVE_BYTES, 'archive uncompressed bound')
        for key in REQUIRED + (('done',) if spec['layout'] == 'partial' else ()):
            require(key + '.npy' in names, 'missing numeric member: ' + key)
    with np.load(p, allow_pickle=False) as z:
        t = axis(z['log10_temperature'], shape[0], 'temperature')
        a = axis(z['alpha'], shape[1], 'alpha')
        l = numeric(z['lyapunov'], 'lyapunov')
        q = numeric(z['period'], 'period')
        expected = shape if spec['layout'] == 'full' else (shape[0] * shape[1],)
        require(l.shape == q.shape == expected, 'science array layout')
        if spec['layout'] == 'partial':
            d = z['done']
            require(d.shape == () and d.dtype.kind in 'iu', 'done must be scalar integer')
            done = int(d)
            require(0 <= done <= l.size, 'done out of allocation')
        else:
            done = l.size
        require(done == spec['expected_done'], 'done differs from frozen metadata')
        arrays = (l.copy().reshape(shape), q.copy().reshape(shape))
    # Detect an input changing while its selected members were read.
    authenticate(root, spec)
    return t, a, *arrays, done


def assemble(spec, loaded):
    """Exact frozen axes; explicit selected ordinal intervals, no tolerance search."""
    nt, na = spec['shape']
    require(len(loaded) == len(spec['segments']) > 0, 'missing/extra segment')
    t, a = loaded[spec['axis_segment']][:2]
    require(t.size == nt and a.size == na, 'canonical axis segment is not full-grid')
    l, q = np.full((nt, na), np.nan), np.full((nt, na), np.nan)
    owner = np.full((nt, na), -1, dtype=np.int64)
    committed = np.zeros((nt, na), dtype=bool)
    actual_t = np.repeat(t[:, None], na, axis=1)
    allocated = [[] for _ in range(nt * na)]
    segments, source_rows = [], []
    ordinal = spec.get('ordinal_amendment')
    if ordinal:
        require(isinstance(ordinal, str) and len(ordinal) == 64, 'ordinal amendment authority')
    for k, (s, row) in enumerate(zip(spec['segments'], loaded)):
        ts, al, ls, qs, done = row
        offset = integer(s['row_offset'], 'row offset')
        require(offset + len(ts) <= nt and len(al) == na, 'tile outside declared grid')
        require([float(x).hex() for x in ts] == s['temperature_hex'], 'frozen temperature axis mismatch')
        require([float(x).hex() for x in al] == s['alpha_hex'], 'frozen alpha axis mismatch')
        require(np.array_equal(a, al), 'literal alpha mismatch')
        if not ordinal:
            require(np.array_equal(t[offset:offset + len(ts)], ts), 'literal temperature mismatch')
        require(ls.shape == qs.shape == (len(ts), na), 'loaded tile shape')
        require(type(done) is int and 0 <= done <= ls.size, 'loaded done')
        lo, hi = s.get('selected_local_flat_range', [0, ls.size])
        integer(lo, 'selection start'); integer(hi, 'selection stop')
        require(0 <= lo <= hi <= ls.size, 'selection outside allocation')
        chosen = owner.reshape(-1)[offset * na + lo:offset * na + hi]
        require((chosen < 0).all(), 'overlapping selected cells')
        chosen[:] = k
        actual_t.reshape(-1)[offset * na + lo:offset * na + hi] = np.repeat(ts, na)[lo:hi]
        end = min(done, hi)
        if end > lo:
            sl = slice(offset * na + lo, offset * na + end)
            committed.reshape(-1)[sl] = True
            l.reshape(-1)[sl], q.reshape(-1)[sl] = ls.reshape(-1)[lo:end], qs.reshape(-1)[lo:end]
        for local in range(ls.size):
            global_flat = offset * na + local
            allocated[global_flat].append(k)
            completed = local < done
            state, issues = disposition(ls.flat[local], qs.flat[local], spec['max_period']) if completed else ('unfinished', [])
            source_rows.append({'dataset': spec['id'], 'segment': k, 'local_flat': local,
                'row': offset + local // na, 'column': local % na,
                'log10_temperature': float(ts[local // na]), 'alpha': float(al[local % na]),
                'selected_ordinal': lo <= local < hi, 'completed': completed,
                'lyapunov': number(ls.flat[local]) if completed else None,
                'period': number(qs.flat[local]) if completed else None,
                'exponent_finite': bool(completed and np.isfinite(ls.flat[local])),
                'period_valid': bool(completed and np.isfinite(qs.flat[local]) and qs.flat[local] == np.floor(qs.flat[local]) and 0 <= qs.flat[local] <= spec['max_period']),
                'disposition': state, 'issues': issues})
        segments.append({'path': s['path'], 'row_offset': offset, 'allocated_cells': ls.size,
            'completed_prefix': done, 'selected_local_flat_range': [lo, hi],
            'uncommitted_allocation': ls.size - done, 'nonselected_completed': done - max(0, end - lo),
            'original_attempt_outcome': s['original_attempt_outcome'],
            'outcome_authority': s['outcome_authority'], 'sha256': s['sha256'],
            'coordinate_rows': [{'global_row': offset + i, 'canonical': float(t[offset + i]),
                'actual': float(x), 'canonical_hex': float(t[offset + i]).hex(), 'actual_hex': float(x).hex(),
                'difference': float(x - t[offset + i])} for i, x in enumerate(ts)]})
    return {'t': t, 'a': a, 'l': l, 'q': q, 'owner': owner, 'committed': committed,
            'actual_t': actual_t, 'allocated': allocated, 'segments': segments, 'source_rows': source_rows}


def number(value):
    """JSON finite number or explicit nonfinite name; no nonstandard JSON NaNs."""
    x = float(value)
    if np.isnan(x):
        return 'NaN'
    if np.isposinf(x):
        return '+Inf'
    if np.isneginf(x):
        return '-Inf'
    return x


def disposition(l, q, max_period):
    issues = []
    if not np.isfinite(l):
        issues.append('exponent_' + number(l))
    if not np.isfinite(q):
        issues.append('period_' + number(q))
    elif q != np.floor(q) or q < 0 or q > max_period:
        issues.append('period_invalid_integer_or_range')
    if issues:
        return 'invalid', issues
    if q > 0:
        return 'detected_period', []
    return ('chaotic' if l > THRESHOLD else 'no_period_not_above_threshold'), []


def cell_records(spec, data):
    t, a, l, q, owner = [data[k] for k in ('t', 'a', 'l', 'q', 'owner')]
    records = []
    for i in range(len(t)):
        for j in range(len(a)):
            k = int(owner[i, j]); done = bool(data['committed'][i, j])
            state, issues = disposition(l[i, j], q[i, j], spec['max_period']) if done else (
                'unfinished' if k >= 0 else 'uncovered', [])
            records.append({'dataset': spec['id'], 'row': i, 'column': j,
                'log10_temperature': float(data['actual_t'][i, j]) if k >= 0 else None,
                'canonical_temperature': float(t[i]), 'alpha': float(a[j]),
                'completed': done, 'source_segment': k if k >= 0 else None,
                'allocated_segments': data['allocated'][i * len(a) + j],
                'lyapunov': number(l[i, j]) if done else None,
                'period': number(q[i, j]) if done else None,
                'exponent_finite': bool(done and np.isfinite(l[i, j])),
                'period_valid': bool(done and np.isfinite(q[i, j]) and q[i, j] == np.floor(q[i, j]) and 0 <= q[i, j] <= spec['max_period']),
                'disposition': state, 'issues': issues})
    return records


def overlap_rows(selected, sources):
    """Retain every nonselected completed observation, not just disagreements."""
    lookup = {(r['row'], r['column']): r for r in selected}
    result = []
    for r in sources:
        if r['completed'] and not r['selected_ordinal']:
            chosen = lookup[(r['row'], r['column'])]
            both_finite = r['exponent_finite'] and chosen['exponent_finite']
            result.append({'row': r['row'], 'column': r['column'], 'nonselected_segment': r['segment'],
                'selected_segment': chosen['source_segment'], 'nonselected_temperature': r['log10_temperature'],
                'selected_temperature': chosen['log10_temperature'],
                'temperature_difference': chosen['log10_temperature'] - r['log10_temperature'] if chosen['log10_temperature'] is not None else None,
                'nonselected_exponent': r['lyapunov'], 'selected_exponent': chosen['lyapunov'],
                'exponent_selected_minus_nonselected': chosen['lyapunov'] - r['lyapunov'] if both_finite else None,
                'nonselected_period': r['period'], 'selected_period': chosen['period'],
                'nonselected_disposition': r['disposition'], 'selected_disposition': chosen['disposition']})
    return result


def summarize(records):
    c = Counter(r['disposition'] for r in records)
    n = len(records)
    completed = sum(r['completed'] for r in records)
    valid = sum(c[k] for k in ('chaotic', 'no_period_not_above_threshold', 'detected_period'))
    period = Counter(str(int(r['period'])) for r in records if r['disposition'] in (
        'chaotic', 'no_period_not_above_threshold', 'detected_period'))
    positive_period = [int(r['period']) for r in records if r['disposition'] == 'detected_period']
    issues = Counter(issue for r in records for issue in r['issues'])
    marginal_period = Counter(str(int(r['period'])) for r in records if r['period_valid'])
    require(sum(c.values()) == n and valid + c['invalid'] == completed, 'denominator reconciliation')
    return {'expected': n, 'completed': completed, 'valid': valid,
            'dispositions': dict(sorted(c.items())), 'issues': dict(sorted(issues.items())),
            'finite_exponent_count': sum(r['exponent_finite'] for r in records),
            'valid_period_field_count': sum(r['period_valid'] for r in records),
            'period_histogram_all_completed_valid_period_fields': dict(sorted(marginal_period.items(), key=lambda kv: int(kv[0]))),
            'chaotic': c['chaotic'], 'chaotic_share_valid': c['chaotic'] / valid if valid else None,
            'chaotic_share_completed': c['chaotic'] / completed if completed else None,
            'completed_share_expected': completed / n if n else None,
            'chaotic_share_expected_lower_bound': c['chaotic'] / n if n else None,
            'period_histogram_including_zero': dict(sorted(period.items(), key=lambda kv: int(kv[0]))),
            'detected_period_count': len(positive_period),
            'median_detected_period': float(np.median(positive_period)) if positive_period else None,
            'detected_period_above_exponent_threshold': sum(
                r['disposition'] == 'detected_period' and r['lyapunov'] > THRESHOLD for r in records)}


def subsets(records, alphas, temperatures):
    bins = []
    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        selected = [r for r in records if lo <= abs(r['alpha']) and
                    (abs(r['alpha']) <= hi if i == len(EDGES) - 2 else abs(r['alpha']) < hi)]
        bins.append({'lower': lo, 'upper': hi, 'upper_inclusive': i == len(EDGES) - 2,
                     'all': summarize(selected),
                     'negative': summarize([r for r in selected if r['alpha'] < 0]),
                     'zero': summarize([r for r in selected if r['alpha'] == 0]),
                     'positive': summarize([r for r in selected if r['alpha'] > 0])})
    negative = [j for j, x in enumerate(alphas) if x < 0]
    positive = [j for j, x in enumerate(alphas) if x > 0]
    near = {}
    for label, indices in [('exact_zero', [j for j, x in enumerate(alphas) if x == 0]),
            ('nearest_negative', negative[-1:]), ('nearest_positive', positive[:1])]:
        near[label] = {'column_indices': indices, 'alpha': [float(alphas[j]) for j in indices],
                       'summary': summarize([r for r in records if r['column'] in indices])}
    return {'absolute_alpha_bins': bins, 'outside_bins': summarize([r for r in records if abs(r['alpha']) > EDGES[-1]]),
            'zero_and_signed_nearest': near,
            'by_temperature_row': [{'row': i, 'canonical_temperature': float(t),
                                    'actual_selected_temperatures': sorted(set(r['log10_temperature'] for r in records if r['row'] == i and r['log10_temperature'] is not None)),
                                    'summary': summarize([r for r in records if r['row'] == i])}
                                   for i, t in enumerate(temperatures)],
            'by_signed_alpha_column': [{'column': j, 'alpha': float(a),
                                      'summary': summarize([r for r in records if r['column'] == j])}
                                     for j, a in enumerate(alphas)]}


def tabulate_dataset(root, spec):
    """Failures remain a failed dataset and every planned cell, never a dropped row."""
    nt, na = spec['shape']
    try:
        loaded = [load_segment(root, s) for s in spec['segments']]
        data = assemble(spec, loaded)
        records = cell_records(spec, data)
        report = {'id': spec['id'], 'admitted': True, 'metadata': spec['metadata'],
                  'shape': [nt, na], 'max_period': spec['max_period'], 'segments': data['segments'],
                  'summary': summarize(records), **subsets(records, data['a'], data['t']),
                  'ordinal_amendment': spec.get('ordinal_amendment'),
                  'nonselected_completed_observations': overlap_rows(records, data['source_rows']),
                  'all_source_prefix_summaries': [summarize([r for r in data['source_rows'] if r['segment'] == k]) for k in range(len(spec['segments']))]}
        sources = data['source_rows']
    except Exception as e:
        records = [{'dataset': spec['id'], 'row': i, 'column': j, 'log10_temperature': None,
                    'alpha': None, 'completed': False, 'source_segment': None, 'allocated_segments': [],
                    'lyapunov': None, 'period': None, 'exponent_finite': False, 'period_valid': False, 'disposition': 'admission_failed',
                    'issues': [type(e).__name__ + ': ' + str(e)]} for i in range(nt) for j in range(na)]
        report = {'id': spec['id'], 'admitted': False, 'metadata': spec.get('metadata'),
                  'shape': [nt, na], 'error': type(e).__name__ + ': ' + str(e), 'summary': summarize(records)}
        sources = [{'dataset': spec['id'], 'segment': k, 'local_flat': i,
                    'disposition': 'admission_failed', 'issues': [report['error']]}
                   for k, s in enumerate(spec['segments']) for i in range(s['shape'][0] * s['shape'][1])]
    return report, records, sources


def matched_groups(all_records, groups):
    """Descriptive counts on literal common valid coordinates, never nearest matching."""
    result = []
    for group in groups:
        maps = []
        for name in group['datasets']:
            maps.append({(r['log10_temperature'], r['alpha']): r for r in all_records[name]
                if r['disposition'] in ('chaotic', 'detected_period', 'no_period_not_above_threshold')})
        common = set.intersection(*(set(m) for m in maps)) if maps else set()
        result.append({'id': group['id'], 'metadata_basis': group['metadata_basis'],
                       'common_valid_exact_coordinates': len(common),
                       'datasets': {name: summarize([m[key] for key in sorted(common)])
                                    for name, m in zip(group['datasets'], maps)}})
    return result
