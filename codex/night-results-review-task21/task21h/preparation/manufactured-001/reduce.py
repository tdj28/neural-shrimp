"""Independent finite saved-grid arithmetic for Task21h; no producer imports."""
import math
from collections import Counter
import numpy as np


def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [clean(x) for x in v]
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return float(v) if math.isfinite(v) else ('NaN' if math.isnan(v) else ('+Infinity' if v > 0 else '-Infinity'))
    return v


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fraction(n, d):
    return None if d == 0 else n / d


def histogram(a):
    return dict(sorted(Counter(str(clean(x)) for x in np.asarray(a).ravel()).items()))


def comparison(name, own, saved):
    own, saved = clean(own), clean(saved)
    numeric = type(own) in (int, float) and type(saved) in (int, float)
    return {'field': name, 'own': own, 'saved': saved, 'equal': own == saved,
            'difference_own_minus_saved': own - saved if numeric else None}


def comparison_leaves(name, own, saved):
    own, saved = clean(own), clean(saved)
    if isinstance(own, dict) and isinstance(saved, dict):
        return [leaf for key in sorted(set(own) | set(saved))
                for leaf in comparison_leaves(name + '/' + key, own.get(key), saved.get(key))]
    if isinstance(own, list) and isinstance(saved, list) and len(own) == len(saved):
        return [leaf for i, (a, b) in enumerate(zip(own, saved))
                for leaf in comparison_leaves(name + '/' + str(i), a, b)]
    return [comparison(name, own, saved)]


def grid(x, y, L, P, *, kind, max_period, saved_chaotic=None):
    x, y, L, P = (np.asarray(z) for z in (x, y, L, P))
    require(x.ndim == y.ndim == 1 and len(x) > 0 and len(y) > 0, 'nonempty coordinate axes required')
    require(np.all(np.isfinite(x)) and np.all(np.isfinite(y)), 'finite coordinates required')
    require(np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0), 'strict ordered unique coordinates required')
    require(L.shape == P.shape == (len(y), len(x)), 'rectangular cell coverage mismatch')
    require(L.dtype.kind == 'f' and P.dtype.kind in 'if', 'numeric LE and period required')
    finite = np.isfinite(L) & np.isfinite(P)
    valid_period = np.isfinite(P) & (P == np.floor(P)) & (P >= 0) & (P <= max_period)
    valid = finite & valid_period
    reconstructed = (L > 1e-3) & ((P == 0) if kind == 'LS' else ~(P > 0))
    if saved_chaotic is None:
        require(kind == 'LS', 'TB saved Boolean mask required')
        literal = reconstructed
    else:
        literal = np.asarray(saved_chaotic)
        require(literal.dtype.kind == 'b' and literal.shape == L.shape, 'Boolean chaotic-mask schema mismatch')
    return {'x': x, 'y': y, 'L': L, 'P': P, 'literal': literal, 'reconstructed': reconstructed,
            'valid': valid, 'valid_period': valid_period, 'kind': kind, 'max_period': max_period}


def counts(g, select=None):
    select = np.ones_like(g['literal'], dtype=bool) if select is None else np.asarray(select, dtype=bool)
    require(select.shape == g['literal'].shape, 'selection shape mismatch')
    n = int(select.sum()); c = g['literal'] & select; valid = g['valid'] & select
    good = c & g['valid']; L = g['L']; P = g['P']
    return {'allocated_cells': n, 'literal_chaotic': int(c.sum()), 'literal_share_all_allocated': fraction(int(c.sum()), n),
            'finite_valid_cells': int(valid.sum()), 'finite_valid_chaotic': int(good.sum()),
            'finite_valid_chaotic_share_all_allocated': fraction(int(good.sum()), n),
            'finite_valid_chaotic_share_valid': fraction(int(good.sum()), int(valid.sum())),
            'nonfinite_LE': int((select & ~np.isfinite(L)).sum()),
            'invalid_period': int((select & ~g['valid_period']).sum()),
            'periodic_valid': int((valid & (P > 0)).sum()),
            'no_detected_period_valid': int((valid & (P == 0)).sum()),
            'LE_positive_literal': int((select & (L > 1e-3)).sum()),
            'positive_LE_and_period': int((select & (L > 1e-3) & (P > 0)).sum()),
            'saved_vs_reconstructed_chaos_mismatches': int((select & (g['literal'] != g['reconstructed'])).sum()),
            'period_histogram': histogram(P[select]),
            'finite_valid_LE_range': [float(np.min(L[valid])), float(np.max(L[valid]))] if valid.any() else None}


def ranges(g, mask):
    rows, cols = np.nonzero(mask)
    return {'x': [float(g['x'][cols].min()), float(g['x'][cols].max())],
            'y': [float(g['y'][rows].min()), float(g['y'][rows].max())]} if len(rows) else None


def summary(g):
    out = counts(g)
    out.update({'literal_chaotic_coordinate_ranges': ranges(g, g['literal']),
                'finite_valid_chaotic_coordinate_ranges': ranges(g, g['literal'] & g['valid']),
                'exact_x_zero_indices': np.flatnonzero(g['x'] == 0).tolist(),
                'exact_y_zero_indices': np.flatnonzero(g['y'] == 0).tolist()})
    out['rows'] = [{'index': i, 'coordinate': float(v), **counts(g, np.indices(g['L'].shape)[0] == i)} for i, v in enumerate(g['y'])]
    out['columns'] = [{'index': i, 'coordinate': float(v), **counts(g, np.indices(g['L'].shape)[1] == i)} for i, v in enumerate(g['x'])]
    if g['kind'] == 'LS':
        out['abs_alpha_below_0.02'] = counts(g, np.broadcast_to(abs(g['x']) < .02, g['L'].shape))
        out['near_zero_definition'] = 'strict abs(alpha)<0.02; this is not exact alpha=0'
    return out


def cells(g, identifier):
    for i, y in enumerate(g['y']):
        for j, x in enumerate(g['x']):
            yield clean({'grid': identifier, 'row': i, 'column': j, 'x': x, 'y': y,
                         'lyapunov': g['L'][i, j], 'period': g['P'][i, j],
                         'literal_chaotic': g['literal'][i, j], 'reconstructed_chaotic': g['reconstructed'][i, j],
                         'finite_valid': g['valid'][i, j], 'valid_period': g['valid_period'][i, j]})


def join_ls(lo, hi):
    require(lo['kind'] == hi['kind'] == 'LS', 'LS halves required')
    require(np.array_equal(lo['x'], hi['x']), 'LS alpha axes differ')
    require(lo['y'][-1] < hi['y'][0], 'LS half ordering or overlap mismatch')
    return grid(lo['x'], np.concatenate([lo['y'], hi['y']]),
                np.concatenate([lo['L'], hi['L']]), np.concatenate([lo['P'], hi['P']]),
                kind='LS', max_period=lo['max_period'])


def mask_count(mask, eligible, qualified):
    n, d, q = int(mask.sum()), int(eligible.sum()), int(qualified.sum())
    return {'count': n, 'all_plane_cells': int(mask.size), 'fraction_all_plane': n / mask.size,
            'eligible_cells': d, 'fraction_eligible': fraction(n, d),
            'finite_valid_count': int((mask & qualified).sum()), 'finite_valid_eligible': int((eligible & qualified).sum()),
            'finite_valid_triplets': q,
            'finite_valid_fraction_all_plane': int((mask & qualified).sum()) / mask.size,
            'finite_valid_fraction_eligible': fraction(int((mask & qualified).sum()), int((eligible & qualified).sum()))}


def compare_controls(plane, a1, a2):
    """Literal saved-mask comparisons plus explicit finite qualification; no causal gate."""
    require(np.array_equal(plane['x'], a1['x']) and np.array_equal(plane['y'], a2['y']), 'control free-axis mismatch')
    require(a1['y'].tolist() == [0.] and a2['x'].tolist() == [0.], 'controls must have singleton exact zero')
    require(not np.any(plane['x'] == 0) and not np.any(plane['y'] == 0), 'expected original plane omits zeros')
    C, P = plane['literal'], plane['P']; shape = C.shape
    A, B = np.broadcast_to(a1['literal'][0], shape), np.broadcast_to(a2['literal'][:, 0, None], shape)
    AP, BP = np.broadcast_to(a1['P'][0], shape), np.broadcast_to(a2['P'][:, 0, None], shape)
    valid = plane['valid'] & a1['valid'][0] & a2['valid'][:, 0, None]
    i0, j0 = int(np.argmin(abs(plane['y']))), int(np.argmin(abs(plane['x'])))
    nearA, nearB = np.broadcast_to(C[i0], shape), np.broadcast_to(C[:, j0, None], shape)
    nearAP, nearBP = np.broadcast_to(P[i0], shape), np.broadcast_to(P[:, j0, None], shape)
    nearvalid = plane['valid'] & plane['valid'][i0] & plane['valid'][:, j0, None]
    both_off_nearest = (np.indices(shape)[0] != i0) & (np.indices(shape)[1] != j0)
    exact_nc, exact_p = ~A & ~B, (AP > 0) & (BP > 0)
    near_nc, near_p = ~nearA & ~nearB, (nearAP > 0) & (nearBP > 0)
    masks = {'exact_controls_nonchaotic': (exact_nc, valid), 'exact_controls_periodic': (exact_p, valid),
             'near_axis_nonchaotic_all_plane': (near_nc, nearvalid), 'near_axis_periodic_all_plane': (near_p, nearvalid),
             'near_axis_nonchaotic_excluding_nearest_row_column': (near_nc & both_off_nearest, nearvalid)}
    result = {'nearest_row': i0, 'nearest_column': j0, 'nearest_alpha2': float(plane['y'][i0]),
              'nearest_alpha1': float(plane['x'][j0]), 'nearest_row_share': float(C[i0].mean()),
              'nearest_column_share': float(C[:, j0].mean()),
              'exact_alpha1_only_share': float(a1['literal'].mean()), 'exact_alpha2_only_share': float(a2['literal'].mean()),
              'alpha1_near_vs_exact_mismatches': int(np.count_nonzero(C[i0] != a1['literal'][0])),
              'alpha2_near_vs_exact_mismatches': int(np.count_nonzero(C[:, j0] != a2['literal'][:, 0])),
              'definitions': 'Nonchaotic means false saved union mask, not periodic; periodic means P>0. All-plane denominator is distinct from eligible controls. Near axes are original argmin slices; exact controls have one zero-valued hook. All labels are finite-horizon.'}
    for name, (eligible, q) in masks.items():
        result[name] = mask_count(C & eligible, eligible, q)
    ledger = []
    for i, j in np.ndindex(shape):
        ledger.append(clean({'row': i, 'column': j, 'alpha1': plane['x'][j], 'alpha2': plane['y'][i],
                             'plane_chaotic': C[i, j], 'a1_chaotic': A[i, j], 'a2_chaotic': B[i, j],
                             'a1_period': AP[i, j], 'a2_period': BP[i, j], 'finite_valid_triplet': valid[i, j],
                             'near_a1_chaotic': nearA[i, j], 'near_a2_chaotic': nearB[i, j],
                             **{name: C[i, j] and eligible[i, j] for name, (eligible, _) in masks.items()}}))
    return result, ledger
