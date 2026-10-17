"""Independent saved-array/scalar analysis of SYNC. No model or producer calls.

All functions consume admitted arrays. Nonfinite rows are explicitly unqualified,
not interpreted as unsynchronized evidence. No native closure is regenerated.
"""
import math

import numpy as np

PRIMARY_TOL = 1e-8


def check_scalars(saved):
    """Read the saved precision diagnostics, without claiming new native checks."""
    rows, seen = [], set()
    for cell in saved['cells']:
        ci = cell['index']
        if ci in seen:
            raise ValueError('duplicate check cell')
        seen.add(ci)
        h = cell['fd_h']; err = cell['fd_rel_err']
        if len(h) != len(err) or len(set(h)) != len(h):
            raise ValueError('FD ladder alignment')
        ratios = [err[i] / err[i+2] if err[i+2] != 0 and math.isfinite(err[i]) and math.isfinite(err[i+2]) else None
                  for i in range(0, 6, 2)] if len(err) >= 7 else []
        diff = cell['max_abs_diff_rel']
        finite = all(math.isfinite(float(v)) and v >= 0 for v in list(diff.values()) + list(err))
        rows.append({'saved': cell, 'finite_nonnegative_diagnostics': finite,
                     'source_ratio_per_decade': ratios,
                     'saved_minus_recomputed_ratios': [a-b if b is not None else None for a,b in zip(cell.get('fd_ratio_per_decade', []), ratios)],
                     'max_recorded_batch_difference': max(diff.values()) if diff and finite else None,
                     'exceeds_1e_minus_12_separate_row_trigger': any(v > 1e-12 for v in diff.values()) if finite else None,
                     'scope': 'inherited one-state checks; no new derivative or layout execution; no invented h2 pass threshold'})
    return rows


def threshold(couplings, synchronized):
    c = np.asarray(couplings, dtype=float)
    flags = list(synchronized)
    if c.ndim != 1 or len(c) != len(flags) or not len(c) or not np.isfinite(c).all():
        raise ValueError('invalid coupling grid')
    if len(set(c.tolist())) != len(c) or any(type(x) is not bool for x in flags):
        raise ValueError('duplicate coupling or non-Boolean flag')
    order = np.argsort(c)
    x, y = c[order], np.asarray(flags)[order]
    first = len(x)
    while first and y[first - 1]:
        first -= 1
    if first == len(x):
        return {'c_lo': float(x[-1]), 'c_hi': None,
                'status': 'largest c not synchronized' if y.any() else 'no synchronized c on the grid'}
    return {'c_lo': float(x[first - 1]) if first else None, 'c_hi': float(x[first]),
            'status': 'bracketed' if first else 'all synchronized', 'non_monotone': bool(y[:first].any())}


def laminae(mask):
    """All maximal True runs, including left/right-censor diagnostics."""
    values = np.asarray(mask, dtype=bool)
    starts, lengths = [], []
    start = None
    for i, yes in enumerate(values):
        if yes and start is None:
            start = i
        if start is not None and (not yes or i == len(values) - 1):
            end = i if not yes else i + 1
            starts.append(start); lengths.append(end - start); start = None
    return {'starts_zero_based': starts, 'lengths': lengths,
            'censored_first': bool(len(values) and values[0]),
            'censored_last': bool(len(values) and values[-1])}


def slope(lengths):
    x = np.asarray(lengths, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or np.any(x < 1) or np.any(x != np.floor(x)):
        raise ValueError('invalid laminar lengths')
    x = x[x >= 2]
    detail = {'eligible_lengths': x.astype(int).tolist(), 'fit': None}
    if len(x) < 100:
        return dict(detail, reason='fewer than 100 lengths >= 2')
    edges = np.unique(np.floor(np.logspace(np.log10(2), np.log10(float(x.max()) + 1), 15)).astype(int))
    detail['edges'] = edges.tolist()
    if len(edges) < 4:
        return dict(detail, reason='fewer than four distinct edges')
    counts = np.histogram(x, bins=edges)[0]
    widths = np.diff(edges)
    density = counts / widths / len(x)
    mids = np.sqrt(edges[:-1] * edges[1:])
    usable = counts >= 3
    detail.update({'counts': counts.tolist(), 'density': density.tolist(), 'midpoints': mids.tolist(),
                   'fit_mask': usable.tolist()})
    if usable.sum() < 3:
        return dict(detail, reason='fewer than three bins with count >= 3')
    # Independent centered OLS; compare every nonzero rounding difference to np.polyfit source.
    xx, yy = np.log(mids[usable]), np.log(density[usable])
    centered = xx - xx.mean()
    beta = float(np.dot(centered, yy - yy.mean()) / np.dot(centered, centered))
    detail['fit'] = {'slope': beta, 'bins_used': int(usable.sum()), 'n': len(x), 'min_len': 2}
    detail['intercept'] = float(yy.mean() - beta * xx.mean())
    detail['reason'] = 'descriptive log-density OLS; no power-law inference or uncertainty calibration'
    return detail


def bursts(values, low=PRIMARY_TOL, high=1e-3):
    armed = False
    events = []
    for step, x in enumerate(values, 1):
        if x < low:
            armed = True
        elif x > high and armed:
            events.append(step); armed = False
    return events


def block_error(values, count=10):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1:
        raise ValueError('log array is not a vector')
    size = len(a) // count
    out = {'blocks': count, 'block_size': size, 'trailing_unused': len(a) - count * size,
           'nonfinite': int((~np.isfinite(a)).sum()), 'se': None, 'block_means': []}
    if size and np.isfinite(a).all():
        b = a[:count * size].reshape(count, size).mean(axis=1)
        out.update(se=float(b.std(ddof=1) / math.sqrt(count)), block_means=b.tolist())
    return out


def critical(lam):
    return .5 * (1 - math.exp(-lam)) if lam is not None and math.isfinite(lam) and lam > 0 else None


def transverse(cells, pairs, tangent_logs, twin_logs, published):
    tangent = np.asarray(tangent_logs, dtype=float)
    twins = np.asarray(twin_logs, dtype=float)
    if tangent.ndim != 2 or twins.ndim != 2 or tangent.shape[0] != len(cells) or twins.shape[0] != len(pairs):
        raise ValueError('transverse array shape')
    ids = [c['index'] for c in cells]
    keys = [(m['cell'], m['c']) for m in pairs]
    if len(set(ids)) != len(ids) or len(set(keys)) != len(keys) or any(i not in ids for i, _ in keys):
        raise ValueError('transverse duplicate/missing identity')
    result, difference_rows, failures = {}, [], []
    for i, cell in enumerate(cells):
        ci = cell['index']; logs = tangent[i]
        error = block_error(logs)
        lam = float(logs.mean()) if len(logs) and not error['nonfinite'] else None
        se = error['se']
        interval = [critical(lam - 2 * se), critical(lam + 2 * se)] if lam is not None and se is not None else [None, None]
        rows, witnesses = [], []
        for k, meta in enumerate(pairs):
            if meta['cell'] != ci:
                continue
            c = float(meta['c'])
            if not math.isfinite(c):
                raise ValueError('nonfinite coupling')
            err = block_error(twins[k])
            measured = float(twins[k].mean()) if twins.shape[1] and not err['nonfinite'] else None
            factor = abs(1 - 2 * c)
            log_factor = math.log(factor) if factor else None
            pred = lam + log_factor if lam is not None and log_factor is not None else None
            delta = measured - pred if measured is not None and pred is not None else None
            row = {'c': c, 'measured': measured, 'measured_se': err['se'], 'predicted': pred,
                   'measured_minus_predicted': delta,
                   'measured_minus_ln_factor': measured - log_factor if measured is not None and log_factor is not None else None}
            rows.append(row)
            witnesses.append({'cell': ci, 'c': c, 'twin_block_error': err, 'zero_factor': factor == 0,
                              'zero_factor_prediction': 'minus infinity; exact one-step collapse' if not factor else None,
                              'coupling_in_frozen_domain': 0 <= c < .5})
            if delta is not None:
                difference_rows.append({'cell': ci, 'c': c, 'signed': delta, 'absolute': abs(delta)})
            else:
                failures.append({'cell': ci, 'c': c, 'reason': 'undefined finite difference', 'witness': witnesses[-1]})
        pub = published.get(ci)
        result[str(ci)] = {**cell, 'lambda1_published': pub, 'lambda1_jvp': lam, 'lambda1_jvp_se': se,
                           'lambda1_jvp_steps': len(logs), 'c_star_published': critical(pub),
                           'c_star_jvp': critical(lam), 'c_star_jvp_2se': interval, 'transverse': rows,
                           'diagnostics': {'tangent_block_error': error, 'pairs': witnesses,
                                           'positive_lam_and_2se_endpoints': all(v is not None for v in [critical(lam), *interval])}}
    return {'cells': result, 'all_grid_differences': difference_rows, 'undefined_differences': failures,
            'max_absolute_measured_minus_predicted': max((x['absolute'] for x in difference_rows), default=None),
            'all_grid_difference_defined': not failures}


def histogram_tv(a, b):
    aa, bb = np.asarray(a), np.asarray(b)
    if not len(aa) or not len(bb):
        return None
    tokens = np.union1d(aa, bb)
    ua, na = np.unique(aa, return_counts=True)
    ub, nb = np.unique(bb, return_counts=True)
    ca, cb = np.zeros(len(tokens)), np.zeros(len(tokens))
    ca[np.searchsorted(tokens, ua)] = na / len(aa)
    cb[np.searchsorted(tokens, ub)] = nb / len(bb)
    return float(.5 * np.abs(ca - cb).sum())


def sweep(meta, distance, tokens, tail=500, text_length=12, final=None):
    d, tok = np.asarray(distance, dtype=float), np.asarray(tokens)
    pairs, steps = meta['pairs'], meta['steps']
    if type(steps) is not int or steps < 1 or d.shape != (len(pairs), steps + 1) or tok.shape != (2 * len(pairs), steps):
        raise ValueError('sweep steps/shape mismatch')
    if not np.issubdtype(tok.dtype, np.integer) or np.any(tok < 0):
        raise ValueError('tokens must be nonnegative integers')
    keys = [(p['cell'], p['c']) for p in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate sweep pair')
    if final is not None:
        final = np.asarray(final, dtype=float)
        if final.shape[0] != len(pairs) * 2:
            raise ValueError('final state pair count')
    out = {}
    for cell in sorted({p['cell'] for p in pairs}):
        indices = [k for k, p in enumerate(pairs) if p['cell'] == cell]
        zero = [k for k in indices if pairs[k]['c'] == 0.]
        if len(zero) > 1:
            raise ValueError('duplicate zero reference')
        ref = np.concatenate([tok[2 * zero[0], steps // 2:], tok[2 * zero[0] + 1, steps // 2:]]) if zero else None
        rows, diagnostics = [], []
        for k in indices:
            values = d[k, 1:]; end = values[-tail:]
            qualified = bool(np.isfinite(d[k]).all() and np.all(d[k] >= 0))
            diag = {'cell': cell, 'c': pairs[k]['c'], 'qualified_distance': qualified,
                    'nonfinite': int((~np.isfinite(d[k])).sum()), 'negative': int((d[k] < 0).sum()),
                    'tail_requested': tail, 'tail_available': len(end), 'initial_distance': float(d[k, 0]),
                    'last_A_token_ids': tok[2*k, -text_length:].tolist(), 'last_B_token_ids': tok[2*k+1, -text_length:].tolist(),
                    'text_scope': 'source tokenizer decoding inherited; these are independently selected saved IDs'}
            if not qualified:
                rows.append({'c': pairs[k]['c'], 'status': 'unqualified nonfinite/negative saved distance'})
                diagnostics.append(diag); continue
            burst_steps = bursts(values)
            row = {'c': pairs[k]['c'], 'mean_dist': float(values.mean()),
                   'mean_dist_second_half': float(values[len(values)//2:].mean()),
                   'frac_below_tol': float((values < PRIMARY_TOL).mean()), 'synced': bool(end.max() < PRIMARY_TOL),
                   'synced_1e-6': bool(end.max() < 1e-6), 'tail_max': float(end.max()), 'tail_median': float(np.median(end)),
                   'first_below_tol': int(np.flatnonzero(values < PRIMARY_TOL)[0]) + 1 if np.any(values < PRIMARY_TOL) else None,
                   'bursts_1e-8_to_1e-3': len(burst_steps),
                   'top1_equal_frac_second_half': float((tok[2*k, steps//2:] == tok[2*k+1, steps//2:]).mean())}
            diag['burst_steps'] = burst_steps
            diag['at_least_five_bursts_frozen_rule'] = len(burst_steps) >= 5
            for th in (1e-4, 1e-6):
                runs = laminae(values < th)
                lengths = runs['lengths'][:-1] if runs['censored_last'] else runs['lengths']
                fit = slope(lengths)
                row[f'laminar_{th:g}'] = {'n': len(runs['lengths']), 'censored_last': runs['censored_last'],
                                        'max': max(runs['lengths'], default=0), 'fit': fit['fit']}
                diag[f'laminar_{th:g}'] = {**runs, 'slope_evidence': fit}
            if ref is not None:
                row['tv_A_vs_c0_tokens'] = histogram_tv(tok[2*k, steps//2:], ref)
            if final is not None:
                a, b = final[2*k].ravel(), final[2*k+1].ravel()
                n = np.linalg.norm(a)
                last = float(np.linalg.norm(a-b)/n) if n > 0 and np.isfinite(a).all() and np.isfinite(b).all() else None
                diag['final_state_relative_distance'] = last
                diag['final_state_minus_saved_last'] = last - float(d[k,-1]) if last is not None else None
            rows.append(row); diagnostics.append(diag)
        all_qualified = all(r.get('qualified_distance') for r in diagnostics)
        grid = sorted(pairs[k]['c'] for k in indices)
        out[str(cell)] = {'start': meta['args']['start'], 'steps': steps, 'complete': meta['complete'],
                          'commit': meta['commit'], 'sweep_npz_sha256': meta['sweep_npz_sha256'],
                          'threshold': threshold([r['c'] for r in rows], [r['synced'] for r in rows]) if all_qualified else None,
                          'threshold_tol_1e-6_secondary': threshold([r['c'] for r in rows], [r['synced_1e-6'] for r in rows]) if all_qualified else None,
                          'tv_c0_A_vs_B_tokens': histogram_tv(tok[2*zero[0],steps//2:], tok[2*zero[0]+1,steps//2:]) if zero else None,
                          'pairs': rows, 'diagnostics': diagnostics, 'all_distances_qualified': all_qualified,
                          'full_record_and_tail': bool(meta['complete'] and steps == meta['args']['record'] and steps >= tail),
                          'coupling_grid': grid, 'grid_gaps': np.diff(grid).tolist(),
                          'threshold_scope': 'finite trailing-suffix bracket, not monotonicity or global-basin proof'}
    return out


def periodic(saved):
    """Check internal saved scalar implications; no unsaved return ladder certification."""
    result = []
    limit, tol = saved['args']['max_period'], saved['args']['tol']
    seen = set()
    for row in saved['pairs']:
        key = (row['sweep'], row['cell'], row['c'])
        if key in seen:
            raise ValueError('duplicate periodic pair')
        seen.add(key)
        p, ret = row['primitive_period'], row['return_distance_at_P']
        finite = all(math.isfinite(float(row[k])) and row[k] >= 0 for k in ['min_return', 'pair_distance_now'])
        gate = {'min_and_pair_distance_finite_nonnegative': finite,
                'argmin_within_search': type(row['argmin_return']) is int and 1 <= row['argmin_return'] <= limit,
                'period_in_range': p is None or type(p) is int and 1 <= p <= limit,
                'return_below_tol': ret is not None and math.isfinite(ret) and 0 <= ret < tol if p is not None else ret is None,
                'minimum_implies_some_closure': (row['min_return'] < tol) == (p is not None),
                'minimum_le_selected_return': ret is None or row['min_return'] <= ret,
                'token_lengths_match': len(row['top1_A']) == len(row['top1_B']) == (p if p is not None else min(12, limit))}
        result.append({'saved': row, 'scalar_gates': gate, 'all_scalar_gates': all(gate.values()),
                       'native_closure': 'inherited: full states and full return ladder not saved by source',
                       'primitive_minimum': 'inherited first qualifying return; absent full ladder prevents independent all-divisor check',
                       'stability': 'no multipliers saved; forward attraction is finite evidence only'})
    return result
