"""Task19 independent finite-record reductions; no producer analysis imports.

SVD fits explicitly retain rank/conditioning and every candidate/fold. A low
value error is an empirical scalar prediction, never a conjugacy proof.
"""
import math
import numpy as np

DEGREES = tuple(range(4, 25, 2))


def finite(x):
    x = np.asarray(x)
    if not np.isfinite(x).all():
        raise ValueError('nonfinite analysis input/return')
    return x


def symbolic_period(tokens, maximum=64):
    a = np.asarray(tokens)
    if a.ndim != 1 or not np.issubdtype(a.dtype, np.integer):
        raise ValueError('one integer token sequence required')
    return next((q for q in range(1, maximum+1)
                 if len(a) > 2*q and np.array_equal(a[q:], a[:-q])), 0)


def features(x, degree, basis):
    x = finite(x); out = np.ones((len(x), degree+1), dtype=np.float64)
    if degree:
        out[:, 1] = x
    for j in range(2, degree+1):
        out[:, j] = (2*x*out[:, j-1]-out[:, j-2] if basis == 'chebyshev'
                     else x*out[:, j-1])
    return out


def polynomial(x, y, degree, basis='chebyshev'):
    """Column-scaled SVD least squares; no numpy polynomial fit helper."""
    x, y = finite(x), finite(y)
    if x.ndim != 1 or x.shape != y.shape or len(x) <= degree or basis not in ('power', 'chebyshev'):
        raise ValueError('overdetermined scalar fit required')
    lo, hi = float(x.min()), float(x.max())
    if hi <= lo:
        raise ValueError('constant scalar coordinate')
    shift = lo if basis == 'power' else (hi+lo)/2
    scale = hi-lo if basis == 'power' else (hi-lo)/2
    v = features((x-shift)/scale, degree, basis)
    norms = np.linalg.norm(v, axis=0)
    u, s, vt = np.linalg.svd(v/norms, full_matrices=False)
    finite(s)
    cutoff = len(x)*np.finfo(np.float64).eps*s[0]
    keep = s > cutoff
    coeff = (vt[keep].T @ ((u[:, keep].T @ y)/s[keep]))/norms
    finite(coeff)
    return dict(basis=basis, degree=degree, shift=shift, scale=scale,
                coefficients=coeff.tolist(), rank=int(keep.sum()),
                singular_values=s.tolist(), condition=float(s[0]/s[-1]),
                rank_full=bool(keep.all()), points=len(x))


def evaluate(fit, x, derivative=0):
    """Polynomial evaluation/derivatives use a standard arithmetic primitive."""
    c = np.asarray(fit['coefficients'])
    if fit['basis'] == 'chebyshev':
        c = np.polynomial.chebyshev.chebder(c, m=derivative)
        out = np.polynomial.chebyshev.chebval((np.asarray(x)-fit['shift'])/fit['scale'], c)
    else:
        c = np.polynomial.polynomial.polyder(c, m=derivative)
        out = np.polynomial.polynomial.polyval((np.asarray(x)-fit['shift'])/fit['scale'], c)
    return finite(out / fit['scale']**derivative)


def details(fit, x, y):
    derivative = evaluate(fit, x, 1)
    absolute = np.abs(derivative)
    zero = int((absolute == 0).sum())
    # Retain the literal producer-floor comparison separately from the true
    # logarithm's undefined/-infinity case. No silent floor in own exponent.
    exp = None if zero else float(np.mean(np.log(absolute)))
    coef = np.asarray(fit['coefficients'])
    if fit['basis'] == 'chebyshev':
        roots = np.polynomial.chebyshev.chebroots(np.polynomial.chebyshev.chebder(coef))
    else:
        roots = np.polynomial.polynomial.polyroots(np.polynomial.polynomial.polyder(coef))
    finite(roots)
    raw = [[float(r.real), float(r.imag)] for r in roots]
    producer_stationary = [r for r in roots
        if abs(r.imag*(fit['scale'] if fit['basis']=='chebyshev' else 1.)) < 1e-9
        and float(x.min()) < r.real*fit['scale']+fit['shift'] < float(x.max())]
    physical = sorted(float(r.real*fit['scale']+fit['shift']) for r in roots
                      if abs(r.imag) < 1e-9 and float(x.min()) < r.real*fit['scale']+fit['shift'] < float(x.max()))
    critical = []
    for i, root in enumerate(physical):
        left = float(x.min()) if i == 0 else physical[i-1]
        right = float(x.max()) if i+1 == len(physical) else physical[i+1]
        dl, dr = float(evaluate(fit, (left+root)/2, 1)), float(evaluate(fit, (root+right)/2, 1))
        curvature = float(evaluate(fit, root, 2))
        kind = ('quadratic_maximum' if dl > 0 > dr and curvature < 0 else
                'quadratic_minimum' if dl < 0 < dr and curvature > 0 else 'unresolved_or_degenerate')
        critical.append(dict(x=root, curvature=curvature, left_derivative=dl,
                             right_derivative=dr, kind=kind))
    residual = evaluate(fit, x)-y
    span = float(np.ptp(y))
    return dict(**fit, residual_relative=float(np.sqrt(np.mean(residual**2))/span),
                exponent_per_repetition=exp, zero_derivatives=zero,
                producer_floor_exponent=float(np.mean(np.log(absolute+1e-300))),
                derivative_min=float(derivative.min()), derivative_max=float(derivative.max()),
                derivative_roots_scaled=raw, critical_points=critical,
                stationary_count=len(critical),
                producer_stationary_count=len(producer_stationary),
                producer_root_imaginary_test=('physical_coordinate' if fit['basis']=='chebyshev' else 'normalized_coordinate'),
                fitted_quadratic_maximum_count=sum(c['kind']=='quadratic_maximum' for c in critical))


def folds(n, kind):
    if kind not in ('random', 'purged_contiguous') or n < 10:
        raise ValueError('fixed fivefold kind/size required')
    ordering = np.random.default_rng(0).permutation(n) if kind == 'random' else np.arange(n)
    result = []
    for test in np.array_split(ordering, 5):
        mask = np.ones(n, dtype=bool); mask[test] = False
        if kind == 'purged_contiguous':
            mask[max(0, int(test[0])-1):min(n, int(test[-1])+2)] = False
        result.append((np.flatnonzero(mask), test))
    return result


def crossvalidate(x, y, kind):
    """Frozen candidate grid; target twin exponent is never an argument."""
    span = float(np.ptp(y)); cap = max(4, (len(x)-1)//6)
    degrees = [d for d in DEGREES if d <= cap]
    partition = folds(len(x), kind); candidates = []
    for degree in degrees:
        records = []
        for train, test in partition:
            fit = polynomial(x[train], y[train], degree)
            errors = evaluate(fit, x[test])-y[test]
            records.append(dict(fold=len(records),
                mse=float(np.mean(errors**2)), rank=fit['rank'], rank_full=fit['rank_full'],
                condition=fit['condition'], train_min=float(x[train].min()), train_max=float(x[train].max()),
                test_outside_train_range=int(((x[test]<x[train].min()) | (x[test]>x[train].max())).sum())))
        candidates.append(dict(degree=degree, folds=records,
            # Equal fold weighting intentionally matches the original amendment.
            cv_residual_relative=float(np.sqrt(np.mean([f['mse'] for f in records]))/span),
            all_fold_ranks_full=all(f['rank_full'] for f in records)))
    selected = min(candidates, key=lambda c: c['cv_residual_relative'])
    fit = details(polynomial(x, y, selected['degree']), x, y)
    return dict(kind=kind, partition=[dict(train=t.tolist(),test=s.tolist()) for t,s in partition],
                candidates=candidates, selected_degree=selected['degree'],
                cv_residual_relative=selected['cv_residual_relative'], fit=fit,
                producer_value_fit_qualified=bool(selected['cv_residual_relative'] < .01),
                qualified_value_fit=bool(selected['all_fold_ranks_full'] and fit['rank_full']
                                        and selected['cv_residual_relative'] < .01))


def phase_fits(probabilities, q):
    p = finite(probabilities)
    if p.ndim != 1 or not 0 < q <= 64 or np.any((p <= 0) | (p > 1)):
        raise ValueError('finite probability sequence and positive period required')
    reps0 = len(p[::q]); original_degree = min(8, max(2, (reps0-1)//6))
    rows = []
    for phase in range(q):
        pk = p[phase::q]; x, y = pk[:-1], pk[1:]
        row = dict(phase=phase, observations=len(pk), pairs=len(x))
        if reps0 < 30 or len(x) < 10:
            row['disposition'] = 'insufficient_repetitions'
        elif np.ptp(x) <= 0 or np.ptp(y) <= 0:
            row['disposition'] = 'constant_coordinate'
        else:
            row.update(disposition='fitted',
                original=details(polynomial(x, y, original_degree, 'power'), x, y),
                amended_random=crossvalidate(x, y, 'random'),
                purged_contiguous=crossvalidate(x, y, 'purged_contiguous'))
        rows.append(row)
    return rows


def aggregate(rows, q, full):
    """Descriptive comparison only; full exponent cannot alter selection."""
    fits = [r for r in rows if r['disposition'] == 'fitted']
    result = {}
    original = min(fits, key=lambda r: r['original']['residual_relative']) if fits else None
    for name in ('original', 'amended_random', 'purged_contiguous'):
        kept = ([original] if original is not None and name == 'original' else
                [r for r in fits if r[name]['producer_value_fit_qualified' if name=='amended_random' else 'qualified_value_fit']] if name != 'original' else [])
        values = [(r['original'] if name == 'original' else r[name]['fit']) for r in kept]
        exponents = [v['producer_floor_exponent']/q for v in values]
        own_exponents = [v['exponent_per_repetition']/q if v['exponent_per_repetition'] is not None else None for v in values]
        estimate = (float(np.median(own_exponents)) if own_exponents and all(v is not None for v in own_exponents) else None) if name=='purged_contiguous' else (float(np.median(exponents)) if exponents else None)
        result[name] = dict(phases=[r['phase'] for r in kept], exponents_per_step=exponents,
            unfloored_exponents_per_step=own_exponents, estimate=estimate,
            spread=float(np.ptp(exponents)) if exponents else None,
            full_twin_mean=full, relative_discrepancy=(estimate-full)/full if estimate is not None and full != 0 else None,
            has_one_stationary_point=any(v['producer_stationary_count']==1 for v in values),
            has_one_quadratic_maximum=any(v['stationary_count']==1 and v['fitted_quadratic_maximum_count']==1 for v in values),
            rank_warning_phases=[r['phase'] for r in kept if (not r['original']['rank_full'] if name=='original' else not r[name]['qualified_value_fit'])],
            original_value_fit_qualified=values[0]['residual_relative'] < .01 if name=='original' and values else None)
    return result


def sample_bands(states, q, block=64):
    """Actual finite-sample cross-phase distances, not centroid substitution."""
    s = finite(states)
    if s.ndim != 2 or not 0 < q <= 64 or block < 1:
        raise ValueError('finite states/period required')
    scale = float(np.linalg.norm(s, axis=1).mean())
    if scale == 0:
        raise ValueError('zero mean state norm')
    centers = np.stack([s[k::q].mean(0) for k in range(q)])
    radii = np.array([np.linalg.norm(s[k::q]-centers[k], axis=1).max() for k in range(q)])
    rows = []
    for a in range(q):
        for b in range(a+1, q):
            sa, sb = s[a::q], s[b::q]; best = math.inf; pair = None
            # Direct differences avoid cancellation in Gram distance subtraction.
            for start in range(0, len(sa), block):
                dist = np.linalg.norm(sa[start:start+block, None, :]-sb[None, :, :], axis=2)
                flat = int(np.argmin(dist)); i,j = np.unravel_index(flat, dist.shape)
                if float(dist[i,j]) < best:
                    best=float(dist[i,j]); pair=[int(a+(start+i)*q), int(b+j*q)]
            centroid = float(np.linalg.norm(centers[a]-centers[b]))
            rows.append(dict(phases=[a,b], sample_distance=best, sample_indices=pair,
                centroid_distance=centroid, bounding_ball_gap=centroid-radii[a]-radii[b]))
    return dict(mean_state_norm=scale, radii_relative=(radii/scale).tolist(), pairs=rows,
                minimum_sample_distance_relative=min(r['sample_distance'] for r in rows)/scale if rows else None,
                minimum_centroid_distance_relative=min(r['centroid_distance'] for r in rows)/scale if rows else None,
                minimum_bounding_ball_gap_relative=min(r['bounding_ball_gap'] for r in rows)/scale if rows else None,
                invariant_set_separation_proved=False)
