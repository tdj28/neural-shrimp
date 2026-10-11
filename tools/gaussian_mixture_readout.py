"""Pure, bounded first-bit Gaussian-mixture probabilities for EXP-020.

No files, neural dynamics, fitting, or research-data access occur here. Stored
binary64 inputs (including coefficients of sums) define exact rationals. Arb
balls then enclose projections, roots, Gaussian CDFs and quadrature bounds.
SciPy's independent nominal quadrature is an estimate, never the certificate.
The process-local Arb precision context assumes one numerical thread per worker.
"""
from __future__ import annotations

from fractions import Fraction
import heapq
import math
import warnings

import numpy as np
from flint import arb, ctx, fmpq
from scipy.integrate import IntegrationWarning, quad
from scipy.special import ndtr, ndtri


PRECISIONS = (128, 256, 512)
MAX_LEAVES = 65536
MAX_ROOT_ITERATIONS = 256
WIDTH = 1e-4
TAIL_RADIUS = 10
ROOT_WIDTH = 1e-14


def contract():
    return {
        'schema': 'gaussian-mixture-readout-v1',
        'precision_bits': list(PRECISIONS), 'maximum_leaves': MAX_LEAVES,
        'maximum_root_iterations': MAX_ROOT_ITERATIONS,
        'probability_width': WIDTH, 'root_absolute_width': ROOT_WIDTH,
        'normal_tail_radius': TAIL_RADIUS, 'quad_epsabs': [1e-9, 1e-11],
        'quad_limits': [512, 1024], 'quad_epsrel': 0.,
        'quad_agreement': 1e-8, 'quad_max_error': 1e-8,
        'exact_binary64_geometry': True, 'rank_snap': False,
        'transition_levels': [-4, -2, -1, 0, 1, 2, 4],
        'nominal_display_clip': [0., 1.],
        'nominal_gates_use_raw_estimate': True,
        'closed_nominal_roundoff_allowance': float(8*np.finfo(float).eps),
        'control_comparison': 'balanced four-class mean',
        'certificate': 'Arb endpoint/extremum Gaussian-mass Darboux enclosure',
    }


class NumericalUnavailable(ValueError):
    def __init__(self, message, *, diagnostics=None, interval=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}
        self.interval = interval


def _float(x, name='scalar'):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name}: finite real scalar required')
    v = float(x)
    if not math.isfinite(v):
        raise ValueError(f'{name}: nonfinite')
    return v


def _vector(x, name='vector'):
    a = np.asarray(x)
    if a.dtype != np.dtype('float64') or a.ndim != 1 or len(a) == 0 or not np.isfinite(a).all():
        raise ValueError(f'{name}: nonempty finite float64 vector required')
    return a


def _q(x):
    return Fraction.from_float(float(x))


def _dot(a, b):
    return sum((_q(x) * _q(y) for x, y in zip(a, b)), Fraction(0))


def _encode(q):
    return [str(q.numerator), str(q.denominator)]


def _decode(pair):
    if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(x, str) for x in pair):
        raise ValueError('invalid exact rational encoding')
    q = Fraction(int(pair[0]), int(pair[1]))
    if _encode(q) != pair:
        raise ValueError('noncanonical rational encoding')
    return q


def _ball(q):
    if isinstance(q, Fraction):
        return arb(fmpq(q.numerator, q.denominator))
    if isinstance(q, float):
        return _ball(_q(q))
    return arb(q)


def _bounds(x):
    """Binary64 endpoints rounded OUTWARD, also when conversion underflows."""
    if not x.is_finite():
        raise NumericalUnavailable('nonfinite Arb ball')
    lo, hi = float(x.lower()), float(x.upper())
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise NumericalUnavailable('ball outside finite binary64 report range')
    # Python conversion rounds to nearest. Compare exactly before adjustment.
    if _ball(lo) > x.lower():
        lo = math.nextafter(lo, -math.inf)
    if _ball(hi) < x.upper():
        hi = math.nextafter(hi, math.inf)
    return [lo, hi]


def _cdf(x):
    return (-x / arb(2).sqrt()).erfc() / 2


def _hull(values):
    out = values[0]
    for v in values[1:]:
        out = out.union(v)
    return out


def _fail(error, *, diagnostics=None, interval=None):
    return {'complete': False, 'available': False, 'qualified': False,
            'status': 'unavailable', 'interval': interval, 'nominal': None,
            'error': str(error), 'diagnostics': diagnostics or {}}


def define_classifier(A, B, sigma=1e-4, *, one_mode_rho=None):
    """Seal reference-only geometry; optional A=rho*B is mathematical/exact.

    Definitions contain arrays plus JSON scalars/strings and can pass unchanged
    through the repository numeric JSON/NPZ codec. No mean or outcome is read.
    Invalid definitions remain explicit unavailable packets.
    """
    try:
        A, B = _vector(A, 'A'), _vector(B, 'B')
        if A.shape != B.shape:
            raise ValueError('A/B dimensions differ')
        sigma = _float(sigma, 'sigma')
        if sigma <= 0:
            raise ValueError('sigma must be positive')
        rho = None if one_mode_rho is None else _float(one_mode_rho, 'rho')
        BB = _dot(B, B)
        if rho is None:
            AA, AB = _dot(A, A), _dot(A, B)
        else:
            AB, AA = _q(rho) * BB, _q(rho)**2 * BB
        D = AA * BB - AB**2
        if D < 0:
            raise ValueError('negative exact Gram determinant')
        branch = ('zero-first' if AA == 0 else 'zero-second' if BB == 0
                  else 'collinear' if D == 0 else 'orthogonal' if AB == 0 else 'rank-two')
        return {'schema': contract()['schema'], 'complete': True, 'available': True,
                'status': 'defined', 'dimension': len(A), 'A': A.copy(), 'B': B.copy(),
                'sigma': sigma, 'one_mode_rho': rho, 'branch': branch,
                'exact': {k: _encode(v) for k, v in {'AA': AA, 'BB': BB, 'AB': AB,
                                                   'D': D, 'sigma': _q(sigma)}.items()}}
    except (ValueError, TypeError, OverflowError) as exc:
        return {'schema': contract()['schema'], 'complete': False, 'available': False,
                'status': 'unavailable', 'error': str(exc)}


def _definition(d):
    if (not isinstance(d, dict) or d.get('available') is not True or
            d.get('complete') is not True or d.get('status') != 'defined'):
        raise ValueError('classifier unavailable')
    rebuilt = define_classifier(d['A'], d['B'], d['sigma'], one_mode_rho=d['one_mode_rho'])
    for key in ['schema', 'dimension', 'sigma', 'one_mode_rho', 'branch', 'exact']:
        if rebuilt.get(key) != d.get(key):
            raise ValueError(f'classifier binding mismatch: {key}')
    return {k: _decode(v) for k, v in d['exact'].items()}


def _terms(terms, n):
    if isinstance(terms, np.ndarray):
        terms = [{'coefficient': 1., 'vector': terms}]
    if not isinstance(terms, (list, tuple)) or not terms:
        raise ValueError('nonempty exact mean_terms required')
    out = []
    for term in terms:
        if not isinstance(term, dict) or set(term) != {'coefficient', 'vector'}:
            raise ValueError('mean term requires coefficient and vector')
        v = _vector(term['vector'], 'mean term')
        if len(v) != n:
            raise ValueError('mean dimension mismatch')
        out.append((_q(_float(term['coefficient'], 'coefficient')), v))
    return out


def _term_dot(vector, terms):
    return sum((coefficient * _dot(vector, v) for coefficient, v in terms), Fraction(0))


def _sign(s):
    if isinstance(s, (bool, np.bool_)) or not isinstance(s, (int, np.integer)) or s not in (-1, 1):
        raise ValueError('first_sign must be integer -1 or +1')
    return int(s)


def _g(t, a, b):
    """Stable odd log-likelihood part, with safe hulls at branch boundaries."""
    if t < 0:
        return -_g(-t, a, b)
    if not t >= 0:
        return _hull([_g(t.lower(), a, b), _g(t.upper(), a, b),
                      2*a*t + (b*(t-a)).cosh().log() - (b*(t+a)).cosh().log()])
    corr = (-2*b*abs(t-a)).exp().log1p() - (-2*b*(t+a)).exp().log1p()
    if t >= a:
        return 2*a*(t-b) + corr
    if t <= a:
        return 2*(a-b)*t + corr
    return _hull([2*a*(t-b)+corr, 2*(a-b)*t+corr])


def _gp(t, a, b):
    return 2*a + b*(b*(t-a)).tanh() - b*(b*(t+a)).tanh()


def _root(function, lo, hi, *, increasing=True):
    """Certified enclosing bisection; uncertain signs retain both endpoints."""
    lo, hi = lo.lower(), hi.upper()
    flo, fhi = function(lo), function(hi)
    if not increasing:
        function0 = function
        function = lambda t: -function0(t)
        flo, fhi = -flo, -fhi
    if not flo < 0 or not fhi > 0:
        raise NumericalUnavailable('root endpoint signs not certified')
    for count in range(MAX_ROOT_ITERATIONS):
        if hi-lo < _ball(ROOT_WIDTH):
            return lo.union(hi), count+1
        mid = ((lo+hi)/2).mid()
        fm = function(mid)
        if fm < 0:
            lo = mid
        elif fm > 0:
            hi = mid
        else:
            # Retain a certified local bracket if evaluation uncertainty hits
            # the midpoint. Never replace an uncertain root by its midpoint.
            q1, q3 = ((lo+mid)/2).mid(), ((mid+hi)/2).mid()
            if function(q1) < 0 and function(q3) > 0:
                lo, hi = q1, q3
            else:
                raise NumericalUnavailable('root sign unresolved at current precision')
    raise NumericalUnavailable('root iteration budget exhausted',
                               diagnostics={'root_bracket': _bounds(lo.union(hi)),
                                            'iterations': MAX_ROOT_ITERATIONS})


def _stationary(a, b, exact):
    if abs(exact['AB']) >= exact['BB']:
        return [], {'stationary_branch': 'monotone', 'root_iterations': 0}
    derivative0 = 2*(a-b*(a*b).tanh())
    if derivative0 > 0 or derivative0.is_zero():
        return [], {'stationary_branch': 'monotone', 'root_iterations': 0}
    if not derivative0 < 0:
        raise NumericalUnavailable('stationary branch unresolved')
    root, count = _root(lambda t: _gp(t, a, b), arb(0), b)
    return [-root, root], {'stationary_branch': 'two-extrema', 'root_iterations': count}


def _range_g(lo, hi, a, b, stationary):
    """Endpoint values plus every possibly intersecting extremum enclosure."""
    values = [_g(lo, a, b), _g(hi, a, b)]
    for root in stationary:
        if not (root.upper() < lo.lower() or root.lower() > hi.upper()):
            values.append(_g(root, a, b))
    return _hull(values)


def _g_float(t, a, b):
    sign = -1. if t < 0 else 1.
    t = abs(t)
    corr = np.logaddexp(0., -2*b*abs(t-a))-np.logaddexp(0., -2*b*(t+a))
    return sign * ((2*a*(t-b) if t >= a else 2*(a-b)*t) + corr)


def _transition_splits(a, b, c, mt, mz, stationary):
    """Retain all seven declared levels on every monotone interval.

    Failed optional efficiency roots are recorded; they never remove mass or
    determine qualification. The certificate independently includes all extrema.
    """
    low, high = (mt-TAIL_RADIUS).lower(), (mt+TAIL_RADIUS).upper()
    cuts = [low]
    for r in stationary:
        for x in (r.lower(), r.upper()):
            if x > low and x < high:
                cuts.append(x)
    cuts.append(high)
    cuts.sort(key=float)
    roots, records = [], []
    for index, (left, right) in enumerate(zip(cuts[:-1], cuts[1:])):
        # Tiny extrema brackets are intentionally not declared monotone.
        if any(not (r.upper() <= left or r.lower() >= right) for r in stationary):
            records.append({'segment': index, 'status': 'extremum-bracket'})
            continue
        gl, gr = _g(left, a, b), _g(right, a, b)
        for level in contract()['transition_levels']:
            target = 2*c*(level-mz)
            fl, fr = gl-target, gr-target
            if (fl < 0 and fr > 0) or (fl > 0 and fr < 0):
                try:
                    r, n = _root(lambda x: _g(x, a, b)-target, left, right,
                                 increasing=bool(fl < 0))
                    roots.append(r)
                    records.append({'segment': index, 'level': level, 'status': 'bracketed',
                                    'bracket': _bounds(r), 'iterations': n})
                except NumericalUnavailable as exc:
                    records.append({'segment': index, 'level': level, 'status': 'unresolved',
                                    'error': str(exc)})
            else:
                records.append({'segment': index, 'level': level, 'status': 'no-certified-crossing'})
    return roots, records


def _enclose(a, b, c, mt, mz, sign, stationary, roots):
    # u is the centered standard normal variable, with exact dyadic endpoints.
    cuts = [-float(TAIL_RADIUS), float(TAIL_RADIUS)]
    for r in stationary + roots:
        for x in _bounds(r-mt):
            if -TAIL_RADIUS < x < TAIL_RADIUS:
                cuts.append(x)
    cuts = sorted(set(cuts))
    heap, total_lo, total_hi, serial = [], arb(0), arb(0), 0

    def leaf(l, r):
        mass = _cdf(_ball(r))-_cdf(_ball(l))
        grange = _range_g(_ball(l)+mt, _ball(r)+mt, a, b, stationary)
        values = _cdf(sign*(mz+grange/(2*c)))
        lower = (mass.lower()*max(arb(0), values.lower())).lower()
        upper = (mass.upper()*min(arb(1), values.upper())).upper()
        return lower, upper

    def push(l, r):
        nonlocal total_lo, total_hi, serial
        lower, upper = leaf(l, r)
        total_lo += lower
        total_hi += upper
        heapq.heappush(heap, (-float(upper-lower), serial, l, r, lower, upper))
        serial += 1

    for l, r in zip(cuts[:-1], cuts[1:]):
        push(l, r)
    tail = 2*_cdf(arb(-TAIL_RADIUS))
    while not total_hi+tail-total_lo <= _ball(WIDTH):
        if len(heap) >= MAX_LEAVES:
            raise NumericalUnavailable(f'subdivision budget exhausted ({len(heap)} leaves)',
                                       diagnostics={'leaves': len(heap), 'tail_bound': _bounds(tail)},
                                       interval=[max(0., _bounds(total_lo)[0]),
                                                 min(1., _bounds(total_hi+tail)[1])])
        _, _, l, r, lower, upper = heapq.heappop(heap)
        total_lo -= lower
        total_hi -= upper
        # Near-equal Gaussian masses improve the Darboux convergence rate;
        # this floating choice never enters the validity of an enclosure.
        q = (float(ndtr(l))+float(ndtr(r)))/2
        mid = float(ndtri(q)) if 0 < q < 1 else (l+r)/2
        if not l < mid < r:
            mid = (l+r)/2
        if not l < mid < r:
            raise NumericalUnavailable('binary64 subdivision endpoints exhausted')
        push(l, mid)
        push(mid, r)
    low = max(0., _bounds(total_lo)[0])
    high = min(1., _bounds(total_hi+tail)[1])
    return [low, high], {'leaves': len(heap), 'tail_bound': _bounds(tail),
                         'initial_intervals': len(cuts)-1}, cuts


def _nominal(a, b, c, mt, mz, sign, cuts):
    a, b, c, mt, mz = map(float, (a, b, c, mt, mz))
    if not all(math.isfinite(x) for x in (a, b, c, mt, mz)) or c <= 0:
        raise NumericalUnavailable('nominal reduction outside finite binary64')
    fun = lambda u: float(ndtr(sign*(mz+_g_float(u+mt, a, b)/(2*c))))*math.exp(-u*u/2)/math.sqrt(2*math.pi)
    # Adjacent outward root endpoints are distinct for the certificate, but
    # QUADPACK can fail on intervals only a few ulps wide. Merge only nominal
    # split hints; this neither changes the integrand nor its full domain.
    points = []
    for point in cuts[1:-1]:
        if not points or point-points[-1] > 1e-10:
            points.append(point)
    results = []
    for tol, limit in [(1e-9, 512), (1e-11, 1024)]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', IntegrationWarning)
            out = quad(fun, -TAIL_RADIUS, TAIL_RADIUS, epsabs=tol, epsrel=0.,
                       limit=limit, points=points, full_output=1)
        if len(out) != 3 or caught or not np.isfinite(out[:2]).all() or out[1] > 1e-8:
            raise NumericalUnavailable('nominal quadrature warning/nonconvergence/error',
                                       diagnostics={'nominal_runs': results,
                                                    'failed_nominal_value': float(out[0]),
                                                    'failed_nominal_error': float(out[1]),
                                                    'message': str(out[3]) if len(out)>3 else None,
                                                    'warnings': [str(w.message) for w in caught]})
        results.append({'value': float(out[0]), 'estimated_error': float(out[1]),
                        'epsabs': tol, 'evaluations': int(out[2]['neval'])})
    if abs(results[0]['value']-results[1]['value']) > 1e-8:
        raise NumericalUnavailable('nominal quadrature disagreement')
    return results[-1]['value'], results


def _finish(interval, nominal, diagnostics, *, nominal_error=0.):
    if (not all(math.isfinite(x) for x in interval+[nominal]) or
            interval[0] < 0 or interval[1] > 1 or interval[1] < interval[0]):
        raise NumericalUnavailable('invalid probability enclosure')
    raw_nominal = nominal
    diagnostics['raw_nominal'] = raw_nominal
    nominal = max(0., min(1., nominal))
    diagnostics['nominal_range_correction'] = nominal-raw_nominal
    qualified = (interval[1]-interval[0] <= WIDTH and
                 -nominal_error <= raw_nominal <= 1+nominal_error and
                 interval[0]-nominal_error <= raw_nominal <= interval[1]+nominal_error)
    return {'complete': True, 'available': True, 'qualified': bool(qualified),
            'status': 'complete' if qualified else 'numerical-gate-failed',
            'interval': interval, 'nominal': nominal,
            'error': None if qualified else 'width or nominal containment gate failed',
            'diagnostics': diagnostics}


def _closed(p, nominal, diagnostics):
    interval = _bounds(p)
    interval = [max(0., interval[0]), min(1., interval[1])]
    # A closed-form scipy CDF uses a separately rounded binary64 projection;
    # report its discrepancy with a prospective 1e-8 agreement bound.
    error = 8*np.finfo(float).eps
    out = _finish(interval, float(nominal), diagnostics, nominal_error=error)
    out['diagnostics']['nominal_method'] = 'SciPy ndtr closed-form CDF'
    out['diagnostics']['nominal_roundoff_allowance'] = error
    return out


def probability(definition, mean_terms, first_sign):
    """Probability of correct first-bit prediction for one actual Gaussian mean.

    A coefficient/vector list is summed exactly, including rho*B and translated
    reference baselines. No finite mean affects the sealed decision rule.
    Exhaustion, invalid input, and arithmetic failures return retained failures.
    """
    attempts = []
    try:
        exact = _definition(definition)
        sign = _sign(first_sign)
        terms = _terms(mean_terms, definition['dimension'])
        BM = _term_dot(definition['B'], terms)
        AM = (_term_dot(definition['A'], terms) if definition['one_mode_rho'] is None
              else _q(definition['one_mode_rho'])*BM)
        for precision in PRECISIONS:
            try:
                with ctx.workprec(precision):
                    return _probability(definition, exact, AM, BM, sign, precision, attempts)
            except NumericalUnavailable as exc:
                attempts.append({'precision_bits': precision, 'error': str(exc),
                                 'partial_interval': exc.interval,
                                 'diagnostics': exc.diagnostics})
        return _fail('precision ladder exhausted', diagnostics={'attempts': attempts,
                                                               'branch': definition['branch']},
                     interval=attempts[-1]['partial_interval'] if attempts else None)
    except (ValueError, TypeError, KeyError, OverflowError, ZeroDivisionError) as exc:
        return _fail(exc, diagnostics={'attempts': attempts})


def _probability(d, exact, AM, BM, sign, precision, attempts):
    sigma, AA, BB, AB, D = [exact[k] for k in ('sigma', 'AA', 'BB', 'AB', 'D')]
    diag = {'branch': d['branch'], 'precision_bits': precision, 'attempts': attempts.copy(),
            'exact_rank': 0 if AA == 0 else 1 if D == 0 else 2,
            'exact_gram_determinant_sign': 0 if D == 0 else 1,
            'reference_unsigned_angle_radians': None}
    if AA > 0 and BB > 0:
        if D == 0:
            angle = arb(0)
        elif AB == 0:
            angle = arb.pi()/2
        else:
            # atan(sqrt(D)/|AB|) avoids subtracting nearly equal cosines and
            # never rounds a nonzero perpendicular direction to rank one.
            angle = (_ball(D).sqrt()/_ball(abs(AB))).atan()
        diag['reference_unsigned_angle_radians'] = _bounds(angle)
    if AA == 0:
        return _closed(arb('1/2'), .5, diag)
    if BB == 0 or AB == 0:
        margin = sign*_ball(AM)/(_ball(sigma)*_ball(AA).sqrt())
        diag['margin'] = _bounds(margin)
        return _closed(_cdf(margin), ndtr(float(margin)), diag)
    a = _ball(abs(AB))/(_ball(sigma)*_ball(BB).sqrt())
    b = _ball(BB).sqrt()/_ball(sigma)
    c = _ball(D/BB).sqrt()/_ball(sigma)
    mt = (1 if AB > 0 else -1)*_ball(BM)/(_ball(sigma)*_ball(BB).sqrt())
    diag['reduction'] = {k: _bounds(v) for k, v in {'a': a, 'b': b, 'c': c, 'mean_parallel': mt}.items()}
    diag['reduction_radius_upper'] = {k: _bounds(v.rad())[1] for k, v in
                                      {'a': a, 'b': b, 'c': c, 'mean_parallel': mt}.items()}
    if D == 0:
        # Exact a>=b includes a=b, whose exponentially small likelihood can
        # otherwise be rounded to a false tie.
        stationary, details = _stationary(a, b, exact)
        diag.update(details)
        if not stationary:
            p = _cdf(sign*mt)
            return _closed(p, ndtr(sign*float(mt)), diag)
        minimum = stationary[-1]
        # Analytically the root is below b. A small exterior endpoint avoids
        # cancellation in g(b) when its strictly positive correction is much
        # smaller than the ball radius of an irrational b. The final bracket
        # is still certified, never rounded or snapped to b.
        exterior = b.upper()+_ball(ROOT_WIDTH)
        root, count = _root(lambda x: _g(x, a, b), minimum.upper(), exterior)
        diag['positive_decision_root'] = _bounds(root)
        diag['decision_root_iterations'] = count
        # Positive prediction intervals are (-r,0) and (r,+infinity).
        pplus = _cdf(-mt)-_cdf(-root-mt)+_cdf(mt-root)
        p = pplus if sign > 0 else 1-pplus
        r, m = float(root), float(mt)
        nomplus = ndtr(-m)-ndtr(-r-m)+ndtr(m-r)
        return _closed(p, nomplus if sign > 0 else 1-nomplus, diag)
    mz = _ball(AM-AB*BM/BB)/(_ball(sigma)*_ball(D/BB).sqrt())
    diag['reduction']['mean_perpendicular'] = _bounds(mz)
    diag['reduction_radius_upper']['mean_perpendicular'] = _bounds(mz.rad())[1]
    stationary, details = _stationary(a, b, exact)
    diag.update(details)
    diag['stationary_brackets'] = [_bounds(r) for r in stationary]
    roots, root_records = _transition_splits(a, b, c, mt, mz, stationary)
    diag['transition_roots'] = root_records
    interval = None
    try:
        interval, enclosure, cuts = _enclose(a, b, c, mt, mz, sign, stationary, roots)
        diag['enclosure'] = enclosure
        nominal, nominal_runs = _nominal(a, b, c, mt, mz, sign, cuts)
    except NumericalUnavailable as exc:
        raise NumericalUnavailable(str(exc), diagnostics={**diag, **exc.diagnostics},
                                   interval=exc.interval or interval) from exc
    diag['nominal_runs'] = nominal_runs
    return _finish(interval, nominal, diag, nominal_error=nominal_runs[-1]['estimated_error'])


def linear_probability(h, mean_terms, first_sign, sigma=1e-4):
    """Certified fixed-linear observer probability for exact mean sums."""
    try:
        h = _vector(h, 'h')
        sigma = _float(sigma, 'sigma')
        if sigma <= 0:
            raise ValueError('sigma must be positive')
        sign = _sign(first_sign)
        terms = _terms(mean_terms, len(h))
        hh, hm = _dot(h, h), _term_dot(h, terms)
        if hh == 0:
            raise ValueError('zero linear observer')
        with ctx.workprec(PRECISIONS[0]):
            margin = sign*_ball(hm)/(_ball(_q(sigma))*_ball(hh).sqrt())
            return _closed(_cdf(margin), ndtr(float(margin)),
                           {'branch': 'fixed-linear', 'precision_bits': PRECISIONS[0],
                            'margin': _bounds(margin)})
    except (ValueError, TypeError, KeyError, OverflowError, ZeroDivisionError) as exc:
        return _fail(exc)
