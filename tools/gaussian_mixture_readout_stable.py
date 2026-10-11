"""EXP-021 draft: higher-precision point likelihood for nominal quadrature.

Pure, no IO/ODE/fitting. The EXP020 input geometry, rank, classifier definitions,
root brackets, Darboux probability certificate and scientific gates are reused
unchanged. No monkeypatch of the frozen module occurs. Only rank-two nominal
point evaluation differs; scalar/linear branches delegate byte-for-byte.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from flint import arb, ctx
from scipy.integrate import IntegrationWarning, quad

import gaussian_mixture_readout as frozen

POINT_ERROR_LIMIT = 1e-15
FROZEN_CORE_SHA256 = '04c586df68f8d06527146243035b1182b6cd7fa75c056b55a33f6794be66e2f3'
define_classifier = frozen.define_classifier
linear_probability = frozen.linear_probability


def contract():
    result = frozen.contract()
    result.update(nominal_implementation='Arb direct-logcosh point CDF plus unchanged QUADPACK',
        inherited_core_sha256=FROZEN_CORE_SHA256,
        point_probability_absolute_error_limit=POINT_ERROR_LIMIT,
        nominal_extra_conditional_cdf_levels=[-8,8],
        nominal_fixed_standard_normal_splits=list(range(-9,10)),
        point_likelihood_algebra_check='Direct logcosh ball must overlap frozen stable-likelihood ball',
        coefficient_rounding='Original exact-input reduction balls retained inside point evaluation',
        original_containment_allowance='Unchanged QUADPACK estimated error only',
        scalar_and_linear_paths='Delegate unchanged to frozen EXP020 core')
    return result


def _point_probability(u, a, b, c, mt, mz, sign):
    """Enclose a single conditional CDF before nearest-float conversion.

    u is the exact binary64 node chosen by SciPy. All reduced coefficients are
    the original Arb balls, not newly rounded double constants. The returned
    bound encloses the float's absolute difference from the mathematical CDF.
    """
    t = frozen._ball(float(u))+mt
    direct = 2*a*t+(b*(t-a)).cosh().log()-(b*(t+a)).cosh().log()
    alternate = frozen._g(t,a,b)
    if not direct.is_finite() or not alternate.is_finite() or not direct.overlaps(alternate):
        raise frozen.NumericalUnavailable('Point likelihood algebraic enclosures disagree/nonfinite')
    probability = frozen._cdf(sign*(mz+direct/(2*c)))
    if not probability.is_finite():
        raise frozen.NumericalUnavailable('Nonfinite nominal point-CDF enclosure')
    value = float(probability.mid())
    if not math.isfinite(value):
        raise frozen.NumericalUnavailable('Nonfinite nominal point probability')
    value = max(0.,min(1.,value))
    represented = frozen._ball(value)
    error = max(abs(represented-probability.lower()).upper(),
                abs(probability.upper()-represented).upper())
    upper_error = frozen._bounds(error)[1]
    if upper_error > POINT_ERROR_LIMIT:
        raise frozen.NumericalUnavailable('Nominal point-CDF error budget exceeded',
            diagnostics={'point_cdf_interval':frozen._bounds(probability),
                         'point_float':value,'point_absolute_error_upper':upper_error,
                         'point_u':float(u)})
    return value,upper_error


def _extra_nominal_splits(a,b,c,mt,mz,stationary):
    """Optional ±8 conditional-CDF hints; no change to certificate or model."""
    low,high=(mt-frozen.TAIL_RADIUS).lower(),(mt+frozen.TAIL_RADIUS).upper()
    cuts=[low,high]
    for root in stationary:
        for x in (root.lower(),root.upper()):
            if x>low and x<high:cuts.append(x)
    cuts.sort(key=float)
    hints=[];records=[]
    for index,(left,right) in enumerate(zip(cuts[:-1],cuts[1:])):
        if any(not (r.upper()<=left or r.lower()>=right) for r in stationary):
            records.append(dict(segment=index,status='extremum-bracket'));continue
        for level in [-8,8]:
            target=2*c*(level-mz)
            fleft=frozen._g(left,a,b)-target;fright=frozen._g(right,a,b)-target
            if (fleft<0 and fright>0) or (fleft>0 and fright<0):
                try:
                    bracket,count=frozen._root(lambda t:frozen._g(t,a,b)-target,left,right,
                                               increasing=bool(fleft<0))
                    coordinate=float(bracket-mt)
                    if -frozen.TAIL_RADIUS<coordinate<frozen.TAIL_RADIUS:hints.append(coordinate)
                    records.append(dict(segment=index,level=level,status='bracketed',
                                        bracket=frozen._bounds(bracket),iterations=count))
                except frozen.NumericalUnavailable as error:
                    records.append(dict(segment=index,level=level,status='unresolved',error=str(error)))
            else:records.append(dict(segment=index,level=level,status='no-certified-crossing'))
    return hints,records


def _nominal(a,b,c,mt,mz,sign,cuts,stationary):
    extra,extra_records=_extra_nominal_splits(a,b,c,mt,mz,stationary)
    raw_points=sorted(cuts[1:-1]+extra+[float(x) for x in range(-9,10)])
    points=[]
    for point in raw_points:
        if not points or point-points[-1]>1e-10:
            points.append(point)
    results=[]
    for tolerance,limit in [(1e-9,512),(1e-11,1024)]:
        stats={'point_attempts':0,'point_evaluations':0,'maximum_point_absolute_error':0.}
        def integrand(u):
            stats['point_attempts']+=1
            value,error=_point_probability(u,a,b,c,mt,mz,sign)
            stats['point_evaluations']+=1
            stats['maximum_point_absolute_error']=max(stats['maximum_point_absolute_error'],error)
            return value*math.exp(-u*u/2)/math.sqrt(2*math.pi)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always',IntegrationWarning)
                out=quad(integrand,-frozen.TAIL_RADIUS,frozen.TAIL_RADIUS,
                    epsabs=tolerance,epsrel=0.,limit=limit,points=points,full_output=1)
        except frozen.NumericalUnavailable as error:
            raise frozen.NumericalUnavailable(str(error),
                diagnostics={**error.diagnostics,'nominal_runs':results,
                    'failed_point_run':dict(epsabs=tolerance,**stats)},
                interval=error.interval) from error
        record=dict(value=float(out[0]),estimated_error=float(out[1]),epsabs=tolerance,
                    evaluations=int(out[2]['neval']),extra_nominal_roots=extra_records,
                    nominal_points=points,**stats)
        if len(out)!=3 or caught or not np.isfinite(out[:2]).all() or out[1]>1e-8:
            raise frozen.NumericalUnavailable('Nominal quadrature warning/nonconvergence/error',
                diagnostics={'nominal_runs':results,'failed_nominal_run':record,
                             'message':str(out[3]) if len(out)>3 else None,
                             'warnings':[str(w.message) for w in caught]})
        results.append(record)
    if abs(results[0]['value']-results[1]['value'])>1e-8:
        raise frozen.NumericalUnavailable('Nominal quadrature disagreement',
                                         diagnostics={'nominal_runs':results})
    return results[-1]['value'],results


def _rank_two(definition,exact,AM,BM,sign,precision,attempts):
    """Same reduction/enclosure as EXP020; only the nominal callback changes."""
    sigma,AA,BB,AB,D=[exact[k] for k in ('sigma','AA','BB','AB','D')]
    if not (AA>0 and BB>0 and AB!=0 and D>0):
        raise ValueError('Exact rank-two nonorthogonal geometry required')
    angle=(frozen._ball(D).sqrt()/frozen._ball(abs(AB))).atan()
    diag={'branch':definition['branch'],'precision_bits':precision,'attempts':attempts.copy(),
          'exact_rank':2,'exact_gram_determinant_sign':1,
          'reference_unsigned_angle_radians':frozen._bounds(angle),
          'nominal_implementation':contract()['nominal_implementation']}
    a=frozen._ball(abs(AB))/(frozen._ball(sigma)*frozen._ball(BB).sqrt())
    b=frozen._ball(BB).sqrt()/frozen._ball(sigma)
    c=frozen._ball(D/BB).sqrt()/frozen._ball(sigma)
    mt=(1 if AB>0 else -1)*frozen._ball(BM)/(frozen._ball(sigma)*frozen._ball(BB).sqrt())
    mz=frozen._ball(AM-AB*BM/BB)/(frozen._ball(sigma)*frozen._ball(D/BB).sqrt())
    reduced={'a':a,'b':b,'c':c,'mean_parallel':mt,'mean_perpendicular':mz}
    diag['reduction']={k:frozen._bounds(v) for k,v in reduced.items()}
    diag['reduction_radius_upper']={k:frozen._bounds(v.rad())[1] for k,v in reduced.items()}
    stationary,details=frozen._stationary(a,b,exact)
    diag.update(details)
    diag['stationary_brackets']=[frozen._bounds(r) for r in stationary]
    roots,records=frozen._transition_splits(a,b,c,mt,mz,stationary)
    diag['transition_roots']=records
    interval=None
    try:
        interval,enclosure,cuts=frozen._enclose(a,b,c,mt,mz,sign,stationary,roots)
        diag['enclosure']=enclosure
        nominal,runs=_nominal(a,b,c,mt,mz,sign,cuts,stationary)
    except frozen.NumericalUnavailable as exc:
        raise frozen.NumericalUnavailable(str(exc),diagnostics={**diag,**exc.diagnostics},
                                         interval=exc.interval or interval) from exc
    diag['nominal_runs']=runs
    # Original containment gate and its allowance are unchanged. The separate
    # point bound is1e-15, four orders smaller than requested1e-11 tolerance.
    return frozen._finish(interval,nominal,diag,nominal_error=runs[-1]['estimated_error'])


def probability(definition,mean_terms,first_sign):
    # Every non-rank-two branch is exactly the prior implementation, including
    # its errors, scalar roots, direct CDFs and outcome-independent definitions.
    if not isinstance(definition,dict) or definition.get('branch')!='rank-two':
        return frozen.probability(definition,mean_terms,first_sign)
    attempts=[]
    try:
        exact=frozen._definition(definition)
        sign=frozen._sign(first_sign)
        terms=frozen._terms(mean_terms,definition['dimension'])
        BM=frozen._term_dot(definition['B'],terms)
        AM=frozen._term_dot(definition['A'],terms)
        for precision in frozen.PRECISIONS:
            try:
                with ctx.workprec(precision):
                    return _rank_two(definition,exact,AM,BM,sign,precision,attempts)
            except frozen.NumericalUnavailable as exc:
                attempts.append(dict(precision_bits=precision,error=str(exc),
                                     partial_interval=exc.interval,diagnostics=exc.diagnostics))
        return frozen._fail('precision ladder exhausted',
            diagnostics={'attempts':attempts,'branch':definition['branch']},
            interval=attempts[-1]['partial_interval'] if attempts else None)
    except (ValueError,TypeError,KeyError,OverflowError,ZeroDivisionError) as exc:
        return frozen._fail(exc,diagnostics={'attempts':attempts})
