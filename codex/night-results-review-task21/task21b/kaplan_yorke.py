"""Finite saved-spectrum Kaplan–Yorke arithmetic; no model or producer imports."""
from decimal import Decimal, localcontext
from fractions import Fraction
import math


def exact(value):
    if isinstance(value, Fraction):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError('nonfinite exponent')
        return Fraction.from_float(value)
    if type(value) not in (int, str, Decimal):
        raise ValueError('real scalar required')
    x = Decimal(value)
    if not x.is_finite():
        raise ValueError('nonfinite exponent')
    return Fraction(x)


def outward(value, places, upper):
    value = exact(value); scale = 10 ** places
    n = -((-value.numerator * scale) // value.denominator) if upper else value.numerator * scale // value.denominator
    sign = '-' if n < 0 else ''; digits = str(abs(n)).rjust(places + 1, '0')
    return sign + (digits[:-places] + '.' + digits[-places:] if places else digits)


def number(value, places=6):
    value = exact(value)
    with localcontext() as context:
        context.prec = 60
        display = str(Decimal(value.numerator) / Decimal(value.denominator))
    return dict(numerator=str(value.numerator), denominator=str(value.denominator), decimal_60=display,
                outward_lower=outward(value, places, False), outward_upper=outward(value, places, True), decimal_places=places)


def estimate(values, *, leading_spectrum, complete_spectrum=False, ambient_dimension=None):
    """Sort retained estimates stably, retaining the original order and every disposition.

    leading_spectrum asserts only the reviewed estimator/source contract that these are
    the leading estimates; it is not a proof of true asymptotic Lyapunov exponents.
    A nonnegative last retained sum with an unknown tail supplies only a conditional
    lower bound. A crossing before the end is evaluable from the retained estimates.
    """
    out = dict(original_values=[str(x) for x in values], retained_count=len(values),
        leading_spectrum=leading_spectrum, complete_spectrum=complete_spectrum,
        ambient_dimension=ambient_dimension,
        interpretation='Finite estimated-spectrum arithmetic, not an exact invariant dimension or proof of chaos.')
    try:
        if type(leading_spectrum) is not bool or type(complete_spectrum) is not bool:
            raise ValueError('explicit Boolean spectrum-scope flags required')
        if not values:
            raise ValueError('empty spectrum')
        x = list(map(exact, values)); m = len(x)
        if ambient_dimension is not None and (type(ambient_dimension) is not int or ambient_dimension < m):
            raise ValueError('ambient dimension must be an integer at least the retained count')
        if complete_spectrum and ambient_dimension not in (None, m):
            raise ValueError('complete-spectrum count disagrees with ambient dimension')
        order = sorted(range(m), key=lambda i: x[i], reverse=True)
        lam = [x[i] for i in order]; sums = []; total = Fraction(0)
        for v in lam:
            total += v; sums.append(total)
        out.update(sorted_to_original_index=order, input_already_nonincreasing=order == list(range(m)),
            sorted_values=[number(v) for v in lam], cumulative_sums=[number(v) for v in sums],
            zero_exponent_indices=[i for i,v in enumerate(lam) if v == 0],
            arithmetic_basis='Exact rational arithmetic on supplied decimal spellings, or exact represented values for supplied Python floats.')
        j = max([0]+[i+1 for i,s in enumerate(sums) if s >= 0]); out['j'] = j
        supported = leading_spectrum or complete_spectrum
        if j == m:
            out.update(dimension=None, denominator=None,
                disposition='full finite spectrum has no negative cumulative sum; dimension equals ambient count' if complete_spectrum else
                    'truncated spectrum has no negative cumulative sum; tail indeterminate',
                conditional_lower_bound=number(m) if supported else None)
            if complete_spectrum:
                out['dimension'] = number(m)
            else:
                out['tail_statement'] = 'D_KY is at least the retained count only under the leading-spectrum assumption; no point estimate is supplied.'
            if not supported:
                out['disposition'] = 'retained cumulative sum is nonnegative; leading-tail authority unavailable'
            return out
        s = sums[j-1] if j else Fraction(0); den = abs(lam[j])
        # With descending finite values and maximal j, the next cumulative sum is
        # negative; a zero denominator cannot be bypassed or replaced by epsilon.
        if den == 0:
            return dict(out, disposition='undefined zero denominator', dimension=None, conditional_lower_bound=None)
        d = Fraction(j) + s / den
        out.update(denominator=number(den), numerator_sum=number(s),
            diagnostic_from_retained_estimates=number(d), dimension=number(d) if supported else None,
            conditional_lower_bound=None,
            disposition=('all retained exponents strictly negative; finite estimated dimension zero' if j == 0 else
                'negative cumulative-sum crossing lies within retained spectrum') if supported else
                'retained arithmetic available but leading-spectrum authority unavailable')
        return out
    except (ValueError, TypeError, ArithmeticError) as exc:
        return dict(out, disposition='invalid saved spectrum', error=str(exc), dimension=None, conditional_lower_bound=None)


def dimension_range(rows, places=3):
    """Extrema of point estimates only, with unresolved/bound-only rows kept separate."""
    admitted=[]; omitted=[]
    for i,row in enumerate(rows):
        d=row.get('dimension')
        if d is None:
            omitted.append(dict(index=i,disposition=row['disposition'],conditional_lower_bound=row.get('conditional_lower_bound')))
        else:
            admitted.append((i,Fraction(int(d['numerator']),int(d['denominator']))))
    result=dict(total_rows=len(rows),point_estimate_count=len(admitted),other_rows=omitted)
    if admitted:
        lo=min(v for _,v in admitted); hi=max(v for _,v in admitted)
        result.update(exact_min=number(lo),exact_max=number(hi),
            min_original_row_indices=[i for i,v in admitted if v==lo],max_original_row_indices=[i for i,v in admitted if v==hi],
            outward_display_min=outward(lo,places,False),outward_display_max=outward(hi,places,True),decimal_places=places)
    else:
        result.update(exact_min=None,exact_max=None,outward_display_min=None,outward_display_max=None)
    return result
