"""Independent Task21a scalar arithmetic. No producer imports, models or array algebra."""
from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction
import math

PF2_REFERENCE = Fraction(4669, 1000)
PF2_LOWER = Fraction(88711, 20000)   # 4.43555 exactly
PF2_UPPER = Fraction(98049, 20000)   # 4.90245 exactly


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def exact(value):
    """An exact rational of the JSON decimal spelling, not a shortened display value."""
    require(type(value) in (int, float, Decimal, str, Fraction), 'real scalar required')
    if isinstance(value, Fraction):
        return value
    if type(value) is float:
        require(math.isfinite(value), 'finite binary64 scalar')
        return Fraction.from_float(value)
    number = Decimal(value)
    require(number.is_finite(), 'finite decimal scalar')
    return Fraction(number)


def f64(value):
    number = float(value)
    require(math.isfinite(number), 'finite represented binary64 value')
    return number


def decimal_string(number, precision=60):
    number = exact(number)
    with localcontext() as context:
        context.prec = precision
        return str(Decimal(number.numerator) / Decimal(number.denominator))


def rational_record(number):
    number = exact(number)
    return dict(numerator=str(number.numerator), denominator=str(number.denominator), decimal_60=decimal_string(number))


def scalar_record(value):
    return dict(saved_decimal=str(value), represented_binary64=f64(value), binary64_hex=f64(value).hex())


def outward(number, places, upper):
    """Exact floor/ceiling at a decimal grid; no binary rounding before bounding."""
    number = exact(number)
    scale = 10 ** places
    integer = -((-number.numerator * scale) // number.denominator) if upper else (number.numerator * scale) // number.denominator
    sign = '-' if integer < 0 else ''
    digits = str(abs(integer)).rjust(places + 1, '0')
    return sign + (digits[:-places] + '.' + digits[-places:] if places else digits)


def interval_record(low, high, places=12):
    low, high = exact(low), exact(high)
    require(low <= high, 'interval ordering')
    return dict(exact_lower=rational_record(low), exact_upper=rational_record(high),
                outward_lower=outward(low, places, False), outward_upper=outward(high, places, True), decimal_places=places)


def ratio(a, b, c):
    a, b, c = map(exact, (a, b, c))
    require(a < b < c, 'ratio needs three strictly ascending parameters')
    return (b - a) / (c - b)


def ratio_record(parameters, saved=None):
    require(len(parameters) == 3, 'ratio uses exactly three parameters')
    x = list(map(f64, parameters))
    require(x[0] < x[1] < x[2], 'represented parameters strictly ascending')
    computed = (x[1] - x[0]) / (x[2] - x[1])
    result = dict(parameters=[scalar_record(v) for v in parameters], independent_binary64=computed,
        exact_decimal_input_ratio=rational_record(ratio(*parameters)),
        exact_binary64_input_ratio=rational_record(ratio(*[Fraction.from_float(v) for v in x])))
    if saved is not None:
        original = f64(saved)
        result.update(saved_ratio=scalar_record(saved), saved_minus_independent_binary64=original - computed,
                      exact_binary64_reproduction=original == computed)
    return result


def coupled_ratio_envelope(brackets):
    """Sensitivity of a shared-middle-point ratio, conditional on ordered true brackets."""
    require(len(brackets) == 3 and all(len(b) == 2 for b in brackets), 'three endpoint pairs')
    bounds = [sorted(map(exact, b)) for b in brackets]
    out = dict(assumption='The intended continuous scalar root lies in each saved clean bracket; not a rigorous model enclosure.',
               brackets=[interval_record(*b, places=18) for b in bounds])
    (al, ah), (bl, bh), (cl, ch) = bounds
    if not ah < bl or not bh < cl:
        return dict(out, disposition='unresolved: independent root intervals are not strictly ordered', envelope=None)
    lower = ratio(ah, bl, ch)
    upper = ratio(al, bh, cl)
    return dict(out, disposition='ordered numerical bracket sensitivity', envelope=interval_record(lower, upper),
        extremal_corners={'lower': [decimal_string(x) for x in (ah, bl, ch)], 'upper': [decimal_string(x) for x in (al, bh, cl)]},
        entire_envelope_within_frozen_pf2=PF2_LOWER <= lower and upper <= PF2_UPPER)


def ratio_sequence(levels, saved_ratios=None):
    """Do not skip failed or mismatched levels; a triple keeps its own disposition."""
    rows = []
    for i in range(max(0, len(levels) - 2)):
        triple = levels[i:i + 3]
        entry = dict(index=i, levels=[dict(attempt=x['attempt'], period=x['period'], disposition=x['disposition']) for x in triple])
        periods = [x['period'] for x in triple]
        admissible = all(x.get('parameter') is not None and x['disposition'] in ('accepted_flip', 'superstable_point_estimate', 'logged_interpolation') for x in triple)
        if not admissible:
            rows.append(dict(entry, disposition='cannot check: missing or refused level'))
            continue
        if not (periods[1] == 2 * periods[0] and periods[2] == 2 * periods[1]):
            rows.append(dict(entry, disposition='cannot check: nonconsecutive doubled periods'))
            continue
        parameters = [x['parameter'] for x in triple]
        point = ratio(*parameters)
        entry.update(disposition='computed', arithmetic=ratio_record(parameters, None if saved_ratios is None else saved_ratios[i]),
                     point_within_frozen_pf2=PF2_LOWER <= point <= PF2_UPPER)
        if all(x.get('clean_bracket') is not None for x in triple):
            entry['clean_bracket_sensitivity'] = coupled_ratio_envelope([x['clean_bracket'] for x in triple])
        rows.append(entry)
    return rows


def pf1(first_ratio_156, first_ratio_152):
    threshold = Fraction(711, 250)
    smaller156 = None if first_ratio_156 is None else exact(first_ratio_156) < threshold
    smaller152 = None if first_ratio_152 is None else exact(first_ratio_152) < threshold
    ordered = None if first_ratio_156 is None or first_ratio_152 is None else exact(first_ratio_156) < exact(first_ratio_152)
    values = [smaller156, smaller152, ordered]
    overall = 'fails' if False in values else 'cannot check' if None in values else 'passes'
    return dict(frozen_threshold='2.844', below_at_minus_156=smaller156, below_at_minus_152=smaller152,
                ratio_lower_at_minus_156=ordered, conjunctive_prediction=overall)


def pf2(sequence, as_of, latest_period):
    require(sequence and sequence[-1]['period'] == latest_period, 'explicit latest completed scoring level')
    require(all(x.get('kind') == 'superstable' for x in sequence), 'P-F2 requires superstable levels, never flip ratios')
    ratios = ratio_sequence(sequence)
    last = ratios[-2:]
    point = None if len(last) != 2 or any(r['disposition'] != 'computed' for r in last) else all(r['point_within_frozen_pf2'] for r in last)
    envelope = None
    if len(last) == 2 and all('clean_bracket_sensitivity' in r and r['clean_bracket_sensitivity']['envelope'] is not None for r in last):
        envelope = all(r['clean_bracket_sensitivity']['entire_envelope_within_frozen_pf2'] for r in last)
    return dict(as_of=as_of, latest_completed_period=latest_period, frozen_reference='4.669', lower='4.43555', upper='4.90245',
        rule='last two superstable ratios within the two-sided 5 percent interval',
        last_two=last, point_estimate_pass=point, entire_numerical_bracket_envelope_pass=envelope,
        distinction='Point scoring and conditional numerical bracket sensitivity are separate; neither is a rigorous model proof.')


def secant(a, b, fa, fb):
    a, b, fa, fb = map(exact, (a, b, fa, fb))
    require(a != b and fa * fb < 0, 'opposite-sign secant endpoints')
    return (a * fb - b * fa) / (fb - fa)


def ols_root(rows, x_key='lambda', y_key='g'):
    """Centered exact arithmetic for the disclosed local-linear diagnostic, not a locator retry."""
    require(len(rows) >= 3, 'at least three saved evaluations for residual degrees of freedom')
    x, y = [exact(r[x_key]) for r in rows], [exact(r[y_key]) for r in rows]
    n = len(x)
    xb, yb = sum(x) / n, sum(y) / n
    sxx = sum((v - xb) ** 2 for v in x)
    require(sxx > 0, 'distinct parameters needed for fit')
    slope = sum((a - xb) * (b - yb) for a, b in zip(x, y)) / sxx
    require(slope != 0, 'nonzero fitted slope required')
    root = xb - yb / slope
    residuals = [b - (yb + slope * (a - xb)) for a, b in zip(x, y)]
    sse = sum(r * r for r in residuals)
    sigma2 = sse / (n - 2)
    root_variance = sigma2 / (slope * slope) * (Fraction(1, n) + (root - xb) ** 2 / sxx)
    with localcontext() as context:
        context.prec = 80
        def sqrt(q):
            return (Decimal(q.numerator) / Decimal(q.denominator)).sqrt()
        sigma = sqrt(sigma2)
        rms = sqrt(sse / n)
        standard_error = sqrt(root_variance)
        # A one-unit conservative decimal arithmetic rounding of this diagnostic;
        # it does not repair unverified statistical error assumptions.
        se_up = standard_error.next_plus().quantize(Decimal('1e-18'), rounding=ROUND_CEILING)
    max_scatter = max(map(abs, residuals))
    empirical_radius = max_scatter / abs(slope)
    return dict(n=n, root=rational_record(root), slope=rational_record(slope), centered_x=rational_record(xb), centered_y=rational_record(yb),
        x_span=interval_record(min(x), max(x), places=18), residuals=[rational_record(r) for r in residuals],
        maximum_absolute_line_residual=rational_record(max_scatter), residual_rms=str(rms), residual_standard_deviation=str(sigma),
        scatter_over_slope_radius=rational_record(empirical_radius), empirical_scatter_band=interval_record(root - empirical_radius, root + empirical_radius, places=18),
        delta_method_root_standard_error=str(standard_error), standard_error_rounded_up_1e18=str(se_up),
        standard_error_assumptions='A locally correct linear model with fixed x and independent homoscedastic errors; deterministic solver errors need not satisfy this. Not a calibrated confidence interval.',
        original_locator_status='Unchanged: any original refusal remains a refusal. This is a separately labelled saved-evaluation interpolation.')


def check_evaluation(record, period, tol, g_accept=None, accepted_flip=False):
    """Read saved scalar roles; no unseen multiplier or half-period witness is invented."""
    require(type(period) is int and period > 0, 'positive source period')
    tolerance = max(f64(tol), 1e-13 * period)
    critical = record['critical_multiplier']
    multipliers = record['multipliers']
    require(len(critical) == 4 and multipliers and all(len(z) == 2 for z in multipliers), 'saved multiplier schema')
    complex_values = [complex(f64(z[0]), f64(z[1])) for z in multipliers]
    selected = min(range(len(complex_values)), key=lambda j: complex_values[j].real)
    mu = complex_values[selected]
    history = list(map(f64, record['newton_residuals']))
    require(history, 'saved Newton history missing')
    reported = complex(f64(critical[0]), f64(critical[1]))
    ritz = f64(record['ritz_residual'])
    other = max((abs(z) for j, z in enumerate(complex_values) if j != selected), default=None)
    result = dict(period=period, parameter=scalar_record(record['lambda']),
        critical_index_from_saved_pairs=selected,
        critical_real_difference=reported.real - mu.real, critical_imag_difference=reported.imag - mu.imag,
        saved_modulus_minus_hypot=f64(critical[2]) - abs(reported),
        saved_turns_minus_atan2=f64(critical[3]) - math.atan2(reported.imag, reported.real) / (2 * math.pi),
        next_modulus_difference=None if other is None or record['next_modulus'] is None else f64(record['next_modulus']) - other,
        newton_history=record['newton_residuals'], effective_newton_tolerance=tolerance,
        newton_converged=history[-1] < tolerance, ritz_residual=ritz,
        g_from_saved_real=reported.real + 1., realness_at_1e8=abs(reported.imag) <= 1e-8,
        next_modulus_below_one=None if record['next_modulus'] is None else f64(record['next_modulus']) < 1,
        half_period_distance='not saved in this record; source guard or original refusal is separate',
        complete_spectrum='not established by the saved partial Ritz pairs')
    if accepted_flip:
        require(g_accept is not None, 'original acceptance threshold required')
        result.update(original_g_accept=f64(g_accept), best_evaluation_g_gate=abs(reported.real + 1.) <= f64(g_accept),
            ritz_acceptance_gate=ritz <= 1e-6)
        result['saved_flip_gates_pass'] = all((result['newton_converged'], result['realness_at_1e8'],
            result['next_modulus_below_one'] is True, result['best_evaluation_g_gate'], result['ritz_acceptance_gate'],
            reported == mu))
    return result
