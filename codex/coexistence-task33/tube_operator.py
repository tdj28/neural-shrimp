"""One conditional infinity-operator enclosure on a nominal convex input cube.

No model loading, source admission, trajectory propagation, or certificate.
The caller supplies a separately reviewed interval-Jet forward and an optional
saving callback. Arb[d,1] directions must contain the whole Cartesian unit cube.
"""
from fractions import Fraction
from flint import arb, arb_mat, ctx


def need(condition, message):
    if not condition:
        raise ValueError(message)


def rational(value):
    return {'numerator': value.numerator, 'denominator': value.denominator}


def native(value):
    need(isinstance(value, arb) and value.is_finite(), 'finite native Arb ball required')
    return {'mid': list(map(int, value.mid().man_exp())),
            'rad': list(map(int, value.rad().man_exp()))}


def dyadic(pair):
    return Fraction(pair[0]) * Fraction(2) ** pair[1]


def endpoints(record):
    mid, rad = dyadic(record['mid']), dyadic(record['rad'])
    need(rad >= 0, 'negative native radius')
    return mid-rad, mid+rad


def positive_radius(radius):
    need(type(radius) is int or isinstance(radius, Fraction),
         'radius must be an exact integer or Fraction, never a float/bool/ball')
    radius = Fraction(radius)
    need(radius > 0 and radius.denominator & (radius.denominator-1) == 0,
         'positive exact dyadic radius required')
    return radius


def centre_records(centres, dimension):
    need(type(dimension) is int and dimension > 0, 'positive exact dimension required')
    need(isinstance(centres, (list, tuple)) and len(centres) == dimension,
         'complete exact centre vector required')
    records = [native(c) for c in centres]
    need(all(c.is_exact() for c in centres), 'uncertain centre is not an exact point')
    return records


def matrix_records(matrix, dimension, label):
    need(isinstance(matrix, arb_mat) and
         (matrix.nrows(), matrix.ncols()) == (dimension, 1),
         label+' must be a native Arb matrix of shape(d,1)')
    return [[native(matrix[i, 0])] for i in range(dimension)]


def admit(centres, radius, primal, directions, dimension):
    """Check actual native endpoints, not reconstructed rounded Arb endpoints."""
    radius = positive_radius(radius)
    centres_saved = centre_records(centres, dimension)
    need(isinstance(primal, (list, tuple)) and len(primal) == dimension,
         'complete live primal vector required')
    primal_saved = [native(x) for x in primal]
    direction_saved = matrix_records(directions, dimension, 'directions')
    for c, x, v in zip(centres_saved, primal_saved, direction_saved):
        nominal = dyadic(c['mid'])
        lower, upper = endpoints(x)
        need(lower <= nominal-radius and upper >= nominal+radius,
             'live primal does not contain the full nominal coordinate interval')
        lower, upper = endpoints(v[0])
        need(lower <= -1 and upper >= 1,
             'each live direction coordinate must contain [-1,1] independently')
    return dict(centres=centres_saved, nominal_radius=rational(radius),
                live_primal=primal_saved, live_directions=direction_saved,
                dimension=dimension, direction_count=1)


def build(centres, radius, dimension):
    """Return owned live inputs; any native construction widening is retained."""
    radius = positive_radius(radius)
    saved = centre_records(centres, dimension)
    rpair = (radius.numerator, -(radius.denominator.bit_length()-1))
    primal = [arb(tuple(c['mid']), rpair) for c in saved]
    directions = arb_mat(dimension, 1, [arb((0, 0), (1, 0)) for _ in saved])
    admit(centres, radius, primal, directions, dimension)
    return primal, directions


def evaluate(centres, radius, forward, *, dimension, primal=None, directions=None,
             save=None):
    """Return JSON-safe conditional evidence from one reviewed forward call.

    forward(live_primal, live_directions) must rigorously enclose F(x), DF(x)v
    simultaneously for every x and v in those Cartesian input enclosures. Its
    source, fixed parameters/prefix, assets, backend, successful domain gates,
    full dimensions and execution/closure must be authenticated separately.
    Before/after snapshots catch changed values, shapes, or arithmetic context;
    they do not defend against a hostile callback that mutates and restores.
    """
    need(callable(forward) and (save is None or callable(save)), 'callable forward/save required')
    need((primal is None) == (directions is None), 'supply both live inputs or neither')
    if primal is None:
        primal, directions = build(centres, radius, dimension)
    initial = admit(centres, radius, primal, directions, dimension)
    context = (ctx.prec, ctx.threads)
    flags = dict(trajectory_enclosure=False, certificate_passed=False,
                 conditional_operator_bound_only=True)
    if save is not None:
        # Construct a separate plain record so save cannot mutate the snapshot.
        save('tube-inputs.json', dict(admit(centres, radius, primal, directions, dimension),
                                     precision=context[0], threads=context[1], **flags))
    try:
        need(admit(centres, radius, primal, directions, dimension) == initial,
             'inputs changed before evaluation')
        need((ctx.prec, ctx.threads) == context, 'arithmetic context changed before evaluation')
        values, derivative = forward(primal, directions)
    finally:
        need(admit(centres, radius, primal, directions, dimension) == initial,
             'primal, directions, or centres mutated across evaluation')
        need((ctx.prec, ctx.threads) == context, 'arithmetic context mutated across evaluation')
    need(isinstance(values, (list, tuple)) and len(values) == dimension,
         'complete signed primal output required')
    output = [native(value) for value in values]
    derivatives = matrix_records(derivative, dimension, 'derivative output')
    row_bounds = [max(abs(q) for q in endpoints(row[0])) for row in derivatives]
    lipschitz = max(row_bounds)
    result = dict(schema='conditional-whole-tube-infinity-operator-v1', **initial,
                  precision=context[0], threads=context[1], full_output=output,
                  signed_derivative=derivatives,
                  row_abs_endpoint_upper=[rational(q) for q in row_bounds],
                  L=rational(lipschitz), **flags,
                  scope='Conditional on authenticated whole-input interval-Jet semantics: for every x in the nominal closed infinity cube and every v in [-1,1]^d, each signed derivative output encloses [DF(x)v]_i. L bounds sup_x ||DF(x)||_infinity. No trajectory error, basin, or certificate is established.')
    if save is not None:
        save('tube-operator.json', result)
    return result
