"""Exact endpoint witnesses for lower-precision whole-phase input boxes.

Only packed dyadics and Arb arithmetic are admitted. This module neither loads
nor evaluates a model and does not prove a derivative enclosure. Consumers must
use the returned decoded balls (or rigorously checked larger balls), never their
midpoints alone. Serialized magnitudes may widen on reconstruction.
"""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
from pathlib import Path

import flint
from flint import arb, ctx

SCHEMA = 'codex-mixed-precision-phase-domain-v1'


def canonical_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _positive_integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f'{name} must be an integer at least {minimum}')
    return value


def _fraction(pair):
    if not isinstance(pair, list) or len(pair) != 2 or any(type(v) is not int for v in pair):
        raise ValueError('integer mantissa/exponent pair required; no floats or booleans')
    mantissa, exponent = pair
    return Fraction(mantissa << exponent, 1) if exponent >= 0 else Fraction(mantissa, 1 << -exponent)


def _pair(value):
    """Canonical exact dyadic representation, including exact rational margins."""
    value = Fraction(value)
    denominator = value.denominator
    if denominator & (denominator-1):
        raise ValueError('a non-dyadic rational cannot be encoded as a dyadic')
    mantissa, exponent = value.numerator, -(denominator.bit_length()-1)
    if mantissa == 0:
        return [0, 0]
    while mantissa % 2 == 0:
        mantissa //= 2
        exponent += 1
    return [mantissa, exponent]


def _record_parts(record, *, exact=False):
    if not isinstance(record, dict) or set(record) != {'mid', 'rad'}:
        raise ValueError('packed Arb midpoint/radius record required')
    midpoint, radius = _fraction(record['mid']), _fraction(record['rad'])
    if radius < 0 or exact and radius != 0:
        raise ValueError('nonnegative radius required; nominal data must be exact points')
    return midpoint, radius


def pack_ball(value):
    if not isinstance(value, arb) or not value.is_finite():
        raise ValueError('finite Arb ball required')
    return dict(mid=[int(v) for v in value.mid().man_exp()],
                rad=[int(v) for v in value.rad().man_exp()])


def _nominal_domain(centers, radii, period, dimension):
    _positive_integer(period, 'period'); _positive_integer(dimension, 'dimension')
    if not isinstance(centers, list) or len(centers) != period:
        raise ValueError('exact expected number of phases required')
    if not isinstance(radii, list) or len(radii) != period:
        raise ValueError('one exact positive nominal radius per phase required')
    points, phase_radii = [], []
    for index, row in enumerate(centers):
        if not isinstance(row, list) or len(row) != dimension:
            raise ValueError('every full state coordinate must be retained')
        points.append([_record_parts(v, exact=True)[0] for v in row])
        radius = _record_parts(radii[index], exact=True)[0]
        if radius <= 0:
            raise ValueError('nominal phase radius must be strictly positive')
        phase_radii.append(radius)
    nominal = dict(period=period, dimension=dimension, centers=centers, radii=radii)
    return points, phase_radii, canonical_hash(nominal)


def _decode(record):
    _record_parts(record)
    value = arb(tuple(record['mid']), tuple(record['rad']))
    if not value.is_finite():
        raise ValueError('decoded input ball is not finite')
    # This verifies constructor non-narrowing without expecting equal radii.
    old_mid, old_radius = _record_parts(record)
    new_mid, new_radius = _record_parts(pack_ball(value))
    if new_mid-new_radius > old_mid-old_radius or new_mid+new_radius < old_mid+old_radius:
        raise ValueError('decoding narrowed the serialized enclosure')
    return value


def containment_witnesses(centers, radii, actual_balls, *, period, dimension):
    """Check any actual receiver balls with exact rational endpoint comparisons.

    This also admits conservatively wider balls from additional reconstruction.
    It returns their actual records and a hash; it does not identify them with
    an earlier encoded radius or discard centre-rounding error.
    """
    points, nominal_radii, domain_hash = _nominal_domain(centers, radii, period, dimension)
    if not isinstance(actual_balls, list) or len(actual_balls) != period:
        raise ValueError('one actual input box per phase required')
    actual_records, witnesses = [], []
    for phase, balls in enumerate(actual_balls):
        if not isinstance(balls, list) or len(balls) != dimension:
            raise ValueError('full-dimensional actual input boxes required')
        records, phase_witnesses = [], []
        for coordinate, ball in enumerate(balls):
            record = pack_ball(ball)
            midpoint, actual_radius = _record_parts(record)
            center, nominal_radius = points[phase][coordinate], nominal_radii[phase]
            nominal_lower, nominal_upper = center-nominal_radius, center+nominal_radius
            actual_lower, actual_upper = midpoint-actual_radius, midpoint+actual_radius
            center_error = abs(center-midpoint)
            lower_margin, upper_margin = nominal_lower-actual_lower, actual_upper-nominal_upper
            radius_margin = actual_radius-nominal_radius-center_error
            if lower_margin < 0 or upper_margin < 0 or radius_margin < 0:
                raise ValueError(f'actual phase {phase} coordinate {coordinate} loses part of the nominal box')
            records.append(record)
            phase_witnesses.append(dict(
                phase=phase, coordinate=coordinate,
                nominal_lower=_pair(nominal_lower), nominal_upper=_pair(nominal_upper),
                actual_lower=_pair(actual_lower), actual_upper=_pair(actual_upper),
                center_rounding_error=_pair(center_error), nominal_radius=_pair(nominal_radius),
                actual_radius=_pair(actual_radius), lower_margin=_pair(lower_margin),
                upper_margin=_pair(upper_margin), radius_margin=_pair(radius_margin)))
        actual_records.append(records); witnesses.append(phase_witnesses)
    return dict(nominal_domain_sha256=domain_hash, actual_balls=actual_records,
                actual_balls_sha256=canonical_hash(actual_records), witnesses=witnesses,
                witnesses_sha256=canonical_hash(witnesses), full_nominal_domain_contained=True)


def build_phase_domains(centers, radii, *, period, dimension, target_precision):
    """Construct one lower-precision input box per exact nominal phase box.

    Centers/radii are saved packed exact dyadics. A phase has one scalar radius
    for all its coordinates, matching the shooting gate's nominal domain.
    Explicit shape arguments prevent dropped coordinates/phases being accepted.
    """
    _positive_integer(target_precision, 'target precision', 16)
    points, nominal_radii, domain_hash = _nominal_domain(centers, radii, period, dimension)
    encoded, decoded = [], []
    with ctx.workprec(target_precision):
        for phase, row in enumerate(points):
            encoded_row, decoded_row = [], []
            for center in row:
                # Adding zero forces an outward arithmetic rounding at the
                # target precision; its midpoint is a proposed input centre.
                rounded = (arb(tuple(_pair(center)))+arb(0)).mid()
                rounded_center = _record_parts(pack_ball(rounded), exact=True)[0]
                required_radius = nominal_radii[phase]+abs(center-rounded_center)
                radius_upper = arb(tuple(_pair(required_radius))).upper()
                encoded_ball = pack_ball(arb(rounded, radius_upper))
                actual = _decode(encoded_ball)
                encoded_row.append(encoded_ball); decoded_row.append(actual)
            encoded.append(encoded_row); decoded.append(decoded_row)
        checked = containment_witnesses(centers, radii, decoded, period=period, dimension=dimension)
    body = dict(schema=SCHEMA, period=period, dimension=dimension,
                target_precision=target_precision, nominal_domain_sha256=domain_hash,
                nominal_centers=json.loads(json.dumps(centers)), nominal_radii=json.loads(json.dumps(radii)),
                encoded_balls=encoded, encoded_balls_sha256=canonical_hash(encoded),
                actual_decoded_balls=checked['actual_balls'], actual_balls_sha256=checked['actual_balls_sha256'],
                containment_witnesses=checked['witnesses'], witnesses_sha256=checked['witnesses_sha256'],
                full_nominal_domain_contained=True,
                versions=dict(python_flint=flint.__version__, flint=flint.__FLINT_VERSION__),
                helper_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                scope='Exact endpoint containment only; no map or whole-box derivative evaluation.')
    return dict(**body, artifact_sha256=canonical_hash(body))


def decode_verified_domains(artifact, centers, radii, *, period, dimension,
                            target_precision, expected_artifact_sha256):
    """Return the actual balls to evaluate, after hash and endpoint verification.

    The expected artifact hash must come from an independent committed plan or
    receipt, not be guessed from a potentially modified artifact. Any subsequent
    reconstruction must be rechecked with containment_witnesses and its actual
    balls/hash bound to the derivative calculation.
    """
    if not isinstance(artifact, dict) or 'artifact_sha256' not in artifact:
        raise ValueError('complete domain artifact required')
    body = {k:v for k,v in artifact.items() if k != 'artifact_sha256'}
    if (not isinstance(expected_artifact_sha256, str) or len(expected_artifact_sha256) != 64
            or artifact['artifact_sha256'] != expected_artifact_sha256
            or canonical_hash(body) != expected_artifact_sha256):
        raise ValueError('domain artifact hash mismatch')
    _positive_integer(target_precision, 'target precision', 16)
    _, _, domain_hash = _nominal_domain(centers, radii, period, dimension)
    if (artifact['schema'] != SCHEMA or artifact['period'] != period or artifact['dimension'] != dimension
            or artifact['target_precision'] != target_precision
            or artifact['nominal_domain_sha256'] != domain_hash
            or artifact['nominal_centers'] != centers or artifact['nominal_radii'] != radii
            or artifact['full_nominal_domain_contained'] is not True):
        raise ValueError('domain artifact does not bind the complete requested nominal domain')
    if artifact['versions'] != dict(python_flint=flint.__version__, flint=flint.__FLINT_VERSION__):
        raise ValueError('different Arb backend; rebuild and rebind the actual domain')
    encoded = artifact['encoded_balls']
    if (not isinstance(encoded, list) or len(encoded) != period
            or any(not isinstance(row, list) or len(row) != dimension for row in encoded)
            or canonical_hash(encoded) != artifact['encoded_balls_sha256']):
        raise ValueError('encoded input boxes have missing coordinates or a wrong binding')
    with ctx.workprec(target_precision):
        actual = [[_decode(v) for v in row] for row in encoded]
        checked = containment_witnesses(centers, radii, actual, period=period, dimension=dimension)
    if (checked['actual_balls'] != artifact['actual_decoded_balls']
            or checked['actual_balls_sha256'] != artifact['actual_balls_sha256']
            or checked['witnesses'] != artifact['containment_witnesses']
            or checked['witnesses_sha256'] != artifact['witnesses_sha256']):
        raise ValueError('actual decoded balls or exact containment witnesses changed')
    return actual, checked
