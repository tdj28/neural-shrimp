"""Conditional exact-real orbit proof gates, independent of any map evaluator.

All supplied scalars/matrices must already be finite Arb enclosures. No float
array, point derivative, or numerical eigenvalue is upgraded to an enclosure.
The evaluator must establish the explicit EnclosureContract below; this module
checks its input/domain bindings and the algebraic sufficient conditions, NOT
whether a transformer implementation discharged that mathematical contract.

For H=F^p-I and X=c+diag(r)[-1,1]^n, strict centered Krawczyk inclusion and a
weighted infinity contraction prove a unique zero throughout X. Divisor-image
separation proves primitive period. A rigorous monodromy power norm below one
proves local attraction, not that all of X lies in the attraction basin.
Context precision/thread settings are caller-owned and must not change mid-call.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path

from flint import arb, arb_mat, ctx


ENCLOSURE_DECLARATION = (
    "The hash-identified exact-real map F is C1 on a neighborhood of the full "
    "box and all required images; Hc encloses F^p(c)-c; JH encloses D(F^p-I)(z) "
    "for every z in the full box; each divisor image encloses F^d of that full "
    "box; M encloses D(F^p) at every zero of H in the box. All evaluator "
    "operations and constants, including elementary functions and roundoff, "
    "are rigorously enclosed. No missing transverse directions are discarded."
)


@dataclass(frozen=True)
class EnclosureContract:
    """Mandatory evaluator assertions/provenance; metadata is not their proof.

    The domain/enclosure digests are computed by helpers below. The other
    digests bind the exact-map manifest, evaluator source and external evidence.
    A caller must independently review/recompute those artifacts. This class
    deliberately has no defaults that silently assert enclosure validity.
    """
    map_sha256: str
    evaluator_source_sha256: str
    evidence_sha256: str
    period: int
    domain_sha256: str
    enclosures_sha256: str
    declaration: str


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _ball(value, name, *, exact=False):
    if not isinstance(value, arb) or not value.is_finite():
        raise ValueError(f'{name} must be a finite Arb enclosure')
    if exact and not value.is_exact():
        raise ValueError(f'{name} must be an exact dyadic Arb point')
    return value


def _vector(values, name, *, n=None, exact=False, positive=False):
    if isinstance(values, arb_mat):
        if values.ncols() != 1:
            raise ValueError(f'{name} must be a column vector')
        values = [values[i, 0] for i in range(values.nrows())]
    if not isinstance(values, (list, tuple)) or not values or n is not None and len(values) != n:
        raise ValueError(f'{name} has invalid vector dimensions')
    result = [_ball(v, f'{name}[{i}]', exact=exact) for i, v in enumerate(values)]
    if positive and any(not v > 0 for v in result):
        raise ValueError(f'{name} entries must be strictly positive')
    return result


def _matrix(value, name, *, n=None, exact=False):
    if not isinstance(value, arb_mat) or value.nrows() < 1 or value.nrows() != value.ncols():
        raise ValueError(f'{name} must be a nonempty square Arb matrix')
    if n is not None and value.nrows() != n:
        raise ValueError(f'{name} dimension mismatch')
    for i in range(value.nrows()):
        for j in range(value.ncols()):
            _ball(value[i, j], f'{name}[{i},{j}]', exact=exact)
    return value


def ball_record(value):
    """Serialize exact dyadic midpoint/radius; no decimal-based proof decisions."""
    _ball(value, 'serialized ball')
    return dict(mid=[int(x) for x in value.mid().man_exp()],
                rad=[int(x) for x in value.rad().man_exp()])


def restore_ball(record):
    if not isinstance(record, dict) or set(record) != {'mid', 'rad'}:
        raise ValueError('invalid ball record')
    for key in ('mid', 'rad'):
        pair = record[key]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(type(x) is not int for x in pair):
            raise ValueError('ball mantissa/exponent must be exact integer pairs')
    if record['rad'][0] < 0:
        raise ValueError('negative radius')
    return arb(arb(tuple(record['mid'])), tuple(record['rad']))


def vector_record(values):
    return [ball_record(v) for v in values]


def matrix_record(matrix):
    return [[ball_record(matrix[i, j]) for j in range(matrix.ncols())]
            for i in range(matrix.nrows())]


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _domain(center, radii):
    c = _vector(center, 'center', exact=True)
    r = _vector(radii, 'radii', n=len(c), exact=True, positive=True)
    # Arb's magnitude radius may round outward. This box encloses the nominal
    # exact c +/- r; strict inclusion is nevertheless tested against nominal r.
    box = [arb(ci, ri) for ci, ri in zip(c, r)]
    return c, r, box


def domain_sha256(center, radii):
    c, r, _ = _domain(center, radii)
    return _digest(dict(center=vector_record(c), radii=vector_record(r)))


def proper_divisors(period):
    p = _integer(period, 'period')
    return [d for d in range(1, p) if p % d == 0]


def enclosures_sha256(h_center, h_jacobian, divisor_images, monodromy):
    """Hash actual enclosure bytes, not merely a producer-supplied label."""
    h = _vector(h_center, 'Hc')
    j = _matrix(h_jacobian, 'JH', n=len(h))
    m = _matrix(monodromy, 'M', n=len(h))
    if not isinstance(divisor_images, dict) or any(type(d) is not int or d < 1 for d in divisor_images):
        raise ValueError('divisor images require positive integer keys')
    images = {str(d): vector_record(_vector(divisor_images[d], f'image {d}', n=len(h)))
              for d in sorted(divisor_images)}
    return _digest(dict(Hc=vector_record(h), JH=matrix_record(j),
                        divisor_images=images, M=matrix_record(m)))


def _eye(n):
    return arb_mat([[int(i == j) for j in range(n)] for i in range(n)])


def _norm_bound(matrix, weights):
    """Outward induced weighted-infinity bound; every returned bound is a point."""
    rows = []
    for i in range(matrix.nrows()):
        total = sum((matrix[i, j].abs_upper()*weights[j]
                     for j in range(matrix.ncols())), arb(0))
        bound = (total/weights[i]).upper()
        _ball(bound, 'row norm upper', exact=True)
        rows.append(bound)
    # Bounds are exact dyadic points, so max is never an ambiguous ball ordering.
    return max(rows), rows


def krawczyk_gate(center, radii, h_center, h_jacobian, preconditioner):
    """Algebraic gate conditional on rigorous H(c), DH(X); no map evaluated.

    R must be an exact dyadic matrix, but need not already have a certified
    inverse: ||I-R DH(X)||_r<1 proves R nonsingular. The region of uniqueness
    is the full nominal box c +/- r, not just the resulting image enclosure.
    """
    c, r, box = _domain(center, radii)
    n = len(c)
    h = _vector(h_center, 'Hc', n=n)
    j = _matrix(h_jacobian, 'JH', n=n)
    R = _matrix(preconditioner, 'R', n=n, exact=True)
    defect = _eye(n)-R*j
    _matrix(defect, 'Krawczyk derivative', n=n)
    q, row_bounds = _norm_bound(defect, r)
    delta = arb_mat(n, 1, [arb(0, ri) for ri in r])
    residual = -R*arb_mat(n, 1, h)
    image = residual+defect*delta
    coordinate = [image[i, 0] for i in range(n)]
    margins = [(r[i]-coordinate[i].abs_upper()).lower() for i in range(n)]
    contraction_margin = (1-q).lower()
    inclusion = [bool(v > 0) for v in margins]
    contract = bool(contraction_margin > 0)
    strict = all(inclusion)
    return dict(gate='krawczyk', passed=bool(strict and contract),
                strict_inclusion=strict, contraction=contract,
                center=vector_record(c), radii=vector_record(r), box_enclosure=vector_record(box),
                Hc=vector_record(h), JH=matrix_record(j), R=matrix_record(R),
                derivative_defect=matrix_record(defect), weighted_row_upper=vector_record(row_bounds),
                contraction_upper=ball_record(q), contraction_margin=ball_record(contraction_margin),
                preconditioned_residual=matrix_record(residual), image_delta=vector_record(coordinate),
                coordinate_inclusions=inclusion, coordinate_margins=vector_record(margins),
                unique_in_full_nominal_box=bool(strict and contract),
                statement='Conditional on the stated full-box enclosure contract; Newton contraction is not dynamical attraction.')


def primitive_gate(center, radii, period, divisor_images):
    """Exclude every proper divisor by strict coordinate separation of F^d(X)."""
    c, r, box = _domain(center, radii)
    divisors = proper_divisors(period)
    if (not isinstance(divisor_images, dict)
            or any(type(d) is not int for d in divisor_images)
            or sorted(divisor_images) != divisors):
        raise ValueError('exactly every proper-divisor image is required; none may be missing or substituted')
    records = []
    for d in divisors:
        image = _vector(divisor_images[d], f'F^{d}(X)', n=len(c))
        differences = [v-b for v, b in zip(image, box)]
        witnesses = []
        for i, difference in enumerate(differences):
            lo, hi = difference.lower(), difference.upper()
            if lo > 0:
                witnesses.append(dict(coordinate=i, direction='image_above_box', gap_lower=ball_record(lo)))
            elif hi < 0:
                witnesses.append(dict(coordinate=i, direction='image_below_box', gap_lower=ball_record(-hi)))
        records.append(dict(divisor=d, passed=bool(witnesses), image=vector_record(image),
                            differences=vector_record(differences), separating_witnesses=witnesses))
    return dict(gate='primitive', passed=all(v['passed'] for v in records), period=int(period),
                required_divisors=divisors, divisors=records,
                statement='Conditional on existence of an F^p fixed point in this box and rigorous full-box divisor images.')


def _matrix_power(matrix, exponent):
    result, factor = _eye(matrix.nrows()), matrix
    k = exponent
    while k:
        if k & 1:
            result = result*factor
        k //= 2
        if k:
            factor = factor*factor
    return result


def attraction_gate(monodromy, *, powers=(1, 2, 4, 8, 16, 32, 64), weights=None):
    """Bound actual monodromy powers by interval matrix products, never spectra."""
    m = _matrix(monodromy, 'monodromy')
    if not isinstance(powers, (list, tuple)) or not powers:
        raise ValueError('nonempty declared power list required')
    exponents = [_integer(p, 'monodromy power') for p in powers]
    if exponents != sorted(set(exponents)):
        raise ValueError('powers must be distinct and increasing')
    w = [arb(1)]*m.nrows() if weights is None else _vector(
        weights, 'stability weights', n=m.nrows(), exact=True, positive=True)
    records, witness = [], None
    for k in exponents:
        power = _matrix_power(m, k)
        _matrix(power, f'monodromy power {k}', n=m.nrows())
        upper, row_bounds = _norm_bound(power, w)
        margin = (1-upper).lower()
        passed = bool(margin > 0)
        if passed and witness is None:
            witness = k
        records.append(dict(power=k, passed=passed, matrix=matrix_record(power),
                            weighted_row_upper=vector_record(row_bounds),
                            induced_infinity_upper=ball_record(upper), margin=ball_record(margin)))
    return dict(gate='attraction', passed=witness is not None, witness_power=witness,
                monodromy=matrix_record(m), weights=vector_record(w), powers=records,
                statement='Conditional on M enclosing the root monodromy; proves local attraction, not an entire-box basin.')


def _check_contract(contract, period, domain_digest, enclosure_digest):
    if not isinstance(contract, EnclosureContract):
        raise ValueError('an explicit EnclosureContract is required; arbitrary matrices are not proof evidence')
    for field in ('map_sha256', 'evaluator_source_sha256', 'evidence_sha256',
                  'domain_sha256', 'enclosures_sha256'):
        value = getattr(contract, field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'contract {field} must be a full lowercase SHA-256')
    if (_integer(contract.period, 'contract period') != period
            or contract.domain_sha256 != domain_digest
            or contract.enclosures_sha256 != enclosure_digest
            or contract.declaration != ENCLOSURE_DECLARATION):
        raise ValueError('enclosure contract does not bind this period, full domain, data and required assertions')


def check_certificate(center, radii, h_center, h_jacobian, preconditioner,
                      divisor_images, monodromy, *, period, contract,
                      powers=(1, 2, 4, 8, 16, 32, 64), stability_weights=None):
    """Return all gates and their conditional conjunction with exact witnesses.

    A True conditional_gates_passed is the algebraic implication under the explicit
    enclosure contract, whose truth must be discharged by the bound evaluator
    and independently checked artifacts. Hash syntax or a producer declaration
    alone cannot establish enclosure validity. This function assigns no rung 3.
    Invalid/missing/nonfinite inputs raise instead of fabricating a verdict.
    """
    p = _integer(period, 'period')
    domain_digest = domain_sha256(center, radii)
    enclosure_digest = enclosures_sha256(h_center, h_jacobian, divisor_images, monodromy)
    _check_contract(contract, p, domain_digest, enclosure_digest)
    root = krawczyk_gate(center, radii, h_center, h_jacobian, preconditioner)
    primitive = primitive_gate(center, radii, p, divisor_images)
    stability = attraction_gate(monodromy, powers=powers, weights=stability_weights)
    passed = root['passed'] and primitive['passed'] and stability['passed']
    return dict(schema='codex-llm-proof-gates-v1', conditional_gates_passed=bool(passed),
                enclosure_contract=asdict(contract), enclosure_validity_checked_by_this_module=False,
                precision=int(ctx.prec), threads=int(ctx.threads),
                gate_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                gates=dict(root=root, primitive=primitive, attraction=stability),
                scope='Conditional algebraic proof of one unique F^p fixed point in the full nominal box, belonging to a primitive attracting exact-real orbit; the enclosure contract must separately be established.',
                not_claimed=['validated evaluator by metadata alone', 'entire-box attraction basin',
                             'parameter interval', 'finite-precision orbit', 'whole shrimp geometry'])
