"""Conditional multiple-shooting root gates with streamed n-square blocks.

No map is evaluated here. The supplied whole-phase-box derivative enclosures
must be established by an independently checked evaluator. Metadata binds that
obligation but does not prove it. No (period*dimension)-square matrix is formed.
Context precision and inputs must remain unchanged throughout each call.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from flint import arb, arb_mat, ctx
from proof_gates import (ball_record, vector_record, proper_divisors,
                         _ball, _vector, _matrix, _eye, _integer)

SHOOTING_DECLARATION = (
    "The hash-identified exact-real map F is C1 on a neighborhood of every "
    "full nominal phase box X_i=c_i+[-r_i,r_i]^n. Each image_i encloses F(c_i), "
    "and each J_i encloses DF(z) for every z in X_i, with all coordinates "
    "and transverse directions retained. All evaluator operations, constants, "
    "elementary functions and roundoff are rigorously enclosed. The exact "
    "dyadic A_i and B are only preconditioner proposals, not asserted inverses."
)


@dataclass(frozen=True)
class ShootingEnclosureContract:
    map_sha256: str
    evaluator_source_sha256: str
    evidence_sha256: str
    period: int
    domain_sha256: str
    enclosures_sha256: str
    proposals_sha256: str
    declaration: str


def _domain(centers, radii):
    if not isinstance(centers, (list, tuple)) or len(centers) < 2:
        raise ValueError('at least two phase centers are required')
    c = [_vector(centers[0], 'center 0', exact=True)]
    n = len(c[0])
    c += [_vector(v, f'center {i}', n=n, exact=True)
          for i, v in enumerate(centers[1:], 1)]
    r = _vector(radii, 'one radius per phase', n=len(c), exact=True, positive=True)
    return c, r, n


def _phases(values, p, name, *, n, matrix=False, exact=False):
    if not isinstance(values, (tuple, list)) or len(values) != p:
        raise ValueError(f'{name} requires one entry per phase')
    validate = _matrix if matrix else _vector
    return [validate(v, f'{name} {i}', n=n, exact=exact) for i,v in enumerate(values)]


def _feed(digest, value):
    digest.update(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode())
    digest.update(b'\n')


def _hash_vector(digest, label, values):
    _feed(digest, [label, 'vector', len(values)])
    for value in values:
        _feed(digest, ball_record(value))


def _hash_matrix(digest, label, matrix):
    _feed(digest, [label, 'matrix', matrix.nrows(), matrix.ncols()])
    for i in range(matrix.nrows()):
        for j in range(matrix.ncols()):
            _feed(digest, ball_record(matrix[i,j]))


def _matrix_sha256(matrix):
    digest = hashlib.sha256()
    _hash_matrix(digest, 'matrix', matrix)
    return digest.hexdigest()


def _domain_digest(c, r):
    digest = hashlib.sha256()
    _feed(digest, ['codex-shooting-domain-v1', len(c), len(c[0])])
    for i, center in enumerate(c):
        _hash_vector(digest, f'center {i}', center)
    _hash_vector(digest, 'nominal phase radii', r)
    return digest.hexdigest()


def domain_sha256(centers, radii):
    c, r, _ = _domain(centers, radii)
    return _domain_digest(c, r)


def _enclosure_digest(images, jacobians):
    digest = hashlib.sha256()
    _feed(digest, ['codex-shooting-enclosures-v1', len(images), len(images[0])])
    for i, (image, jacobian) in enumerate(zip(images, jacobians)):
        _hash_vector(digest, f'image {i}', image)
        _hash_matrix(digest, f'whole-phase-box Jacobian {i}', jacobian)
    return digest.hexdigest()


def enclosures_sha256(images, jacobians):
    if not isinstance(images, (list, tuple)) or len(images) < 2:
        raise ValueError('at least two image vectors required')
    first = _vector(images[0], 'image 0')
    v = _phases(images, len(images), 'images', n=len(first))
    j = _phases(jacobians, len(v), 'Jacobians', n=len(first), matrix=True)
    return _enclosure_digest(v, j)


def _proposal_digest(A, B):
    digest = hashlib.sha256()
    _feed(digest, ['codex-shooting-proposals-v1', len(A), B.nrows()])
    for i, matrix in enumerate(A):
        _hash_matrix(digest, f'A {i}', matrix)
    _hash_matrix(digest, 'B', B)
    return digest.hexdigest()


def proposals_sha256(derivatives, inverse_cycle):
    B = _matrix(inverse_cycle, 'B', exact=True)
    if not isinstance(derivatives, (list, tuple)) or len(derivatives) < 2:
        raise ValueError('at least two derivative proposals required')
    A = _phases(derivatives, len(derivatives), 'A', n=B.nrows(), matrix=True, exact=True)
    return _proposal_digest(A, B)


def _check_contract(contract, period, domain, enclosures, proposals):
    if not isinstance(contract, ShootingEnclosureContract):
        raise ValueError('explicit ShootingEnclosureContract required; metadata is not proof')
    for field in ('map_sha256', 'evaluator_source_sha256', 'evidence_sha256',
                  'domain_sha256', 'enclosures_sha256', 'proposals_sha256'):
        value = getattr(contract, field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'{field} must be a lowercase SHA-256')
    if (_integer(contract.period, 'period', 2) != period
            or contract.domain_sha256 != domain
            or contract.enclosures_sha256 != enclosures
            or contract.proposals_sha256 != proposals
            or contract.declaration != SHOOTING_DECLARATION):
        raise ValueError('contract does not bind this period/domain/enclosures/proposals/declaration')


def _row_norms(matrix):
    rows = []
    for i in range(matrix.nrows()):
        upper = sum((matrix[i,j].abs_upper() for j in range(matrix.ncols())), arb(0)).upper()
        _ball(upper, 'matrix row bound', exact=True)
        rows.append(upper)
    return max(rows), rows


def _defect_blocks(A, B, jacobians):
    """Yield (i,j,K_ij), only O(n^2) internal temporaries besides inputs.

    Inputs are validated by the public gate. Every product encloses the exact
    product of the original dyadic proposals; no rounded midpoint is reused as
    an exact product. This generator is also useful for tiny dense cross-checks.
    """
    p, n = len(A), B.nrows()
    identity = _eye(n)
    product = identity
    for matrix in A:
        product = matrix*product
    D = identity-B*(identity-product)
    _matrix(D, 'cyclic inverse defect', n=n)
    suffix = identity
    for j in range(p-1, -1, -1):
        R = -B*suffix
        delta = jacobians[j]-A[j]
        K = D if j == 0 else None
        for i in range(p):
            defect = -R*delta
            if j == 0:
                defect = K+defect
            _matrix(defect, f'defect block {i},{j}', n=n)
            yield i, j, defect
            if i+1 < p:
                R = A[i]*R
                if i == j:
                    R = R-identity
                if j == 0:
                    K = A[i]*K
        if j:
            suffix = suffix*A[j]



def _factored_bounds(A, B, jacobians):
    """Yield conservative scalar block bounds, without constructing R or K.

    Signed P_i, Q_j and P_i D are formed before taking their norms. Remaining
    products use submultiplicativity. This can be much looser than the default
    signed-block mode, but avoids two dense products for every block pair.
    No matrix enclosure/hash is fabricated for a scalar factored bound.
    """
    p, n = len(A), B.nrows()
    identity = _eye(n)
    prefixes = [identity]
    for matrix in A:
        prefixes.append(matrix*prefixes[-1])
    D = identity-B*(identity-prefixes[-1])
    _matrix(D, 'cyclic inverse defect', n=n)
    prefix_norms = [_row_norms(matrix)[0] for matrix in prefixes[:-1]]
    defect_norms = [_row_norms(matrix*D)[0] for matrix in prefixes[:-1]]
    matrix_norms = [_row_norms(matrix)[0] for matrix in A]
    b_norm = _row_norms(B)[0]
    suffix = identity
    for j in range(p-1, -1, -1):
        suffix_norm = _row_norms(suffix)[0]
        delta_norm = _row_norms(jacobians[j]-A[j])[0]
        for i in range(p):
            segment = arb(0)
            if j < i:
                segment = arb(1)
                for k in range(j+1, i):
                    segment = (segment*matrix_norms[k]).upper()
            product_term = (prefix_norms[i]*b_norm*suffix_norm).upper()
            r_bound = (product_term+segment).upper()
            inverse_defect_term = defect_norms[i] if j == 0 else arb(0)
            perturbation_term = (r_bound*delta_norm).upper()
            bound = (inverse_defect_term+perturbation_term).upper()
            _ball(bound, 'factored block norm upper', exact=True)
            yield i, j, bound, dict(
                prefix_norm_upper=ball_record(prefix_norms[i]),
                inverse_proposal_norm_upper=ball_record(b_norm),
                suffix_norm_upper=ball_record(suffix_norm),
                remainder_segment_norm_product_upper=ball_record(segment),
                preconditioner_block_norm_upper=ball_record(r_bound),
                jacobian_difference_norm_upper=ball_record(delta_norm),
                inverse_defect_term_upper=ball_record(inverse_defect_term),
                jacobian_perturbation_term_upper=ball_record(perturbation_term))
        if j:
            suffix = suffix*A[j]


def _residual_correction(c, images, A, B):
    p, n = len(c), len(c[0])
    h = [arb_mat(n,1,[images[i][a]-c[(i+1)%p][a] for a in range(n)])
         for i in range(p)]
    s = arb_mat(n,1)
    for matrix, residual in zip(A,h):
        s = matrix*s+residual
    correction = B*s
    values = []
    for i in range(p):
        value = _vector([correction[a,0] for a in range(n)], f'correction {i}', n=n)
        values.append(value)
        if i+1 < p:
            correction = A[i]*correction+h[i]
    return h, values


def shooting_root_gate(centers, radii, images, jacobians, derivatives, inverse_cycle, *,
                       contract, bound_method='streamed'):
    """Strict contraction/inclusion for H_i=F(x_i)-x_(i+1) on product boxes.

    q_i=sum_j ||K_ij||_inf*r_j/r_i is conservative: block row maxima are
    taken before summation. The phase residual bound likewise takes a maximum
    over all coordinates. The exact nominal radii, not Arb's rounded box
    storage radii, define every strict comparison. No attraction is tested.
    bound_method='factored' uses conservative scalar product bounds; 'streamed'
    (default) retains each signed block product and its exact enclosure hash.
    """
    if bound_method not in ('streamed', 'factored'):
        raise ValueError('bound_method must be streamed or factored')
    c, r, n = _domain(centers, radii)
    p = len(c)
    v = _phases(images, p, 'images', n=n)
    J = _phases(jacobians, p, 'whole-phase-box Jacobians', n=n, matrix=True)
    A = _phases(derivatives, p, 'A', n=n, matrix=True, exact=True)
    B = _matrix(inverse_cycle, 'B', n=n, exact=True)
    domain = _domain_digest(c, r)
    enclosures = _enclosure_digest(v, J)
    proposals = _proposal_digest(A, B)
    _check_contract(contract, p, domain, enclosures, proposals)
    h, corrections = _residual_correction(c, v, A, B)
    sums = [arb(0) for _ in range(p)]
    blocks = []
    if bound_method == 'streamed':
        for i, j, block in _defect_blocks(A, B, J):
            bound, rows = _row_norms(block)
            contribution = (bound*r[j]).upper()
            sums[i] = (sums[i]+contribution).upper()
            blocks.append(dict(phase_row=i, phase_column=j, bound_method='streamed',
                               matrix_sha256=_matrix_sha256(block), row_norm_upper=vector_record(rows),
                               induced_infinity_upper=ball_record(bound),
                               radius_weighted_upper=ball_record(contribution)))
    else:
        for i, j, bound, factors in _factored_bounds(A, B, J):
            contribution = (bound*r[j]).upper()
            sums[i] = (sums[i]+contribution).upper()
            blocks.append(dict(phase_row=i, phase_column=j, bound_method='factored',
                               induced_infinity_upper=ball_record(bound), factors=factors,
                               radius_weighted_upper=ball_record(contribution)))
    phase_records = []
    for i in range(p):
        residual_upper = max(v.abs_upper() for v in corrections[i])
        q = (sums[i]/r[i]).upper()
        image_upper = (residual_upper+sums[i]).upper()
        inclusion_margin = (r[i]-image_upper).lower()
        contraction_margin = (1-q).lower()
        for bound in (residual_upper,q,image_upper,inclusion_margin,contraction_margin):
            _ball(bound, 'phase bound', exact=True)
        phase_records.append(dict(phase=i, radius=ball_record(r[i]),
            preconditioned_residual=vector_record(corrections[i]),
            preconditioned_residual_inf_upper=ball_record(residual_upper),
            derivative_image_inf_upper=ball_record(sums[i]),
            weighted_contraction_upper=ball_record(q), contraction_margin=ball_record(contraction_margin),
            image_delta_inf_upper=ball_record(image_upper), inclusion_margin=ball_record(inclusion_margin),
            strict_inclusion=bool(inclusion_margin>0), contraction=bool(contraction_margin>0)))
    passed = all(v['strict_inclusion'] and v['contraction'] for v in phase_records)
    max_q = max((sums[i]/r[i]).upper() for i in range(p))
    return dict(schema='codex-conditional-shooting-root-v1', gate='shooting_root',
        conditional_gates_passed=bool(passed), enclosure_validity_checked_by_this_module=False,
        period=p, dimension=n, precision_bits=ctx.prec, bound_method=bound_method,
        gate_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        contract=asdict(contract), domain_sha256=domain, enclosures_sha256=enclosures,
        proposals_sha256=proposals, centers=[vector_record(v) for v in c],
        nominal_phase_radii=vector_record(r), residuals=[vector_record([v[a,0] for a in range(n)]) for v in h],
        block_bounds=blocks, phases=phase_records, contraction_upper=ball_record(max_q),
        unique_root_in_full_product_box=bool(passed), attraction_checked=False,
        statement='Conditional on the bound evaluator contract: a unique root tuple of the cyclic shooting equations in the full nominal product box. This does not exclude F^p fixed points in X_0 whose intermediate phases leave the other boxes, and proves no attraction. Metadata does not establish enclosure validity.')


def primitive_phase_gate(centers, radii):
    """Separate phase boxes at every proper divisor, conditional on a root.

    Once a shooting root exists, its x_d=F^d(x_0) lies in X_d. These boxes
    are NOT asserted to enclose F^d(X_0), and no map is evaluated here.
    """
    c, r, n = _domain(centers, radii)
    p = len(c)
    records = []
    for d in proper_divisors(p):
        gaps = []
        witnesses = []
        for a in range(n):
            gap = ((c[d][a]-c[0][a]).abs_lower()-r[0]-r[d]).lower()
            _ball(gap, 'separation margin', exact=True)
            gaps.append(gap)
            if gap > 0:
                witnesses.append(dict(coordinate=a, gap_lower=ball_record(gap)))
        records.append(dict(divisor=d, phase_separated=bool(witnesses),
                            separation_margins=vector_record(gaps), witnesses=witnesses))
    return dict(gate='shooting_phase_primitive', period=p, domain_sha256=_domain_digest(c,r),
        conditional_primitive_period=all(v['phase_separated'] for v in records),
        root_existence_checked=False, requires_shooting_root_existence=True,
        required_divisors=proper_divisors(p), divisors=records,
        statement='Conditional on existence of a shooting root in this same product box. Separation excludes every proper divisor for that root; X_d is not claimed to enclose F^d(X_0).')
