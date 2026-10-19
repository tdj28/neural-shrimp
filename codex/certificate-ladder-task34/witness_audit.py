"""Independent exact-rational audit of saved conditional gate witnesses.

No Arb, model, native matrix products, artifact reader or certificate finalizer.
This rechecks scalar inequalities, separation and complete saved matrix row
norms. The validity of producer matrix products and enclosure inputs remains
an explicit independently reviewed source/execution premise.
"""
from fractions import Fraction
import hashlib
import json


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_value(value):
    require(type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'lowercase SHA256 required')
    return value


def integer(value):
    require(type(value) is int, 'exact integer required')
    return value


def dyadic(pair):
    require(type(pair) is list and len(pair) == 2, 'dyadic pair required')
    n, e = map(integer, pair)
    return Fraction(n * (1 << e), 1) if e >= 0 else Fraction(n, 1 << -e)


def endpoints(record):
    require(type(record) is dict and set(record) == {'mid', 'rad'}, 'complete ball record required')
    m, r = dyadic(record['mid']), dyadic(record['rad'])
    require(r >= 0, 'negative interval radius')
    return m-r, m+r


def point(record, *, nonnegative=False):
    lo, hi = endpoints(record)
    require(lo == hi, 'exact point bound required')
    require(not nonnegative or lo >= 0, 'negative norm bound')
    return lo


def absolute(record):
    lo, hi = endpoints(record)
    return max(abs(lo), abs(hi))


def vector(records, n):
    require(type(records) is list and len(records) == n, 'complete vector required')
    for record in records:
        endpoints(record)
    return records


def matrix(records, n):
    require(type(records) is list and len(records) == n, 'complete square matrix required')
    for row in records:
        vector(row, n)
    return records


def feed(digest, value):
    digest.update(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()+b'\n')


def matrix_digest(records, n):
    matrix(records, n)
    digest=hashlib.sha256();feed(digest, ['matrix', 'matrix', n, n])
    for row in records:
        for v in row:
            feed(digest, v)
    return digest.hexdigest()


def domain_digest(centers, radii, p, n):
    require(type(centers) is list and len(centers) == p, 'all external centers required')
    vector(radii, p)
    digest=hashlib.sha256();feed(digest, ['codex-shooting-domain-v1', p, n])
    for i, row in enumerate(centers):
        vector(row, n);feed(digest, [f'center {i}', 'vector', n])
        for v in row:
            point(v);feed(digest, v)
    feed(digest, ['nominal phase radii', 'vector', p])
    for r in radii:
        require(point(r) > 0, 'strictly positive nominal radius required');feed(digest, r)
    return digest.hexdigest()


def audit(result, *, centers, radii, images, contract, jacobian_hashes):
    """External exact centers/images/contract/J hashes must come from admission.

    A passing audit is consistent witness arithmetic, not a completed proof.
    Valid failed sufficient-condition gates are retained as failed.
    """
    p, n=integer(result['period']), integer(result['dimension'])
    require(p >= 2 and n >= 1, 'positive dimensions and nontrivial period required')
    require(result['schema'] == 'codex-conditional-shooting-certificate-v1', 'conditional result schema')
    require(result['certificate_passed'] is False, 'producer must remain conditional')
    require(type(contract) is dict and result['contract'] == contract, 'external gate contract differs')
    require(integer(contract['period']) == p, 'external contract period differs')
    dh=domain_digest(centers, radii, p, n)
    require(type(images) is list and len(images) == p, 'all external point images required')
    for row in images: vector(row,n)
    require(result['domain_sha256'] == contract['domain_sha256'] == dh, 'external domain identity differs')
    for field in ('enclosures_sha256','proposals_sha256'):
        require(result[field] == contract[field], 'external input identity differs')
    r=[point(v) for v in radii]
    root=result['root']
    require(integer(root['period']) == p and integer(root['dimension']) == n and root['bound_method'] == 'factored', 'frozen factored root scope')
    require(root['contract'] == contract and root['centers'] == centers and root['nominal_phase_radii'] == radii, 'root external data differs')
    for field in ('domain_sha256','enclosures_sha256','proposals_sha256'):
        require(root[field] == result[field], 'root identity differs')
    require(len(root['residuals']) == p, 'all residual phases required')
    for i in range(p):
        for a, saved in enumerate(vector(root['residuals'][i], n)):
            lo,hi=endpoints(saved);il,iu=endpoints(images[i][a]);c=point(centers[(i+1)%p][a])
            require(lo <= il-c and hi >= iu-c, 'residual narrows external image-minus-center')
    require(type(root['block_bounds']) is list and len(root['block_bounds']) == p*p, 'complete block grid required')
    sums=[Fraction(0)]*p;seen=set()
    for b in root['block_bounds']:
        i,j=integer(b['phase_row']),integer(b['phase_column'])
        require(0 <= i < p and 0 <= j < p and (i,j) not in seen, 'duplicate or invalid block')
        seen.add((i,j));require(b['bound_method'] == 'factored', 'factored block required')
        f={key:point(v, nonnegative=True) for key,v in b['factors'].items()}
        require(set(f) == {'prefix_norm_upper','inverse_proposal_norm_upper','suffix_norm_upper',
            'remainder_segment_norm_product_upper','preconditioner_block_norm_upper',
            'jacobian_difference_norm_upper','inverse_defect_term_upper','jacobian_perturbation_term_upper'}, 'complete factor witnesses required')
        if i == 0: require(f['prefix_norm_upper'] >= 1, 'identity prefix norm understated')
        if j == p-1: require(f['suffix_norm_upper'] >= 1, 'identity suffix norm understated')
        if j >= i: require(f['remainder_segment_norm_product_upper'] == 0, 'noncausal segment must vanish')
        if j == i-1: require(f['remainder_segment_norm_product_upper'] >= 1, 'empty segment norm bound')
        if j != 0: require(f['inverse_defect_term_upper'] == 0, 'inverse defect only in first block column')
        pre=f['prefix_norm_upper']*f['inverse_proposal_norm_upper']*f['suffix_norm_upper']+f['remainder_segment_norm_product_upper']
        require(f['preconditioner_block_norm_upper'] >= pre, 'preconditioner scalar bound understated')
        perturb=f['preconditioner_block_norm_upper']*f['jacobian_difference_norm_upper']
        require(f['jacobian_perturbation_term_upper'] >= perturb, 'perturbation scalar bound understated')
        bound=point(b['induced_infinity_upper'], nonnegative=True)
        require(bound >= f['inverse_defect_term_upper']+f['jacobian_perturbation_term_upper'], 'block norm sum understated')
        contribution=point(b['radius_weighted_upper'], nonnegative=True)
        require(contribution >= bound*r[j], 'weighted block bound understated');sums[i]+=contribution
    require(type(root['phases']) is list and len(root['phases']) == p, 'all strict gate phases required')
    phase_flags=[];contractions=[]
    for i, row in enumerate(root['phases']):
        require(type(row['phase']) is int and row['phase'] == i and row['radius'] == radii[i], 'phase identity differs')
        corr=max(absolute(v) for v in vector(row['preconditioned_residual'], n))
        residual=point(row['preconditioned_residual_inf_upper'], nonnegative=True)
        total=point(row['derivative_image_inf_upper'], nonnegative=True)
        q=point(row['weighted_contraction_upper'], nonnegative=True)
        image=point(row['image_delta_inf_upper'], nonnegative=True)
        margin=point(row['inclusion_margin']);cmargin=point(row['contraction_margin'])
        require(residual >= corr and total >= sums[i], 'phase upper bound understated')
        require(q >= total/r[i] and image >= residual+total, 'phase recombination understated')
        require(margin <= r[i]-image and cmargin <= 1-q, 'strict margin overstated')
        require(row['strict_inclusion'] is (margin > 0) and row['contraction'] is (cmargin > 0), 'phase boolean disagrees with exact inequality')
        phase_flags.append(margin > 0 and cmargin > 0);contractions.append(q)
    root_pass=all(phase_flags)
    require(root['conditional_gates_passed'] is root_pass and root['unique_root_in_full_product_box'] is root_pass, 'root aggregate differs')
    require(point(root['contraction_upper']) >= max(contractions), 'aggregate contraction bound understated')
    primitive=result['primitive'];divisors=[d for d in range(1,p) if p%d == 0]
    require(primitive['period'] == p and primitive['domain_sha256'] == dh and primitive['required_divisors'] == divisors, 'primitive domain or divisor coverage differs')
    require(type(primitive['divisors']) is list and len(primitive['divisors']) == len(divisors), 'all proper divisors required')
    primitive_flags=[]
    for d, row in zip(divisors, primitive['divisors']):
        require(type(row['divisor']) is int and row['divisor'] == d, 'primitive divisor identity differs')
        margins=[point(v) for v in vector(row['separation_margins'],n)]
        for a,margin in enumerate(margins):
            gap=abs(point(centers[d][a])-point(centers[0][a]))-r[0]-r[d]
            require(margin <= gap, 'primitive gap lower bound overstated')
        witnesses=[{'coordinate':a,'gap_lower':row['separation_margins'][a]} for a,v in enumerate(margins) if v > 0]
        require(row['witnesses'] == witnesses and row['phase_separated'] is bool(witnesses), 'primitive witnesses disagree')
        primitive_flags.append(bool(witnesses))
    primitive_pass=all(primitive_flags)
    require(primitive['conditional_primitive_period'] is primitive_pass, 'primitive aggregate differs')
    mon=result['monodromy'];attr=result['attraction']
    require(mon['order'] == 'J_(p-1) ... J_1 J_0', 'monodromy order differs')
    require(type(jacobian_hashes) is list and len(jacobian_hashes) == p, 'external full Jacobian identities required')
    chain=mon['phase_product_witnesses'];require(type(chain) is list and len(chain) == p, 'complete product chain required')
    for i,row in enumerate(chain):
        digest_value(jacobian_hashes[i]);digest_value(row['jacobian_sha256']);digest_value(row['partial_product_sha256'])
        require(type(row['phase']) is int and row['phase'] == i and row['jacobian_sha256'] == jacobian_hashes[i], 'product phase/input identity differs')
    m=matrix(attr['monodromy'],n)
    require(matrix_digest(m,n) == mon['matrix_sha256'] == chain[-1]['partial_product_sha256'], 'saved monodromy digest differs')
    weights=[point(v) for v in vector(attr['weights'],n)]
    require(all(v == 1 for v in weights), 'frozen unweighted attraction required')
    require(type(attr['powers']) is list and [row['power'] for row in attr['powers']] == [1,2,4,8], 'declared powers one, two, four and eight required')
    power_flags=[];matrix_entries=0
    for row in attr['powers']:
        k=integer(row['power']);mat=matrix(row['matrix'],n)
        saved=[point(v, nonnegative=True) for v in vector(row['weighted_row_upper'],n)]
        for i,entries in enumerate(mat):
            exact=sum((absolute(v) for v in entries),Fraction(0));matrix_entries+=n
            require(saved[i] >= exact, 'saved matrix row norm understated')
            if k == 1:
                for j,v in enumerate(entries):
                    lo,hi=endpoints(v);ml,mh=endpoints(m[i][j]);require(lo <= ml and hi >= mh, 'first power narrows monodromy')
        upper=point(row['induced_infinity_upper'], nonnegative=True);margin=point(row['margin'])
        require(upper >= max(saved) and margin <= 1-upper, 'attraction bound or margin understated')
        require(row['passed'] is (margin > 0), 'attraction boolean differs');power_flags.append(margin > 0)
    attraction_pass=any(power_flags);witness=next((k for k,ok in zip((1,2,4,8),power_flags) if ok),None)
    require(attr['passed'] is attraction_pass and attr['witness_power'] == witness, 'attraction witness differs')
    flags=dict(root_existence_and_local_uniqueness=root_pass,primitive_period=primitive_pass,local_attraction=attraction_pass)
    require(type(result['conditional_gates']) is dict and set(result['conditional_gates']) == set(flags), 'all aggregate gates required')
    require(all(result['conditional_gates'][k] is v for k,v in flags.items()) and result['conditional_gates_passed'] is all(flags.values()), 'overall gate flags differ')
    return dict(schema='codex-exact-rational-gate-witness-audit-v1',certificate_passed=False,
        witness_arithmetic_consistent=True,conditional_gates=flags,conditional_gates_passed=all(flags.values()),
        phase_rows_checked=p,scalar_blocks_checked=p*p,proper_divisors_checked=divisors,
        matrix_row_norm_entries_checked=matrix_entries,dense_matrix_products_recomputed=False,
        preconditioned_residual_products_recomputed=False,source_enclosure_hypotheses_established=False,
        scope='Independent exact rational scalar recombination, strict margins, primitive separation and complete saved power-matrix row norms. Matrix products and evaluator enclosures still require their source/execution/evidence authorities; this is not certificate finalization.')
