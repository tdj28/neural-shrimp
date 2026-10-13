import sys
from dataclasses import replace
from pathlib import Path

import pytest
from flint import arb, arb_mat, ctx

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'codex/certify-llm'))
from shooting_gates import (
    SHOOTING_DECLARATION, ShootingEnclosureContract, domain_sha256,
    enclosures_sha256, proposals_sha256, shooting_root_gate,
    primitive_phase_gate, _defect_blocks, _eye, _matrix_sha256,
)
from proof_gates import restore_ball, attraction_gate


@pytest.fixture(autouse=True)
def precision():
    old=ctx.prec; ctx.prec=256
    yield
    ctx.prec=old


def contract(c,r,v,J,A,B):
    return ShootingEnclosureContract('a'*64,'b'*64,'c'*64,len(c),
        domain_sha256(c,r),enclosures_sha256(v,J),proposals_sha256(A,B),SHOOTING_DECLARATION)


def call(c,r,v,J,A,B,**kwargs):
    return shooting_root_gate(c,r,v,J,A,B,contract=kwargs.pop('contract',contract(c,r,v,J,A,B)),**kwargs)


def basic():
    c=[[arb(2)],[arb(2)]];r=[arb(1),arb(1)]
    v=[[arb(2)],[arb(2)]];A=[arb_mat([[arb(1)/2]]) for _ in range(2)]
    return c,r,v,A,A,arb_mat([[arb(5)/4]])


def test_unique_repeated_fixed_point_root_passes_without_primitive_claim():
    args=basic(); result=call(*args)
    assert result['conditional_gates_passed']
    assert result['unique_root_in_full_product_box']
    assert result['enclosure_validity_checked_by_this_module'] is False
    assert result['attraction_checked'] is False
    assert 'certificate_passed' not in result
    assert not primitive_phase_gate(*args[:2])['conditional_primitive_period']
    assert restore_ball(result['contraction_upper']) == arb(1)/16


def test_streamed_blocks_equal_dense_noncommuting_exact_system():
    p,n=4,2;I=_eye(n)
    A=[arb_mat([[1,arb(1)/2],[0,1]]),arb_mat([[1,0],[-1,1]]),
       arb_mat([[arb(1)/2,0],[0,arb(1)/4]]),arb_mat([[0,arb(1)/4],[0,0]])]
    B=arb_mat([[arb(1)/2,arb(1)/8],[0,arb(1)/2]])
    J=[A[i]+arb_mat([[arb(i+1)/64,arb(-1)/128],[0,arb(1)/256]]) for i in range(p)]
    R=arb_mat(p*n,p*n)
    for col in range(p*n):
        b=[arb_mat(n,1,[int(i*n+a==col) for a in range(n)]) for i in range(p)]
        s=arb_mat(n,1)
        for Ai,bi in zip(A,b):s=Ai*s-bi
        z=B*s
        for i in range(p):
            for a in range(n):R[i*n+a,col]=z[a,0]
            z=A[i]*z-b[i]
    H=arb_mat(p*n,p*n)
    for i in range(p):
        for a in range(n):
            for b in range(n):H[i*n+a,i*n+b]=J[i][a,b]
            H[i*n+a,((i+1)%p)*n+a]=-1
    expected=_eye(p*n)-R*H
    seen=set()
    for i,j,block in _defect_blocks(A,B,J):
        seen.add((i,j))
        for a in range(n):
            for b in range(n):
                assert block[a,b].is_exact()
                assert block[a,b] == expected[i*n+a,j*n+b]
    assert len(seen)==p*p


def test_signed_linear_cancellation_preserved_before_absolute_norms():
    c=[[arb(0),arb(0)] for _ in range(4)]
    A=[arb_mat([[1,32],[0,1]]),arb_mat([[1,-32],[0,1]]),
       arb_mat([[0,16],[0,0]]),arb_mat(2,2)]
    result=call(c,[arb(1)]*4,c,A,A,_eye(2))
    assert result['conditional_gates_passed']
    assert restore_ball(result['contraction_upper']) == 0
    assert all(restore_ball(v['induced_infinity_upper'])==0 for v in result['block_bounds'])


def test_box_derivative_uncertainty_is_not_ignored():
    c,r,v,J,A,B=basic()
    wide=[arb_mat([[arb(0,2)]]) for _ in range(2)]
    result=call(c,r,v,wide,A,B)
    assert not result['conditional_gates_passed']
    assert restore_ball(result['contraction_upper']) > 1


def test_strict_inclusion_rejects_boundary_and_accepts_interior():
    c=[[arb(0)],[arb(0)]];r=[arb(1)]*2;A=[arb_mat([[0]]) for _ in range(2)];B=_eye(1)
    result=call(c,r,[[arb(0)],[arb(1)]],A,A,B)
    assert not result['conditional_gates_passed']
    assert restore_ball(result['phases'][0]['inclusion_margin'])==0
    result=call(c,r,[[arb(0)],[arb(1)/2]],A,A,B)
    assert result['conditional_gates_passed']


def test_nominal_radius_used_not_outward_rounded_storage_radius():
    radius=arb(1)+arb((1,-100))
    stored=arb(0,radius).rad()
    assert stored > radius
    c=[[arb(0)],[arb(0)]];A=[arb_mat([[0]]) for _ in range(2)]
    result=call(c,[radius]*2,[[arb(0)],[stored]],A,A,_eye(1))
    assert not result['conditional_gates_passed']


def test_singular_preconditioner_has_q_one_and_fails_strict_contraction():
    c=[[arb(0)],[arb(0)]];A=[arb_mat([[0]]) for _ in range(2)]
    result=call(c,[arb(1)]*2,c,A,A,arb_mat([[0]]))
    assert not result['conditional_gates_passed']
    assert restore_ball(result['contraction_upper'])==1


def test_unstable_map_root_can_pass_but_attraction_must_fail_separately():
    c=[[arb(0)],[arb(0)]];A=[arb_mat([[2]]) for _ in range(2)]
    result=call(c,[arb(1)]*2,c,A,A,arb_mat([[-arb(5)/16]]))
    assert result['conditional_gates_passed']
    assert result['attraction_checked'] is False
    assert not attraction_gate(arb_mat([[4]]),powers=(1,2,4))['passed']


def test_phase_radii_weights_and_all_block_witness_hashes():
    c,r,v,J,A,B=basic();r=[arb(1),arb(1)/4]
    result=call(c,r,v,J,A,B)
    assert restore_ball(result['contraction_upper'])==arb(1)/8
    blocks={(i,j):M for i,j,M in _defect_blocks(A,B,J)}
    for record in result['block_bounds']:
        assert record['matrix_sha256']==_matrix_sha256(blocks[record['phase_row'],record['phase_column']])
    assert len(result['block_bounds'])==4


@pytest.mark.parametrize('field,value',[
    ('domain_sha256','d'*64),('enclosures_sha256','e'*64),('proposals_sha256','f'*64),
    ('map_sha256','x'*64),('evaluator_source_sha256','a'*63),('evidence_sha256',''),
    ('period',3),('period',True),('declaration','point derivatives only'),
])
def test_contract_mutations_rejected(field,value):
    args=basic();ct=replace(contract(*args),**{field:value})
    with pytest.raises(ValueError):shooting_root_gate(*args,contract=ct)


def test_missing_contract_rejected():
    with pytest.raises(ValueError):shooting_root_gate(*basic(),contract=None)


def test_actual_input_mutations_rejected_after_contract_binding():
    c,r,v,J,A,B=basic();ct=contract(c,r,v,J,A,B)
    changed=[arb_mat([[arb(1)/2+arb((1,-100))]]),J[1]]
    with pytest.raises(ValueError):shooting_root_gate(c,r,v,changed,A,B,contract=ct)
    with pytest.raises(ValueError):shooting_root_gate(c,[arb(2),r[1]],v,J,A,B,contract=ct)
    with pytest.raises(ValueError):shooting_root_gate(c,r,v,J,A,_eye(1),contract=ct)


@pytest.mark.parametrize('bad',[arb('nan'),arb('inf'),0.0])
def test_invalid_center_values_refused(bad):
    c,r,v,J,A,B=basic();c[0][0]=bad
    with pytest.raises(ValueError):call(c,r,v,J,A,B)


def test_inexact_center_and_proposal_refused():
    c,r,v,J,A,B=basic();c[0][0]=arb(2,(1,-100))
    with pytest.raises(ValueError):call(c,r,v,J,A,B)
    c,r,v,J,A,B=basic();B[0,0]=arb(1,(1,-100))
    with pytest.raises(ValueError):call(c,r,v,J,A,B)


@pytest.mark.parametrize('r',[arb(0),arb(-1),arb(1,(1,-100)),1.0])
def test_nonpositive_or_inexact_radius_refused(r):
    c,_,v,J,A,B=basic()
    with pytest.raises(ValueError):call(c,[r,arb(1)],v,J,A,B)


def test_missing_phase_and_transverse_dimension_refused():
    c,r,v,J,A,B=basic()
    with pytest.raises(ValueError):call(c,r,v,J[:1],A,B)
    with pytest.raises(ValueError):call(c,r,v,[arb_mat(2,2)]*2,A,B)


def test_primitive_phase_boxes_all_proper_divisors_and_root_dependency():
    c=[[arb(i*4),arb(-i)] for i in range(8)];r=[arb(1)]*8
    result=primitive_phase_gate(c,r)
    assert result['required_divisors']==[1,2,4]
    assert result['conditional_primitive_period']
    assert result['root_existence_checked'] is False
    assert result['requires_shooting_root_existence'] is True
    assert result['domain_sha256']==domain_sha256(c,r)
    c[4]=c[0][:]
    result=primitive_phase_gate(c,r)
    assert not result['conditional_primitive_period']
    assert result['divisors'][-1]['divisor']==4 and not result['divisors'][-1]['phase_separated']


def test_primitive_touching_phase_boxes_fail_strict_separation():
    result=primitive_phase_gate([[arb(0)],[arb(2)]],[arb(1)]*2)
    assert not result['conditional_primitive_period']
    assert restore_ball(result['divisors'][0]['separation_margins'][0])==0


def test_domain_and_enclosure_hashes_bind_exact_tiny_radius_changes():
    c,r,v,J,A,B=basic()
    assert domain_sha256(c,r)!=domain_sha256(c,[arb(1)+arb((1,-200)),arb(1)])
    v2=[[arb(2,(1,-200))],v[1]]
    assert enclosures_sha256(v,J)!=enclosures_sha256(v2,J)


def test_factored_mode_valid_bound_and_explicit_nonmatrix_witnesses():
    args=basic()
    result=call(*args,bound_method='factored')
    assert result['conditional_gates_passed']
    assert result['bound_method']=='factored'
    assert restore_ball(result['contraction_upper'])==arb(1)/16
    for record in result['block_bounds']:
        assert record['bound_method']=='factored'
        assert 'matrix_sha256' not in record and 'row_norm_upper' not in record
        assert 'factors' in record
    assert result['proposals_sha256']==proposals_sha256(args[-2],args[-1])


def test_factored_scalar_bounds_dominate_exact_noncommuting_blocks():
    from shooting_gates import _factored_bounds, _row_norms
    A=[arb_mat([[1,arb(1)/2],[0,1]]),arb_mat([[1,0],[-1,1]]),
       arb_mat([[arb(1)/2,0],[0,arb(1)/4]]),arb_mat([[0,arb(1)/4],[0,0]])]
    B=arb_mat([[arb(1)/2,arb(1)/8],[0,arb(1)/2]])
    J=[A[i]+arb_mat([[arb(i+1)/64,arb(-1)/128],[0,arb(1)/256]]) for i in range(4)]
    exact={(i,j):_row_norms(M)[0] for i,j,M in _defect_blocks(A,B,J)}
    seen=set()
    for i,j,bound,factors in _factored_bounds(A,B,J):
        seen.add((i,j));assert bound>=exact[i,j]
        assert bound.is_exact() and bound.is_finite()
        if i==j+1:
            assert restore_ball(factors['remainder_segment_norm_product_upper'])==1
        elif i<=j:
            assert restore_ball(factors['remainder_segment_norm_product_upper'])==0
    assert len(seen)==16


def test_factored_mode_strict_boundary_and_derivative_uncertainty():
    c=[[arb(0)],[arb(0)]];r=[arb(1)]*2;A=[arb_mat([[0]]) for _ in range(2)];B=_eye(1)
    assert not call(c,r,[[arb(0)],[arb(1)]],A,A,B,bound_method='factored')['conditional_gates_passed']
    wide=[arb_mat([[arb(0,2)]]) for _ in range(2)]
    assert not call(c,r,c,wide,A,B,bound_method='factored')['conditional_gates_passed']


def test_factored_mode_can_be_more_conservative_than_streamed_mode():
    # At this invertible two-phase linear proposal, two R block terms cancel.
    # The factored estimate deliberately discards that last cancellation.
    c=[[arb(0)],[arb(0)]];A=[arb_mat([[8]]),arb_mat([[arb(1)/4]])]
    B=arb_mat([[-1]])
    J=[arb_mat([[8+arb(1)/8]]),arb_mat([[arb(1)/4]])]
    streamed=call(c,[arb(1)]*2,c,J,A,B)
    factored=call(c,[arb(1)]*2,c,J,A,B,bound_method='factored')
    assert restore_ball(factored['contraction_upper'])>=restore_ball(streamed['contraction_upper'])
    assert restore_ball(factored['contraction_upper'])>restore_ball(streamed['contraction_upper'])


def test_unknown_bound_method_refused():
    with pytest.raises(ValueError):call(*basic(),bound_method='float32_heuristic')
