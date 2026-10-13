"""Exact synthetic controls; no transformer, checkpoint, or target data imports."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from flint import arb, arb_mat, ctx
import pytest

PATH = Path(__file__).resolve().parents[1]/'codex/certify-llm/proof_gates.py'
SPEC = importlib.util.spec_from_file_location('codex_llm_proof_gates', PATH)
gates = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gates
SPEC.loader.exec_module(gates)


@pytest.fixture(autouse=True)
def arithmetic_context():
    previous = ctx.prec, ctx.threads
    ctx.prec, ctx.threads = 128, 1
    yield
    ctx.prec, ctx.threads = previous


def point(n, exponent=0):
    return arb((n, exponent))


def restored(record):
    return gates.restore_ball(record)


def synthetic_cycle(period=8, unstable=False):
    """F(x)=20(j+1)+s(x-20j) in disjoint phase neighborhoods, cyclically.

    It admits a smooth extension between those neighborhoods. On X around
    zero, G'=s**period and all supplied image intervals are exact enclosures.
    """
    slope_power = point(1, period if unstable else -period)
    center, radii = [arb(0)], [point(1, -4)]
    h, j = [arb(0)], arb_mat([[slope_power-1]])
    R = arb_mat([[point(1, -period) if unstable else arb(-1)]])
    images = {d: [arb(20*d, point(1, d-4 if unstable else -d-4))]
              for d in gates.proper_divisors(period)}
    M = arb_mat([[slope_power]])
    digest = lambda text: hashlib.sha256(text.encode()).hexdigest()
    contract = gates.EnclosureContract(
        map_sha256=digest('synthetic local affine phase map'),
        evaluator_source_sha256=digest('exact displayed phase-map formulas'),
        evidence_sha256=digest('test fixture is the symbolic enclosure derivation'),
        period=period, domain_sha256=gates.domain_sha256(center, radii),
        enclosures_sha256=gates.enclosures_sha256(h,j,images,M),
        declaration=gates.ENCLOSURE_DECLARATION)
    return dict(center=center, radii=radii, h_center=h, h_jacobian=j,
                preconditioner=R, divisor_images=images, monodromy=M,
                period=period, contract=contract, powers=(1,2,4))


def refresh_contract(data):
    data['contract'] = replace(data['contract'],
        domain_sha256=gates.domain_sha256(data['center'], data['radii']),
        enclosures_sha256=gates.enclosures_sha256(data['h_center'], data['h_jacobian'],
                                                 data['divisor_images'], data['monodromy']))


def test_stable_primitive_eight_cycle_passes_all_conditional_gates_and_serializes():
    result = gates.check_certificate(**synthetic_cycle())
    assert result['conditional_gates_passed']
    assert not result['enclosure_validity_checked_by_this_module']
    assert all(g['passed'] for g in result['gates'].values())
    assert result['gates']['root']['unique_in_full_nominal_box']
    assert result['gates']['primitive']['required_divisors'] == [1,2,4]
    assert result['gates']['attraction']['witness_power'] == 1
    assert restored(result['gates']['root']['contraction_upper']) == point(1,-8)
    json.dumps(result, allow_nan=False)


def test_unstable_cycle_root_and_primitive_pass_but_certificate_does_not():
    result = gates.check_certificate(**synthetic_cycle(unstable=True))
    assert result['gates']['root']['passed']
    assert result['gates']['primitive']['passed']
    assert not result['gates']['attraction']['passed']
    assert not result['conditional_gates_passed']


def test_nonzero_primal_residual_is_not_discarded():
    result = gates.krawczyk_gate([arb(0)], [arb(1)], [point(1,-2)],
                                arb_mat([[2]]), arb_mat([[point(1,-1)]]))
    assert result['passed']
    assert restored(result['image_delta'][0]) == point(-1,-3)
    assert restored(result['coordinate_margins'][0]) > point(7,-4)


def test_uncertain_primal_residual_and_jacobian_enclosures_are_retained():
    result = gates.krawczyk_gate([arb(0)], [arb(1)], [arb(0,point(1,-3))],
                                arb_mat([[arb(1,point(1,-4))]]), arb_mat([[1]]))
    assert result['passed']
    assert restored(result['image_delta'][0]).contains(point(3,-4))
    assert not restored(result['Hc'][0]).is_exact()
    assert restored(result['contraction_upper']) >= point(1,-4)


def test_large_primal_residual_refuses_inclusion_despite_zero_derivative_defect():
    result = gates.krawczyk_gate([arb(0)], [arb(1)], [arb(2)], arb_mat([[1]]), arb_mat([[1]]))
    assert result['contraction']
    assert not result['strict_inclusion'] and not result['passed']
    assert restored(result['coordinate_margins'][0]) == -1


def test_exact_inclusion_boundary_equality_fails():
    result = gates.krawczyk_gate([arb(0)], [arb(1)], [arb(1)], arb_mat([[1]]), arb_mat([[1]]))
    assert result['contraction']
    assert not result['passed'] and not result['coordinate_inclusions'][0]
    assert restored(result['coordinate_margins'][0]) == 0


@pytest.mark.parametrize('jac,R', [(0,1),(1,0)])
def test_contraction_boundary_and_singular_preconditioner_fail(jac,R):
    result = gates.krawczyk_gate([arb(0)],[arb(1)],[arb(0)],arb_mat([[jac]]),arb_mat([[R]]))
    assert not result['contraction'] and not result['passed']
    assert restored(result['contraction_upper']) == 1


def test_weighted_full_box_root_gate_uses_radii_not_unweighted_norm():
    # P'=[[0,2],[0,0]]; its weighted norm with r=(4,1) is 1/2.
    j = arb_mat([[1,-2],[0,1]])
    result = gates.krawczyk_gate([arb(0),arb(0)], [arb(4),arb(1)],
                                [arb(0),arb(0)], j, arb_mat([[1,0],[0,1]]))
    assert result['passed']
    assert restored(result['contraction_upper']) == point(1,-1)


def test_nilpotent_nonnormal_attraction_requires_matrix_cancellation():
    M = arb_mat([[1,1],[-1,-1]])
    result = gates.attraction_gate(M,powers=(1,2))
    assert not result['powers'][0]['passed'] and result['powers'][1]['passed']
    assert result['witness_power'] == 2
    assert restored(result['powers'][0]['induced_infinity_upper']) == 2
    assert restored(result['powers'][1]['induced_infinity_upper']) == 0
    naive = M*(M*arb_mat([[arb(0,1)],[arb(0,1)]]))
    assert naive[0,0].abs_upper() >= 4  # Matrix-free interval wrapping can fail.


def test_nonnormal_positive_power_witness_and_all_declared_powers_retained():
    result = gates.attraction_gate(arb_mat([[point(1,-1),8],[0,point(1,-1)]]),powers=(1,2,4,8,16))
    assert result['witness_power'] == 8
    assert len(result['powers']) == 5
    assert restored(result['powers'][3]['induced_infinity_upper']) == point(129,-8)


def test_interval_matrix_power_encloses_nonzero_uncertainty():
    M = arb_mat([[arb(point(1,-1),point(1,-6)),arb(0,point(1,-6))],
                 [arb(0,point(1,-6)),arb(point(1,-1),point(1,-6))]])
    result = gates.attraction_gate(M,powers=(1,2,3))
    assert result['passed']
    # Diagonal upper corner has row sum 17/32, so its square has norm (17/32)^2.
    upper = restored(result['powers'][1]['induced_infinity_upper'])
    assert upper >= point(289,-10)


@pytest.mark.parametrize('entry', [arb(1),arb(-1),arb(2)])
def test_attraction_equality_or_instability_never_passes(entry):
    result = gates.attraction_gate(arb_mat([[entry]]),powers=(1,2,4))
    assert not result['passed'] and result['witness_power'] is None


def test_weighted_attraction_is_an_actual_induced_norm():
    M=arb_mat([[point(1,-1),2],[0,point(1,-1)]])
    assert not gates.attraction_gate(M,powers=(1,))['passed']
    result=gates.attraction_gate(M,powers=(1,),weights=[arb(8),arb(1)])
    assert result['passed']
    assert restored(result['powers'][0]['induced_infinity_upper']) == point(3,-2)


def test_precision_floor_refuses_before_higher_precision_proves_strict_margin():
    M=arb_mat([[point(2**200-1,-200)]])
    ctx.prec=128
    assert not gates.attraction_gate(M,powers=(1,))['passed']
    ctx.prec=256
    result=gates.attraction_gate(M,powers=(1,))
    assert result['passed']
    assert restored(result['powers'][0]['margin']) > 0


def test_primitive_coordinate_separation_accepts_either_direction():
    result=gates.primitive_gate([arb(0)],[point(1,-2)],8,
        {1:[arb(2)],2:[arb(-2)],4:[arb(4)]})
    assert result['passed']
    assert [r['separating_witnesses'][0]['direction'] for r in result['divisors']] == [
        'image_above_box','image_below_box','image_above_box']


def test_lower_period_counterexample_fails_even_with_root_and_attraction():
    data=synthetic_cycle();data['divisor_images'][4]=[arb(0,point(1,-8))];refresh_contract(data)
    result=gates.check_certificate(**data)
    assert result['gates']['root']['passed'] and result['gates']['attraction']['passed']
    assert not result['gates']['primitive']['passed'] and not result['conditional_gates_passed']


def test_primitive_touching_boundary_and_overlap_fail():
    for image in [arb(1),arb(0,2)]:
        result=gates.primitive_gate([arb(0)],[arb(1)],2,{1:[image]})
        assert not result['passed']


@pytest.mark.parametrize('images', [{1:[arb(2)]}, {1:[arb(2)],2:[arb(4)],3:[arb(6)],4:[arb(8)]},
                                    {'1':[arb(2)],2:[arb(4)],4:[arb(8)]}])
def test_missing_extra_or_noninteger_divisor_cannot_be_dropped(images):
    with pytest.raises(ValueError,match='proper-divisor'):
        gates.primitive_gate([arb(0)],[arb(1)],8,images)


@pytest.mark.parametrize('field,value', [('center',[0.]),('radii',[arb(0)]),
    ('radii',[arb(-1)]),('radii',[arb(1,point(1,-4))]),
    ('center',[arb(0,point(1,-4))]),('h_center',[arb('nan')]),
    ('h_jacobian',[[1.]]),('h_jacobian',arb_mat([[1,2]])),
    ('preconditioner',arb_mat([[arb(1,point(1,-4))]])),
    ('preconditioner',arb_mat([[arb('inf')]]))])
def test_invalid_or_nonexact_root_inputs_fail_closed(field,value):
    data=dict(center=[arb(0)],radii=[arb(1)],h_center=[arb(0)],
              h_jacobian=arb_mat([[1]]),preconditioner=arb_mat([[1]]))
    data[field]=value
    with pytest.raises(ValueError):gates.krawczyk_gate(**data)


@pytest.mark.parametrize('powers', [(),(0,),(True,),(2,1),(1,1),(1.5,)])
def test_invalid_power_schedule_refused(powers):
    with pytest.raises(ValueError):gates.attraction_gate(arb_mat([[0]]),powers=powers)


def test_contract_is_required_and_binds_all_mathematical_inputs():
    data=synthetic_cycle()
    for contract in [None,{},replace(data['contract'],declaration='trust me'),
                     replace(data['contract'],period=4),
                     replace(data['contract'],domain_sha256='0'*64),
                     replace(data['contract'],enclosures_sha256='f'*64),
                     replace(data['contract'],evidence_sha256='short')]:
        changed={**data,'contract':contract}
        with pytest.raises(ValueError):gates.check_certificate(**changed)
    data['h_center']=[point(1,-30)]
    with pytest.raises(ValueError,match='bind'):
        gates.check_certificate(**data)


def test_ball_witness_roundtrip_and_context_remains_caller_owned():
    ctx.prec=97
    before=ctx.prec,ctx.threads
    result=gates.check_certificate(**synthetic_cycle())
    assert (ctx.prec,ctx.threads)==before
    for record in result['gates']['root']['image_delta']+result['gates']['root']['coordinate_margins']:
        assert gates.ball_record(gates.restore_ball(record)) == record
    with pytest.raises(ValueError):gates.restore_ball({'mid':[0,0],'rad':[-1,0]})


def test_period_one_has_empty_divisor_inventory_without_invented_cycle_claim():
    result=gates.primitive_gate([arb(0)],[arb(1)],1,{})
    assert result['passed'] and result['required_divisors']==[]
