import sys
from dataclasses import replace
from pathlib import Path

import pytest
from flint import arb, arb_mat, ctx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'codex/certify-llm'))
import proof_gates as proof
import shooting_gates as shooting
from shooting_certificate import check_shooting_certificate


@pytest.fixture(autouse=True)
def precision():
    previous=(ctx.prec,ctx.threads);ctx.prec=256;ctx.threads=1
    yield
    ctx.prec,ctx.threads=previous


def case(matrices=None):
    A=matrices or [arb_mat([[arb(1)/2]]) for _ in range(2)]
    n=A[0].nrows();p=len(A)
    c=[[arb(4*i) for _ in range(n)] for i in range(p)]
    r=[arb(1)/4]*p
    images=[list(c[(i+1)%p]) for i in range(p)]
    product=proof._eye(n)
    for a in A:product=a*product
    # For these exact synthetic examples, a dyadic inverse or dyadic proposal
    # is explicitly supplied by each test, not inferred from a floating solve.
    B=arb_mat([[arb(5)/4]]) if n==1 else proof._eye(n)
    return c,r,images,A,A,B


def run(args,**options):
    c,r,v,J,A,B=args
    contract=shooting.ShootingEnclosureContract('a'*64,'b'*64,'c'*64,len(c),
        shooting.domain_sha256(c,r),shooting.enclosures_sha256(v,J),
        shooting.proposals_sha256(A,B),shooting.SHOOTING_DECLARATION)
    return check_shooting_certificate(*args,contract=options.pop('contract',contract),**options)


def test_all_gates_pass_without_external_certificate_claim():
    result=run(case())
    assert result['conditional_gates_passed']
    assert all(result['conditional_gates'].values())
    assert result['certificate_passed'] is False
    assert result['external_authentication_performed'] is False
    assert result['attraction']['witness_power']==1
    assert result['monodromy']['matrix_sha256']==shooting._matrix_sha256(arb_mat([[arb(1)/4]]))


def test_repeated_fixed_point_does_not_become_primitive_orbit():
    args=list(case());args[0]=[[arb(0)],[arb(0)]];args[2]=[[arb(0)],[arb(0)]]
    result=run(args)
    assert result['conditional_gates']['root_existence_and_local_uniqueness']
    assert result['conditional_gates']['local_attraction']
    assert not result['conditional_gates']['primitive_period']
    assert not result['conditional_gates_passed']


def test_unstable_shooting_root_fails_attraction():
    args=list(case([arb_mat([[2]]),arb_mat([[2]])]));args[-1]=arb_mat([[-arb(5)/16]])
    result=run(args)
    assert result['conditional_gates']['root_existence_and_local_uniqueness']
    assert result['conditional_gates']['primitive_period']
    assert not result['conditional_gates']['local_attraction']
    assert not result['conditional_gates_passed']


def test_failed_inclusion_keeps_the_other_gates_without_promoting_them():
    args=list(case());args[2]=[[arb(100)],[arb(0)]]
    result=run(args)
    assert not result['conditional_gates']['root_existence_and_local_uniqueness']
    assert result['conditional_gates']['primitive_period']
    assert result['conditional_gates']['local_attraction']
    assert not result['conditional_gates_passed']


def test_noncommuting_product_order_and_power_attraction():
    # Exact A1*A0=[[0,2],[0,0]] has infinity norm2 but squares to zero.
    A=[arb_mat([[1,0],[0,2]]),arb_mat([[0,1],[0,0]])]
    args=list(case(A));args[-1]=arb_mat([[1,2],[0,1]])
    result=run(args)
    assert result['conditional_gates_passed']
    assert result['attraction']['witness_power']==2
    assert not result['attraction']['powers'][0]['passed']
    assert result['attraction']['powers'][1]['passed']
    expected=arb_mat([[0,2],[0,0]])
    assert result['monodromy']['matrix_sha256']==shooting._matrix_sha256(expected)
    assert result['monodromy']['matrix_sha256']!=shooting._matrix_sha256(A[0]*A[1])
    assert [v['phase'] for v in result['monodromy']['phase_product_witnesses']]==[0,1]


def test_all_proper_divisors_must_be_separated():
    A=[arb_mat([[0]]) for _ in range(8)];args=list(case(A));args[-1]=proof._eye(1)
    args[0][4]=list(args[0][0]);args[2]=[list(args[0][(i+1)%8]) for i in range(8)]
    result=run(args)
    assert result['primitive']['required_divisors']==[1,2,4]
    assert not result['conditional_gates_passed']
    assert result['conditional_gates']['root_existence_and_local_uniqueness']
    assert result['conditional_gates']['local_attraction']


def test_interval_derivative_width_survives_composition():
    args=list(case());args[3]=[arb_mat([[arb(0,2)]]),arb_mat([[arb(0,2)]])]
    result=run(args)
    assert not result['conditional_gates_passed']
    assert not result['conditional_gates']['local_attraction']


def test_input_contract_mismatch_refused_before_other_gates():
    args=case();c,r,v,J,A,B=args
    ct=shooting.ShootingEnclosureContract('a'*64,'b'*64,'c'*64,2,
        shooting.domain_sha256(c,r),shooting.enclosures_sha256(v,J),
        shooting.proposals_sha256(A,B),shooting.SHOOTING_DECLARATION)
    with pytest.raises(ValueError):run(args,contract=replace(ct,enclosures_sha256='d'*64))
    with pytest.raises(ValueError):run(args,contract=None)


def test_invalid_power_plan_and_missing_direction_refused():
    with pytest.raises(ValueError):run(case(),powers=())
    with pytest.raises(ValueError):run(case(),powers=(2,1))
    args=list(case());args[3]=[arb_mat(2,2),args[3][1]]
    with pytest.raises(ValueError):run(args)


def test_gate_call_does_not_change_inputs_or_context():
    args=case();before=[repr(v) for v in args];context=(ctx.prec,ctx.threads)
    result=run(args)
    assert before==[repr(v) for v in args]
    assert context==(ctx.prec,ctx.threads)
    assert result['precision']==256 and result['threads']==1
