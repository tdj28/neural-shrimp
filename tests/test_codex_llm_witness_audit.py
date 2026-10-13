from pathlib import Path
import copy
import importlib.util
import sys
import pytest
from flint import arb, arb_mat, ctx
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'codex/certify-llm'))
import witness_audit as audit
import proof_gates as proof
import shooting_gates as shooting
import shooting_certificate as certificate


def fixture(slope=0.25, uncertain=False):
    ctx.prec=192;ctx.threads=1;p,n=8,2
    c=[[arb(10*i),arb(-3*i)] for i in range(p)];r=[arb(1)/2]*p
    v=[list(c[(i+1)%p]) for i in range(p)]
    a=[arb_mat([[slope,0],[0,slope]]) for _ in range(p)]
    b=arb_mat([[1,0],[0,1]])
    j=[arb_mat([[arb(slope, (1,-30)) if uncertain else arb(slope),arb(0,(1,-32)) if uncertain else 0],[0,slope]]) for _ in range(p)]
    contract=shooting.ShootingEnclosureContract('a'*64,'b'*64,'c'*64,p,shooting.domain_sha256(c,r),shooting.enclosures_sha256(v,j),shooting.proposals_sha256(a,b),shooting.SHOOTING_DECLARATION)
    result=certificate.check_shooting_certificate(c,r,v,j,a,b,contract=contract,powers=(1,2))
    return result,dict(centers=[proof.vector_record(row) for row in c],radii=proof.vector_record(r),images=[proof.vector_record(row) for row in v],contract=result['contract'].copy(),jacobian_hashes=[shooting._matrix_sha256(m) for m in j])


def test_full_nontrivial_synthetic_chain_with_exact_rational_audit():
    result,external=fixture();out=audit.audit(result,**external)
    assert out['conditional_gates_passed'] and out['witness_arithmetic_consistent']
    assert out['scalar_blocks_checked']==64 and out['matrix_row_norm_entries_checked']==8
    assert out['certificate_passed'] is False and out['dense_matrix_products_recomputed'] is False


def test_valid_failed_gates_are_retained_without_fabricating_success():
    result,external=fixture(1.0);out=audit.audit(result,**external)
    assert out['witness_arithmetic_consistent'] and not out['conditional_gates_passed']
    assert not out['conditional_gates']['root_existence_and_local_uniqueness']
    assert not out['conditional_gates']['local_attraction']


@pytest.mark.parametrize('alter',[
 lambda r,e:r['root']['phases'][0].update(contraction=True,contraction_margin={'mid':[2,0],'rad':[0,0]}),
 lambda r,e:r['root']['phases'][0].update(inclusion_margin={'mid':[2,0],'rad':[0,0]}),
 lambda r,e:r['root']['block_bounds'][-8].update(radius_weighted_upper={'mid':[0,0],'rad':[0,0]}),
 lambda r,e:r['root']['block_bounds'].pop(),
 lambda r,e:r['root']['block_bounds'].__setitem__(1,copy.deepcopy(r['root']['block_bounds'][0])),
 lambda r,e:r['primitive']['divisors'][0]['separation_margins'].__setitem__(0,{'mid':[1000,0],'rad':[0,0]}),
 lambda r,e:r['primitive']['divisors'].pop(),
 lambda r,e:r['attraction']['powers'][0]['weighted_row_upper'].__setitem__(0,{'mid':[0,0],'rad':[0,0]}),
 lambda r,e:r['attraction']['powers'][0].update(passed=False),
 lambda r,e:r['attraction']['powers'].reverse(),
 lambda r,e:r['monodromy'].update(order='J_0 ... J_(p-1)'),
 lambda r,e:e['jacobian_hashes'].__setitem__(0,'f'*64),
 lambda r,e:e['centers'][0][0]['mid'].__setitem__(0,1),
 lambda r,e:e['images'][0][0]['rad'].__setitem__(0,1),
 lambda r,e:r.update(certificate_passed=True),
 lambda r,e:r.update(conditional_gates_passed=False),
])
def test_corrupted_bound_binding_coverage_or_gate_refused(alter):
    result,external=fixture();alter(result,external)
    with pytest.raises(ValueError):audit.audit(result,**external)


@pytest.mark.parametrize('record',[
 {'mid':[True,0],'rad':[0,0]}, {'mid':[1,0],'rad':[-1,0]},
 {'mid':[1.0,0],'rad':[0,0]}, {'mid':[1,0]},
])
def test_malformed_original_dyadic_refused(record):
    with pytest.raises(ValueError):audit.endpoints(record)


def test_strict_boundary_is_not_success():
    result,external=fixture(1.0)
    assert result['attraction']['powers'][0]['margin']=={'mid':[0,0],'rad':[0,0]}
    out=audit.audit(result,**external)
    assert not out['conditional_gates']['local_attraction']


def test_interval_matrix_norms_and_nonzero_perturbation_bounds():
    result,external=fixture(uncertain=True)
    out=audit.audit(result,**external)
    assert out['conditional_gates_passed']
    result['root']['block_bounds'][0]['factors']['jacobian_perturbation_term_upper']={'mid':[0,0],'rad':[0,0]}
    with pytest.raises(ValueError,match='perturbation scalar bound understated'):
        audit.audit(result,**external)


@pytest.mark.parametrize('which',['prefix','suffix','partial_digest'])
def test_known_identity_factor_or_intermediate_digest_corruption_refused(which):
    result,external=fixture()
    block=result['root']['block_bounds'][0]
    if which=='prefix':block['factors']['prefix_norm_upper']={'mid':[0,0],'rad':[0,0]}
    elif which=='suffix':block['factors']['suffix_norm_upper']={'mid':[0,0],'rad':[0,0]}
    else:result['monodromy']['phase_product_witnesses'][0]['partial_product_sha256']='not a digest'
    with pytest.raises(ValueError):audit.audit(result,**external)
