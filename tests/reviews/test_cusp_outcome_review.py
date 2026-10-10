import sys
from pathlib import Path
import pytest
from copy import deepcopy
from flint import arb,ctx
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools/reviews'),str(ROOT/'tests/reviews')]
import review_cusp_math as R
from neural_shrimp.rigorous_cusps import cusp_proof
from test_rigorous_cusps_independent import CoupledCusp

def fixture():
    with ctx.workprec(256):return cusp_proof(CoupledCusp('1/2'),[arb(0),arb(0),arb(1),arb(1),arb(1),arb(0),arb(0),arb(4)],1,newton_steps=1)

def test_independent_saved_reduction_recovers_coupled_constants():
    r=R.arithmetic(fixture());assert r['passed'];assert R.restore(r['cubic_coefficient']).contains(5);assert R.restore(r['unfolding_determinant']).contains(4)

@pytest.mark.parametrize('field',['cubic_coefficient','unfolding_determinant'])
def test_changed_numeric_claim_is_rejected(field):
    r=fixture();r[field]={'mid':[99,0],'rad':[0,0],'display':'99'}
    with pytest.raises(ValueError):R.arithmetic(r)

def test_reduced_monodromy_cannot_detach_from_saved_gradients():
    r=fixture();r['monodromy'][0][0]={'mid':[2,0],'rad':[0,0],'display':'2'}
    with pytest.raises(ValueError):R.arithmetic(r)

def test_singular_spectral_basis_fails_closed():
    r=fixture();z={'real':{'mid':[0,0],'rad':[0,0],'display':'0'},'imag':{'mid':[0,0],'rad':[0,0],'display':'0'}};r['spectral_exclusion']['proposal_basis']=[[z,z],[z,z]]
    with pytest.raises(ZeroDivisionError):R.arithmetic(r)

def test_strict_nesting_does_not_accept_equal_or_protruding_box():
    a=[{'mid':[0,0],'rad':[1,-8],'display':'x'}];b=[{'mid':[0,0],'rad':[1,-10],'display':'x'}]
    assert R.strict_nested(a,b) and not R.strict_nested(a,a) and not R.strict_nested(b,a)


def test_bounded_driver_import_and_resource_metadata_fail_closed():
    import importlib.util
    path=ROOT/'tools/reviews/review_s2_cusps.py'
    spec=importlib.util.spec_from_file_location('cusp_audit_driver',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.finite(0,1)
    for x in (-1,float('inf'),float('nan'),True):assert not module.finite(x,1)
