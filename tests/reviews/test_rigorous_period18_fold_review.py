"""Small synthetic controls; no selected target, model data or file reads."""
import copy,importlib.util
from pathlib import Path
from fractions import Fraction
import pytest
from flint import arb,ctx
from neural_shrimp.rigorous_folds import _proof
from neural_shrimp.rigorous_orbits import ball_json,vector_json
spec=importlib.util.spec_from_file_location('fold_review',Path(__file__).parents[2]/'tools/reviews/review_rigorous_period18_fold.py')
R=importlib.util.module_from_spec(spec);spec.loader.exec_module(R)

def test_exact_dyadic_interval_and_norm():
    b={'mid':[3,-2],'rad':[1,-3]}
    assert R.interval(b)==(Fraction(5,8),Fraction(7,8))
    assert R.exact_norm([[b,b],[b,b]])==Fraction(7,4)
    with pytest.raises(ValueError):R.interval({'mid':[0,0],'rad':[-1,-3]})

def test_strict_nested_not_overlap():
    outer=[{'mid':[0,0],'rad':[1,-2]}]
    inner=[{'mid':[0,0],'rad':[1,-3]}]
    assert R.nested(inner,outer)
    assert not R.nested(outer,outer)
    assert not R.nested([{'mid':[1,0],'rad':[1,-3]}],outer)

def fixture():
    # A thirteen-dimensional polynomial fold, then synthetic phase labels only
    # to exercise the saved-divisor arithmetic join. This is NOT a period18 map.
    class Fold:
        n=13
        def jet_step(self,x,h):return [x[0]+h-1+x[0]*x[0]]+[a*(arb(1)/2) for a in x[1:]]
    with ctx.workprec(256):
        z=[arb(0)]*13+[arb(1),arb(1)]+[arb(0)]*12
        r=_proof(Fold(),z,1,-100,0,4096)
        orbit=[vector_json([arb(1)]+[arb(0)]*12) for _ in range(19)]
        orbit[0]=copy.deepcopy(r['box'][:13]);orbit[-1]=copy.deepcopy(r['box'][:13])
        r['orbit_enclosures']=orbit
        r['divisors']=[{'divisor':d,'excluded':True} for d in (1,2,3,6,9)]
        return r

def test_saved_arithmetic_independent_reductions():
    result=R.saved_math(fixture())
    assert result['complement_power']==1 and result['proper_divisors']==[1,2,3,6,9]
    assert R.interval(result['a'])[0]>0 and R.interval(result['c'])[0]>0

def test_divisor_and_full_pass_tampering_rejected():
    r=fixture();r['orbit_enclosures'][9]=copy.deepcopy(r['box'][:13])
    with pytest.raises(ValueError,match='divisor'):R.saved_math(r)
    r=fixture();r['gates']['other_modes_attracting']=False
    with pytest.raises(ValueError,match='full-fold'):R.saved_math(r)
