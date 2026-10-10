from fractions import Fraction
import numpy as np
import pytest
from neural_shrimp.eigenpair_radius_diagnostic import positive_inflation


def test_diagonal_known_resolvent_exact_floor():
    r=[Fraction(1,16),Fraction(1,32)];out=positive_inflation([.9/16,.1/16],[[.2,0],[0,.1]],r)
    new=list(map(Fraction,out['exact_radii']))
    assert new[0]>Fraction(1,16)+Fraction(17,16)*Fraction.from_float(.1249999999999999/16)
    assert new[1]==r[1]*(1+Fraction(2)**-60)
    assert max(out['proposed_old_domain_ratios'])<1
    assert out['changed_coordinates']==1


def test_coupling_propagates_inflation_and_never_shrinks():
    out=positive_inflation([.8/16,.1/16],[[.3,.1],[.2,.1]],[Fraction(1,16),Fraction(1,16)])
    r=list(map(Fraction,out['exact_radii']))
    assert r[0]>Fraction(1,16) and r[1]>Fraction(1,16)
    assert max(out['proposed_old_domain_ratios'])<1
    assert max(map(abs,out['linear_residual']))<1e-15


def test_exact_radius_bits_not_float_coerced():
    r=Fraction(1,16)+Fraction(2)**-80
    out=positive_inflation([0],[[0]],[r]);new=Fraction(out['exact_radii'][0])
    assert new==r*(1+Fraction(2)**-60)


def test_nonpositive_resolvent_retained_then_rejected():
    seen=[]
    with pytest.raises(ValueError):positive_inflation([2],[[2]],[1],lambda k,v:seen.append((k,v)))
    assert seen[0][0]=='inflation_linear_return' and seen[0][1]['solution']==[-3.]


def test_nonfinite_solve_retained_before_validation(monkeypatch):
    seen=[];monkeypatch.setattr(np.linalg,'solve',lambda *_:np.array([np.nan]))
    with pytest.raises(ValueError):positive_inflation([2],[[0]],[1],lambda k,v:seen.append((k,v)))
    assert seen[0][1]['solution']==[None]


@pytest.mark.parametrize('a,B,r',[([1],[[np.nan]],[1]),([-1],[[0]],[1]),([1],[[0]],[0]),([1],[[0,0]],[1])])
def test_reject_bad_inputs_before_solve(a,B,r,monkeypatch):
    monkeypatch.setattr(np.linalg,'solve',lambda *_:pytest.fail('must not solve'))
    with pytest.raises(ValueError):positive_inflation(a,B,r)
