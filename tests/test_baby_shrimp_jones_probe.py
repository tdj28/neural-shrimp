import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
import baby_shrimp_jones_probe as probe


def test_lift_tangent_has_domain_scaling():
    coef=np.array([[1,2],[3,4],[.1,.2]])
    u=np.array([.1,.3,.8]);h=1e-5
    fd=(probe.lift(coef,np.zeros(2),u+h)-probe.lift(coef,np.zeros(2),u-h))/(2*h)
    assert np.allclose(probe.lift(coef,np.zeros(2),u,True),fd,rtol=1e-10)


def test_critical_brackets_require_actual_sign_change_and_finite_support():
    u=np.linspace(0,1,201)
    roots=probe.brackets(u,(u-.233)*(u-.701))
    assert len(roots)==2 and roots[0][0]<.233<roots[0][1]
    assert not probe.brackets(u,np.ones(201))
    with pytest.raises(ValueError,match="finite"):
        probe.brackets(u,np.full(201,np.nan))


def test_precision_extension_checks_quadratic_regime_and_accuracy():
    def evaluate(u,tangent=True):
        return u**4,4*u**3 if tangent else None,(u**4)[:,None]
    rec,raw=probe.precision_test(evaluate,[.01,.003,.001,.0003,.0001])
    assert rec["derivative_qualified"]
    assert np.allclose(raw["orders"],2,atol=.01)
    def wrong(u,tangent=True):
        return u**4,4*u**3+.01 if tangent else None,(u**4)[:,None]
    assert not probe.precision_test(wrong,[.01,.003,.001,.0003,.0001])[0]["derivative_qualified"]
