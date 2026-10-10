import importlib.util
from pathlib import Path
import pytest
from flint import arb,arb_mat,ctx
spec=importlib.util.spec_from_file_location('audit',Path(__file__).parents[1]/'tools/audit_generalized_chain_independent.py');A=importlib.util.module_from_spec(spec);spec.loader.exec_module(A)
def test_strict_krawczyk_linear_and_outside_root():
    with ctx.workprec(384):
        c=[arb(0),arb(0)];b=[arb(0,1),arb(0,1)];j=arb_mat([[2,0],[0,3]]);pre=arb_mat([[arb(1)/2,0],[0,arb(1)/3]])
        assert A.krawczyk(c,b,arb_mat([[0],[0]]),j,pre)['passed']
        assert not A.krawczyk(c,b,arb_mat([[3],[0]]),j,pre)['passed']
def test_contraction_and_finite_rejection():
    c=[arb(0)];b=[arb(0,1)]
    assert not A.krawczyk(c,b,arb_mat([[0]]),arb_mat([[0]]),arb_mat([[1]]))['passed']
    with pytest.raises(ValueError):A.krawczyk(c,b,arb_mat([[arb('nan')]]),arb_mat([[1]]),arb_mat([[1]]))
def test_divisor_checks_all_divisors_and_rejects_collapse():
    b=[arb(0,arb(2)**-40)];o=[[arb(i)]for i in range(19)]
    r=A.divisors(o,b,18,1);assert [x['divisor']for x in r]==[1,2,3,6,9]and all(x['passed']for x in r)
    o[6]=b;assert not A.divisors(o,b,18,1)[3]['passed']
def test_dyadic_decode_outward_and_bad_radii():
    x=arb(arb(3)/8,arb(2)**-80);assert A.ball(A.enc(x)).contains(x)
    with pytest.raises(ValueError):A.ball({'mid':[1,0],'rad':[-1,-8]})
    with pytest.raises(ValueError):A.ball({'mid':[1.,0],'rad':[0,0]})
