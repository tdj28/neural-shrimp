"""Independent synthetic controls for the lower-gap adapter; no target reads."""
import math
import numpy as np
import pytest
from neural_shrimp import s1r_gap_diagnostic as c


def budget():
    records=[]
    return c.Budget(lambda label,payload: records.append((label,payload))),records


def test_nonlinear_log_hidden_chain_against_synthetic_difference():
    Q=np.array([[1.2,.2,-.1],[.2,.7,.15],[-.1,.15,.9]])
    def derivative(x):
        return float(x@Q@x/2+np.sum(x**4)/4),Q@x+x**3,Q+np.diag(3*x*x)
    x=np.array([.3,-.2,.4]);rates=np.array([.07,.13,.04])
    b,_=budget();a=c.evaluate_state(x,rates,2,derivative,b,'independent')
    def residual(h):
        r=rates*np.array([10.**h,10.**h,1.])
        y=x-r*derivative(x)[1]
        return y-r*derivative(y)[1]-x
    eps=2e-6
    expected=(residual(eps)-residual(-eps))/(2*eps)
    assert np.allclose(a['b'],expected,rtol=2e-8,atol=2e-11)
    assert b.attempted=={'maps':2,'jacobians':2,'svds':0}


def test_nonnormal_linear_equation_has_independent_scalar_residual():
    A=np.array([[1.,1e4,0.],[0.,2.,3.],[0.,0.,.5]])
    p=np.array([.3,-.00002,-.4]);R=-A@p+np.array([1e-9,-2e-9,3e-9])
    result=c.saved_linear_check(A,R,np.ones(3),p,R)
    e=[math.fsum([float(A[i,j]*p[j]) for j in range(3)]+[float(R[i])]) for i in range(3)]
    independent=math.sqrt(math.fsum(x*x for x in e))
    assert math.isclose(result['absolute_linear_residual'],independent,rel_tol=1e-7,abs_tol=1e-16)
    assert result['verdict']=='linear_equation_resolved'
    damaged=c.saved_linear_check(A,R,np.ones(3),p+np.array([0.,1e-3,0.]),R)
    assert damaged['verdict']=='material_linear_equation_mismatch'


def test_augmented_scaled_kernel_converts_back_before_normalization():
    A=np.array([[2.,3.],[0.,5.]])
    b=np.array([7.,11.]);D=np.array([17.,2.])
    expected=np.array([-.2,-2.2,1.]);expected/=np.linalg.norm(expected)
    bu,_=budget();g=c.geometry({'A':A,'b':b},D,expected,bu,'triangular')
    for key in ('raw_B','scaled_B'):
        t=g[key]['null_tangent']['tangent']
        assert np.allclose(t,expected,rtol=1e-9,atol=1e-11)
        assert np.linalg.norm(np.column_stack((A,b))@t)<1e-10
    assert bu.attempted['svds']==4


def test_rank_band_boundaries_and_isolation_abstentions():
    assert c.rank_summary(np.array([1.,1e-12]))['status']=='numerically_rank_deficient'
    assert c.rank_summary(np.array([1.,1e-10]))['status']=='ambiguous_rank'
    assert c.rank_summary(np.array([1.,np.nextafter(1e-10,np.inf)]))['status']=='full_row_rank_resolved'
    assert not c.rank_summary(np.array([1.,2e-11,1e-12]))['isolated_soft_direction']


def test_parameter_coupling_is_not_a_fold_verdict():
    A=np.diag([1e-6,1.]);b=np.array([1.,0.]);bu,_=budget()
    g=c.geometry({'A':A,'b':b},np.ones(2),np.array([-1e6,0.,1.]),bu,'regular')
    assert g['raw_A']['rank']['status']=='full_row_rank_resolved'
    assert abs(g['raw_A']['smallest_left_parameter_coupling'])==1.
    assert all('fold_verified' not in row for row in g.values())


def test_phase_matching_reverses_native_assignment_not_h_direction():
    parent=np.array([[2.,3.],[-2.,-3.]])
    child=np.array([[-2.1,-3.1],[2.2,3.2]])
    s=c.carried_secants(parent,child,.9925,.992,1)
    assert np.allclose(s['parent',0],[.2,.2,-.0005])
    assert np.array_equal(s['parent',0],s['child',1])
    assert np.allclose(s['parent',1],[-.1,-.1,-.0005])


def test_nearzero_and_reconstructed_residual_uncertainty_abstain():
    A=np.eye(2);x=np.ones(2)
    r=c.saved_linear_check(A,np.array([1e-15,0.]),x,np.ones(2),np.zeros(2))
    assert r['verdict']=='abstain_nearzero_residual' and r['true_relative_forcing'] is None
    R=np.array([.01,.02])
    r=c.saved_linear_check(A,R,x,-R,R+1e-6)
    assert r['verdict']=='arithmetic_resolution_limited'


def test_returned_svd_survives_sticky_interruption(monkeypatch):
    real=np.linalg.svd;bu,records=budget()
    def interrupt(*a,**kw):
        result=real(*a,**kw);c.request_stop(None,None);return result
    monkeypatch.setattr(c,'STOP',False);monkeypatch.setattr(np.linalg,'svd',interrupt)
    c.decompose(np.eye(3),bu,'returned')
    assert bu.returned['svds']==1 and any(k=='returned_svd' for k,v in records)
    with pytest.raises(InterruptedError):bu.check()
    monkeypatch.setattr(c,'STOP',False)
