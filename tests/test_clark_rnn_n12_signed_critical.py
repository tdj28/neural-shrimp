import numpy as np
import pytest
from neural_shrimp.clark_rnn import iterate
from neural_shrimp.clark_rnn_symmetry import g_jets,critical_fixed_gain
@pytest.mark.parametrize('sign',[-1,1])
def test_fixed_gain_signed_critical_full_analytic_jacobian(sign):
    rng=np.random.default_rng(173);j=rng.normal(size=(12,12))*.1;x=rng.normal(size=12)*.2;q=rng.normal(size=12);q/=np.linalg.norm(q);z=np.r_[x,.72,q];g=2.1
    h,b,a=critical_fixed_gain(z,j,g,sign)
    eps=2e-6;fd=np.column_stack([(critical_fixed_gain(z+eps*np.eye(25)[k],j,g,sign)[0]-critical_fixed_gain(z-eps*np.eye(25)[k],j,g,sign)[0])/(2*eps)for k in range(25)])
    np.testing.assert_allclose(b,fd,rtol=2e-6,atol=2e-8)
    np.testing.assert_allclose(h[:12],a['endpoint']-x)
    np.testing.assert_allclose(h[12:24],(a['monodromy']-sign*np.eye(12))@q)
def test_global_signed_half_square_matches_unquotiented_native_six():
    rng=np.random.default_rng(174);j=rng.normal(size=(12,12))*.15;x=rng.normal(size=12);g=2.5;d=.81
    gx,du,_,_,_=g_jets(x,j,g,d,3);ggx,dv,_,_,_=g_jets(gx,j,g,d,3)
    end,m,cycle,phase=iterate(x,j,g,d,6)
    np.testing.assert_allclose(ggx,end,rtol=1e-12,atol=1e-13)
    np.testing.assert_allclose(dv[:,:12]@du[:,:12],m,rtol=1e-11,atol=1e-13)
    # M6=DG(x)^2 needs an actual fixed point; it is not true for this arbitrary x.
    assert np.max(np.abs(du[:,:12]@du[:,:12]-m))>1e-5
