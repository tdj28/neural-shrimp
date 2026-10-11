"""Analytic non-target systems test pulse chronology and second derivatives."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import two_write_sensitivity as s


@pytest.mark.parametrize('method', list(s.METHODS))
def test_bilinear_integrator_exact_cross_term_and_postwrite_convention(method):
    def rhs(t,x): return np.array([0.,0.,x[0]*x[1]])
    def jac(t,x): return np.array([[0.,0.,0.],[0.,0.,0.],[x[1],x[0],0.]])
    def hess(t,x,u,v): return np.array([0.,0.,u[0]*v[1]+u[1]*v[0]])
    times=np.array([1.,1.25,2.])
    initial=np.zeros(3);q1=np.array([1.,0.,0.]);q2=np.array([0.,1.,0.])
    ref=s.reference(rhs,jac,hess,initial,q1,q2,1.,times,method)
    expected=np.zeros((3,6,3));expected[:,1,0]=1;expected[:,2,1]=1
    expected[:,5,2]=times-1
    np.testing.assert_allclose(ref['samples'],expected,atol=2e-13)
    np.testing.assert_array_equal(ref['event_before'][2],np.zeros(3))
    np.testing.assert_array_equal(ref['event_after'][2],q2)
    trace=s.finite(rhs,initial,q1,q2,1.,times,.2,-.3,method)['samples']
    np.testing.assert_allclose(trace,np.column_stack((times*0+.2,times*0-.3,-.06*(times-1))),atol=2e-13)
    c=s.coefficients(ref['samples'],.2)
    np.testing.assert_allclose(c['C'].reshape(3,3)[:,2],.04*(times-1),atol=2e-13)


def test_self_terms_and_factor_two_with_scalar_quadratic_forcing():
    def rhs(t,x):return np.array([0.,x[0]**2])
    def jac(t,x):return np.array([[0.,0.],[2*x[0],0.]])
    def hess(t,x,u,v):return np.array([0.,2*u[0]*v[0]])
    times=np.array([1.,1.25,2.]);q=np.array([1.,0.]);initial=np.zeros(2)
    ref=s.reference(rhs,jac,hess,initial,q,q,1.,times,'primary')['samples']
    np.testing.assert_allclose(ref[:,3,1],2*times,atol=2e-13)
    np.testing.assert_allclose(ref[:,4,1],2*(times-1),atol=2e-13)
    np.testing.assert_allclose(ref[:,5,1],2*(times-1),atol=2e-13)
    c=s.coefficients(ref,.1)
    np.testing.assert_allclose(c['m'].reshape(3,2)[:,1],.01*(2*times-1),atol=1e-14)
    np.testing.assert_allclose(c['C'].reshape(3,2)[:,1],.02*(times-1),atol=1e-14)
    for a,b in ((1,1),(1,-1),(-1,1),(-1,-1)):
        trace=s.finite(rhs,initial,q,q,1.,times,a*.1,b*.1,'primary')['samples'].ravel()
        np.testing.assert_allclose(trace,c['m']+a*c['A']+b*c['B']+a*b*c['C'],atol=2e-13)


def test_linear_diagonal_system_has_no_hessian_or_parity():
    lam=np.array([-.2,-.7]);jac=lambda t,x:np.diag(lam)
    rhs=lambda t,x:lam*x;hess=lambda t,x,u,v:np.zeros(2)
    initial=np.array([.4,-.3]);q1=np.array([1.,2.]);q2=np.array([3.,-1.])
    times=np.array([1.,1.5,2.])
    ref=s.reference(rhs,jac,hess,initial,q1,q2,1.,times,'primary')['samples']
    np.testing.assert_allclose(ref[:,0],initial*np.exp(times[:,None]*lam),atol=2e-12)
    np.testing.assert_allclose(ref[:,1],q1*np.exp(times[:,None]*lam),atol=2e-12)
    np.testing.assert_allclose(ref[:,2],q2*np.exp((times[:,None]-1)*lam),atol=2e-12)
    np.testing.assert_array_equal(ref[:,3:],0)


def test_tanh_hessian_against_centered_jacobian_difference():
    W=np.array([[.2,-.7],[.3,.1]])
    rhs,jac,hess=s.tanh_flow(W)
    x=np.array([.4,-.8]);u=np.array([.7,.3]);v=np.array([-.2,.5]);epsilon=1e-5
    direct=((jac(0,x+epsilon*v)-jac(0,x-epsilon*v))/(2*epsilon))@u
    np.testing.assert_allclose(hess(0,x,u,v),direct,rtol=1e-9,atol=1e-11)
    np.testing.assert_allclose(hess(0,x,u,v),hess(0,x,v,u),atol=1e-16)


def test_activation_ablation_preserves_static_observation_interaction():
    samples=np.zeros((1,6,1));samples[0,:,0]=[.7,.2,.3,.4,.5,.6]
    amp=.1;z=np.tanh(.7);gp=1-z*z;gpp=-2*z*gp
    full=s.coefficients(samples,amp,'activation','second')
    ablated=s.coefficients(samples,amp,'activation','ablated')
    first=s.coefficients(samples,amp,'activation','first')
    assert full['C'][0]==pytest.approx(amp**2*(gp*.6+gpp*.2*.3))
    assert ablated['C'][0]==pytest.approx(amp**2*gpp*.2*.3)
    assert ablated['m'][0]==full['m'][0]
    assert first['C'][0]==0 and first['m'][0]==pytest.approx(z)


def test_harmonic_phase_contexts_match_exact_states_gradients_and_null_writes():
    J=np.array([[0.,-1.],[1.,0.]])
    rhs=lambda t,x:J@x;jac=lambda t,x:J
    fractions=np.array([0.,.125,.625]);T=2*np.pi
    packet=s.phase_contexts(rhs,jac,np.array([1.,0.]),np.array([0.,1.]),T,fractions,'primary')
    assert packet['closure']<1e-10
    for row,f in zip(packet['contexts'],fractions):
        np.testing.assert_allclose(row['initial'],[np.cos(f*T),np.sin(f*T)],atol=1e-10)
        np.testing.assert_allclose(row['phase_gradient'],[-np.sin(f*T),np.cos(f*T)],atol=1e-10)
        assert abs(row['normalization_residual'])<1e-10
        assert abs(row['direction_leakage'])<1e-14
        assert np.linalg.norm(row['direction'])==pytest.approx(1)


def test_times_and_shapes_are_explicit():
    times=s.sample_times(2.)
    assert times.shape==(64,) and times[0]==2 and times[31]==2.5 and times[32]==3.5 and times[-1]==4
    with pytest.raises(ValueError):s.coefficients(np.zeros((3,5,2)),.1)
    with pytest.raises(ValueError):s.phase_contexts(None,None,np.ones(2),np.ones(2),1.,np.array([1.]),'primary')
    with pytest.raises(ValueError):s.finite(None,np.ones(2),np.ones(2),np.ones(2),1.,np.array([.5,2.]),.1,.1,'primary')
