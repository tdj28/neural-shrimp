import numpy as np
import pytest
from neural_shrimp.cifar_amplitude import equations,jacobian_action,physical_residuals


def test_full_nonlinear_derivative_and_symmetric_bordered_action():
    # L=.5(x^2+3y^2)+.2 x^2 y+.1 x^4; noncommuting phase Hessians.
    def grad(x):return np.array([x[0]+.4*x[0]*x[1]+.4*x[0]**3,3*x[1]+.2*x[0]**2])
    def hvp(x,v):return np.array([[1+.4*x[1]+1.2*x[0]**2,.4*x[0]],[.4*x[0],3.]])@v
    m=np.array([.2,-.1]);d=np.array([.05,.08]);tau=2.4;a=.1;z=np.r_[m,d,tau]
    J=np.column_stack([jacobian_action(m,d,tau,v,hvp)for v in np.eye(5)])
    assert np.allclose(J,J.T,atol=1e-15)
    for k in range(5):
        h=1e-5;zp=z+h*np.eye(5)[k];zm=z-h*np.eye(5)[k]
        fd=(equations(zp[:2],zp[2:4],zp[4],a,grad)-equations(zm[:2],zm[2:4],zm[4],a,grad))/(2*h)
        assert np.allclose(fd,J[:,k],rtol=2e-9,atol=1e-9)


def test_known_attracting_quartic_cycle_and_exact_shooting_identity():
    a=.2;tau=2-.5*a*a;m=np.zeros(2);d=np.array([a,0.]);eta=2/tau
    def grad(x):return np.array([2*x[0]-.5*x[0]**3,.5*x[1]])
    r=equations(m,d,tau,a,grad);assert np.linalg.norm(r)<1e-15
    assert np.allclose(m+d-eta*grad(m+d),m-d)
    assert max(abs(1-eta*(2-1.5*a*a)),abs(1-.5*eta))<1
    m=np.array([.02,.03]);d=np.array([.18,.04]);r=equations(m,d,tau,a,grad)
    first,second=physical_residuals(r,tau)
    assert np.allclose(first,m+d-eta*grad(m+d)-(m-d))
    assert np.allclose(second,m-d-eta*grad(m-d)-(m+d))


def test_zero_amplitude_is_separate_stationary_eigenpair_problem():
    with pytest.raises(ValueError):equations(np.zeros(2),np.zeros(2),2.,0.,lambda x:x)
    with pytest.raises(ValueError):physical_residuals(np.zeros(5),0.)
