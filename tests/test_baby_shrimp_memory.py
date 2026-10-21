import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from baby_shrimp_memory import MemoryOrbit, qualified_ritz


def test_memory_map_and_derivative_are_same_intervention():
    import torch
    class Base:
        shape=(1,2); n=2
        start=torch.tensor([[[.2,.3]]],dtype=torch.float64)
        def f(self,x): return x*x
        def jvp_rows(self,x,v): return 2*x*v
        def jacobian(self,x): return np.diag(2*x.numpy().ravel())
    Base.torch=torch
    base=Base(); x=base.start; v=torch.tensor([[[.7,-.1]]],dtype=torch.float64)
    zero=MemoryOrbit(base,0)
    assert torch.equal(zero.f(x),base.f(x))
    assert torch.equal(zero.jvp_rows(x,v),base.jvp_rows(x,v))
    for beta in [-.01,.01]:
        orb=MemoryOrbit(base,beta); h=1e-5
        fd=(orb.f(x+h*v)-orb.f(x-h*v))/(2*h)
        assert np.allclose(fd,orb.jvp_rows(x,v),rtol=1e-10,atol=1e-11)
        assert np.allclose(orb.jacobian(x)@v.numpy().ravel(),fd.numpy().ravel())
        assert torch.equal(orb.iterate(x,3),orb.f(orb.f(orb.f(x))))


def test_ritz_nonnormal_small_opposite_sign_pair():
    rng=np.random.default_rng(4)
    transform=np.eye(9)+.1*rng.normal(size=(9,9))
    values=np.array([.00052,-.000057,3e-12,1e-13,1e-14,1e-15,1e-16,1e-17,1e-18])
    m=transform@np.diag(values)@np.linalg.inv(transform)
    result=qualified_ritz(lambda q:m@q,rng.normal(size=(9,4)))
    assert np.allclose(result['values'][:2],values[:2],rtol=1e-9,atol=1e-14)
    assert max(result['relative_residuals'])<1e-6


def test_ritz_refuses_unisolated_pair():
    m=np.diag([.5,-.4,.3,.2,.1])
    with pytest.raises(ValueError,match='separation'):
        qualified_ritz(lambda q:m@q,np.eye(5))
