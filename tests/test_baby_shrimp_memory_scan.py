import sys
from pathlib import Path
import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import baby_shrimp_memory_scan as scan


def test_scan_memory_matches_polynomial_map_and_tangent(monkeypatch):
    def step(self,x,temps): return x*x, torch.softmax(x[:,0],dim=-1)
    def jvp(self,x,temps,v):
        y,p=step(self,x,temps); return y,p,2*x*v
    monkeypatch.setattr(scan.soft.SoftLoop,"step",step)
    monkeypatch.setattr(scan.soft.SoftLoop,"step_jvp",jvp)
    obj=object.__new__(scan.MemoryScanLoop)
    x=torch.tensor([[[.2,-.3]],[[.4,.1]]],dtype=torch.float64)
    v=torch.tensor([[[.7,.2]],[[-.1,.3]]],dtype=torch.float64)
    for beta in [0.,.006448531026086656]:
        obj.beta=beta
        y,p,jv=obj.step_jvp(x,[1.,1.],v)
        h=1e-5
        fd=(obj.step(x+h*v,[1.,1.])[0]-obj.step(x-h*v,[1.,1.])[0])/(2*h)
        assert torch.allclose(jv,fd,rtol=1e-10,atol=1e-11)
        assert torch.equal(p,step(obj,x,None)[1])
        assert torch.equal(y,obj.step(x,[1.,1.])[0])
        assert torch.allclose(y, (1-beta)*x*x+beta*x, rtol=0, atol=4e-17)
        if beta == 0:
            assert torch.equal(y,x*x)


@pytest.mark.parametrize("beta",[float('nan'),-.01,.051,float('inf')])
def test_scan_memory_rejects_invalid_beta(beta):
    with pytest.raises(ValueError,match="finite"):
        scan.MemoryScanLoop(beta=beta)


def test_scan_memory_rejects_longer_window(monkeypatch):
    def init(self,*a,**kw): self.K=2
    monkeypatch.setattr(scan.soft.SoftLoop,"__init__",init)
    with pytest.raises(ValueError,match="one-token"):
        scan.MemoryScanLoop(beta=.01)


def test_reference_applies_retention_once_and_restores_scan(monkeypatch):
    monkeypatch.setattr(scan.soft.SoftLoop,"step",lambda self,x,t:(x*x,None))
    loop=object.__new__(scan.MemoryScanLoop); loop.beta=.006448531026086656
    class Base:
        shape=(1,2); n=2; start=None
        def f(self,x): return loop.step(x,None)[0]
    Base.torch=torch
    reference=scan.MemoryOrbit(Base(),loop.beta)
    x=torch.tensor([[[.2,-.3]]],dtype=torch.float64)
    expected=scan.unretained_reference_step(loop,reference,x)
    assert torch.equal(expected,loop.step(x,None)[0])
    assert loop.beta==.006448531026086656
    assert not torch.equal(reference.f(x),expected)  # erroneous double mixing
    class Failure:
        def f(self,x): raise RuntimeError('reference failure')
    with pytest.raises(RuntimeError): scan.unretained_reference_step(loop,Failure(),x)
    assert loop.beta==.006448531026086656
