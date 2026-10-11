import numpy as np
import torch
import pytest
from neural_shrimp.cifar import MLP,Problem,as_tensor
from neural_shrimp.cifar_residual_adjoint import residual_adjoint

def tiny():
 m=MLP(3,2,2);theta=as_tensor(.7*np.sin(np.arange(m.n_parameters)+.31));x=np.cos(np.arange(15).reshape(5,3)*.27);y=1.4+np.sin(np.arange(10).reshape(5,2)*.43);p=Problem(m,x,y,.03);return p,theta,.23

def test_full_autodiff_residual_jacobian_transpose_and_pairing():
 p,t,eta=tiny();u=p.update(t,eta);v=as_tensor(np.cos(np.arange(len(t))*.6));w=as_tensor(np.sin(np.arange(len(t))*.4+.1))
 full=torch.func.jacrev(lambda z:p.update(p.update(z,eta),eta)-z)(t)
 assert torch.linalg.matrix_norm(full-full.T)>1e-3
 av=residual_adjoint(p,t,u,v,eta)
 torch.testing.assert_close(av,full.T@v,rtol=2e-12,atol=2e-13)
 torch.testing.assert_close(torch.dot(av,w),torch.dot(v,full@w),rtol=2e-12,atol=2e-13)
 wrong=p.jvp_two_step(t,v,eta,theta_next=u)-v
 assert torch.linalg.vector_norm(wrong-av)>1e-4

def test_merit_gradient_matches_direct_autodiff_and_directional_difference():
 p,t,eta=tiny();u=p.update(t,eta);r=p.update(u,eta)-t
 g=residual_adjoint(p,t,u,r,eta)
 def merit(z):
  r=p.update(p.update(z,eta),eta)-z;return torch.dot(r,r)/2
 torch.testing.assert_close(g,torch.func.grad(merit)(t),rtol=2e-12,atol=2e-13)
 w=as_tensor(np.cos(np.arange(len(t))*.71));eps=1e-5
 fd=(merit(t+eps*w)-merit(t-eps*w))/(2*eps)
 torch.testing.assert_close(torch.dot(g,w),fd,rtol=2e-8,atol=2e-9)

def test_exactly_two_hvps_reverse_phase_order_no_hidden_map_calls():
 p,t,eta=tiny();u=p.update(t,eta);calls=[];old=p.hvp
 def hvp(x,v):calls.append(x.clone());return old(x,v)
 p.hvp=hvp
 residual_adjoint(p,t,u,torch.ones_like(t),eta)
 assert len(calls)==2 and torch.equal(calls[0],u) and torch.equal(calls[1],t)

@pytest.mark.parametrize('bad',[0.,-1.,float('nan'),[.2,.3]])
def test_nonpositive_or_nonscalar_rate_rejected(bad):
 p,t,eta=tiny()
 with pytest.raises(ValueError):residual_adjoint(p,t,t,t,bad)

def test_float32_or_nonfinite_rejected():
 p,t,eta=tiny()
 with pytest.raises(ValueError):residual_adjoint(p,t,t,t.float(),eta)
 bad=t.clone();bad[0]=float('nan')
 with pytest.raises(ValueError):residual_adjoint(p,t,t,bad,eta)
