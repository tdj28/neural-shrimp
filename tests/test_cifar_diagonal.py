import numpy as np
import torch
import pytest
from neural_shrimp.cifar import MLP,Problem,as_tensor
from neural_shrimp.cifar_diagonal import loss_hessian_diagonal,scaling
from neural_shrimp.cifar_linear_diagnostic import solve_direction

@pytest.mark.parametrize('d,h,k,n,lam',[(3,2,2,5,.03),(2,3,1,4,.01)])
def test_exact_mse_diagonal_matches_autograd_with_nonzero_residual(d,h,k,n,lam):
 model=MLP(d,h,k);theta=np.sin(np.arange(model.n_parameters)+.3)*.7;x=np.cos(np.arange(n*d).reshape(n,d)*.31);y=np.sin(np.arange(n*k).reshape(n,k)*.7)+2.;p=Problem(model,x,y,lam);full=torch.func.hessian(p.loss)(as_tensor(theta)).numpy();diag,aux=loss_hessian_diagonal(theta,x,y,d,h,k,lam)
 np.testing.assert_allclose(diag,np.diag(full),rtol=3e-13,atol=3e-13);assert np.max(np.abs(aux['error']))>1.
 zero,_=loss_hessian_diagonal(theta,x,y,d,h,k,0.);np.testing.assert_allclose(diag-zero,lam,rtol=1e-12,atol=1e-14)

def test_output_bias_and_packing():
 model=MLP(2,2,3);t=np.zeros(model.n_parameters);x=np.ones((4,2));y=np.ones((4,3));d,a=loss_hessian_diagonal(t,x,y,2,2,3,.1)
 np.testing.assert_array_equal(d[-3:],np.full(3,1.1));np.testing.assert_array_equal(d[:-3],np.full(len(d)-3,.1))

def test_proxy_is_not_exact_product_diagonal():
 h0=np.array([[2.,3.],[3.,4.]]);h1=np.array([[1.,5.],[5.,6.]]);eta=.01
 inv,proxy,r=scaling(np.diag(h0),np.diag(h1),eta,.1);actual=np.diag((np.eye(2)-eta*h1)@(np.eye(2)-eta*h0)-np.eye(2));assert not np.allclose(actual,proxy);assert np.all(inv>0)

def test_ridge_floor_and_invalid_scaling():
 inv,proxy,r=scaling(np.zeros(3),np.zeros(3),.01,.02);np.testing.assert_allclose(inv,1/r['ridge_floor']);assert r['floor_coordinates']==3
 for eta,lam in[(0.,.01),(.01,0.),(float('nan'),.01)]:
  with pytest.raises(ValueError):scaling(np.ones(3),np.ones(3),eta,lam)

def test_right_scaled_solve_preserves_physical_residual():
 a=np.diag([-.01,-1.,-10.]);scale=np.array([100.,1.,.1]);r=np.array([1.,2.,3.]);saved=[]
 y,ap,error,linear=solve_direction(lambda v:a@(scale*v),r,restart=3,cycles=1,rtol=1e-10,forcing_limit=.5,max_matvec=5,explicit_zero_start=True,save_direction=lambda y,i:saved.append((y.copy(),scale*y)))
 p=scale*y;np.testing.assert_allclose(a@p,ap,atol=1e-14);np.testing.assert_allclose(error,a@p+r,atol=1e-14);assert linear['forcing_test_pass']and linear['descent_direction']and saved
