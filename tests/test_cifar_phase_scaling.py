import numpy as np
import pytest
from neural_shrimp.cifar_phase_scaling import transform,scales,physical,dual,direction


def test_orthogonal_transform_and_nonscalar_symmetric_congruence():
    rng=np.random.default_rng(541913);x=rng.normal(size=(6,6));x=(x+x.T)/2;y=rng.normal(size=(6,6));y=(y+y.T)/2
    a=np.block([[x,-np.eye(6)],[-np.eye(6),y]]);q=np.column_stack([transform(v)for v in np.eye(12)]);r=(x+y)/2;s=(x-y)/2
    assert np.allclose(q.T@a@q,np.block([[r-np.eye(6),s],[s,r+np.eye(6)]]),rtol=0,atol=2e-15)
    w,_=scales(.01,.03);p=np.column_stack([physical(v,w)for v in np.eye(12)]);pt=np.column_stack([dual(v,w)for v in np.eye(12)])
    assert np.allclose(pt,p.T,rtol=0,atol=1e-15)
    assert np.allclose(pt@a@p,(pt@a@p).T,rtol=1e-14,atol=1e-11)
    v=rng.normal(size=12);assert np.allclose(transform(transform(v)),v,rtol=1e-14,atol=1e-15)


def test_ridge_only_balance_and_true_physical_elimination():
    eta,lam=.00827753,.01;d=eta*lam;n=7;j=(1-d)*np.eye(n);a=np.block([[j,-np.eye(n)],[-np.eye(n),j]]);w,_=scales(eta,lam);p=np.column_stack([physical(v,w)for v in np.eye(2*n)])
    assert np.allclose(p.T@a@p,np.diag([-1]*n+[1]*n),rtol=1e-11,atol=1e-11)
    r=np.arange(1.,n+1)*1e-5;step,ap,z,az,q,info=direction(j.__matmul__,j.__matmul__,r,eta,lam,iterations=8)
    assert info['forcing_pass']and info['true_forcing']<1e-8
    assert np.allclose(ap,(j@j-np.eye(n))@step,atol=1e-14)
    assert np.allclose(az,a@z,atol=1e-14)


def test_invalid_scaling_and_true_forcing_not_solver_status():
    for args in [(0,.01),(.01,-1),(3,1),(float('nan'),.1)]:
        with pytest.raises(ValueError):scales(*args)
    rng=np.random.default_rng(751);x=rng.normal(size=(5,5));x=x+x.T;y=rng.normal(size=(5,5));y=y+y.T;r=np.ones(5)
    p,ap,z,az,q,rec=direction(x.__matmul__,y.__matmul__,r,.1,.2,iterations=1)
    assert np.isclose(rec['true_forcing'],np.linalg.norm(r+ap)/np.linalg.norm(r))
    assert rec['forcing_pass']==(rec['true_forcing']<=.5)
