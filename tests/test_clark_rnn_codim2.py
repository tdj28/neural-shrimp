import numpy as np
import pytest
import neural_shrimp.clark_rnn_codim2 as C

def test_native_augmented_jacobian_against_central_difference():
    j=np.array([[.2,-.3],[.7,.1]]);z=np.array([.1,-.2,1.7,.6,.8,.6,-.6,.8]);h,dh,_=C.fold_flip_augmented(z,j,3)
    eps=1e-6;fd=np.column_stack([(C.fold_flip_augmented(z+eps*np.eye(8)[k],j,3)[0]-C.fold_flip_augmented(z-eps*np.eye(8)[k],j,3)[0])/(2*eps) for k in range(8)])
    np.testing.assert_allclose(dh,fd,atol=2e-10,rtol=2e-8)

def test_known_decoupled_fold_flip_full_rank_and_normalizations(monkeypatch):
    def toy(x,j,g,d,p):
        a,b=x;end=np.array([a+g+a*a,-(1+d)*b+b**3]);u=np.array([[1+2*a,0,1,0],[0,-1-d+3*b*b,0,-b]])
        v=np.zeros((2,4,4));v[0,0,0]=2;v[1,1,1]=6*b;v[1,1,3]=v[1,3,1]=-1
        return end,u,v,np.array([x]),np.array([u[:,:2]])
    monkeypatch.setattr(C,'jets',toy)
    z=np.array([0.,0.,0.,0.,1.,0.,0.,1.]);h,dh,a=C.fold_flip_augmented(z,np.eye(2),1)
    np.testing.assert_array_equal(h,np.zeros(8));assert np.linalg.matrix_rank(dh)==8
    assert dh[4,6]==2 and dh[5,7]==0 and dh[2,4]==0 and dh[3,5]==-2
    bad=z.copy();bad[6:]=[1,0];assert np.linalg.norm(C.fold_flip_augmented(bad,np.eye(2),1)[0])>1

def test_dimension_fail_closed():
    with pytest.raises(ValueError):C.fold_flip_augmented(np.zeros(7),np.eye(2),1)
