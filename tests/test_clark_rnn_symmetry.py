import numpy as np
import pytest
import neural_shrimp.clark_rnn_symmetry as S
from neural_shrimp.clark_rnn import iterate

def test_G_squared_matches_native_even_return():
    j=np.array([[.2,-.7],[.6,.1]]);x=np.array([.3,-.2]);g=1.8;d=.7
    y=-iterate(x,j,g,d,5)[0];z=-iterate(y,j,g,d,5)[0]
    np.testing.assert_allclose(z,iterate(x,j,g,d,10)[0],atol=1e-15)

def test_G_fold_augmented_derivatives():
    j=np.array([[.2,-.3],[.7,.1]]);z=np.array([.1,-.2,1.7,.6,.8,.6]);h,b,a=S.fold_family(z,j,3);eps=1e-6
    fd=np.column_stack([(S.fold_family(z+eps*np.eye(6)[k],j,3)[0]-S.fold_family(z-eps*np.eye(6)[k],j,3)[0])/(2*eps) for k in range(6)])
    np.testing.assert_allclose(b,fd,rtol=2e-8,atol=2e-10)
    assert np.max(abs(a['endpoint']+iterate(z[:2],j,z[2],z[3],3)[0]))<1e-15

def test_G_interaction_augmented_derivatives():
    j=np.array([[.2,-.3],[.7,.1]]);z=np.array([.1,-.2,1.7,.6,.8,.6,-.6,.8]);h,b,a=S.fold_flip(z,j,3);eps=1e-6
    fd=np.column_stack([(S.fold_flip(z+eps*np.eye(8)[k],j,3)[0]-S.fold_flip(z-eps*np.eye(8)[k],j,3)[0])/(2*eps) for k in range(8)])
    np.testing.assert_allclose(b,fd,rtol=2e-8,atol=2e-10)

def test_known_G_fold_flip_and_native_square(monkeypatch):
    def toy(x,j,g,d,p):
        a,b=x;end=np.array([a+g+a*a,-(1+d)*b+b**3]);u=np.array([[1+2*a,0,1,0],[0,-1-d+3*b*b,0,-b]])
        v=np.zeros((2,4,4));v[0,0,0]=2;v[1,1,1]=6*b;v[1,1,3]=v[1,3,1]=-1
        return end,u,v,np.array([x]),np.array([u[:,:2]])
    monkeypatch.setattr(S,'g_jets',toy)
    z=np.array([0.,0.,0.,0.,1.,0.,0.,1.]);h,b,a=S.fold_flip(z,np.eye(2),1)
    np.testing.assert_array_equal(h,np.zeros(8));assert np.linalg.matrix_rank(b)==8
    np.testing.assert_array_equal(a['monodromy']@a['monodromy'],np.eye(2))

def test_dimension_fail_closed():
    with pytest.raises(ValueError):S.fold_family(np.zeros(5),np.eye(2))
    with pytest.raises(ValueError):S.fold_flip(np.zeros(7),np.eye(2))


def test_driver_retains_complete_synthetic_interaction(tmp_path):
    import importlib.util
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'tools/continue_clark_rnn_p10_symmetric_fold.py'
    spec=importlib.util.spec_from_file_location('symdriver',path);D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)
    w=D.Work(tmp_path);saved=[];w.emit=lambda k,v:saved.append((k,v))
    x=np.array([.1,.2,.3,.4]);m=np.diag([1.,-1.,.2,.3]);half=np.arange(20).reshape(5,4)/20+1.;cycle=np.r_[half,-half]
    a={'theta':x,'endpoint':x,'monodromy':m,'native_monodromy':m@m,'native_endpoint':x,'cycle':cycle}
    w.full=lambda *args:(np.zeros(9),np.zeros((9,10)),a,0.)
    w.returned=lambda *args:(np.zeros(14),np.eye(14),a,0.)
    y=np.r_[x,15.5,.45,[1.,0,0,0]]/D.COL
    w.correct=lambda *args:((y.copy(),np.zeros((9,10)),a),'closed')
    out=w.localize(y,np.eye(4),'toy',np.eye(10)[4])
    assert out['status']=='numerical_G_plus_minus_interaction' and out['native_plus_one_count_1e6']==2
    assert saved[-1][0]=='interaction_result' and out['native_square_relative_defect']==0
