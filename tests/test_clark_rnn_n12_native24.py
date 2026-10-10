import importlib.util
from pathlib import Path
import numpy as np
import pytest
import neural_shrimp.clark_rnn_daughters as C
from neural_shrimp.clark_rnn import iterate

def runner():
    p=Path(__file__).parents[1]/'tools/explore_clark_rnn_n12_native24.py';s=importlib.util.spec_from_file_location('native12',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_full_native_pair_jacobian_and_positive_returns():
    rng=np.random.default_rng(805);j=rng.normal(size=(12,12))*.1;z=np.r_[rng.normal(size=24)*.3,.73];q=rng.normal(size=12);q/=np.linalg.norm(q)
    h,b,a=C.pair_fixed_amplitude(z,j,2.4,.02,q);eps=2e-6
    fd=np.column_stack([(C.pair_fixed_amplitude(z+eps*np.eye(25)[k],j,2.4,.02,q)[0]-C.pair_fixed_amplitude(z-eps*np.eye(25)[k],j,2.4,.02,q)[0])/(2*eps)for k in range(25)])
    np.testing.assert_allclose(b,fd,rtol=1e-6,atol=2e-8)
    ex=iterate(z[:12],j,2.4,z[-1],6)[0];ey=iterate(z[12:24],j,2.4,z[-1],6)[0]
    np.testing.assert_allclose(a['Px'],ex);np.testing.assert_allclose(a['Py'],ey)
    np.testing.assert_allclose(h[:12],ex-z[12:24]);np.testing.assert_allclose(h[12:24],ey-z[:12])
    assert h[-1]==pytest.approx(q@(z[:12]-z[12:24])-.04)

@pytest.mark.parametrize('c',[1.,-1.])
def test_native_supercritical_and_subcritical_daughters(monkeypatch,c):
    def normal(x,j,g,d,period):
        assert period==6
        end=np.array([-(1+d)*x[0]+c*x[0]**3,.2*x[1]]);u=np.zeros((2,4));u[0,0]=-(1+d)+3*c*x[0]**2;u[1,1]=.2;u[0,3]=-x[0]
        v=np.zeros((2,4,4));v[0,0,0]=6*c*x[0];v[0,0,3]=v[0,3,0]=-1
        return end,u,v,np.array([x]),np.array([u[:,:2]])
    monkeypatch.setattr(C,'jets',normal);amp=.03;z=np.array([amp,0,-amp,0,c*amp**2]);h,b,a=C.pair_fixed_amplitude(z,np.eye(2),1,amp,np.array([1.,0.]))
    assert np.max(abs(h))<1e-15 and np.linalg.matrix_rank(b)==5
    assert (max(abs(np.linalg.eigvals(a['first_y'][:,:2]@a['first_x'][:,:2])))<1)==(c>0)
    z[:4]=0;assert C.pair_fixed_amplitude(z,np.eye(2),1,amp,np.array([1.,0.]))[0][-1]==-2*amp

def test_native24_rejects_repeated_primitive12(tmp_path,monkeypatch):
    m=runner();w=m.Work(tmp_path);base=np.arange(144,dtype=float).reshape(12,12)/20;cycle=np.tile(base,(2,1))
    monkeypatch.setattr(w,'it',lambda *args:(cycle[0],np.eye(12)*.5,cycle,np.tile(np.eye(12),(24,1,1)),.001));monkeypatch.setattr(w,'emit',lambda *args:None)
    result=w.native(cycle[0],np.eye(12),2.,.8,'synthetic')
    assert set(result['proper_divisors'])=={'1','2','3','4','6','8','12'} and not result['passed'] and result['proper_divisors']['12']==0

def test_four_native_contexts_positive_twelve_shift(tmp_path,monkeypatch):
    m=runner();w=m.Work(tmp_path);x=np.arange(288,dtype=float).reshape(24,12)/50;y=np.roll(x,12,axis=0)
    def check(bad=False):
        cycles=iter([x,np.roll(x,-1,axis=0),-y if bad else y,np.roll(-y if bad else y,-1,axis=0)])
        monkeypatch.setattr(w,'native',lambda *args:dict(cycle=next(cycles),radius=.5,eigenvalues=np.arange(1,13)/24,passed=True))
        monkeypatch.setattr(w,'emit',lambda *args:None)
        return w.qualify(np.r_[x[0],y[0],.8],np.zeros(25),np.eye(12),2.,.1,np.eye(12)[0],'synthetic')
    row,saved=check();assert row['passed']and row['cross_identity_error']==0
    with pytest.raises(ValueError,match='qualification'):check(True)

def test_native_and_pair_caps_before_calls(tmp_path):
    m=runner();w=m.Work(tmp_path);w.native_calls=512
    with pytest.raises(RuntimeError,match='512 native'):w.it(np.zeros(12),np.eye(12),2.,.8,'synthetic')
    w.augmented_calls=2560
    with pytest.raises(RuntimeError,match='1280 paired'):w.pair(np.zeros(25),np.eye(12),2.,.1,np.eye(12)[0],'synthetic')
    assert w.calls==0

def test_profile_complete_native12_pair_solve():
    j=np.eye(12)*.2+np.ones((12,12))*.025;x=np.arange(1,13)/20
    h,b,_=C.pair_fixed_amplitude(np.r_[x,-x,.8],j,2.,.02,np.eye(12)[0],12);sol=np.linalg.solve(b,-h)
    assert np.isfinite(sol).all()and max(abs(b@sol+h))<1e-8

def test_birth_actual_mode_and_untransformed_doubled_parent():
    m=runner();x=np.arange(12)*.1;q=np.eye(12)[0];a=.001;z=np.r_[x+a*q,x-a*q,.8]
    assert m.birth_check(z,x,q,a,.8)['passed']and not m.birth_check(z,x,-q,a,.8)['passed']
    cycle=np.tile(np.arange(72).reshape(6,12),(2,1));assert m.paired_join([cycle,cycle],[cycle,cycle])['passed']
    assert not m.paired_join([cycle,cycle],[np.roll(cycle,1,axis=0),cycle])['passed']


def test_full_native12_pair_jacobian_and_positive_return():
 rng=np.random.default_rng(106);j=rng.normal(size=(12,12))*.09;z=np.r_[rng.normal(size=24)*.2,.74];q=np.eye(12)[0];eps=1e-5
 h,b,a=C.pair_fixed_amplitude(z,j,2.3,.02,q,12)
 fd=np.column_stack([(C.pair_fixed_amplitude(z+eps*np.eye(25)[k],j,2.3,.02,q,12)[0]-C.pair_fixed_amplitude(z-eps*np.eye(25)[k],j,2.3,.02,q,12)[0])/(2*eps)for k in range(25)])
 np.testing.assert_allclose(b,fd,rtol=2e-6,atol=1e-9)
 np.testing.assert_allclose(a['Px'],iterate(z[:12],j,2.3,.74,12)[0])
