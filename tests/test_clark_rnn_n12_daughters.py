import numpy as np
import pytest
import neural_shrimp.clark_rnn_symmetry as S

def test_paired_amplitude_full_native_jacobian():
    rng=np.random.default_rng(481);j=rng.normal(size=(12,12))*.1;x=rng.normal(size=12)*.3;y=rng.normal(size=12)*.3;q=rng.normal(size=12);q/=np.linalg.norm(q);z=np.r_[x,y,.73]
    h,b,a=S.pair_fixed_amplitude(z,j,2.4,.02,q);eps=2e-6
    fd=np.column_stack([(S.pair_fixed_amplitude(z+eps*np.eye(25)[k],j,2.4,.02,q)[0]-S.pair_fixed_amplitude(z-eps*np.eye(25)[k],j,2.4,.02,q)[0])/(2*eps)for k in range(25)])
    np.testing.assert_allclose(b,fd,rtol=1e-6,atol=2e-8)
    np.testing.assert_allclose(h[:12],a['Gx']-y);np.testing.assert_allclose(h[12:24],a['Gy']-x)
    assert h[-1]==pytest.approx(q@(x-y)-.04)

@pytest.mark.parametrize('c',[1.,-1.])
def test_known_supercritical_and_subcritical_flip_daughters(monkeypatch,c):
    def normal(x,j,g,d,halfperiod):
        n=2;end=np.array([-(1+d)*x[0]+c*x[0]**3,.2*x[1]]);u=np.zeros((2,4));u[0,0]=-(1+d)+3*c*x[0]**2;u[1,1]=.2;u[0,3]=-x[0]
        v=np.zeros((2,4,4));v[0,0,0]=6*c*x[0];v[0,0,3]=v[0,3,0]=-1
        return end,u,v,np.array([x]),np.array([u[:,:2]])
    monkeypatch.setattr(S,'g_jets',normal)
    amp=.03;z=np.array([amp,0.,-amp,0.,c*amp**2]);h,b,a=S.pair_fixed_amplitude(z,np.eye(2),1.,amp,np.array([1.,0.]))
    assert np.max(abs(h))<1e-15 and np.linalg.matrix_rank(b)==5
    m=a['first_y'][:,:2]@a['first_x'][:,:2]
    assert (max(abs(np.linalg.eigvals(m)))<1)==(c>0)
    # A collapsed G fixed point cannot satisfy positive prescribed amplitude.
    z[:4]=0;hh,_,_=S.pair_fixed_amplitude(z,np.eye(2),1.,amp,np.array([1.,0.]));assert hh[-1]==-2*amp

def test_invalid_amplitude_or_projection_rejected():
    for amp,q in[(0.,np.eye(2)[0]),(.1,np.ones(2)),(.1,np.array([np.nan,0]))]:
        with pytest.raises(ValueError):S.pair_fixed_amplitude(np.zeros(5),np.eye(2),1.,amp,q)

def runner():
    import importlib.util
    from pathlib import Path
    p=Path(__file__).parents[1]/'tools/explore_clark_rnn_n12_daughters.py';s=importlib.util.spec_from_file_location('daughter_runner',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_profile_complete_paired_solve():
    j=np.eye(12)*.2+np.ones((12,12))*.025;x=np.arange(1,13)/20;q=np.eye(12)[0]
    h,b,_=S.pair_fixed_amplitude(np.r_[x,-x,.8],j,2.,.02,q)
    sol=np.linalg.solve(b,-h);assert np.isfinite(sol).all()and np.max(abs(b@sol+h))<1e-8

def test_birth_requires_actual_parent_mode_and_quadratic_midpoint():
    m=runner();x=np.arange(12)*.1;q=np.eye(12)[0];a=.001;z=np.r_[x+a*q,x-a*q,.8]
    assert m.birth_check(z,x,q,a,.8)['passed']
    bad=z.copy();bad[:24]+=1e-3;assert not m.birth_check(bad,x,q,a,.8)['passed']
    assert not m.birth_check(z,x,-q,a,.8)['passed']

def test_paired_return_cap_before_model(tmp_path):
    m=runner();w=m.Work(tmp_path);w.augmented_calls=2560
    with pytest.raises(RuntimeError,match='1280 paired'):w.pair(np.zeros(25),np.eye(12),2.,.1,np.eye(12)[0],'synthetic')
    assert w.calls==0


def test_complete_four_context_qualification_accepts_native_cycle_lists(tmp_path,monkeypatch):
    m=runner();w=m.Work(tmp_path);x=np.arange(72,dtype=float).reshape(6,12)/30
    y=-np.roll(x,3,axis=0);cycles=iter([x,np.roll(x,-1,axis=0),y,np.roll(y,-1,axis=0)])
    def native(*args):
        return dict(cycle=next(cycles),radius=.5,eigenvalues=np.arange(1,13)/24,passed=True)
    monkeypatch.setattr(w,'native',native);monkeypatch.setattr(w,'emit',lambda *args:None)
    row,saved=w.qualify(np.r_[x[0],y[0],.8],np.zeros(25),np.eye(12),2.,.1,np.eye(12)[0],'synthetic')
    assert row['passed'] and row['native_stability']=='attracting'
    assert row['cross_identity_error']==0 and len(saved)==2
