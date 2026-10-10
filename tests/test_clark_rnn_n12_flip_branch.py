import importlib.util
from pathlib import Path
import numpy as np
import pytest
from neural_shrimp.clark_rnn_symmetry import critical_fixed_gain

def runner():
    p = Path(__file__).parents[1]/'tools/explore_clark_rnn_n12_flip_branch.py'
    spec = importlib.util.spec_from_file_location('branch_runner', p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def test_critical_family_gain_derivative_all_equations():
    m=runner();rng=np.random.default_rng(312);j=rng.normal(size=(12,12))*.1
    q=rng.normal(size=12);q/=np.linalg.norm(q);z=np.r_[rng.normal(size=12)*.3,.73,q];g=2.4;eps=2e-6
    _,_,a=critical_fixed_gain(z,j,g,-1)
    fd=(critical_fixed_gain(z,j,g+eps,-1)[0]-critical_fixed_gain(z,j,g-eps,-1)[0])/(2*eps)
    np.testing.assert_allclose(m.gain_derivative(a,q),fd,rtol=1e-6,atol=2e-8)

@pytest.mark.parametrize('cubic',[-.4,0.,.3])
def test_known_flip_normal_form_including_degenerate_coefficient(cubic):
    m=runner();gm=np.eye(12)*.2;gm[0,0]=-1;q=np.eye(12)[0];second=np.zeros((12,14,14));second[0,0,13]=2.;second[0,13,0]=2.
    for sign in [-1.,1.]:
        alpha,c,*_=m.flip_coefficients(gm,second,np.zeros(12),sign*q,sign*q,sign*6*cubic*q)
        assert alpha==2. and c==pytest.approx(cubic)

def test_native_join_rejects_sign_phase_or_parameter_displacement():
    m=runner();a=np.arange(72).reshape(6,12)/12
    assert m.native_join(a,a,.7,.7)['passed']
    assert not m.native_join(a,-a,.7,.7)['passed']
    assert not m.native_join(a,np.roll(a,1,axis=0),.7,.7)['passed']
    assert not m.native_join(a,a,.7,.70001)['passed']

def test_numerical_cubic_bracket_keeps_exact_zero():
    m=runner();rows=[{'gain':i,'cubic':c}for i,c in enumerate([.2,0.,-.1,-.2])]
    assert [(x['left'],x['right'])for x in m.cubic_crossings(rows)]==[(0,1),(1,2)]

def test_continued_mode_rejects_sign_reversal_before_model(tmp_path):
    m=runner();w=m.Work(tmp_path);z=np.r_[np.ones(12),.7,np.eye(12)[0]];bad=z.copy();bad[13:]*=-1
    with pytest.raises(ValueError,match='continued eigenvector'):
        w.correct_critical(bad,z,np.eye(12),2.,'synthetic')
    assert w.calls==0

def test_caps_precede_model_calls(tmp_path):
    m=runner();w=m.Work(tmp_path);x=np.ones(12);j=np.eye(12);q=np.eye(12)[0]
    w.native_calls=256
    with pytest.raises(RuntimeError,match='256 native'):w.it(x,j,2.,.8,'synthetic')
    w.augmented_calls=1024
    with pytest.raises(RuntimeError,match='1024 half'):w.half(np.r_[x,.8,q],j,2.,-1,'synthetic')
    w.third_calls=43
    with pytest.raises(RuntimeError,match='43 third'):w.third(x,q,j,2.,.8,'synthetic')
    assert w.calls==0

def test_flip_coefficient_includes_fixed_point_and_quadratic_corrections():
    m=runner();gm=np.eye(12)*.2;gm[0,0]=-1;q=np.eye(12)[0];second=np.zeros((12,14,14));second[0,0,0]=1.4;second[0,0,13]=2.
    param=.3*q;third=-2.4*q
    alpha,c,*_=m.flip_coefficients(gm,second,param,q,q,third)
    assert alpha==pytest.approx(2.+.7*.3)
    assert c==pytest.approx(-.4+.7**2)

def test_unrelated_full_size_profile_linear_solve():
    j=np.eye(12)*.2+np.ones((12,12))*.025;x=np.arange(1,13)/20;q=np.eye(12)[0]
    h,b,_=critical_fixed_gain(np.r_[x,.8,q],j,2.,-1)
    solution=np.linalg.solve(b,-h)
    assert np.isfinite(solution).all()
    assert np.max(np.abs(b@solution+h))<1e-8
