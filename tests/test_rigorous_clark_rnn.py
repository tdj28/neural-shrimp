import numpy as np
import pytest
from flint import arb,ctx
from neural_shrimp.rigorous_orbits import dyadic
from neural_shrimp.rigorous_clark_rnn import BallClarkRNN,check_rnn_orbit
from neural_shrimp.clark_rnn import step,jacobian

def data(j,g,d,p,x):return {'schema':'clark-rnn-c1-input-v1','j0':[[dyadic(v) for v in row] for row in j],'gain':dyadic(g),'delta':dyadic(d),'period':p,'approximation':[dyadic(v) for v in x]}

def test_native_map_and_jacobian_agree_with_independent_float_form():
    j=np.array([[.2,-.7],[.3,.4]]);x=np.array([.3,-.2]);d=data(j,1.7,.6,1,x)
    with ctx.workprec(256):
        model=BallClarkRNN(d);v,m=model.step([arb(t) for t in d['approximation']])
        np.testing.assert_allclose([float(y) for y in v],step(x,j,1.7,.6),atol=1e-16)
        np.testing.assert_allclose([[float(m[i,k]) for k in range(2)] for i in range(2)],jacobian(x,j,1.7,.6),atol=2e-16)

def test_known_primitive_four_and_smaller_divisor_rejection():
    a=2.
    for _ in range(15):a-=(a-2*np.tanh(a))/(1-2*(1-np.tanh(a)**2))
    d=data(np.array([[0.,-2.],[2.,0.]]),1.,1.,4,[a,a]);r=check_rnn_orbit(d)
    assert r['passed'] and [x['divisor'] for x in r['divisors']]==[1,2]
    assert not r['odd_companion_distinct'] # p4 cycle contains its own antipode.
    r=check_rnn_orbit(dict(d,period=8));assert not r['passed'] and not r['exact_statement']['primitive_period_excluded_divisors']

def test_unstable_origin_is_not_attracting_and_invalid_model_fails_closed():
    d=data(np.array([[2.]]),1.,1.,1,[0.]);r=check_rnn_orbit(d,max_power=8)
    assert r['exact_statement']['existence_and_local_uniqueness'] and not r['passed']
    for changed in [{'delta':'0'},{'gain':'-1'},{'schema':'wrong'},{'j0':[['1','2']]}]:
        with pytest.raises(ValueError):check_rnn_orbit(dict(d,**changed))

def test_stable_nonzero_fixedpoint_has_distinct_odd_companion():
    d=data(np.array([[2.]]),1.,1.,1,[1.9]);r=check_rnn_orbit(d)
    assert r['passed'] and r['odd_companion_distinct']
