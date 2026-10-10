import numpy as np
import pytest
from neural_shrimp.tiny_lm_two_snapshot import forecast,candidate_gate

def test_order_and_information_budget():
    a=np.array([[1.,.2],[.2,.6]]);b=np.array([[.7,-.1],[-.1,.8]])
    t0=np.array([0.]);t1=np.array([1.]);u=np.array([.6,.8]);seen=[];saved={}
    def hvp(t,v):
        seen.append(int(t[0]));return (np.eye(2)-(b if t[0] else a))@v
    result=forecast(a@u,1.,t0,t1,hvp,saved.__setitem__,lambda:None)
    assert seen==[1,0]*7+[1]+[1]*15
    exact=u.copy()
    for h in range(16):
        exact=(a if h%2==0 else b)@exact
        np.testing.assert_allclose(result['alternating'][h],exact,atol=1e-15)
    np.testing.assert_allclose(result['latest'][-1],np.linalg.matrix_power(b,15)@a@u)
    assert set(saved)=={'alternating','latest'}

def test_interruption_retains_last_acquired_prediction():
    saved={};calls=0
    def h(t,v):
        nonlocal calls
        calls+=1
        if calls==3:raise TimeoutError('control')
        return v*.1
    with pytest.raises(TimeoutError):forecast(np.ones(2),1,np.zeros(2),np.ones(2),h,saved.__setitem__,lambda:None)
    assert saved['alternating'].shape==(3,2)

def test_gate_requires_improvement_over_equal_information_and_persistence():
    rows=[dict(seed=s,arm=256,direction='gradient',horizon=16,mechanism_qualified=True,alternating_vector_error=.01,latest_vector_error=.2,persistence_vector_error=.2)for s in [28161,28162]]
    assert candidate_gate(rows)
    rows[0]['latest_vector_error']=.05
    assert not candidate_gate(rows)
