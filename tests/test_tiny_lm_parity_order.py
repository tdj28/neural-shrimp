import numpy as np
import pytest
from neural_shrimp import tiny_lm_parity_order as H


def dense(ms,u,order):
    rows=[u.copy()]
    for k in order:rows.append(ms[k]@rows[-1])
    return np.asarray(rows)


def report(ms,u):
    v={k:dense(ms,u,order)[-1]for k,order in H.ORDERS.items()}
    return H.compare(u,v['P'],v['R'],v['Q'],v['T'],np.linalg.matrix_power(ms[0],16)@u,np.ones(len(u)))


def test_commuting_variation_can_fail_mean_but_zero_order_contrast():
    ms=np.array([np.diag([.5+.07*k,1.2-.025*k])for k in range(16)]);u=np.array([.6,.8]);r=report(ms,u)
    assert r['qualified']and r['even_norm']<1e-13
    assert np.linalg.norm(dense(ms,u,H.P)[-1]-np.linalg.matrix_power(ms.mean(0),16)@u)>.01


def test_reversal_even_contrast_not_bounded_by_odd():
    a=np.diag([.7,1.2]);b=np.array([[.95,.15],[.15,.85]]);half=[a,b]*4;ms=np.array(half+half[::-1]);r=report(ms,np.array([.6,.8]))
    assert r['qualified']and r['chronological_odd_norm']<1e-13 and r['even_norm']>1e-3


def test_actual_orders_and_full_retention():
    states=np.arange(34,dtype=float).reshape(17,2);u=np.array([.6,.8]);ms=np.array([np.array([[.6+.01*k,.02],[.02,1.1-.01*k]])for k in range(16)]);saved=[];calls=[]
    def hvp(s,v):
        k=int(s[0]/2);calls.append(k);return (v-ms[k]@v)/.2
    for key in H.ORDERS:
        out=H.propagate(states,u,.2,key,hvp,lambda a,ids:saved.append((a,ids)),lambda:None)
        assert np.allclose(out,dense(ms,u,H.ORDERS[key]),rtol=1e-14,atol=1e-15)
        assert saved[-1][1]==list(H.ORDERS[key])
    assert 2*sum(16*len(H.paths(r))for r in('original','quarter_step'))==224


def test_failure_keeps_partial_trace():
    saved=[];counter=[0]
    def bad(s,v):
        counter[0]+=1
        if counter[0]==3:return np.full(2,np.nan)
        return v
    with pytest.raises(ValueError):H.propagate(np.ones((17,2)),np.ones(2),.1,'Q',bad,lambda a,i:saved.append((a,i)),lambda:None)
    assert saved[0][0].shape==(3,2)and saved[0][1]==[0,2]


def test_replay_and_fixed_primary_gate():
    x=np.ones((17,3));assert H.replay(x,x)['passed'];y=x.copy();y[4,0]+=.01;assert not H.replay(y,x)['passed']
    cases=[dict(seed=s,rate_role=r,qualified=True,comparison=dict(even_relative_ordered=.2))for s in(28161,28162)for r in('original','quarter_step')]
    assert H.descriptive_gate(cases);cases[2]['qualified']=False;assert not H.descriptive_gate(cases)
    with pytest.raises(ValueError):H.descriptive_gate(cases[:-1])


def test_cold_cost_floor_full_retention_forecast_and_invalid_data():
    v=15776*8+128;f=H.forecast([.1,.2],[v,v],.4,1000)
    assert f['seconds']==2*224*.4+30 and f['vectors']==528 and f['bytes']==2*528*v+64*1024**2+1000
    assert H.forecast([.8,.2],[v,v],.4)['seconds']>240
    with pytest.raises(ValueError):H.forecast([.1,.2],[v,v-1],.4)
    with pytest.raises(ValueError):H.paths('changed')
