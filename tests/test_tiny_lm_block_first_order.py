import numpy as np
import pytest
from neural_shrimp import tiny_lm_block_first_order as H

def example(seed=18):
    rng=np.random.default_rng(seed);h=rng.normal(size=(16,4,4));h=(h+h.transpose(0,2,1))/5;eta=.15;J=np.eye(4)-eta*h;blocks=np.asarray([J[2*k+1]@J[2*k]for k in range(8)]);mean=blocks.mean(axis=0);u=rng.normal(size=4);v=[u]
    for _ in range(8):v.append(mean@v[-1])
    return h,eta,blocks,mean,np.asarray(v)

def test_independent_dense_product_derivative_and_call_identity():
    h,eta,blocks,mean,v=example();states=np.tile(np.arange(16)[:,None],(1,4));states=np.vstack([states,np.zeros(4)]);saved={};calls=[]
    def hvp(state,direction,j,k,phase):
        assert np.array_equal(state,states[2*k+phase]);calls.append((j,k,phase,direction.copy()));return h[2*k+phase]@direction
    answer=H.correct(states,v,np.asarray([blocks[k]@v[k]for k in range(8)]),eta,hvp,lambda k,x:saved.update({k:x.copy()}),lambda:None)
    # Independent explicit sum of product derivatives, with exact chronological order.
    direct=np.zeros(4)
    for k in range(8):direct+=np.linalg.matrix_power(mean,7-k)@(blocks[k]-mean)@np.linalg.matrix_power(mean,k)@v[0]
    np.testing.assert_allclose(answer,v[-1]+direct,rtol=1e-12,atol=1e-12)
    assert len(calls)==112 and [a[:3]for a in calls]==[(j,k,p)for j in range(1,8)for k in range(8)for p in range(2)]
    for a,b in zip(calls[::2],calls[1::2]):
        np.testing.assert_allclose(b[3],a[3]-eta*(h[2*a[1]]@a[3]),rtol=0,atol=0)
    for j in range(1,8):
        vectors=[a[3]for a in calls if a[:1]==(j,)and a[2]==0]
        assert all(np.array_equal(x,saved['first-order-corrections'][j])for x in vectors)

def test_expansion_has_second_order_remainder_and_constant_block_control():
    h,eta,blocks,mean,v=example(19);direct=sum((np.linalg.matrix_power(mean,7-k)@(blocks[k]-mean)@v[k]for k in range(8)),np.zeros(4))
    def ordered(e):
        z=v[0].copy()
        for b in blocks:z=(mean+e*(b-mean))@z
        return z
    errors=[np.linalg.norm(ordered(e)-v[-1]-e*direct)for e in(.01,.005,.0025)]
    assert all(3.8<a/b<4.2 for a,b in zip(errors,errors[1:]))
    # Same Jacobian at every step; no temporal variation means zero correction.
    states=np.zeros((17,4));hs=h[0];J=np.eye(4)-eta*hs;B=J@J;p=[v[0]]
    for _ in range(8):p.append(B@p[-1])
    p=np.asarray(p);out=H.correct(states,p,p[1:].copy(),eta,lambda s,u,*_:hs@u,lambda *_:None,lambda:None)
    np.testing.assert_array_equal(out,p[-1])

def test_failure_retains_partial_correction_and_complete_cost_cap():
    _,eta,blocks,_,v=example();saved={}
    def fail(*_):raise RuntimeError('native failure')
    with pytest.raises(RuntimeError):H.correct(np.zeros((17,4)),v,np.asarray([blocks[k]@v[k]for k in range(8)]),eta,fail,lambda k,x:saved.update({k:x.copy()}),lambda:None)
    assert saved['first-order-corrections'].shape==(2,4)and saved['correction-1-block-ends'].shape==(0,4)
    f=H.forecast([.1,.2],[126208]*2,.8502848679199815)
    assert f['seconds']>790 and f['bytes']==470974464 and H.TARGET_COUNTS=={'hvp':448}
