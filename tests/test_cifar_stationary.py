import numpy as np
from neural_shrimp.cifar_stationary import stationary,positive_mode,SETTINGS


def test_stationary_newton_closes_full_quadratic_with_current_actions():
    H=np.array([[3.,.2],[.2,.6]]);root=np.array([1.,-.2]);x=root+np.array([.02,-.015]);calls=[]
    def g(z):calls.append(('g',z.copy()));return H@(z-root)
    def hv(z,v):calls.append(('h',z.copy(),v.copy()));return H@v
    answer,grad,report=stationary(x,g(x),g,hv)
    assert report['passed']and np.linalg.norm(answer-root)<1e-12
    assert all(np.array_equal(c[1],x)for c in calls if c[0]=='h')
    assert report['steps'][0]['forcing']<1e-12 and report['steps'][0]['accepted_index']==0


def test_nonlinear_stationarity_requires_actual_merit_backtracking():
    # Starting beside a shallow inflection makes a full Newton step overshoot.
    x=np.array([.7]);g=lambda z:z**3-z;h=lambda z,v:(3*z*z-1)*v
    answer,grad,r=stationary(x,g(x),g,h)
    assert any(not t['within_trust']for t in r['steps'][0]['trials'])
    assert r['relative_origin_distance']<=SETTINGS['trust_relative']+1e-14
    assert not r['passed'] # bounded attempt is honestly incomplete
    for row in r['steps']:
        if row['accepted_index']is not None:
            t=row['trials'][-1];assert t['accepted']and t['merit']<=t['armijo_bound']


def test_bad_fresh_linear_action_stops_before_any_trial():
    x=np.array([1.,0.]);g=np.array([.01,.005]);count=0;gradcalls=[]
    def hv(z,v):
        nonlocal count
        count+=1
        # Identity MINRES takes one action; independent final check fails.
        return v if count==1 else -v
    _,_,r=stationary(x,g,lambda z:gradcalls.append(z)or g,hv)
    assert r['status']=='true_forcing_failed'and not gradcalls and not r['steps'][0]['trials']


def test_positive_eigenmode_is_fresh_and_negative_mode_rejects():
    H=np.array([[4.,.1],[.1,.5]]);x=np.array([1.,2.]);calls=[]
    q,hq,r=positive_mode(x,[1.,1.],lambda z,v:calls.append(z.copy())or H@v)
    assert len(calls)==33 and all(np.array_equal(z,x)for z in calls)
    assert r['passed']and np.linalg.norm(hq-r['tau']*q)<1e-10
    assert not positive_mode(x,[1.,0.],lambda z,v:-v)[2]['passed']
