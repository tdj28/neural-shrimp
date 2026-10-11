import numpy as np
import pytest
from neural_shrimp.cifar_reorthogonalized import solve,qualify,orthogonalize

def test_full_indefinite_linear_system_and_physical_action():
    A=np.diag(np.r_[np.linspace(-3,-.1,12),np.linspace(.01,200,20)]);g=np.random.default_rng(15).normal(size=32);rows=[]
    p,hp,G,r=solve(g,-g/np.linalg.norm(g),lambda x:A@x,event=lambda k,v:rows.append(v))
    assert len(rows)==r['actions']==32 and r['orthogonality_qualified']and np.linalg.norm(A@p+g)/np.linalg.norm(g)<1e-9
    assert np.linalg.norm(hp-A@p)/np.linalg.norm(g)<1e-10
    assert not qualify(g,hp,A@p,r,.78)['qualified'] # not equal256 actions

def test_exact_breakdown_retained_without_false_full_budget_claim():
    g=np.arange(1.,9);p,hp,G,r=solve(g,-g/np.linalg.norm(g),lambda x:2*x)
    assert r['actions']==1 and r['status']=='numerical_breakdown'and not r['equal256_action_count'];assert np.linalg.norm(g+2*p)<1e-12

def test_two_passes_remove_nearly_aligned_components():
    Q,_=np.linalg.qr(np.random.default_rng(42).normal(size=(60,12)));w=Q@np.arange(12.)+np.random.default_rng(2).normal(size=60)*1e-8
    q,c=orthogonalize(Q,w);assert len(c)==2 and np.max(abs(Q.T@q))<1e-20 and np.linalg.norm(q)>0

def test_wrong_first_direction_and_actual_action_rejected():
    g=np.arange(1.,9)
    with pytest.raises(ValueError):solve(g,g/np.linalg.norm(g),lambda x:x)
    report=dict(equal256_action_count=True,orthogonality_qualified=True,status='maximum_actions')
    assert qualify(g,-g,-g,report,.78)['strict_forcing_pass']
    assert not qualify(g,-g,g,report,.78)['qualified']
