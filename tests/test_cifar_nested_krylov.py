import numpy as np
import pytest
from neural_shrimp.cifar_nested_krylov import extend,qualify
from neural_shrimp.cifar_reorthogonalized import orthogonalize


def prefix(A,g,m):
    V=[-g/np.linalg.norm(g)];W=[]
    for k in range(m):
        w=A@V[-1];W.append(w)
        if k+1<m:
            q,_=orthogonalize(np.column_stack(V),w);V.append(q/np.linalg.norm(q))
    return np.column_stack(V),np.column_stack(W)


def test_nested_actions_match_uninterrupted_indefinite_basis():
    A=np.diag(np.r_[np.linspace(-3,-.1,8),np.linspace(.1,10,16)]);g=np.random.default_rng(93).normal(size=24);V,W=prefix(A,g,8);V0=V.copy();W0=W.copy();calls=[]
    def action(v):calls.append(v.copy());return A@v
    p,hp,gram,r=extend(g,V,W,action,max_columns=24)
    full,_=prefix(A,g,24)
    assert len(calls)==r['new_basis_actions']==16 and r['total_columns']==24 and r['completed_fixed_extension']
    assert np.allclose(np.column_stack(calls),full[:,8:],atol=1e-12)and np.array_equal(V,V0)and np.array_equal(W,W0)
    assert r['orthogonality_qualified']and np.linalg.norm(g+hp)/np.linalg.norm(g)<1e-11
    assert qualify(g,hp,A@p,r,.6)['strict_forcing_pass']and not qualify(g,hp,-A@p,r,.6)['qualified']


def test_early_breakdown_never_claims_full_extension():
    g=np.arange(1.,9);v=(-g/np.linalg.norm(g))[:,None];calls=[]
    p,hp,gram,r=extend(g,v,2*v,lambda a:calls.append(a),max_columns=8)
    assert not calls and r['status']=='numerical_breakdown'and r['new_basis_actions']==0
    assert not qualify(g,hp,2*p,r,.6)['qualified']


def test_prefix_validation_and_fixed_shape_fail_closed():
    g=np.arange(1.,9);v=(-g/np.linalg.norm(g))[:,None]
    for V,W,m in[(v.astype(np.float32),v,8),(v,-v,1),(np.column_stack([v,v]),np.column_stack([v,v]),8),(-v,v,8),(v,np.full_like(v,np.nan),8)]:
        with pytest.raises(ValueError):extend(g,V,W,lambda a:a,max_columns=m)


def test_material_relative_to_qualified256_baseline_and_strict_separate():
    g=np.array([1.,0.]);r=dict(completed_fixed_extension=True,orthogonality_qualified=True,status='maximum_columns');hp=np.array([-.49,0.]);a=qualify(g,hp,hp,r,.65)
    assert a['material_improvement']and not a['strict_forcing_pass']
    r['completed_fixed_extension']=False;assert not qualify(g,hp,hp,r,.65)['qualified']
