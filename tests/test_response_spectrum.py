import numpy as np
import pytest
from neural_shrimp.response_spectrum import (response, difference_matrix,difference_operator,references,classify_count,matrix_free_probe)

@pytest.mark.parametrize('boundary',['free','periodic','causal'])
def test_response_equals_independent_dense_svd_ridge(boundary):
    maps=np.array([[[.4,.7],[0,1.4]],[[.8,0],[.2,1.1]]]); eta=.03
    k=difference_matrix(maps,8,.1,boundary).toarray(); u,s,vh=np.linalg.svd(k,full_matrices=False)
    rhs=np.zeros((16,2));rhs[8:10]=np.eye(2)
    v=(vh.T*(s/(s*s+eta*eta)))@u.T@rhs
    idx=4 if boundary=='causal' else 5
    r=response(maps,8,.1,eta,boundary)
    assert np.allclose(r['response_block'],v[idx*2:(idx+1)*2],rtol=1e-10,atol=1e-11)
    assert r['solve_qualified']

@pytest.mark.parametrize('boundary',['free','periodic','causal'])
def test_callback_transpose_and_dense_action(boundary):
    # Unequal rate ordering is essential: (I-DH)^T = I-HD, not I-DH.
    d=np.diag([.2,.7]); h=np.array([[2.,.9],[.9,1.]])
    m=np.eye(2)-d@h; mt=np.eye(2)-h@d
    op,c=difference_operator([lambda v:m@v],[lambda v:mt@v],2,8,.1,boundary)
    k=difference_matrix([m],8,.1,boundary); v=np.arange(k.shape[1])/7;w=np.arange(16)/9
    assert np.allclose(op@v,k@v);assert np.allclose(op.T@w,k.T@w)
    assert np.isclose(w@(op@v),(op.T@w)@v); assert c['JVP']>0 and c['VJP']>0

@pytest.mark.parametrize('boundary',['free','periodic'])
def test_nonnormal_oblique_stable_count(boundary):
    t=np.array([[1,4.],[0,1.]]); m=t@np.diag([.7,1.3])@np.linalg.inv(t)
    r=response([m],128,0,1e-6,boundary)
    assert abs(r['count']-1)<1e-7
    # Projector is oblique. Merely using singular values would get this wrong.
    assert abs(r['response_block'][0,1]+4)<1e-6


def test_fixed_causal_zero_regularizer_limit_is_not_cdf():
    r=response([[[1.2]]],20,0,1e-9,'causal')
    assert abs(r['count']-1)<1e-10  # unstable has zero infinite two-sided count
    assert abs(response([[[1.2]]],128,0,1e-6,'free')['count'])<1e-7


def test_window_and_regularizer_bias_are_visible():
    a=[[[1.002]]]
    short=response(a,8,0,1e-6,'free')['count'];long=response(a,1024,0,1e-6,'free')['count']
    assert abs(long)<abs(short);assert abs(short)>.1
    assert abs(response([[[.8]]],64,0,.1,'free')['count']-1)>.01

@pytest.mark.parametrize('value',[1.,-1.])
def test_neutral_is_unresolved_even_when_finite_response_exists(value):
    ref=references([[[value]]],(8,16)); r=response([[[value]]],32,0,1e-4,'periodic')
    assert not classify_count(ref,0,r['count'])['identity_assumptions_numerically_screened']
    assert r['solve_qualified']


def test_sign_angle_information_is_retained_but_not_in_counts():
    a=.8*np.array([[np.cos(.7),-np.sin(.7)],[np.sin(.7),np.cos(.7)]])
    ref=references([a],(8,16));assert np.allclose(sorted(ref['multiplier_angles']),[-.7,.7])
    assert np.allclose(sorted(ref['exponents_per_step']),[np.log(.8)]*2)
    rp=references([[[.8]]],(8,16));rm=references([[[-.8]]],(8,16))
    assert np.allclose(rp['exponents_per_step'],rm['exponents_per_step']);assert rm['multiplier_angles'][0]==np.pi


def test_singular_map_abstention():
    ref=references([np.diag([0.,.8])],(8,16));assert not ref['numerically_invertible_phases']
    assert not classify_count(ref,0,2.)['identity_assumptions_numerically_screened']


def test_noncommuting_period_product_reference():
    a=np.array([[[.7,.2],[0,1.1]],[[.9,0],[.2,1.]]]);ref=references(a,(128,512))
    assert np.allclose(ref['monodromy'],a[1]@a[0]);assert max(ref['eigen_residuals'])<1e-14
    assert np.max(abs(np.sort(ref['exponents_per_step'])-ref['qr_sorted_per_step']['512']))<.003


def test_matrix_free_reference_column_and_counters():
    a=[np.diag([.6,1.4])];r=response(a,16,0,.05,'free');mf=matrix_free_probe(a,16,0,.05,'free')
    assert np.allclose(mf['readout_column'],r['response_block'][:,0],atol=1e-8)
    assert mf['true_normal_residual_norm']<1e-8;assert mf['counts']['JVP']==16*mf['counts']['K']

@pytest.mark.parametrize('maps',[[],np.empty((1,0,0)),[[[np.nan]]],[[[np.inf]]]])
def test_invalid_empty_nonfinite(maps):
    with pytest.raises(ValueError):response(maps,8,0,.1,'free')

@pytest.mark.parametrize('eta',[0,-1,np.nan,np.inf])
def test_invalid_regularizer(eta):
    with pytest.raises(ValueError):response([[[.8]]],8,0,eta,'free')
