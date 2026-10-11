"""Synthetic algebra only; no empirical record loading or decompositions."""
from pathlib import Path
import importlib.util
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('saved_plane',ROOT/'tools/explain_curvature_plane.py');D=importlib.util.module_from_spec(spec);spec.loader.exec_module(D)


def fixture(M,right,values,u=None,v=None):
    n=len(M);u=np.eye(n)[0] if u is None else u;v=np.eye(n)[1] if v is None else v
    return dict(target=np.asarray(M),phases=[dict(u=u,S0=.6*np.eye(n),E=.3*np.eye(n)),dict(u=v,S0=.8*np.eye(n),E=np.asarray(M)/.9-.8*np.eye(n))],pair_product=np.asarray(M).copy(),model_product=.48*np.eye(n),target_pair=dict(indices=np.array([0,1]),values=np.asarray(values[:2]).copy()),target_eigensystem=dict(values=np.asarray(values),right=np.asarray(right)),models=dict(rotating=dict(leading=dict(values=np.array([8.,9.])))),comparisons=dict(rotating=dict(normalized=7.)),status='analysis_complete',attribution=dict(qualified=False))


def test_exact_invariant_rotation_plane_and_bad_original_coefficients(monkeypatch):
    for name in ('eig','eigh','eigvals','svd','qr','norm','cond','inv'):
        monkeypatch.setattr(np.linalg,name,lambda *a,**k:(_ for _ in ()).throw(AssertionError('forbidden decomposition')))
    M=np.array([[.5,-.3,0],[.3,.5,0],[0,0,.1]])
    right=np.array([[1,1,0],[-1j,1j,0],[0,0,1]],dtype=complex);values=np.array([.5+.3j,.5-.3j,.1])
    r=D.diagnose(fixture(M,right,values));np.testing.assert_allclose(r['capture_fractions'],1.)
    assert r['leakage_frobenius']==0 and r['compression_pair_error']<1e-15 and r['original_pair_error']==7.
    np.testing.assert_allclose(np.sort_complex(r['compression_roots']),np.sort_complex(values[:2]))


def test_missing_target_eigenspace_does_not_select_another_pair():
    M=np.diag([.3,.2,2.,1.5]);right=np.eye(4)[:,[2,3,0,1]];values=np.array([2.,1.5,.3,.2])
    r=D.diagnose(fixture(M,right,values));np.testing.assert_array_equal(r['capture_fractions'],[0.,0.]);assert r['compression_pair_error']>.5
    np.testing.assert_array_equal(r['target_values'],values[:2])


def test_leakage_array_is_full_native_product_residual():
    M=np.array([[.5,0,0],[0,.3,0],[.2,.4,.1]]);r=D.diagnose(fixture(M,np.eye(3),np.array([.5,.3,.1])))
    np.testing.assert_allclose(r['leakage'],[[0,0],[0,0],[.2,.4]])
    assert r['leakage_frobenius']==pytest.approx(np.sqrt(.2**2+.4**2))


def test_eigenvector_scaling_and_top_vector_signs_preserve_energies():
    M=np.diag([2.,1.,.2]);right=np.array([[2.,0,0],[1.,3,0],[1.,4,1.]])
    a=fixture(M,right,np.array([2.,1.,.2]),u=np.array([-2.,0,0]),v=np.array([0,-3.,0]))
    r=D.diagnose(a);np.testing.assert_allclose(r['capture_fractions'],[5/6,9/25]);np.testing.assert_allclose(r['capture_fractions']+r['uncaptured_energy_fractions'],1)


def test_aligned_threshold_keeps_one_dimensional_abstention():
    a=fixture(np.eye(3),np.eye(3),np.ones(3),v=np.array([1.,1e-6,0]))
    r=D.diagnose(a);assert r['dimension']==1 and r['compression_pair_error'] is None and len(r['compression_roots'])==1


def test_nonorthogonal_top_vectors_use_fixed_gram_schmidt():
    a=fixture(np.diag([2.,1.,.1]),np.eye(3),np.array([2.,1.,.1]),v=np.array([.6,.8,0]))
    r=D.diagnose(a);np.testing.assert_allclose(r['Q'],np.eye(3)[:,:2]);assert r['orthogonality_residual']<1e-15


def test_stored_target_pair_binding_is_required():
    a=fixture(np.eye(3),np.eye(3),np.ones(3));a['target_pair']['values'][0]=2
    with pytest.raises(ValueError,match='target-pair binding'):D.diagnose(a)


def test_zero_top_vector_rejected():
    a=fixture(np.eye(3),np.eye(3),np.ones(3),u=np.zeros(3))
    with pytest.raises(ValueError,match='nonzero top'):D.diagnose(a)


def test_all_four_residual_action_terms_and_cancellation_identity():
    a=fixture(np.diag([1.4,.6,.2]),np.eye(3),np.array([1.4,.6,.2]))
    a['target']=a['target']+np.diag([.01,.02,.03])
    r=D.diagnose(a);d=r['residual_action'];assert set(d['terms'])=={'native_formation','first_phase_omission','second_phase_omission','interacting_omissions'}
    assert all(np.any(np.abs(t)>0) for t in d['terms'].values())
    np.testing.assert_allclose(d['sum'],d['actual_difference_action'],atol=3e-16)
    assert np.max(d['identity_residual_norms'])<3e-16
