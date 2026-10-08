"""Data-free curvature controls. No model/input/acquisition imports."""
import math
import numpy as np
import pytest
from neural_shrimp import curvature_mechanism as C


def supplied(n=4,c=.6,a=.5,b=.5,alpha=1.2,beta=1.2,rates=None):
    u=np.eye(n)[0];v=c*u+math.sqrt(1-c*c)*np.eye(n)[1]
    S0=a*np.eye(n)-alpha*np.outer(u,u);S1=b*np.eye(n)-beta*np.outer(v,v)
    p=np.ones(n) if rates is None else rates;root=np.sqrt(p)
    H=np.stack([(np.eye(n)-S)/root[:,None]/root[None,:] for S in (S0,S1)])
    actual=np.stack([root[:,None]*M/root[None,:] for M in (S1@S0,S0@S1)])
    return np.zeros((2,n)),H,p,actual


def test_exact_rank_one_fit_and_explicit_decomposition_budget(monkeypatch):
    def forbidden(*_,**__):raise AssertionError('hidden norm/condition decomposition')
    monkeypatch.setattr(np.linalg,'norm',forbidden);monkeypatch.setattr(np.linalg,'cond',forbidden)
    out=C.analyze_curvature(*supplied())
    assert out['status']=='analysis_complete',out['reasons']
    assert out['decomposition_counts']==dict(eigh=2,eig=1,svd=1)
    assert all(r['status']=='returned' for r in out['decompositions'])
    assert out['comparisons']['rotating']['normalized']<1e-14
    assert out['models']['rotating']['leading']['kind']=='complex_conjugate_pair'
    assert out['strength_only']['orientation_credit_for_modulus'] is False


def test_aligned_and_orthogonal_full_spectra_keep_bulk_multiplicities():
    aligned=C.full_model_spectrum(5,.5,.5,1.2,1.2,1.)
    orthogonal=C.full_model_spectrum(5,.5,.5,1.2,1.2,0.)
    np.testing.assert_allclose(np.sort(aligned['values'].real),[.25,.25,.25,.25,.49])
    np.testing.assert_allclose(np.sort(orthogonal['values'].real),[-.35,-.35,.25,.25,.25])
    assert aligned['span_case']=='aligned_one_dimensional'
    assert orthogonal['span_case']=='discriminant_unresolved'
    assert aligned['leading']['cutoff_modulus_tie']


def test_bulk_leading_cannot_select_favorable_plane():
    model=C.full_model_spectrum(4,2.,2.,1.,1.,.4)
    np.testing.assert_array_equal(model['leading']['values'],np.array([4.+0j,4.+0j]))
    assert all(model['components'][i]=='bulk' for i in model['leading']['indices'])


@pytest.mark.parametrize('values,kind',[(np.array([1.,.5+.2j,.5-.2j]),'mixed_pair'),(np.array([1.+2j,1.+1j,.1]),'nonconjugate_pair')])
def test_fixed_leading_two_modes_abstain_without_substitution(values,kind):
    p=C.leading_pair(values);assert not p['qualified'] and p['kind']==kind
    np.testing.assert_array_equal(p['indices'],[0,1])


def test_nonsymmetric_hessian_rejected_before_decomposition():
    g,H,p,M=supplied();H[0,0,1]+=.01
    out=C.analyze_curvature(g,H,p,M)
    assert out['status']=='unresolved' and 'nonsymmetric_hessian' in out['reasons'][0]
    assert out['decomposition_counts']==dict(eigh=0,eig=0,svd=0)
    assert out['inputs']['hessians'][0,0,1]==H[0,0,1]


def test_ambiguous_top_direction_retains_return_then_abstains():
    n=4;out=C.analyze_curvature(np.zeros((2,n)),np.stack([np.eye(n)]*2),np.ones(n))
    assert out['status']=='unresolved' and 'ambiguous_top_curvature' in out['reasons'][0]
    assert out['decomposition_counts']['eigh']==1 and out['decompositions'][0]['status']=='returned'


def test_complex_modulus_is_orientation_invariant():
    values=[C.full_model_spectrum(4,.5,.5,1.2,1.2,c) for c in (.3,.6,.8)]
    assert all(v['span_case']=='complex' for v in values)
    np.testing.assert_allclose([v['complex_plane_modulus'] for v in values],[.35]*3,atol=1e-15)
    assert len({round(float(np.angle(v['values'][0])),8) for v in values})==3


def test_eigenbasis_residual_term_detects_perturbed_numerical_vectors(monkeypatch):
    original=C.eig
    def perturbed(*a,**k):
        w,left,right=original(*a,**k);right[0,0]+=1e-5;return w,left,right
    monkeypatch.setattr(C,'eig',perturbed)
    out=C.analyze_curvature(*supplied());diag=out['spectral_diagnostic']
    assert diag['delta_diag']>1e-7 and not diag['eigen_residuals_qualified']
    expected=C.fro(diag['R_V'])/diag['sigma_min']
    assert diag['delta_diag']==expected
    assert out['status']=='unresolved' and not out['attribution']['qualified']


def test_external_matrix_formation_error_enters_total_operator_diagnostic():
    g,H,p,M=supplied();M[0]+=1e-5*np.eye(len(p))
    out=C.analyze_curvature(g,H,p,M)
    assert out['target_kind']=='supplied_external_native0' and not out['actual_pair_product_agreement']
    np.testing.assert_allclose(out['formation_error_frobenius'],2e-5,rtol=1e-10)
    assert out['delta_op_total']==out['delta_fit']+out['formation_error_frobenius']
    assert not out['attribution']['qualified']


def test_diagonal_rate_whitening_preserves_target_and_reports_metric():
    p=np.array([.2,.7,1.3,2.]);out=C.analyze_curvature(*supplied(rates=p))
    assert out['formation_error_frobenius']<1e-14 and out['opposite_phase_formation_error_frobenius']<1e-14
    assert out['whitening_condition']==math.sqrt(10.)


def test_decomposition_returns_are_observed_before_later_rejection():
    events=[];out=C.analyze_curvature(np.zeros((2,2)),np.stack([np.eye(2)]*2),np.ones(2),observer=lambda kind,row:events.append((kind,row['status'])))
    assert events==[('decomposition_start','started'),('decomposition_return','returned')]
    assert out['status']=='unresolved'


def test_unordered_pair_metric_keeps_fixed_target_modes():
    p=np.array([.2+.3j,.2-.3j]);row=C.pair_error(p,p[::-1],.4)
    assert row['normalized']==0 and np.array_equal(row['assignment'],[1,0])
