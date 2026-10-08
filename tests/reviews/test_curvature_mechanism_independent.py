"""Independent analytic/adversarial supplied-array controls, no empirical data."""
import copy,math
import numpy as np
import pytest
from neural_shrimp import curvature_mechanism as C


def matrices(n=5,c=.4,a=.7,b=.4,alpha=1.4,beta=.9,p=None):
    u=np.eye(n)[0];v=np.array([c,math.sqrt(1-c*c)]+[0.]*(n-2));A=a*np.eye(n)-alpha*np.outer(u,u);B=b*np.eye(n)-beta*np.outer(v,v)
    rates=np.ones(n)if p is None else np.asarray(p,dtype=np.float64);root=np.sqrt(rates)
    H=np.stack([(np.eye(n)-S)/root[:,None]/root[None,:]for S in(A,B)])
    native=np.stack([root[:,None]*X/root[None,:]for X in(B@A,A@B)])
    return np.zeros((2,n)),H,rates,native,A,B

@pytest.mark.parametrize('n,c',[(2,0.),(2,.6),(5,-.45),(7,1.),(6,-1.)])
def test_full_spectrum_matches_direct_plane_characteristic_and_bulk(n,c):
    g,H,p,native,A,B=matrices(n=n,c=c);model=C.full_model_spectrum(n,.7,.4,1.4,.9,c);P=(B@A)[:2,:2]
    tr=float(P[0,0]+P[1,1]);det=float(P[0,0]*P[1,1]-P[0,1]*P[1,0]);np.testing.assert_allclose(model['trace'],tr,atol=2e-15);np.testing.assert_allclose(model['determinant'],det,atol=2e-15)
    # Complete ambient characteristic moments, including multiplicities.
    np.testing.assert_allclose(sum(model['values']),np.trace(B@A),atol=3e-15)
    np.testing.assert_allclose(sum(model['values']**2),np.trace((B@A)@(B@A)),atol=3e-15)
    assert len(model['values'])==n


def test_largest_algebraic_curvature_not_largest_magnitude():
    H=np.stack([np.diag([-8.,.1,1.8]),np.diag([-7.,.2,1.7])]);out=C.analyze_curvature(np.zeros((2,3)),H,np.ones(3))
    assert 'phases'in out
    for phase in out['phases']:np.testing.assert_allclose(np.abs(phase['u']),[0,0,1],atol=1e-15)
    assert out['phases'][0]['bulk_curvature']==pytest.approx(-3.95)


def test_exactly_four_decompositions_and_no_inverse_norm_cond(monkeypatch):
    seen=[];old_eigh=np.linalg.eigh;old_svd=np.linalg.svd;old_eig=C.eig
    def eigh(x,*a,**k):seen.append(('eigh',x.shape));return old_eigh(x,*a,**k)
    def svd(x,*a,**k):seen.append(('svd',x.shape));return old_svd(x,*a,**k)
    def eig(x,*a,**k):seen.append(('eig',x.shape));return old_eig(x,*a,**k)
    def forbidden(*a,**k):raise AssertionError('hidden inverse/norm/cond operation')
    monkeypatch.setattr(np.linalg,'eigh',eigh);monkeypatch.setattr(np.linalg,'svd',svd);monkeypatch.setattr(C,'eig',eig)
    for name in('norm','cond','inv','pinv','solve','lstsq'):monkeypatch.setattr(np.linalg,name,forbidden)
    data=matrices();out=C.analyze_curvature(*data[:4]);assert seen==[('eigh',(5,5)),('eigh',(5,5)),('eig',(5,5)),('svd',(5,5))]
    assert out['decomposition_counts']==dict(eigh=2,eig=1,svd=1)


@pytest.mark.parametrize('values,kind',[(np.array([1+1e-12j,1-1e-12j,.1]),'near_real_unresolved'),(np.array([2+0j,1+.5j,1-.5j]),'mixed_pair'),(np.array([1+2j,1+1j,.1]),'nonconjugate_pair')])
def test_fixed_selection_never_searches_for_favorable_pair(values,kind):
    p=C.leading_pair(values);assert p['kind']==kind and not p['qualified'];np.testing.assert_array_equal(p['indices'],[0,1])


def test_equal_cutoff_detected_without_discarding_multiplicity():
    p=C.leading_pair(np.array([.8+.6j,.8-.6j,1.,.1]));assert p['cutoff_modulus_tie'];assert set(p['indices'])=={0,2}


def test_nearly_aligned_is_explicit_limiting_case():
    c=math.sqrt(1-5e-11);m=C.full_model_spectrum(5,.7,.4,1.4,.9,c)
    assert m['span_case']=='aligned_one_dimensional' and len(m['values'])==5
    assert m['components'].count('bulk')==4 and m['complex_plane_modulus']is None


def test_delta_fit_bounds_independent_full_product_frobenius():
    g,H,p,native,A,B=matrices(n=5);H[0,2,2]+=.02;H[1,3,3]-=.03
    out=C.analyze_curvature(g,H,p)
    error=C.fro(out['pair_product']-out['model_product']);assert error<=out['delta_fit']+1e-14
    assert out['target_kind']=='Hessian_pair_product_only'and out['formation_error_frobenius']==0


def test_nonuniform_whitening_and_external_reverse_mismatch_fail_closed():
    data=matrices(p=[.03,.2,.5,2.,9.]);g,H,p,M,A,B=data;M[1,2,3]+=.002
    out=C.analyze_curvature(g,H,p,M);np.testing.assert_allclose(out['target'],B@A,atol=2e-15)
    assert out['formation_error_frobenius']<1e-14 and out['opposite_phase_formation_error_frobenius']>0
    assert out['status']=='unresolved'and not out['attribution']['qualified']


def test_small_native_mismatch_is_included_even_inside_agreement_gate():
    g,H,p,M,*_=matrices();M[0,0,0]+=1e-12;out=C.analyze_curvature(g,H,p,M)
    assert out['actual_pair_product_agreement']and out['formation_error_frobenius']>0
    np.testing.assert_allclose(out['delta_op_total'],out['delta_fit']+out['formation_error_frobenius'],rtol=0,atol=0)
    d=out['spectral_diagnostic'];np.testing.assert_allclose(d['rho_num'],d['delta_diag']+d['kappa2']*out['delta_op_total'],rtol=0,atol=0)


def test_residual_aware_radius_for_explicit_nonnormal_eigenbasis():
    # Analytic exact eigenbasis, deliberately supplied off-eigenvalue residual.
    M=np.array([[1.,3.],[0.,2.]]);V=np.array([[1.,3.],[0.,1.]],dtype=complex);left=np.array([[1.,0.],[-3.,1.]],dtype=complex);w=np.array([1.+1e-5,2.],dtype=complex)
    d=C.Decompositions();r=C.numerical_radius(M,M,.02,d,dict(values=w,left=left,right=V));smax=math.sqrt((11+math.sqrt(117))/2);smin=1/smax
    assert d.counts==dict(eigh=0,eig=0,svd=1)
    np.testing.assert_allclose([r['sigma_min'],r['sigma_max']],[smin,smax],rtol=2e-15)
    np.testing.assert_allclose(r['rho_num'],1e-5/smin+.02*smax/smin,rtol=1e-11)
    assert not r['eigen_residuals_qualified']


def test_numerical_return_preserved_on_svd_exception(monkeypatch):
    events=[]
    def fail(*a,**k):raise np.linalg.LinAlgError('synthetic SVD failure')
    monkeypatch.setattr(np.linalg,'svd',fail);data=matrices();out=C.analyze_curvature(*data[:4],observer=lambda k,p:events.append((k,copy.deepcopy(p))))
    assert out['status']=='unresolved'and out['decomposition_counts']==dict(eigh=2,eig=1,svd=1)
    assert len([v for k,v in events if k=='decomposition_return'])==3 and events[-1][0]=='decomposition_exception'
    assert out['decompositions'][-1]['status']=='raised'


def test_nonfinite_input_rejected_without_decomposition():
    g,H,p,*_=matrices();H[0,0,0]=np.nan;out=C.analyze_curvature(g,H,p)
    assert out['status']=='unresolved'and out['decomposition_counts']==dict(eigh=0,eig=0,svd=0)
