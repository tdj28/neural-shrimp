import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/reviews'))
from cifar_remainder_span_math import diagnose


def fixture(b):
 V=np.eye(5)[:,:2];W=V*np.array([2.,3.]);x=np.array([1.,0,0,0,0]);g=np.array([.01,.02,0,0,0]);p=np.array([-.01,-.02,0,0,0]);hp=W@p[:2];ts=[dict(alpha=a,state=x+a*p,gradient=g+a*hp+a*a*b)for a in[.125,.0625,.03125,.015625]];return V,W,x,g,p,hp,x.copy(),ts


def test_representable_and_irreducible_quadratic_terms():
 a=fixture(np.array([.3,.4,.5,0,0]));r=diagnose(*a);s=r['summary'];assert s['numerical_rank']==2 and s['fixed_reference_alpha']==1/64
 for row,v in zip(s['rows'],r['vectors']):
  assert row['irreducible_fraction']==pytest.approx(1/np.sqrt(2),abs=1e-11)
  np.testing.assert_allclose(v['acceleration'],[-.15,-.4/3,0,0,0],atol=1e-11)
  assert row['assembled_projection_error']<1e-11
 assert max(v['coefficient_relative_drift']for v in s['reference_cross_checks'])<1e-10


def test_full_image_and_basis_coordinate_invariance():
 a=fixture(np.array([.3,.4,0,0,0]));r=diagnose(*a);T=np.array([[2.,1.],[0.,.5]]);b=list(a);b[0]=a[0]@T;b[1]=a[1]@T;s=diagnose(*b)
 assert max(v['irreducible_fraction']for v in r['summary']['rows'])<1e-10
 for u,v in zip(r['vectors'],s['vectors']):np.testing.assert_allclose(u['acceleration'],v['acceleration'],atol=1e-10)


def test_nonconstant_remainder_is_not_silently_extrapolated():
 a=list(fixture(np.array([.3,.4,0,0,0])));a[-1][0]['gradient']=a[-1][0]['gradient']+np.array([0,0,.125**2,0,0]);r=diagnose(*a)
 assert r['summary']['reference_cross_checks'][0]['residual_fraction']>.8
 assert r['summary']['rows'][0]['irreducible_fraction']>.8


def test_invalid_state_nonfinite_and_rank_rejected():
 a=list(fixture(np.zeros(5)));a[-1][0]['state']=a[-1][0]['state']+1e-9
 with pytest.raises(ValueError,match='literal'):diagnose(*a)
 a=list(fixture(np.ones(5)));a[1]=np.zeros_like(a[1])
 with pytest.raises(ValueError,match='nonzero'):diagnose(*a)
 a=list(fixture(np.ones(5)));a[1][0,0]=np.nan
 with pytest.raises(ValueError,match='finite'):diagnose(*a)
