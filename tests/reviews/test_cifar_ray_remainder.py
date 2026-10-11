import importlib.util
from pathlib import Path
import numpy as np
import pytest
p=Path(__file__).resolve().parents[2]/'tools/reviews/cifar_ray_remainder.py'
s=importlib.util.spec_from_file_location('R',p);R=importlib.util.module_from_spec(s);s.loader.exec_module(R)

def test_exact_quadratic_and_all_terms():
 g=np.array([2.,-1.]);h=np.array([-.5,.2]);k=np.array([1.,3.])
 report,arrays=R.diagnose(g,h,[g+a*h+a*a*k for a in R.ALPHAS])
 assert report['quadratic_sufficiency_qualified']
 assert np.max(abs(arrays['coefficient']-k))<1e-12
 assert all(z['quadratic_improves_full_gradient']for z in report['rows'])
 assert [z['role']for z in report['rows']]==['check','fit','fit','check','check']

def test_cubic_extrapolation_failure_is_preserved():
 g=np.array([1.,1.]);h=np.array([.2,-.1]);k=np.array([500.,-300.])
 report,_=R.diagnose(g,h,[g+a*h+a**3*k for a in R.ALPHAS])
 assert not report['quadratic_sufficiency_qualified']
 assert report['rows'][0]['remainder_relative_error']>.1

def test_zero_remainder_abstains_and_nonfinite_rejects():
 g=np.array([1.,2.]);h=np.array([.5,-.25]);ys=[g+a*h for a in R.ALPHAS]
 report,_=R.diagnose(g,h,ys)
 assert not report['quadratic_sufficiency_qualified'] and all(z['remainder_relative_error']is None for z in report['rows'])
 ys[2]=np.array([np.nan,1.])
 with pytest.raises(AssertionError):R.diagnose(g,h,ys)

def test_fixed_fit_does_not_use_check_vectors():
 g=np.array([1.,2.]);h=np.array([.5,-.25]);ys=[g+a*h+a*a*np.array([2.,3.])for a in R.ALPHAS]
 r,a=R.diagnose(g,h,ys);ys[0]=ys[0]+100
 r2,a2=R.diagnose(g,h,ys)
 assert np.array_equal(a['coefficient'],a2['coefficient']) and not r2['quadratic_sufficiency_qualified']
