import numpy as np
import pytest
from neural_shrimp.cifar_recycled import fit,actual_gate

def test_exact_three_column_fit_and_saved_triplets():
 p=np.diag([2.,3.,4.]);a=np.diag([4.,9.,16.]);r=np.array([1.,2.,3.]);events={};f,d=fit(p,a,r,lambda k,v:events.update({k:v}));assert d['qualified'];np.testing.assert_allclose(f['predicted_action'],-r,atol=1e-15);np.testing.assert_allclose(a@f['original_coefficients'],-r,atol=1e-15);assert set(events)=={'scales','normalized','svd','svd_checks','rank','fit'}
 g=actual_gate(f['predicted_action'],f['predicted_action'],r);assert g['qualified']and g['true_forcing_ratio']<1e-14

def test_rank_failure_retains_full_svd_and_abstains():
 p=np.eye(3);a=np.ones((3,3));events={};f,d=fit(p,a,np.ones(3),lambda k,v:events.update({k:v}));assert f is None and not d['qualified']and 'svd'in events

def test_condition_gate_is_separate_from_rank():
 f,d=fit(np.eye(3),np.diag([1.,1.,1e-9]),np.ones(3));assert f is None and d['rank']==3 and d['condition']>1e8

def test_true_action_not_predicted_gate_controls_acceptance():
 r=np.ones(3);g=actual_gate(-.4*r,-r,r);assert not g['qualified']and not g['forcing_test_pass']and not g['action_linearity_pass']
 g=actual_gate(.1*r,.1*r,r);assert not g['qualified']and not g['descent_direction']

def test_nonfinite_and_zero_columns_fail():
 for a in(np.full((3,3),np.nan),np.zeros((3,3))):
  with pytest.raises(ValueError):fit(a,np.eye(3),np.ones(3))

def test_mathematical_span_is_invariant_to_direction_rescaling():
 p=np.eye(3);a=np.array([[2.,.2,.1],[.3,3.,.4],[.1,.5,4.]]);r=np.array([1.,2.,3.]);f,_=fit(p,a,r);g,_=fit(p*np.array([1e-4,2.,-10.]),a*np.array([1e-4,2.,-10.]),r);np.testing.assert_allclose(f['direction'],g['direction'],rtol=1e-13,atol=1e-13)


def test_wrong_saved_svd_fails_before_fit(monkeypatch):
 monkeypatch.setattr(np.linalg,'svd',lambda *a,**k:(np.eye(3),np.ones(3),np.eye(3)))
 with pytest.raises(ValueError,match='SVD residual'):fit(np.eye(3),np.eye(3)*2,np.ones(3))
