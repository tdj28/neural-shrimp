import importlib.util
from pathlib import Path
import numpy as np
import pytest

def module():
 p=Path(__file__).parents[1]/'tools/diagnose_eigenpair_locus.py';s=importlib.util.spec_from_file_location('diag',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_nonnormal_row_bound_can_be_fixed_by_positive_weights():
 m=module();b=np.array([[.4,3.],[0.,.3]]);r=m.proposals(b,np.array([1e-8,1e-8]))
 assert r['all_ones_row_bound']>1 and abs(r['spectral_radius_estimate']-.4)<1e-12
 assert r['perron_weighted_row_bound']<1 and r['linear_radii_proposal']['maximum_ratio']<1
 assert 'requires_reacquisition' in r['linear_radii_proposal']

def test_neutral_and_expanding_proposals_do_not_become_proofs():
 m=module()
 for b in (np.eye(2),2*np.eye(2)):
  r=m.proposals(b,np.ones(2));assert r['spectral_radius_estimate']>=1 and r['linear_radii_proposal']is None

def test_invalid_and_empty_arrays_reject():
 m=module()
 for b,a in ((np.array([[np.nan]]),np.ones(1)),(np.array([[-1.]]),np.ones(1)),(np.eye(2),np.ones(3))):
  with pytest.raises(ValueError):m.proposals(b,a)

def test_zero_gauge_coordinate_gets_declared_positive_floor():
 m=module();b=np.array([[.25,.5],[0.,0.]])
 r=m.proposals(b,np.array([1e-8,0.]))
 assert r['linear_solve_return'][1]==0
 assert all(v>0 for v in r['linear_radii_proposal']['radii'])
 assert r['linear_radii_proposal']['maximum_ratio']<1
 # Negative solves are still rejected without clipping.
 bad=m.proposals(2*np.eye(2),np.ones(2))
 assert all(v<0 for v in bad['linear_solve_return']) and bad['linear_radii_proposal']is None
 with pytest.raises(ValueError):m.proposals(np.empty((0,0)),np.empty(0))
