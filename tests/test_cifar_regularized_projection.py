import importlib.util
from pathlib import Path
import numpy as np
import pytest
p=Path(__file__).resolve().parents[1]/'src/neural_shrimp/cifar_regularized_projection.py';spec=importlib.util.spec_from_file_location('regularized',p);H=importlib.util.module_from_spec(spec);spec.loader.exec_module(H)

def test_diagonal_discrepancy_and_shorter_physical_direction():
 V=np.eye(3);W=np.diag([1.,.1,.01]);g=np.array([.2,.3,1.]);r=H.propose(V,W,g)
 assert abs(r['summary']['predicted_forcing']-.49)<1e-12
 assert H.norm(g+W@r['coefficients'])<.5*H.norm(g)
 assert r['summary']['norm_ratio']<.7 and r['summary']['ridge']>0
 assert len(r['summary']['bisection'])==64

def test_actual_basis_metric_invariant_under_rescaling_and_mixing():
 V=np.eye(4)[:,:3];W=np.diag([1.,.3,.02,2.])@V;g=np.array([.2,.3,1.,.01]);S=np.array([[2.,.3,0.],[0.,.2,.1],[0.,0.,3.]])
 a=H.propose(V,W,g);b=H.propose(V@S,W@S,g)
 np.testing.assert_allclose(a['direction'],b['direction'],rtol=1e-10,atol=1e-11)
 np.testing.assert_allclose(a['predicted_action'],b['predicted_action'],rtol=1e-10,atol=1e-11)

def test_no_unsupported_action_span_or_nonfinite_candidate():
 with pytest.raises(ValueError,match='achievable'):H.propose(np.eye(2)[:,:1],np.eye(2)[:,:1],np.array([0.,1.]))
 with pytest.raises(ValueError,match='rank'):H.propose(np.eye(2),np.diag([1.,0.]),np.ones(2))
 with pytest.raises(ValueError):H.propose(np.eye(2),np.eye(2),np.array([np.nan,1.]))
