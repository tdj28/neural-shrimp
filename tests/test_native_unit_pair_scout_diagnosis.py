import sys
from pathlib import Path
import numpy as np
import pytest
sys.path[:0]=[str(Path(__file__).parents[1]/'tools'),str(Path(__file__).parents[1]/'src')]
from diagnose_native_unit_pair_scout import diagnostic

def test_ellipticity_and_full_rate_tangent():
 z=np.zeros(11);z[5]=1;z[9]=2;h=np.arange(11,dtype=float);J=np.diag(np.arange(1,12))
 r=diagnostic(z,.25,J,h);assert r['ellipticity_gauge_factor']==0
 np.testing.assert_allclose(r['tangent'],-h/np.arange(1,12));assert r['actual_rate_tangent']==r['tangent'][3:5]
 assert r['solve_relative_residual']<1e-15

def test_nonfinite_and_singular_diagnostics_do_not_make_event_proofs():
 with pytest.raises(ValueError):diagnostic(np.zeros(11),.1,np.ones((2,2)),np.ones(11))
 with pytest.raises(np.linalg.LinAlgError):diagnostic(np.zeros(11),.1,np.zeros((11,11)),np.ones(11))
