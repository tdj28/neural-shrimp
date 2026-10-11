import numpy as np
import pytest
from neural_shrimp.cifar_saved_geometry import assess

def test_optimal_actual_line_and_endpoint_secant_are_exact():
 x=np.array([1.,2.,3.]);d=np.array([.01,0,0]);r=np.array([.003,.004,0]);p=np.array([1.,0,0]);ap=np.array([-.01,0,0]);z=np.r_[p,p];az=np.r_[p,p];row=assess(x,x+d,x+r,p,ap,z,az,.1,.01)
 assert row['line']['unconstrained_beta']==pytest.approx(.3)
 assert row['line']['best_predicted_forcing']==pytest.approx(.8)
 assert not row['line']['forcing_point5_reachable']
 assert row['cycle_geometry']['residual_parallel_to_phase_fraction']==pytest.approx(.36)
 assert row['cycle_geometry']['residual_transverse_fraction']==pytest.approx(.64)
 assert row['cycle_geometry']['secant_curvature']==pytest.approx(17)
 assert row['cycle_geometry']['curvature_to_threshold']==pytest.approx(.85)

def test_non_descent_is_not_candidate_and_old_direction_mismatch_rejected():
 x=np.ones(3);d=np.array([.01,0,0]);r=np.array([.003,.004,0]);p=np.ones(3);z=np.r_[p,p];az=np.arange(6.)
 row=assess(x,x+d,x+r,p,r,z,az,.1,.01);assert row['line']['bounded_beta']==0 and row['line']['best_predicted_forcing']==pytest.approx(1)
 with pytest.raises(ValueError):assess(x,x+d,x+r,p*2,r,z,az,.1,.01)
