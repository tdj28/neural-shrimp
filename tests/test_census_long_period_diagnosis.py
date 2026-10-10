import importlib.util
from pathlib import Path
import pytest
p=Path(__file__).parents[1]/'tools/diagnose_census_long_period.py';s=importlib.util.spec_from_file_location('diagnosis',p);D=importlib.util.module_from_spec(s);s.loader.exec_module(D)
def test_exact_dyadic_exponents_and_nonfinite_retention():
 assert D.exponent([3,-10])==-9 and D.exponent([-8,3])==6 and D.exponent([0,7])is None
 r=D.summary([{'mid':[1,0],'rad':[3,-20]},{'finite':False}]);assert r['finite_count']==1 and r['nonfinite_count']==1 and r['maximum_radius_floor_log2']==-19
 with pytest.raises(ValueError):D.summary([{'mid':[1,0],'rad':[-1,-10]}])
def test_full_failed_native_phase_diagnosis_keeps_missing_newton_history():
 ball=lambda e:{'mid':[1,0],'rad':[1,e]}
 r=dict(settings=dict(precision=512,radius_exponent=-160,newton_steps=5,max_power=4096),passed=False,center=[ball(-200)]*72,box=[ball(-160)]*72,orbit_enclosures=[[ball(-160+i*4)]*72 for i in range(59)],jacobians=[[[ball(-90)]]]*58,residual=[[ball(-300)]],preconditioner=[[ball(-200)]],divisors=[],power_bounds=[],contraction_norm_upper={'finite':False})
 r['orbit_enclosures'][42]=[{'finite':False}]*72;d=D.diagnose(r);assert d['first_nonfinite_state']==42 and d['first_radius_at_least_2_to_minus20']==35 and not d['newton_history_available']
