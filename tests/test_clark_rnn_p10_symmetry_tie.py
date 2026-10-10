import importlib.util
from pathlib import Path
import numpy as np
s=importlib.util.spec_from_file_location('p10tie',Path(__file__).parents[1]/'tools/extend_clark_rnn_new_p10_slices.py');M=importlib.util.module_from_spec(s);s.loader.exec_module(M)
def cycle():
 a=np.arange(1,21,dtype=float).reshape(5,4)/20
 return np.r_[a,-a]
def test_exact_odd_half_period_tie_keeps_native_states():
 a=cycle();b=a.copy();r=M.phase_match(a,b)
 assert not r['original_argmin_passed'] and r['passed'] and r['antipodal_half_period_tie']
 assert r['native_error']==0 and not r['state_transform_applied']
 assert all(x>1e-6 for x in M.B.primitive_separations(a).values())
def test_small_native_step_with_symmetry_is_allowed():
 a=cycle();b=a*1.001;r=M.phase_match(a,b)
 assert r['passed'] and r['native_error']<.05
def test_phase_shift_is_not_quotiented_to_native_identity():
 a=cycle();r=M.phase_match(a,np.roll(a,1,axis=0))
 assert not r['passed'] and r['native_error']>.05
def test_odd_companion_is_not_silently_transformed():
 a=cycle();r=M.phase_match(a,-a)
 assert not r['passed'] and not r['state_transform_applied']
def test_approximate_odd_symmetry_outside_tolerance_rejected():
 a=cycle();b=a.copy();b[0,0]+=1e-6
 r=M.phase_match(a,b)
 assert not r['antipodal_half_period_tie']
def test_nonprimitive_two_cycle_is_still_rejected_by_native_divisors():
 a=np.tile([[1.,2.,3.,4.],[-1.,-2.,-3.,-4.]],(5,1))
 assert M.B.primitive_separations(a)['2']==0
