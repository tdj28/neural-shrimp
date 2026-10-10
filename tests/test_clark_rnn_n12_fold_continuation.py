import importlib.util
from pathlib import Path
import numpy as np
import pytest

def module():
 p=Path(__file__).parents[1]/'tools/continue_clark_rnn_n12_daughter_fold.py';s=importlib.util.spec_from_file_location('fold_continuation',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

@pytest.mark.parametrize('fold',[True,False])
def test_full_gain_column_against_central_difference(fold):
 m=module();j=np.eye(12)*.2+np.ones((12,12))*.025;z=np.r_[np.arange(1,13)/20,.8,np.eye(12)[0]]
 def call(g):return m.M.augmented(z,j,g,6,1)if fold else m.B.C.critical_fixed_gain(z,j,g,-1,3)
 h,b,a=call(2.);fd=(call(2.+1e-5)[0]-call(2.-1e-5)[0])/2e-5
 np.testing.assert_allclose(m.B.gain_derivative(a,z[13:]),fd,atol=1e-8,rtol=1e-6)

@pytest.mark.parametrize('index,value',[(12,.5),(0,1.),(13,-1.)])
def test_continued_state_rate_and_mode_guards(index,value):
 m=module();a=np.r_[np.zeros(12),.8,np.eye(12)[0]];z=a.copy();z[index]=value
 assert m.guard(a,a)and not m.guard(z,a)

def test_failed_candidate_is_saved_before_any_native_acquisition(tmp_path):
 m=module();w=m.Work(tmp_path);events=[];w.emit=lambda k,v:events.append((k,v));a=np.r_[np.zeros(12),.8,np.eye(12)[0]];bad=a.copy();bad[12]=.5
 with pytest.raises(ValueError,match='continued eigenvector'):w.correct(bad,a,np.eye(12),2.,'bad',True)
 assert w.calls==0 and len(events)==1 and np.array_equal(events[0][1]['physical'],bad)

def test_combined_derivative_quota_precedes_native_model(tmp_path):
 m=module();w=m.Work(tmp_path);w.augmented_calls=2559
 with pytest.raises(ValueError,match='2560'):w.aug(np.zeros(25),np.eye(12),2.,'bad')
 assert w.calls==0
 w.augmented_calls=2560
 with pytest.raises(ValueError,match='2560'):w.half(np.zeros(25),np.eye(12),2.,-1,'bad')
 assert w.calls==0

def test_comparison_preserves_literal_phase_and_signed_amplitude():
 m=module();q=np.eye(12)[0];p=np.arange(12,dtype=float);flip=np.r_[p,.8,q];fold=np.r_[p+.2*q,p-.2*q,.801];row=m.comparison(fold,flip,q)
 assert row['midpoint_to_signed_root']==0 and row['signed_amplitude']==.2
 assert abs(row['fold_minus_signed_leak']-.001)<1e-15
 shifted=flip.copy();shifted[:12]=-p;assert m.comparison(fold,shifted,q)['midpoint_to_signed_root']>1


def test_solve_quota_checked_before_linear_algebra(tmp_path):
 m=module();w=m.Work(tmp_path);w.solve_calls=1024
 with pytest.raises(ValueError,match='1024 linear solve'):w.solve(np.zeros((2,2)),np.zeros(2))
 assert w.solve_calls==1024
